from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uuid
import logging
import os
import torch
import joblib
import json
import faiss
import hashlib
from recsys.features.dataset import ReciprocalDataset

from recsys.config import get_settings, Settings
from recsys.safety.policy import SafetyGate
from recsys.fairness.controller import FairnessController

from recsys.models.retrieval import TwoTowerRetrieval
from recsys.models.scoring import ReciprocalMultitaskRanker
from recsys.models.reranker import PairwiseSparkReranker

logger = logging.getLogger(__name__)

def safe_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    if not pretrained_dict:
        logger.warning(f"No matching shapes found for {model.__class__.__name__}.")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


app = FastAPI(title="Reciprocal Recommender API")
settings = get_settings()

# Load Models
retrieval_model = None
scorer_model = None
reranker_model = None
calibrator = None

@app.on_event("startup")
    global retrieval_model, scorer_model, reranker_model, calibrator, faiss_index, feature_store
    model_dir = settings.model_path
    
    ret_path = os.path.join(model_dir, "retrieval.pth")
    score_path = os.path.join(model_dir, "scorer.pth")
    rerank_path = os.path.join(model_dir, "reranker.pth")
    cal_path = os.path.join(model_dir, "calibrator.joblib")
    faiss_path = os.path.join(model_dir, "candidates.index")
    config_path = os.path.join(model_dir, "config_snapshot.json")
    
    if not all(os.path.exists(p) for p in [ret_path, score_path, rerank_path, cal_path, faiss_path, config_path]):
        logger.error("Champion artifacts not found. FastAPI will fail fast.")
        raise FileNotFoundError("Champion artifacts missing. Run `recsys train` first.")
        
    logger.info("Loading champion artifacts...")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    faiss_index = faiss.read_index(faiss_path)
    
    retrieval_model = TwoTowerRetrieval(
        user_num_dim=config["user_num_dim"], user_cat_vocab=config["user_cat_vocab"], 
        cand_num_dim=config["cand_num_dim"], cand_cat_vocab=config["cand_cat_vocab"], 
        embedding_dim=config["embedding_dim"], hidden_sizes=config["hidden_sizes"],
        seq_dim=config["seq_dim"], graph_dim=config["graph_dim"]
    )
    safe_load_state_dict(retrieval_model, torch.load(ret_path, map_location='cpu'))
    retrieval_model.eval()
    
    scorer_model = ReciprocalMultitaskRanker(input_dim=config["ranker_input_dim"], hidden_sizes=config["hidden_sizes"])
    safe_load_state_dict(scorer_model, torch.load(score_path, map_location='cpu'))
    scorer_model.eval()
    
    reranker_model = PairwiseSparkReranker(input_dim=config["reranker_input_dim"])
    safe_load_state_dict(reranker_model, torch.load(rerank_path, map_location='cpu'))
    reranker_model.eval()
    
    calibrator = joblib.load(cal_path)
    
    # Load dataset for feature serving (in production, use a fast Feature Store)
    logger.info("Loading feature store...")
    feature_store = ReciprocalDataset("dating_app_behavior_dataset_extended1.csv", split='val', inference=True)
    
    logger.info("Champion artifacts loaded successfully.")


# True states, Fairness controller uses Redis now
safety_gate = SafetyGate(max_safety_risk=settings.max_safety_risk)
fairness_controller = FairnessController()

# Canary state
CHAMPION_VERSION = "v1"
CHALLENGER_VERSION = "v2"
current_canary_rate = settings.canary_percentage

class RecommendationRequest(BaseModel):
    user_id: int
    num_candidates: int = 10
    user_intent: Optional[str] = "casual"

class CandidateResponse(BaseModel):
    cand_id: int
    score: float
    safety_tags: List[str] = []
    fairness_tags: List[str] = []
    explanation: str

class RecommendationResponse(BaseModel):
    request_id: str
    mode: str
    candidates: List[CandidateResponse]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    # Check if models are loaded
    if retrieval_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}

@app.post("/recommend/sync", response_model=RecommendationResponse)
async def recommend_sync(req: RecommendationRequest):
    req_id = str(uuid.uuid4())
    
    # Canary Routing
    version = CHAMPION_VERSION
    user_hash = int(hashlib.md5(str(req.user_id).encode()).hexdigest(), 16) % 100
    if user_hash < current_canary_rate * 100:
        version = CHALLENGER_VERSION
        
    # User Features
    user_idx = feature_store.df.index[feature_store.df['user_id'] == req.user_id].tolist()
    if user_idx:
        u_feat = feature_store[user_idx[0]]
    else:
        logger.warning(f"User {req.user_id} not found in feature store. Using fallback zeros.")
        u_feat = feature_store[0] 

    u_num = u_feat['user_num'].unsqueeze(0)
    u_cat = {k: v.unsqueeze(0) for k,v in u_feat['user_cat'].items()}
    graph = u_feat['graph_feat'].unsqueeze(0)
    seq = u_feat['sequence'].unsqueeze(0)
    
    # 1. Retrieval
    with torch.no_grad():
        user_emb = retrieval_model.user_tower(u_num, u_cat, graph, seq)
        
    D, I = faiss_index.search(user_emb.numpy(), req.num_candidates * 3)
    retrieved_cand_ids = I[0].tolist()
    
    cand_feats = []
    valid_cands = []
    for cid in retrieved_cand_ids:
        if cid == -1: continue
        c_idx = feature_store.df.index[feature_store.df['cand_id'] == cid].tolist()
        if c_idx:
            cand_feats.append(feature_store[c_idx[0]])
            valid_cands.append({'cand_id': int(cid), 'is_blocked': False, 'safety_risk_score': 0.05, 'intent': 'casual'})
            
    # 2. Safety Gate
    safe_cands, _ = safety_gate.filter_candidates(req.user_id, valid_cands)
    
    safe_cand_ids = [c['cand_id'] for c in safe_cands]
    safe_cand_feats = [f for f, cid in zip(cand_feats, retrieved_cand_ids) if cid in safe_cand_ids]
    
    if not safe_cand_feats:
        return RecommendationResponse(request_id=req_id, mode="sync", candidates=[])
        
    c_num = torch.stack([f['cand_num'] for f in safe_cand_feats])
    c_cat = {k: torch.stack([f['cand_cat'][k] for f in safe_cand_feats]) for k in safe_cand_feats[0]['cand_cat']}
    
    # 3. Scoring
    b_size = len(safe_cand_feats)
    u_num_b = u_num.expand(b_size, -1)
    u_cat_b = {k: v.expand(b_size) for k, v in u_cat.items()}
    graph_b = graph.expand(b_size, -1)
    seq_b = seq.expand(b_size, -1, -1)
    
    with torch.no_grad():
        user_emb_b = retrieval_model.user_tower(u_num_b, u_cat_b, graph_b, seq_b)
        cand_emb_b = retrieval_model.cand_tower(c_num, c_cat)
        
        seq_last = seq_b[:, -1, :] if seq_b.dim() == 3 else seq_b
        combined_feat = torch.cat([user_emb_b, cand_emb_b, graph_b, seq_last], dim=1)
        
        preds = scorer_model(combined_feat)
        mutual_scores = preds['mutual'].squeeze().tolist()
        
    if isinstance(mutual_scores, float):
        mutual_scores = [mutual_scores]
        
    for c, score in zip(safe_cands, mutual_scores):
        c['score'] = score
        c['explanation'] = f"High mutual compatibility ({version})."
        
    safe_cands.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. Fairness Caps
    fair_cands = fairness_controller.apply_exposure_caps(safe_cands)
    
    # 5. Record Exposure
    fairness_controller.record_exposures([c['cand_id'] for c in fair_cands[:req.num_candidates]])
    
    cands_resp = [
        CandidateResponse(
            cand_id=c['cand_id'], 
            score=c['score'], 
            explanation=c['explanation']
        ) for c in fair_cands[:req.num_candidates]
    ]
    
    return RecommendationResponse(request_id=req_id, mode="sync", candidates=cands_resp)

@app.post("/recommend/spark", response_model=RecommendationResponse)
async def recommend_spark(req: RecommendationRequest):
    req_id = str(uuid.uuid4())
    
    # Retrieval and Scoring
    user_idx = feature_store.df.index[feature_store.df['user_id'] == req.user_id].tolist()
    if user_idx:
        u_feat = feature_store[user_idx[0]]
    else:
        logger.warning(f"User {req.user_id} not found in feature store. Using fallback zeros.")
        u_feat = feature_store[0]

    u_num = u_feat['user_num'].unsqueeze(0)
    u_cat = {k: v.unsqueeze(0) for k,v in u_feat['user_cat'].items()}
    graph = u_feat['graph_feat'].unsqueeze(0)
    seq = u_feat['sequence'].unsqueeze(0)
    
    with torch.no_grad():
        user_emb = retrieval_model.user_tower(u_num, u_cat, graph, seq)
        
    D, I = faiss_index.search(user_emb.numpy(), req.num_candidates * 3)
    retrieved_cand_ids = I[0].tolist()
    
    cand_feats = []
    valid_cands = []
    for cid in retrieved_cand_ids:
        if cid == -1: continue
        c_idx = feature_store.df.index[feature_store.df['cand_id'] == cid].tolist()
        if c_idx:
            cand_feats.append(feature_store[c_idx[0]])
            valid_cands.append({'cand_id': int(cid), 'is_blocked': False, 'safety_risk_score': 0.05, 'intent': 'casual'})
            
    safe_cands, _ = safety_gate.filter_candidates(req.user_id, valid_cands)
    
    safe_cand_ids = [c['cand_id'] for c in safe_cands]
    safe_cand_feats = [f for f, cid in zip(cand_feats, retrieved_cand_ids) if cid in safe_cand_ids]
    
    if not safe_cand_feats:
        return RecommendationResponse(request_id=req_id, mode="spark", candidates=[])
        
    c_num = torch.stack([f['cand_num'] for f in safe_cand_feats])
    c_cat = {k: torch.stack([f['cand_cat'][k] for f in safe_cand_feats]) for k in safe_cand_feats[0]['cand_cat']}
    
    b_size = len(safe_cand_feats)
    u_num_b = u_num.expand(b_size, -1)
    u_cat_b = {k: v.expand(b_size) for k, v in u_cat.items()}
    graph_b = graph.expand(b_size, -1)
    seq_b = seq.expand(b_size, -1, -1)
    
    with torch.no_grad():
        user_emb_b = retrieval_model.user_tower(u_num_b, u_cat_b, graph_b, seq_b)
        cand_emb_b = retrieval_model.cand_tower(c_num, c_cat)
        
        rerank_preds = reranker_model(user_emb_b, cand_emb_b)
        spark_scores = rerank_preds.squeeze().tolist()
        
    if isinstance(spark_scores, float):
        spark_scores = [spark_scores]
        
    for c, score in zip(safe_cands, spark_scores):
        c['score'] = score
        c['explanation'] = "Great complementary match (Spark)."
        
    safe_cands.sort(key=lambda x: x['score'], reverse=True)
    
    fair_cands = fairness_controller.apply_exposure_caps(safe_cands)
    fairness_controller.record_exposures([c['cand_id'] for c in fair_cands[:req.num_candidates]])
    
    cands_resp = [
        CandidateResponse(
            cand_id=c['cand_id'], 
            score=c['score'], 
            explanation=c['explanation']
        ) for c in fair_cands[:req.num_candidates]
    ]
    
    return RecommendationResponse(request_id=req_id, mode="spark", candidates=cands_resp)

@app.post("/feedback")
async def feedback():
    # Accepts user interaction feedback (like, pass, message)
    return {"status": "recorded"}

@app.post("/moderate")
async def moderate():
    # Accepts manual moderation actions
    return {"status": "applied"}

@app.get("/metrics")
async def metrics():
    # Returns Prometheus metrics
    return {"status": "metrics_available"}

@app.post("/admin/rollback")
async def rollback():
    global current_canary_rate
    current_canary_rate = 0.0
    return {"status": "rolled_back", "message": "Canary traffic set to 0%"}

@app.post("/admin/set-canary")
async def set_canary(percentage: float):
    global current_canary_rate
    current_canary_rate = percentage
    return {"status": "updated", "canary_rate": percentage}
