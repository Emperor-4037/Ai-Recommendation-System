"""
End-to-End Evaluation Pipeline for Sync and Spark Modes.

Loads saved model checkpoints and evaluates against a proper test set
where each user's positive items are ranked against a large pool of
negative candidates (not just the items they interacted with).
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import faiss
from collections import defaultdict
from torch.utils.data import DataLoader

from recsys.config import get_settings
from recsys.features.dataset import ReciprocalDataset
from recsys.models.retrieval import TwoTowerRetrieval
from recsys.models.scoring import ReciprocalMultitaskRanker, Calibrator
from recsys.models.reranker import PairwiseSparkReranker
from recsys.evaluation.offline import ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)


def load_models(device: str = "cuda"):
    """Load all saved model artifacts from disk."""
    settings = get_settings()
    model_dir = settings.model_path

    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No model_config.json found in {model_dir}. Run training first.")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Reconstruct model architectures from saved config
    h = cfg["hidden_sizes"]
    retrieval_model = TwoTowerRetrieval(
        cfg["user_num_dim"], cfg["user_cat_vocab"],
        cfg["cand_num_dim"], cfg["cand_cat_vocab"],
        embedding_dim=cfg["embedding_dim"], hidden_sizes=h,
        dropout=cfg["dropout"],
        graph_dim=cfg["graph_dim"],
        seq_dim=cfg["seq_dim"], seq_hidden=cfg["seq_dim"]
    ).to(device)

    scorer_model = ReciprocalMultitaskRanker(
        input_dim=cfg["ranker_input_dim"],
        hidden_sizes=h,
        dropout=cfg["dropout"]
    ).to(device)

    reranker_model = PairwiseSparkReranker(
        input_dim=cfg["reranker_input_dim"]
    ).to(device)

    retrieval_model.load_state_dict(torch.load(os.path.join(model_dir, "retrieval.pth"), map_location=device, weights_only=True))
    scorer_model.load_state_dict(torch.load(os.path.join(model_dir, "scorer.pth"), map_location=device, weights_only=True))
    reranker_model.load_state_dict(torch.load(os.path.join(model_dir, "reranker.pth"), map_location=device, weights_only=True))

    calibrator = joblib.load(os.path.join(model_dir, "calibrator.joblib"))

    retrieval_model.eval()
    scorer_model.eval()
    reranker_model.eval()

    logger.info("All models loaded from disk.")
    return retrieval_model, scorer_model, reranker_model, calibrator, cfg


def build_candidate_pool(retrieval_model, val_dataset, val_loader, device):
    """
    Build a FAISS index of ALL unique candidate embeddings from the validation set.
    Also returns a mapping from sequential FAISS ID -> (candidate row index, cand_id).
    """
    logger.info("Building candidate embedding pool...")
    cand_embeddings = []
    cand_meta = []  # list of (row_idx_in_dataset, cand_id)
    seen_cands = set()

    retrieval_model.eval()
    row_offset = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_dev[k] = {col: (tensor.to(device, non_blocking=True) if hasattr(tensor, 'to') else tensor) for col, tensor in v.items()}
                elif hasattr(v, 'to'):
                    batch_dev[k] = v.to(device, non_blocking=True)
                else:
                    batch_dev[k] = v

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                c_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])

            c_emb_np = c_emb.float().cpu().numpy()
            cands = batch_dev['cand_id']
            if isinstance(cands, torch.Tensor):
                cands = cands.cpu().numpy()

            bs = len(c_emb_np)
            for i in range(bs):
                cid = cands[i]
                cid_key = str(cid)
                if cid_key not in seen_cands:
                    seen_cands.add(cid_key)
                    cand_embeddings.append(c_emb_np[i])
                    cand_meta.append((row_offset + i, cid))

            row_offset += bs

    cand_embeddings = np.vstack(cand_embeddings).astype(np.float32)
    faiss.normalize_L2(cand_embeddings)

    index = faiss.IndexFlatIP(cand_embeddings.shape[1])  # Inner product after L2 norm = cosine sim
    index.add(cand_embeddings)

    logger.info(f"Candidate pool built: {len(cand_meta)} unique candidates, embedding dim={cand_embeddings.shape[1]}")
    return index, cand_embeddings, cand_meta


def evaluate_end_to_end(dataset_name: str, max_users: int = 5000, k: int = 10, retrieval_k: int = 200):
    """
    End-to-end evaluation pipeline for Sync and Spark modes using a 1+99 
    Sampled Negatives protocol.
    
    For each sampled user:
      1. Pool: Take their positive items, and pad up to 100 with random negative items.
      2. Sync Mode: Score the 100 items with Multitask Ranker, take top-k.
      3. Spark Mode: Rerank the 100 items with the Spark Reranker, fuse scores, take top-k.
    """
    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrieval_model, scorer_model, reranker_model, calibrator, cfg = load_models(device)

    logger.info("Loading validation dataset...")
    val_dataset = ReciprocalDataset(dataset_name, split='val')
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=0)

    logger.info("Building per-user ground truth and extracting candidate embeddings...")
    user_positives = defaultdict(set)
    user_all_items = defaultdict(set)
    user_row_indices = defaultdict(list)
    
    cand_id_to_row = {}

    for idx in range(len(val_dataset)):
        uid = str(val_dataset.user_ids[idx])
        cid = str(val_dataset.cand_ids[idx])
        label = val_dataset.mutual_labels[idx]
        
        user_all_items[uid].add(cid)
        if label > 0.5:
            user_positives[uid].add(cid)
        user_row_indices[uid].append(idx)
        
        if cid not in cand_id_to_row:
            cand_id_to_row[cid] = idx

    all_cids = list(cand_id_to_row.keys())
    eligible_users = [u for u in user_positives if len(user_all_items[u]) >= 3]
    logger.info(f"Total users: {len(user_all_items)}, Users with positives: {len(eligible_users)}")

    if len(eligible_users) == 0:
        logger.error("No eligible users found for evaluation!")
        return {}

    rng = np.random.RandomState(42)
    sampled_users = rng.choice(eligible_users, size=min(max_users, len(eligible_users)), replace=False)

    sync_ndcg_scores, sync_recall_scores, sync_mmr_scores = [], [], []
    spark_ndcg_scores, spark_recall_scores, spark_diversity_scores = [], [], []
    
    processed = 0

    for u_idx, uid in enumerate(sampled_users):
        if (u_idx + 1) % 500 == 0:
            logger.info(f"Evaluating user {u_idx+1}/{len(sampled_users)}...")

        positives = list(user_positives[uid])
        # Sample negatives
        negatives = []
        while len(negatives) < 100 - len(positives):
            c = rng.choice(all_cids)
            if c not in user_all_items[uid]:
                negatives.append(c)
                
        candidate_pool_cids = positives + negatives
        num_cands = len(candidate_pool_cids)
        
        # Build batch for these candidates
        user_row = user_row_indices[uid][0]
        user_sample = val_dataset[user_row]
        
        # Expand user features
        user_num = user_sample['user_num'].unsqueeze(0).expand(num_cands, -1).to(device)
        user_cat = {k: v.unsqueeze(0).expand(num_cands).to(device) for k, v in user_sample['user_cat'].items()}
        graph_feat = user_sample['graph_feat'].unsqueeze(0).expand(num_cands, -1).to(device)
        seq_feat = user_sample['sequence'].unsqueeze(0).expand(num_cands, -1, -1).to(device)
        
        # Build candidate features
        cand_nums = []
        cand_cats = defaultdict(list)
        for cid in candidate_pool_cids:
            crow = cand_id_to_row[cid]
            csample = val_dataset[crow]
            cand_nums.append(csample['cand_num'])
            for k, v in csample['cand_cat'].items():
                cand_cats[k].append(v)
                
        cand_num_batch = torch.stack(cand_nums).to(device)
        cand_cat_batch = {k: torch.stack(v).to(device) for k, v in cand_cats.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                user_emb = retrieval_model.user_tower(user_num, user_cat, graph_feat, seq_feat)
                cand_emb = retrieval_model.cand_tower(cand_num_batch, cand_cat_batch)
                
                seq_val = seq_feat[:, -1, :] if seq_feat.dim() == 3 else seq_feat
                combined_feat = torch.cat([user_emb, cand_emb, graph_feat, seq_val], dim=1)
                
                preds = scorer_model(combined_feat)
                mutual_scores = torch.sigmoid(preds['mutual']).float().cpu().numpy().flatten()
                
                spark_scores_raw = reranker_model(user_emb, cand_emb).float().cpu().numpy().flatten()

        # === SYNC MODE ===
        sync_ranked_indices = np.argsort(-mutual_scores)
        sync_ranked_cids = [candidate_pool_cids[i] for i in sync_ranked_indices]

        sync_ndcg = ndcg_at_k(positives, sync_ranked_cids, k)
        sync_recall = recall_at_k(positives, sync_ranked_cids, k)
        sync_mmr = len([c for c in sync_ranked_cids[:k] if c in positives]) / k

        sync_ndcg_scores.append(sync_ndcg)
        sync_recall_scores.append(sync_recall)
        sync_mmr_scores.append(sync_mmr)

        # === SPARK MODE ===
        spark_min, spark_max = spark_scores_raw.min(), spark_scores_raw.max()
        spark_scores_norm = (spark_scores_raw - spark_min) / (spark_max - spark_min + 1e-8)
        
        fused_scores = (1 - settings.lambda_spark) * mutual_scores + settings.lambda_spark * spark_scores_norm
        
        spark_ranked_indices = np.argsort(-fused_scores)
        spark_ranked_cids = [candidate_pool_cids[i] for i in spark_ranked_indices]
        
        spark_ndcg = ndcg_at_k(positives, spark_ranked_cids, k)
        spark_recall = recall_at_k(positives, spark_ranked_cids, k)
        
        # Diversity
        cand_emb_np = cand_emb.float().cpu().numpy()
        topk_embs = cand_emb_np[[spark_ranked_indices[i] for i in range(min(k, 100))]]
        if len(topk_embs) >= 2:
            norms = np.linalg.norm(topk_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = topk_embs / norms
            sim_matrix = normed @ normed.T
            n = len(sim_matrix)
            total_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
            diversity = 1.0 - total_sim
        else:
            diversity = 0.0

        spark_ndcg_scores.append(spark_ndcg)
        spark_recall_scores.append(spark_recall)
        spark_diversity_scores.append(diversity)

        processed += 1

    results = {
        "num_users_evaluated": processed,
        "k": k,
        "sync": {
            "NDCG@k": float(np.mean(sync_ndcg_scores)) if sync_ndcg_scores else 0.0,
            "Recall@k": float(np.mean(sync_recall_scores)) if sync_recall_scores else 0.0,
            "Mutual_Match_Rate@k": float(np.mean(sync_mmr_scores)) if sync_mmr_scores else 0.0,
        },
        "spark": {
            "NDCG@k": float(np.mean(spark_ndcg_scores)) if spark_ndcg_scores else 0.0,
            "Recall@k": float(np.mean(spark_recall_scores)) if spark_recall_scores else 0.0,
            "Intra_List_Diversity": float(np.mean(spark_diversity_scores)) if spark_diversity_scores else 0.0,
        }
    }

    print("\n" + "=" * 70)
    print("           END-TO-END MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"  Users evaluated:       {results['num_users_evaluated']}")
    print(f"  k (top-k cutoff):      {results['k']}")
    print(f"  Evaluation Protocol:   1 Positive + 99 Sampled Negatives")
    print()
    print("-" * 70)
    print("  SYNC MODE  (Relevance-focused ranking)")
    print("-" * 70)
    print(f"    NDCG@{k}:              {results['sync']['NDCG@k']:.4f}")
    print(f"    Recall@{k}:            {results['sync']['Recall@k']:.4f}")
    print(f"    Mutual Match Rate@{k}: {results['sync']['Mutual_Match_Rate@k']:.4f}")
    print()
    print("-" * 70)
    print("  SPARK MODE  (Diversity-enhanced reranking)")
    print("-" * 70)
    print(f"    NDCG@{k}:              {results['spark']['NDCG@k']:.4f}")
    print(f"    Recall@{k}:            {results['spark']['Recall@k']:.4f}")
    print(f"    Intra-List Diversity:  {results['spark']['Intra_List_Diversity']:.4f}")
    print()
    print("-" * 70)
    print("  COMPARISON")
    print("-" * 70)
    print(f"    NDCG  change (Spark vs Sync):   {results['spark']['NDCG@k'] - results['sync']['NDCG@k']:+.4f}")
    print(f"    Recall change (Spark vs Sync):  {results['spark']['Recall@k'] - results['sync']['Recall@k']:+.4f}")
    print(f"    Diversity gain (Spark):          {results['spark']['Intra_List_Diversity']:.4f}")
    print("=" * 70)
    print()

    return results
