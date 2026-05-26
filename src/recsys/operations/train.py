import os
import torch
import torch.nn as nn
import logging
import joblib
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any
import torch.nn.functional as F

from recsys.config import get_settings
from recsys.features.dataset import ReciprocalDataset
from recsys.models.retrieval import TwoTowerRetrieval
from recsys.models.scoring import ReciprocalMultitaskRanker, Calibrator
from recsys.models.debias import DebiasedLoss, PropensityScorer
from recsys.models.reranker import PairwiseSparkReranker
import faiss

logger = logging.getLogger(__name__)

def train_pipeline(dataset_name: str, params: Dict[str, Any] = None, trial=None):
    """
    Executes the full training loop for Phase 2 models.
    Accepts an optional Optuna trial for intermediate reporting & pruning.
    """
    settings = get_settings()
    if params is None:
        params = {}
        
    lr = params.get('learning_rate', 1e-3)
    weight_decay = params.get('weight_decay', 1e-4)
    epochs = params.get('epochs', 20)
    batch_size = params.get('batch_size', 256)
    
    emb_dim = params.get('embedding_dim', settings.retrieval_embedding_dim)
    hidden_size = params.get('hidden_size', settings.retrieval_hidden_sizes[0])
    dropout = params.get('dropout', settings.scorer_dropout)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for training.")
    
    # 1. Load Data
    train_dataset = ReciprocalDataset(dataset_name, split='train')
    val_dataset = ReciprocalDataset(dataset_name, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    
    # 2. Initialize Models
    user_num_dim = train_dataset.user_features_num.shape[1]
    cand_num_dim = train_dataset.cand_features_num.shape[1]
    
    user_cat_vocab = {k: len(v) for k, v in train_dataset.vocabularies.items() if not (k.startswith('target_') or k.startswith('cand_'))}
    cand_cat_vocab = {k: len(v) for k, v in train_dataset.vocabularies.items() if (k.startswith('target_') or k.startswith('cand_'))}
    
    # Ensure hidden sizes decrease
    h2 = max(hidden_size // 2, 32)
    h3 = max(h2 // 2, 16)
    tower_hidden_sizes = [hidden_size, h2, h3]
    
    retrieval_model = TwoTowerRetrieval(
        user_num_dim, user_cat_vocab, cand_num_dim, cand_cat_vocab,
        embedding_dim=emb_dim, hidden_sizes=tower_hidden_sizes,
        dropout=dropout, graph_dim=settings.graph_embedding_dim,
        seq_dim=settings.sequence_hidden_dim, seq_hidden=settings.sequence_hidden_dim
    ).to(device)
    
    # Input dim to ranker: user + cand emb from retrieval + graph + sequence
    ranker_input_dim = h3 + h3 + settings.graph_embedding_dim + settings.sequence_hidden_dim 
    scorer_model = ReciprocalMultitaskRanker(input_dim=ranker_input_dim, hidden_sizes=[hidden_size, h2, h3], dropout=dropout).to(device)
    
    # Reranker takes user_emb and cand_emb only
    reranker_input_dim = h3 + h3
    reranker_model = PairwiseSparkReranker(input_dim=reranker_input_dim).to(device)
    
    # 3. Optimizers & Losses
    opt_retrieval = torch.optim.AdamW(retrieval_model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_scorer = torch.optim.AdamW(scorer_model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_reranker = torch.optim.AdamW(reranker_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    sched_retrieval = torch.optim.lr_scheduler.CosineAnnealingLR(opt_retrieval, T_max=epochs)
    sched_scorer = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scorer, T_max=epochs)
    sched_reranker = torch.optim.lr_scheduler.CosineAnnealingLR(opt_reranker, T_max=epochs)
    
    debiased_loss = DebiasedLoss(clipping=settings.propensity_clipping)
    mse_loss = nn.MSELoss()
    
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    best_retrieval_state = None
    best_scorer_state = None
    best_reranker_state = None
    patience = 3
    patience_counter = 0
    
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        retrieval_model.train()
        scorer_model.train()
        reranker_model.train()
        
        for batch in train_loader:
            opt_retrieval.zero_grad()
            opt_scorer.zero_grad()
            opt_reranker.zero_grad()
            
            # Move to device
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_dev[k] = {col: (tensor.to(device, non_blocking=True) if hasattr(tensor, 'to') else tensor) for col, tensor in v.items()}
                elif hasattr(v, 'to'):
                    batch_dev[k] = v.to(device, non_blocking=True)
                else:
                    batch_dev[k] = v
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Forward retrieval to get embeddings
                user_emb = retrieval_model.user_tower(batch_dev['user_num'], batch_dev['user_cat'], batch_dev.get('graph_feat'), batch_dev.get('sequence'))
                cand_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])
                
                # Retrieval Loss
                scores = torch.matmul(user_emb, cand_emb.T)
                loss_retrieval = -F.log_softmax(scores / 0.1, dim=1).diag().mean()
                
            scaler.scale(loss_retrieval).backward()
            scaler.unscale_(opt_retrieval)
            torch.nn.utils.clip_grad_norm_(retrieval_model.parameters(), max_norm=1.0)
            scaler.step(opt_retrieval)
            
            user_emb_det = user_emb.detach()
            cand_emb_det = cand_emb.detach()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                seq_feat = batch_dev.get('sequence')
                seq_val = seq_feat[:, -1, :] if (seq_feat is not None and seq_feat.dim() == 3) else seq_feat
                combined_feat = torch.cat([user_emb_det, cand_emb_det, batch_dev.get('graph_feat'), seq_val], dim=1)
                
                # Scorer training
                preds = scorer_model(combined_feat)
                
                # Reranker training
                reranker_preds = reranker_model(user_emb_det, cand_emb_det)

                propensity = batch_dev['propensity'].view(-1, 1).float()
                mutual_labels = batch_dev['mutual_label'].view(-1, 1).float()
                forward_labels = batch_dev['forward_label'].view(-1, 1).float()
                reverse_labels = batch_dev['reverse_label'].view(-1, 1).float()
                
                loss_mutual = debiased_loss(preds['mutual'], mutual_labels, propensity)
                loss_forward = debiased_loss(preds['forward'], forward_labels, propensity)
                loss_reverse = debiased_loss(preds['reverse'], reverse_labels, propensity)
                
                w = settings.multitask_weights
                total_scorer_loss = w['mutual']*loss_mutual + w['forward']*loss_forward + w['reverse']*loss_reverse
                
                loss_rerank = mse_loss(reranker_preds.float(), mutual_labels.squeeze())
                
            scaler.scale(total_scorer_loss).backward()
            scaler.unscale_(opt_scorer)
            torch.nn.utils.clip_grad_norm_(scorer_model.parameters(), max_norm=1.0)
            scaler.step(opt_scorer)
            
            scaler.scale(loss_rerank).backward()
            scaler.unscale_(opt_reranker)
            torch.nn.utils.clip_grad_norm_(reranker_model.parameters(), max_norm=1.0)
            scaler.step(opt_reranker)
            
            scaler.update()
            
        sched_retrieval.step()
        sched_scorer.step()
        sched_reranker.step()
        
        # Simple Validation Loop for Best-Checkpoint Selection
        retrieval_model.eval()
        scorer_model.eval()
        reranker_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                batch_dev = {}
                for k, v in val_batch.items():
                    if isinstance(v, dict):
                        batch_dev[k] = {col: (tensor.to(device, non_blocking=True) if hasattr(tensor, 'to') else tensor) for col, tensor in v.items()}
                    elif hasattr(v, 'to'):
                        batch_dev[k] = v.to(device, non_blocking=True)
                    else:
                        batch_dev[k] = v
                        
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    user_emb = retrieval_model.user_tower(batch_dev['user_num'], batch_dev['user_cat'], batch_dev.get('graph_feat'), batch_dev.get('sequence'))
                    cand_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])
                    combined_feat = torch.cat([user_emb, cand_emb, batch_dev.get('graph_feat'), batch_dev.get('sequence')[:, -1, :] if batch_dev.get('sequence').dim() == 3 else batch_dev.get('sequence')], dim=1)
                    preds = scorer_model(combined_feat)
                    
                propensity = batch_dev['propensity'].view(-1, 1).float()
                mutual_labels = batch_dev['mutual_label'].view(-1, 1).float()
                loss_mutual = debiased_loss(preds['mutual'], mutual_labels, propensity)
                val_loss += loss_mutual.item()
                
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} completed. Val Mutual Loss: {val_loss:.4f}")
        
        # Detect NaN — gradient explosion, abort immediately
        import math
        if math.isnan(val_loss) or math.isinf(val_loss):
            logger.warning(f"NaN/Inf val loss at epoch {epoch+1}. Aborting trial.")
            if trial is not None:
                import optuna
                raise optuna.exceptions.TrialPruned()
            break
        
        # Report to Optuna for Hyperband pruning
        if trial is not None:
            import optuna
            trial.report(-val_loss, epoch)
            if trial.should_prune():
                logger.info(f"Trial pruned at epoch {epoch+1}.")
                raise optuna.exceptions.TrialPruned()
        
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_retrieval_state = {k: v.cpu().clone() for k, v in retrieval_model.state_dict().items()}
            best_scorer_state = {k: v.cpu().clone() for k, v in scorer_model.state_dict().items()}
            best_reranker_state = {k: v.cpu().clone() for k, v in reranker_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break
            
    # Load best checkpoint
    retrieval_model.load_state_dict(best_retrieval_state)
    scorer_model.load_state_dict(best_scorer_state)
    reranker_model.load_state_dict(best_reranker_state)
        
    # 4. Calibration
    logger.info("Fitting calibrator on validation data...")
    calibrator = Calibrator()
    val_preds_list = []
    val_labels_list = []
    
    scorer_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_dev[k] = {col: (tensor.to(device) if hasattr(tensor, 'to') else tensor) for col, tensor in v.items()}
                elif hasattr(v, 'to'):
                    batch_dev[k] = v.to(device)
                else:
                    batch_dev[k] = v
                    
            user_emb = retrieval_model.user_tower(batch_dev['user_num'], batch_dev['user_cat'], batch_dev['graph_feat'], batch_dev['sequence'])
            cand_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])
            combined_feat = torch.cat([user_emb, cand_emb, batch_dev['graph_feat'], batch_dev['sequence'][:, -1, :] if batch_dev['sequence'].dim() == 3 else batch_dev['sequence']], dim=1)
            
            preds = scorer_model(combined_feat)
            
            # Format: [u_likes_c, c_likes_u, mutual, utility]
            batch_preds = torch.cat([torch.sigmoid(preds['forward']), torch.sigmoid(preds['reverse']), torch.sigmoid(preds['mutual']), preds['utility']], dim=1).cpu().numpy()
            f_lab = batch_dev['forward_label'].view(-1, 1).cpu().numpy()
            r_lab = batch_dev['reverse_label'].view(-1, 1).cpu().numpy()
            m_lab = batch_dev['mutual_label'].view(-1, 1).cpu().numpy()
            
            batch_labels_3 = np.hstack([f_lab, r_lab, m_lab])
            
            val_preds_list.append(batch_preds)
            val_labels_list.append(batch_labels_3)
            
    all_preds = np.vstack(val_preds_list)
    all_labels = np.vstack(val_labels_list)
    calibrator.fit(all_preds, all_labels)
    
    # 5. Build FAISS Index
    logger.info("Building FAISS index of candidates...")
    retrieval_model.eval()
    cand_embeddings = []
    cand_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_dev[k] = {col: (tensor.to(device) if hasattr(tensor, 'to') else tensor) for col, tensor in v.items()}
                elif hasattr(v, 'to'):
                    batch_dev[k] = v.to(device)
                else:
                    batch_dev[k] = v
            c_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])
            cand_embeddings.append(c_emb.cpu().numpy())
            
            cands = batch_dev['cand_id'].cpu().numpy() if isinstance(batch_dev['cand_id'], torch.Tensor) else batch_dev['cand_id']
            cand_ids.append(cands)
            
    if cand_embeddings:
        all_cand_embeddings = np.vstack(cand_embeddings)
        concatenated_ids = np.concatenate(cand_ids)
        try:
            all_cand_ids = concatenated_ids.astype(np.int64)
        except ValueError:
            logger.warning("Candidate IDs cannot be cast to int64. Using sequential IDs for FAISS.")
            all_cand_ids = np.arange(len(concatenated_ids), dtype=np.int64)
        
        
        flat_index = faiss.IndexFlatL2(all_cand_embeddings.shape[1])
        faiss_index = faiss.IndexIDMap(flat_index)
        faiss_index.add_with_ids(all_cand_embeddings, all_cand_ids)
    else:
        faiss_index = None

    model_config = {
        "user_num_dim": user_num_dim,
        "cand_num_dim": cand_num_dim,
        "user_cat_vocab": user_cat_vocab,
        "cand_cat_vocab": cand_cat_vocab,
        "embedding_dim": emb_dim,
        "hidden_sizes": tower_hidden_sizes,
        "dropout": dropout,
        "graph_dim": settings.graph_embedding_dim,
        "seq_dim": settings.sequence_hidden_dim,
        "ranker_input_dim": ranker_input_dim,
        "reranker_input_dim": reranker_input_dim
    }
    
    # Save models to disk
    import json
    model_dir = settings.model_path
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(retrieval_model.state_dict(), os.path.join(model_dir, "retrieval.pth"))
    torch.save(scorer_model.state_dict(), os.path.join(model_dir, "scorer.pth"))
    torch.save(reranker_model.state_dict(), os.path.join(model_dir, "reranker.pth"))
    joblib.dump(calibrator, os.path.join(model_dir, "calibrator.joblib"))
    
    if faiss_index is not None:
        faiss.write_index(faiss_index, os.path.join(model_dir, "candidates.index"))
    
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Saved all model artifacts to {model_dir}")
    
    return retrieval_model, scorer_model, reranker_model, calibrator, val_loader, faiss_index, model_config

