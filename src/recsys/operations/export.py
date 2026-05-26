import os
import torch
import joblib
import json
import logging
from typing import Dict, Any

from recsys.config import get_settings

logger = logging.getLogger(__name__)
import faiss

def export_artifacts(retrieval_model, scorer_model, reranker_model, calibrator, metrics: Dict[str, Any], params: Dict[str, Any] = None, faiss_index=None, model_config=None):
    settings = get_settings()
    model_dir = settings.model_path
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    torch.save(retrieval_model.state_dict(), os.path.join(model_dir, "retrieval.pth"))
    torch.save(scorer_model.state_dict(), os.path.join(model_dir, "scorer.pth"))
    torch.save(reranker_model.state_dict(), os.path.join(model_dir, "reranker.pth"))
    
    # Save calibrator
    joblib.dump(calibrator, os.path.join(model_dir, "calibrator.joblib"))
    
    # Save config snapshot and metrics
    export_info = {
        "metrics": metrics,
        "hyperparameters": params or {},
        "config_snapshot": {
            "embedding_dim": settings.retrieval_embedding_dim,
            "lambda_spark": settings.lambda_spark,
            "multitask_weights": settings.multitask_weights,
            "model_architecture": model_config or {}
        }
    }
    with open(os.path.join(model_dir, "export_info.json"), "w") as f:
        json.dump(export_info, f, indent=4)
        
    if model_config:
        with open(os.path.join(model_dir, "config_snapshot.json"), "w") as f:
            json.dump(model_config, f, indent=4)
            
    if faiss_index:
        faiss.write_index(faiss_index, os.path.join(model_dir, "candidates.index"))
        
    logger.info(f"Successfully exported champion artifacts to {model_dir}")
