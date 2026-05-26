import optuna
import mlflow
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Orchestrates Bayesian optimization via Optuna.
    Tracks all trials in MLflow.
    """
    def __init__(self, study_name: str, n_trials: int = 20, mlflow_uri: str = None):
        self.study_name = study_name
        self.n_trials = n_trials
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            
    def optimize(self, objective_fn: Callable[[optuna.Trial], float]):
        study = optuna.create_study(
            direction="maximize", 
            study_name=self.study_name,
            storage="sqlite:///hyperopt_study.db",
            load_if_exists=True,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,
                max_resource=30,
                reduction_factor=3
            )
        )
        
        def mlflow_callback(study, trial):
            if trial.value is None:
                return  # Pruned trial, no metric to log
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("objective", trial.value)
        
        study.optimize(objective_fn, n_trials=self.n_trials, callbacks=[mlflow_callback])
        
        logger.info(f"Best trial: {study.best_trial.params}")
        return study.best_params

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Expanded parameter sampling for Two-Tower / Multitask Ranker."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lambda_spark": trial.suggest_float("lambda_spark", 0.1, 0.5),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        "epochs": 30
    }

def get_objective(dataset_name: str) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        from recsys.operations.train import train_pipeline
        from recsys.evaluation.offline import OfflineEvaluator, counterfactual_evaluation
        import torch
        import numpy as np
        
        params = sample_params(trial)
        
        # Train
        retrieval_model, scorer_model, reranker_model, calibrator, val_loader, _, _ = train_pipeline(dataset_name, params, trial=trial)
        
        # Evaluate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer_model.eval()
        
        from collections import defaultdict
        
        user_preds = defaultdict(list)
        user_actuals = defaultdict(list)
        
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
                    user_emb = retrieval_model.user_tower(batch_dev['user_num'], batch_dev['user_cat'], batch_dev.get('graph_feat'), batch_dev.get('sequence'))
                    cand_emb = retrieval_model.cand_tower(batch_dev['cand_num'], batch_dev['cand_cat'])
                    seq_feat = batch_dev.get('sequence')
                    seq_val = seq_feat[:, -1, :] if (seq_feat is not None and seq_feat.dim() == 3) else seq_feat
                    combined_feat = torch.cat([user_emb, cand_emb, batch_dev.get('graph_feat'), seq_val], dim=1)
                    
                    preds = scorer_model(combined_feat)
                mutual_preds = torch.sigmoid(preds['mutual']).cpu().numpy()
                mutual_labels = batch_dev['mutual_label'].cpu().numpy()
                
                users = batch_dev['user_id'].cpu().numpy() if isinstance(batch_dev['user_id'], torch.Tensor) else batch_dev['user_id']
                cands = batch_dev['cand_id'].cpu().numpy() if isinstance(batch_dev['cand_id'], torch.Tensor) else batch_dev['cand_id']
                
                for u, c, p, l in zip(users, cands, mutual_preds, mutual_labels):
                    user_preds[u].append((c, p))
                    user_actuals[u].append((c, l))
                    
        evaluator = OfflineEvaluator(k=10)
        
        for u in user_preds:
            # Sort candidates by predicted score descending
            sorted_preds = [str(c) for c, p in sorted(user_preds[u], key=lambda x: x[1], reverse=True)]
            # Actual positive candidates (label > 0.5)
            actual_positives = [str(c) for c, l in user_actuals[u] if l > 0.5]
            
            mutual_matches = len([c for c in sorted_preds[:10] if c in actual_positives])
            
            evaluator.evaluate_ranking('sync', actual_positives, sorted_preds, mutual_matches=mutual_matches)
            
        metrics = evaluator.aggregate()
        ndcg_score = metrics.get('sync', {}).get('ndcg', 0.0)
        return float(ndcg_score)
        
    return objective
