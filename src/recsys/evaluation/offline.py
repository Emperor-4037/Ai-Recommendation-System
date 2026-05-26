import numpy as np
from typing import List, Dict, Any

def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    pred_k = set(predicted[:k])
    act_set = set(actual)
    return len(act_set.intersection(pred_k)) / len(act_set)

def ndcg_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    if not actual:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

class OfflineEvaluator:
    """
    Offline evaluation pipeline.
    Reports Sync and Spark metrics separately.
    """
    def __init__(self, k: int = 10):
        self.k = k
        self.metrics = {
            'sync': {'recall': [], 'ndcg': [], 'mutual_match_rate': []},
            'spark': {'recall': [], 'ndcg': [], 'intra_list_diversity': [], 'complementarity_score': []}
        }
        
    def evaluate_ranking(self, mode: str, actual: List[int], predicted: List[int], mutual_matches: int = 0, diversity_score: float = 0.0, comp_score: float = 0.0):
        rec = recall_at_k(actual, predicted, self.k)
        ndcg = ndcg_at_k(actual, predicted, self.k)
        
        self.metrics[mode]['recall'].append(rec)
        self.metrics[mode]['ndcg'].append(ndcg)
        
        if mode == 'sync':
            # Simplified proxy: match rate in top K
            mm_rate = mutual_matches / self.k if self.k > 0 else 0
            self.metrics[mode]['mutual_match_rate'].append(mm_rate)
        elif mode == 'spark':
            self.metrics[mode]['intra_list_diversity'].append(diversity_score)
            self.metrics[mode]['complementarity_score'].append(comp_score)
            
    def aggregate(self) -> Dict[str, Dict[str, float]]:
        agg = {}
        for mode, mets in self.metrics.items():
            agg[mode] = {}
            for met_name, values in mets.items():
                if values:
                    agg[mode][met_name] = float(np.mean(values))
                else:
                    agg[mode][met_name] = 0.0
        return agg

def counterfactual_evaluation(actual_labels: np.ndarray, predicted_scores: np.ndarray, propensities: np.ndarray) -> float:
    """
    Computes Inverse Propensity Scoring (IPS) weighted Mean Squared Error.
    IPS weight = 1 / P(Exposure).
    """
    weights = 1.0 / np.clip(propensities, 0.05, 1.0)
    # MSE = weighted average of (pred - actual)^2
    errors = (predicted_scores - actual_labels) ** 2
    return np.mean(errors * weights)
