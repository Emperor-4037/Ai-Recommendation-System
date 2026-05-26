import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

class SparkMMRReranker:
    """
    Maximal Marginal Relevance (MMR) based reranker.
    Rewards complementarity and novelty while penalizing risk, redundancy, and popularity.
    """
    def __init__(self, lambda_spark: float = 0.2, lambda_comp: float = 0.2, lambda_risk: float = 1.0, lambda_pop: float = 0.1):
        self.lambda_spark = lambda_spark
        self.lambda_comp = lambda_comp
        self.lambda_risk = lambda_risk
        self.lambda_pop = lambda_pop

    def rerank(self, 
               candidate_ids: List[int], 
               reciprocal_scores: np.ndarray, 
               similarity_matrix: np.ndarray, 
               complementarity_scores: np.ndarray,
               safety_risk_scores: np.ndarray,
               popularity_scores: np.ndarray,
               k: int = 10) -> List[int]:
        """
        candidate_ids: List of candidate IDs
        reciprocal_scores: Array of shape (N,) base calibrated reciprocal scores
        similarity_matrix: Array of shape (N, N) pairwise candidate similarities (for redundancy penalty)
        complementarity_scores: Array of shape (N,) how complementary they are to the user
        safety_risk_scores: Array of shape (N,) safety risk [0, 1]
        popularity_scores: Array of shape (N,) normalized popularity [0, 1]
        """
        N = len(candidate_ids)
        if N == 0:
            return []
            
        selected_indices = []
        unselected_indices = list(range(N))
        
        # Initial score includes complementarity, risk, and popularity, but no redundancy penalty yet
        base_scores = reciprocal_scores + \
                      self.lambda_comp * complementarity_scores - \
                      self.lambda_risk * safety_risk_scores - \
                      self.lambda_pop * popularity_scores
                      
        while len(selected_indices) < min(k, N):
            best_idx = -1
            best_score = -float('inf')
            
            for idx in unselected_indices:
                # Redundancy penalty (max similarity to already selected candidates)
                redundancy = 0.0
                if selected_indices:
                    redundancy = np.max([similarity_matrix[idx, s_idx] for s_idx in selected_indices])
                    
                # MMR formula: (1 - lambda_spark) * base_score - lambda_spark * redundancy
                # Wait, the prompt formula: final_score = reciprocal_score + λspark*novelty + λcomp*complementarity - λrisk*safety_risk - λpop*popularity_bias
                # Novelty here can be interpreted as (1 - redundancy)
                
                novelty = 1.0 - redundancy
                
                score = base_scores[idx] + self.lambda_spark * novelty
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
            
        return [candidate_ids[i] for i in selected_indices]

class PairwiseSparkReranker(nn.Module):
    """
    Learned Pairwise Reranker for Spark Mode.
    Trained to rank complementary candidates higher than redundant ones.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, user_feat: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate user and candidate features for pairwise scoring
        if user_feat.size(0) == cand_feat.size(0):
            combined = torch.cat([user_feat, cand_feat], dim=1)
        else:
            # Broadcast user feature to match candidates
            combined = torch.cat([user_feat.repeat(cand_feat.size(0), 1), cand_feat], dim=1)
        return self.net(combined).squeeze(-1)
    
    def rerank(self, user_feat: torch.Tensor, candidates_feat: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        scores = self.forward(user_feat, candidates_feat)
        _, indices = torch.topk(scores, min(top_k, len(scores)))
        return indices
