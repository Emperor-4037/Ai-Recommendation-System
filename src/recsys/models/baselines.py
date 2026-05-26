import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any

class PopularityBaseline:
    """Recommends candidates strictly by popularity (number of historical matches/likes)."""
    def __init__(self):
        self.candidate_scores = {}
        
    def fit(self, interactions_df: pd.DataFrame, cand_col: str = 'cand_id', match_col: str = 'match'):
        # Count positive interactions per candidate
        pop = interactions_df[interactions_df[match_col] == 1].groupby(cand_col).size()
        # Normalize
        if pop.max() > 0:
            pop = pop / pop.max()
        self.candidate_scores = pop.to_dict()
        
    def predict(self, candidate_ids: List[int]) -> np.ndarray:
        return np.array([self.candidate_scores.get(c, 0.0) for c in candidate_ids])

class CosineSimilarityBaseline:
    """Recommends candidates based on cosine similarity of their profile features."""
    def __init__(self):
        self.cand_features = {}
        
    def fit(self, candidates_df: pd.DataFrame, cand_col: str = 'cand_id', feature_cols: List[str] = None):
        if feature_cols is None:
            feature_cols = [c for c in candidates_df.columns if c.startswith('latent_')]
        
        self.cand_features = {
            row[cand_col]: row[feature_cols].values 
            for _, row in candidates_df.iterrows()
        }
        
    def predict(self, user_features: np.ndarray, candidate_ids: List[int]) -> np.ndarray:
        scores = []
        for c in candidate_ids:
            c_feat = self.cand_features.get(c, None)
            if c_feat is not None:
                sim = cosine_similarity(user_features.reshape(1, -1), c_feat.reshape(1, -1))[0, 0]
                scores.append(sim)
            else:
                scores.append(0.0)
        return np.array(scores)

class LogisticRegressionBaseline:
    """Predicts mutual match probability using a simple linear model over concatenated features."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]
