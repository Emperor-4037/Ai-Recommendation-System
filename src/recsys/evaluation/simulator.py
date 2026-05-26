import numpy as np
import pandas as pd
from typing import Dict, Any

class ReciprocalMarketSimulator:
    """
    Deterministic reciprocal-market simulator for synthetic stress testing.
    Generates:
    - latent user profiles
    - similarity and complementarity clusters
    - asymmetric preferences
    - exposure bias
    - match probability
    """
    
    def __init__(self, 
                 seed: int = 42, 
                 n_users: int = 1000, 
                 n_candidates: int = 1000,
                 latent_dim: int = 8,
                 cluster_count: int = 4,
                 homophily_strength: float = 0.6,
                 complementarity_strength: float = 0.4,
                 noise_level: float = 0.1,
                 exposure_bias: float = 0.5):
        self.seed = seed
        self.n_users = n_users
        self.n_candidates = n_candidates
        self.latent_dim = latent_dim
        self.cluster_count = cluster_count
        self.homophily_strength = homophily_strength
        self.complementarity_strength = complementarity_strength
        self.noise_level = noise_level
        self.exposure_bias = exposure_bias
        
        self.rng = np.random.default_rng(seed)
        
    def generate_profiles(self, n_profiles: int) -> pd.DataFrame:
        clusters = self.rng.integers(0, self.cluster_count, size=n_profiles)
        
        # Base cluster centers
        cluster_centers = self.rng.normal(0, 1, size=(self.cluster_count, self.latent_dim))
        
        # Profile latents
        latents = cluster_centers[clusters] + self.rng.normal(0, self.noise_level, size=(n_profiles, self.latent_dim))
        
        # Normalize
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        latents = latents / (norms + 1e-9)
        
        # Calculate base popularity (exposure bias proxy)
        popularity = self.rng.beta(2, 5, size=n_profiles)
        
        df = pd.DataFrame(latents, columns=[f'latent_{i}' for i in range(self.latent_dim)])
        df['cluster'] = clusters
        df['popularity'] = popularity
        
        return df
        
    def generate_interactions(self, users: pd.DataFrame, candidates: pd.DataFrame, n_interactions: int = 10000) -> pd.DataFrame:
        user_indices = self.rng.choice(len(users), size=n_interactions, replace=True)
        
        # Apply exposure bias: popular candidates are sampled more
        cand_probs = candidates['popularity'].values
        cand_probs = cand_probs / cand_probs.sum()
        cand_indices = self.rng.choice(len(candidates), size=n_interactions, p=cand_probs, replace=True)
        
        u_latents = users[[f'latent_{i}' for i in range(self.latent_dim)]].values[user_indices]
        c_latents = candidates[[f'latent_{i}' for i in range(self.latent_dim)]].values[cand_indices]
        
        # Similarity score (dot product)
        similarity = np.sum(u_latents * c_latents, axis=1)
        
        # Complementarity score (heuristic: distance between cross-cluster mapped latents)
        # For simplicity, we create a deterministic "complementary" matrix mapping
        comp_matrix = self.rng.normal(0, 1, size=(self.latent_dim, self.latent_dim))
        u_comp = u_latents @ comp_matrix
        u_comp_norm = u_comp / (np.linalg.norm(u_comp, axis=1, keepdims=True) + 1e-9)
        complementarity = np.sum(u_comp_norm * c_latents, axis=1)
        
        # Asymmetric preferences
        p_u_likes_c = self.homophily_strength * similarity + self.complementarity_strength * complementarity + self.rng.normal(0, self.noise_level, size=n_interactions)
        p_c_likes_u = self.homophily_strength * similarity + self.complementarity_strength * complementarity + self.rng.normal(0, self.noise_level, size=n_interactions)
        
        # Sigmoid activation
        p_u_likes_c = 1 / (1 + np.exp(-p_u_likes_c * 5))
        p_c_likes_u = 1 / (1 + np.exp(-p_c_likes_u * 5))
        
        # Binary outcomes
        u_likes_c = (p_u_likes_c > 0.5).astype(int)
        c_likes_u = (p_c_likes_u > 0.5).astype(int)
        mutual_match = u_likes_c * c_likes_u
        
        df = pd.DataFrame({
            'user_id': user_indices,
            'cand_id': cand_indices,
            'u_likes_c': u_likes_c,
            'c_likes_u': c_likes_u,
            'match': mutual_match,
            'similarity': similarity,
            'complementarity': complementarity
        })
        
        return df

    def run(self) -> Dict[str, pd.DataFrame]:
        users = self.generate_profiles(self.n_users)
        candidates = self.generate_profiles(self.n_candidates)
        interactions = self.generate_interactions(users, candidates)
        
        users['user_id'] = np.arange(self.n_users)
        candidates['cand_id'] = np.arange(self.n_candidates)
        
        return {
            'users': users,
            'candidates': candidates,
            'interactions': interactions
        }
