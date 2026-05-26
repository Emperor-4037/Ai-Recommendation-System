import torch
import torch.nn as nn
from typing import Dict, List, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression

class ReciprocalMultitaskRanker(nn.Module):
    """
    Upgraded Multitask Neural Ranker predicting:
    - forward_pref, reverse_pref, mutual_match, reply_prob, expected_utility
    Includes Dropout for MC-Uncertainty estimation.
    """
    def __init__(self, input_dim: int, hidden_sizes: List[int] = [256, 128, 64], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim
            
        self.shared_bottom = nn.Sequential(*layers)
        
        # Heads: forward, reverse, mutual, reply, utility
        self.forward_head = nn.Linear(curr_dim, 1)
        self.reverse_head = nn.Linear(curr_dim, 1)
        self.mutual_head = nn.Linear(curr_dim, 1)
        self.reply_head = nn.Linear(curr_dim, 1)
        self.utility_head = nn.Linear(curr_dim, 1)
        
    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> Dict[str, torch.Tensor]:
        if not mc_dropout:
            self.eval()
        else:
            self.train() # Enable dropout for uncertainty
            
        bottom_out = self.shared_bottom(x)
        
        return {
            "forward": self.forward_head(bottom_out),
            "reverse": self.reverse_head(bottom_out),
            "mutual": self.mutual_head(bottom_out),
            "reply": self.reply_head(bottom_out),
            "utility": self.utility_head(bottom_out)
        }
        
    def get_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Estimate uncertainty via MC Dropout variance on mutual match head."""
        samples = []
        for _ in range(n_samples):
            samples.append(self.forward(x, mc_dropout=True)["mutual"])
        samples = torch.stack(samples)
        return torch.var(samples, dim=0)

class Calibrator:
    """Isotonic regression calibration for predicted probabilities."""
    def __init__(self):
        self.iso_u_likes_c = IsotonicRegression(out_of_bounds='clip')
        self.iso_c_likes_u = IsotonicRegression(out_of_bounds='clip')
        self.iso_mutual = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        
    def fit(self, preds: np.ndarray, labels: np.ndarray):
        """
        preds shape: (N, 4) [u_likes_c, c_likes_u, mutual, utility]
        labels shape: (N, 3) [u_likes_c, c_likes_u, mutual]
        """
        self.iso_u_likes_c.fit(preds[:, 0], labels[:, 0])
        self.iso_c_likes_u.fit(preds[:, 1], labels[:, 1])
        self.iso_mutual.fit(preds[:, 2], labels[:, 2])
        self.is_fitted = True
        
    def calibrate(self, preds: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return preds
            
        calibrated = np.zeros_like(preds)
        calibrated[:, 0] = self.iso_u_likes_c.transform(preds[:, 0])
        calibrated[:, 1] = self.iso_c_likes_u.transform(preds[:, 1])
        calibrated[:, 2] = self.iso_mutual.transform(preds[:, 2])
        calibrated[:, 3] = preds[:, 3] # Utility is uncalibrated or scaled separately
        return calibrated
