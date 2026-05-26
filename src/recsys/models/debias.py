import torch
import torch.nn as nn
import numpy as np

class PropensityScorer:
    """
    Estimates the propensity of a user being exposed to a candidate.
    Used for Inverse Propensity Scoring (IPS) to debias logs.
    """
    def __init__(self, clipping: float = 0.05):
        self.clipping = clipping
        
    def compute_ips_weights(self, propensities: torch.Tensor) -> torch.Tensor:
        """
        IPS Weight = 1 / P(Exposure)
        Clip to prevent high variance from low propensities.
        """
        clipped_p = torch.clamp(propensities, min=self.clipping)
        return 1.0 / clipped_p

class DebiasedLoss(nn.Module):
    """
    BCE Loss weighted by IPS to correct exposure bias.
    """
    def __init__(self, clipping: float = 0.05):
        super().__init__()
        self.clipping = clipping
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, propensities: torch.Tensor) -> torch.Tensor:
        weights = 1.0 / torch.clamp(propensities, min=self.clipping)
        loss = self.bce(logits, labels)
        return (loss * weights).mean()
