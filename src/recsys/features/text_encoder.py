import logging
import torch
import torch.nn as nn
from typing import List, Union

logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """
    Compact frozen sentence encoder with fallback.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embedding_dim: int = 384, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
            # Freeze the encoder by default
            for param in self.model.parameters():
                param.requires_grad = False
            self.has_encoder = True
            logger.info(f"Loaded text encoder {model_name} successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer '{model_name}'. Using fallback. Error: {e}")
            self.has_encoder = False
            
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
            
        if self.has_encoder and any(t.strip() for t in texts):
            # Encode and return tensor
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return embeddings.to(self.device)
        else:
            # Cold start / Fallback: Zero embeddings
            return torch.zeros((len(texts), self.embedding_dim), device=self.device)

