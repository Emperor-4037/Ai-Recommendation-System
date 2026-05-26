import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Any
from recsys.models.sequential import SequentialEncoder

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        out = self.fc(x)
        out = self.ln(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + self.proj(x)

class Tower(nn.Module):
    def __init__(self, num_features_dim: int, cat_features_vocab: Dict[str, int], embedding_dim: int, hidden_sizes: List[int], dropout: float = 0.1, 
                 graph_dim: int = 0, seq_dim: int = 0, seq_hidden: int = 64):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim, padding_idx=0)
            for col, vocab_size in cat_features_vocab.items()
        })
        
        # Sequence Encoder (only if seq_dim > 0)
        self.seq_encoder = SequentialEncoder(seq_dim, seq_hidden) if seq_dim > 0 else None
        
        # Input dim = numeric + concatenated embeddings + graph + sequence_hidden
        input_dim = num_features_dim + (len(cat_features_vocab) * embedding_dim) + graph_dim + (seq_hidden if seq_dim > 0 else 0)
        
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(ResidualBlock(curr_dim, hidden_dim, dropout))
            curr_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, num_features: torch.Tensor, cat_features: Dict[str, torch.Tensor], 
                graph_feat: torch.Tensor = None, sequence: torch.Tensor = None) -> torch.Tensor:
        # Process categorical features
        emb_list = [num_features]
        for col, emb_layer in self.cat_embeddings.items():
            if col in cat_features:
                emb_list.append(emb_layer(cat_features[col]))
                
        # Add graph features
        if graph_feat is not None:
            emb_list.append(graph_feat)
            
        # Add sequence features
        if self.seq_encoder is not None and sequence is not None:
            seq_state = self.seq_encoder(sequence)
            emb_list.append(seq_state)
            
        # Concatenate everything
        x = torch.cat(emb_list, dim=1)
            
        # MLP
        out = self.mlp(x)
        # Normalize output vectors for cosine/dot-product similarity stability
        out = F.normalize(out, p=2, dim=1)
        return out

class TwoTowerRetrieval(nn.Module):
    def __init__(self, 
                 user_num_dim: int, 
                 user_cat_vocab: Dict[str, int], 
                 cand_num_dim: int, 
                 cand_cat_vocab: Dict[str, int], 
                 embedding_dim: int = 64, 
                 hidden_sizes: List[int] = [256, 128, 64], 
                 dropout: float = 0.1,
                 graph_dim: int = 32,
                 seq_dim: int = 64,
                 seq_hidden: int = 64):
        super().__init__()
        self.user_tower = Tower(user_num_dim, user_cat_vocab, embedding_dim, hidden_sizes, dropout, graph_dim, seq_dim, seq_hidden)
        self.cand_tower = Tower(cand_num_dim, cand_cat_vocab, embedding_dim, hidden_sizes, dropout)
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        user_emb = self.user_tower(batch['user_num'], batch['user_cat'], batch.get('graph_feat'), batch.get('sequence'))
        cand_emb = self.cand_tower(batch['cand_num'], batch['cand_cat'])
        
        # Dot product for similarity
        scores = torch.sum(user_emb * cand_emb, dim=1)
        return scores
        
    def get_user_embedding(self, num_features: torch.Tensor, cat_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.user_tower(num_features, cat_features)
        
    def get_candidate_embedding(self, num_features: torch.Tensor, cat_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cand_tower(num_features, cat_features)

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -F.logsigmoid(pos_scores - neg_scores).mean()


