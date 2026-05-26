import torch
import torch.nn as nn
from typing import List

class SequentialEncoder(nn.Module):
    """
    Lightweight GRU-based sequential encoder for user activity history.
    Encodes a sequence of historical interaction embeddings into a fixed-size state vector.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        lengths: (batch_size,) actual sequence lengths for packing
        """
        if lengths is not None:
            # Sort lengths for packing
            lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
            _, idx_unsort = torch.sort(idx_sort)
            
            x_sorted = x[idx_sort]
            packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True)
            _, hidden = self.gru(packed)
            
            # hidden is (n_layers, batch, hidden_dim), take top layer
            state = hidden[-1]
            # Unsort
            state = state[idx_unsort]
        else:
            _, hidden = self.gru(x)
            state = hidden[-1]
            
        return self.ln(state)
