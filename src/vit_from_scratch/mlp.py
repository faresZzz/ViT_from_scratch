"""Feed-forward network used inside the Vision Transformer encoder."""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """Two-layer MLP with GELU activations and dropout."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.fc1(tokens)
        tokens = self.act(tokens)
        tokens = self.dropout1(tokens)
        tokens = self.fc2(tokens)
        tokens = self.dropout2(tokens)
        return tokens
