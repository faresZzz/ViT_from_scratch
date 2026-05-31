"""Transformer encoder blocks for the Vision Transformer."""

from __future__ import annotations

from torch import Tensor, nn

from vit_from_scratch.attention import MultiHeadSelfAttention
from vit_from_scratch.mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with residual connections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_rope: bool = False,
        rope_mode: str = "none",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_rope=use_rope,
            rope_mode=rope_mode,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(
        self, tokens: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        attention_output = self.attention(
            self.norm1(tokens), return_attention=return_attention
        )
        if return_attention:
            attention_tokens, attention_probs = attention_output
        else:
            attention_tokens = attention_output

        tokens = tokens + attention_tokens
        tokens = tokens + self.mlp(self.norm2(tokens))
        if return_attention:
            return tokens, attention_probs
        return tokens
