"""Multi-head self-attention used by the Vision Transformer encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vit_from_scratch.embedding import apply_rope, build_rope_cache, build_rope_cache_2d


class MultiHeadSelfAttention(nn.Module):
    """Explicit multi-head self-attention with qkv projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_rope: bool = False,
        rope_mode: str = "none",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads: "
                f"got embed_dim={embed_dim}, num_heads={num_heads}."
            )

        # Accept legacy use_rope=True as rope_mode="1d"
        if use_rope and rope_mode == "none":
            rope_mode = "1d"

        if rope_mode not in {"none", "1d", "2d"}:
            raise ValueError(
                f"rope_mode must be one of {{'none', '1d', '2d'}}: got {rope_mode!r}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope_mode = rope_mode
        self.scale = self.head_dim ** -0.5

        if rope_mode != "none" and self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even per-head dimension: "
                f"got head_dim={self.head_dim}."
            )
        if rope_mode == "2d" and self.head_dim % 4 != 0:
            raise ValueError(
                "RoPE 2D requires head_dim divisible by 4: "
                f"got head_dim={self.head_dim}."
            )

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention to ``tokens``.

        When ``return_attention`` is ``True``, the returned attention map is the
        probability tensor after softmax and attention dropout, with shape
        ``[B, H, N, N]``.
        """
        if tokens.ndim != 3:
            raise ValueError(
                "MultiHeadSelfAttention expects tokens with shape [B, N, D]: "
                f"got tensor with shape {tuple(tokens.shape)}."
            )

        batch_size, num_tokens, embed_dim = tokens.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                "Last dimension of tokens must match embed_dim: "
                f"got {embed_dim}, expected {self.embed_dim}."
            )

        qkv = self.qkv(tokens)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)
        if self.rope_mode == "1d":
            cos, sin = build_rope_cache(
                seq_len=num_tokens,
                head_dim=self.head_dim,
                device=queries.device,
                dtype=queries.dtype,
            )
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)
        elif self.rope_mode == "2d":
            num_patches = num_tokens - 1  # subtract CLS
            grid_size = int(num_patches ** 0.5)
            assert grid_size * grid_size == num_patches, (
                f"RoPE 2D requires a square grid of patches, got {num_patches} patches."
            )
            cos, sin = build_rope_cache_2d(
                grid_height=grid_size,
                grid_width=grid_size,
                head_dim=self.head_dim,
                device=queries.device,
                dtype=queries.dtype,
            )
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        output = self.proj(context)
        output = self.proj_dropout(output)
        if return_attention:
            return output, attention_probs
        return output
