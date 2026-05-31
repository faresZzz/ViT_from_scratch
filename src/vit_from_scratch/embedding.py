"""Positional embedding utilities for the pedagogical Vision Transformer."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

PositionEmbeddingType = Literal["learned", "cosine", "rope", "rope2d"]


def _build_1d_sincos_embedding(length: int, embed_dim: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(embed_dim, 1))
    )

    embedding = torch.zeros(length, embed_dim, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(positions * div_term)
    embedding[:, 1::2] = torch.cos(positions * div_term[: embedding[:, 1::2].shape[1]])
    return embedding


def _build_2d_sincos_embedding(num_patches: int, embed_dim: int) -> Tensor:
    grid_size = int(math.isqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError("num_patches must be a perfect square for 2D sin/cos.")

    row_dim = embed_dim // 2
    col_dim = embed_dim - row_dim
    row_embedding = _build_1d_sincos_embedding(grid_size, row_dim)
    col_embedding = _build_1d_sincos_embedding(grid_size, col_dim)

    rows = torch.arange(grid_size).repeat_interleave(grid_size)
    cols = torch.arange(grid_size).repeat(grid_size)
    patch_embedding = torch.cat(
        [row_embedding[rows], col_embedding[cols]],
        dim=-1,
    )
    cls_embedding = torch.zeros(1, embed_dim, dtype=torch.float32)
    return torch.cat([cls_embedding, patch_embedding], dim=0)


class LearnedPositionEmbedding(nn.Module):
    """Add learned absolute positional embeddings to token sequences."""

    def __init__(self, num_tokens: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "LearnedPositionEmbedding expects tokens with shape [B, N, D]: "
                f"got tensor with shape {tuple(tokens.shape)}."
            )
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                "Token count does not match num_tokens: "
                f"got {tokens.shape[1]}, expected {self.num_tokens}."
            )
        if tokens.shape[2] != self.embed_dim:
            raise ValueError(
                "Embedding dimension does not match embed_dim: "
                f"got {tokens.shape[2]}, expected {self.embed_dim}."
            )

        return self.dropout(tokens + self.position_embedding)


class CosinePositionEmbedding(nn.Module):
    """Add fixed sin/cos absolute embeddings, using 2D patches when possible."""

    def __init__(self, num_tokens: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        if num_tokens > 1:
            num_patches = num_tokens - 1
            grid_size = int(math.isqrt(num_patches))
            if grid_size * grid_size == num_patches:
                position_embedding = _build_2d_sincos_embedding(num_patches, embed_dim)
            else:
                position_embedding = _build_1d_sincos_embedding(num_tokens, embed_dim)
        else:
            position_embedding = _build_1d_sincos_embedding(num_tokens, embed_dim)

        self.register_buffer(
            "position_embedding",
            position_embedding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "CosinePositionEmbedding expects tokens with shape [B, N, D]: "
                f"got tensor with shape {tuple(tokens.shape)}."
            )
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                "Token count does not match num_tokens: "
                f"got {tokens.shape[1]}, expected {self.num_tokens}."
            )
        if tokens.shape[2] != self.embed_dim:
            raise ValueError(
                "Embedding dimension does not match embed_dim: "
                f"got {tokens.shape[2]}, expected {self.embed_dim}."
            )

        position_embedding = self.position_embedding.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self.dropout(tokens + position_embedding)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Build broadcastable cos/sin caches for rotary position embeddings."""

    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")
    if head_dim <= 0:
        raise ValueError("head_dim must be a positive integer.")
    if head_dim % 2 != 0:
        raise ValueError(
            "RoPE requires an even head_dim: "
            f"got head_dim={head_dim}."
        )

    cache_dtype = dtype if dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype)) else torch.float32
    positions = torch.arange(seq_len, device=device, dtype=cache_dtype)
    inverse_frequency = 1.0 / (
        10_000
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=cache_dtype)
            / head_dim
        )
    )
    angles = torch.outer(positions, inverse_frequency)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def build_rope_cache_2d(
    grid_height: int,
    grid_width: int,
    head_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build broadcastable cos/sin caches for 2D rotary position embeddings.

    The first half of ``head_dim`` encodes row positions, the second half
    encodes column positions.  A CLS token (identity rotation) is prepended
    so the cache covers ``grid_height * grid_width + 1`` positions.

    Returns tensors shaped ``[1, 1, N+1, head_dim]`` for direct broadcast
    over (batch, heads, seq, dim).
    """

    if head_dim % 4 != 0:
        raise ValueError(
            "RoPE 2D requires head_dim divisible by 4: "
            f"got head_dim={head_dim}."
        )

    half = head_dim // 2  # each spatial axis gets half the dimensions

    # Inverse frequencies — one set per spatial axis, each of length half//2
    inv_freq_row = 1.0 / (
        10_000 ** (torch.arange(0, half, 2, device=device, dtype=dtype) / half)
    )
    inv_freq_col = 1.0 / (
        10_000 ** (torch.arange(0, half, 2, device=device, dtype=dtype) / half)
    )

    # Patch positions in raster order: row index repeated, col index tiled
    row_positions = torch.arange(grid_height, device=device, dtype=dtype).repeat_interleave(grid_width)
    col_positions = torch.arange(grid_width, device=device, dtype=dtype).repeat(grid_height)

    # Outer products → [N, half//2]
    angles_row = torch.outer(row_positions, inv_freq_row)
    angles_col = torch.outer(col_positions, inv_freq_col)

    # Expand each angle pair to adjacent even/odd slots → [N, half]
    cos_row = torch.repeat_interleave(torch.cos(angles_row), repeats=2, dim=-1)
    sin_row = torch.repeat_interleave(torch.sin(angles_row), repeats=2, dim=-1)
    cos_col = torch.repeat_interleave(torch.cos(angles_col), repeats=2, dim=-1)
    sin_col = torch.repeat_interleave(torch.sin(angles_col), repeats=2, dim=-1)

    # Concatenate row and column halves → [N, head_dim]
    cos_patches = torch.cat([cos_row, cos_col], dim=-1)
    sin_patches = torch.cat([sin_row, sin_col], dim=-1)

    # Prepend CLS token: identity rotation (cos=1, sin=0)
    cos_cls = torch.ones(1, head_dim, device=device, dtype=dtype)
    sin_cls = torch.zeros(1, head_dim, device=device, dtype=dtype)
    cos_full = torch.cat([cos_cls, cos_patches], dim=0)   # [N+1, head_dim]
    sin_full = torch.cat([sin_cls, sin_patches], dim=0)   # [N+1, head_dim]

    # Add batch and heads dims for broadcasting → [1, 1, N+1, head_dim]
    return cos_full.unsqueeze(0).unsqueeze(0), sin_full.unsqueeze(0).unsqueeze(0)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to [B, heads, N, head_dim] tensors."""

    if x.ndim != 4:
        raise ValueError(
            "apply_rope expects x with shape [B, heads, N, head_dim]: "
            f"got tensor with shape {tuple(x.shape)}."
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "RoPE requires an even head_dim: "
            f"got head_dim={x.shape[-1]}."
        )
    if cos.shape != sin.shape:
        raise ValueError(
            "cos and sin caches must have matching shapes: "
            f"got cos={tuple(cos.shape)} and sin={tuple(sin.shape)}."
        )
    if cos.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            "RoPE cache shape must match sequence and head dimensions: "
            f"got cache={tuple(cos.shape)}, x={tuple(x.shape)}."
        )

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)
    return (x * cos) + (rotated * sin)
