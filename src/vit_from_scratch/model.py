"""Pedagogical Vision Transformer implementation in PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vit_from_scratch.config import ViTConfig
from vit_from_scratch.embedding import (
    CosinePositionEmbedding,
    LearnedPositionEmbedding,
)
from vit_from_scratch.encoder import TransformerEncoderBlock
from vit_from_scratch.patch_embedding import PatchEmbedding


class VisionTransformer(nn.Module):
    """Vision Transformer aligned with the standard encoder stack."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        num_tokens = config.num_patches + 1
        if config.position_embedding == "learned":
            self.position_embedding: nn.Module | None = LearnedPositionEmbedding(
                num_tokens=num_tokens,
                embed_dim=config.embed_dim,
                dropout=config.dropout,
            )
        elif config.position_embedding == "cosine":
            self.position_embedding = CosinePositionEmbedding(
                num_tokens=num_tokens,
                embed_dim=config.embed_dim,
                dropout=config.dropout,
            )
        else:
            self.position_embedding = None

        if config.position_embedding == "rope":
            rope_mode = "1d"
        elif config.position_embedding == "rope2d":
            rope_mode = "2d"
        else:
            rope_mode = "none"

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    rope_mode=rope_mode,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def encode_tokens(self, images: Tensor) -> Tensor:
        patch_tokens = self.patch_embed(images)
        batch_size = patch_tokens.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        if self.position_embedding is not None:
            tokens = self.position_embedding(tokens)

        for block in self.encoder:
            tokens = block(tokens)

        return self.norm(tokens)

    def forward_with_attention(self, images: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return logits and per-block attention maps shaped ``[B, H, N, N]``."""
        patch_tokens = self.patch_embed(images)
        batch_size = patch_tokens.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        if self.position_embedding is not None:
            tokens = self.position_embedding(tokens)

        attention_maps: list[Tensor] = []
        for block in self.encoder:
            tokens, attention_probs = block(tokens, return_attention=True)
            attention_maps.append(attention_probs)

        tokens = self.norm(tokens)
        cls_representation = tokens[:, 0]
        logits = self.head(cls_representation)
        return logits, tuple(attention_maps)

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.encode_tokens(images)
        cls_representation = tokens[:, 0]
        logits = self.head(cls_representation)
        return logits
