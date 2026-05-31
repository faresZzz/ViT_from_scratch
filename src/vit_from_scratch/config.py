"""Configuration objects for the pedagogical Vision Transformer."""

from __future__ import annotations

from dataclasses import dataclass

from vit_from_scratch.embedding import PositionEmbeddingType


@dataclass(frozen=True)
class ViTConfig:
    """Minimal Vision Transformer configuration."""

    image_size: int
    patch_size: int
    in_channels: int
    num_classes: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float = 0.0
    attention_dropout: float = 0.0
    position_embedding: PositionEmbeddingType = "learned"
    decoder_embed_dim: int | None = None
    decoder_depth: int | None = None
    decoder_num_heads: int | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size: "
                f"got image_size={self.image_size}, patch_size={self.patch_size}."
            )
        if self.in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads: "
                f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}."
            )
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be strictly positive.")
        if self.dropout < 0.0:
            raise ValueError("dropout must be greater than or equal to 0.0.")
        if self.attention_dropout < 0.0:
            raise ValueError(
                "attention_dropout must be greater than or equal to 0.0."
            )
        if self.position_embedding not in {"learned", "cosine", "rope", "rope2d"}:
            raise ValueError(
                "position_embedding must be one of {'learned', 'cosine', 'rope', 'rope2d'}: "
                f"got {self.position_embedding!r}."
            )
        if self.position_embedding == "rope":
            head_dim = self.embed_dim // self.num_heads
            if head_dim % 2 != 0:
                raise ValueError(
                    "RoPE requires an even per-head dimension: "
                    f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                    f"head_dim={head_dim}."
                )
        if self.position_embedding == "rope2d":
            head_dim = self.embed_dim // self.num_heads
            if head_dim % 4 != 0:
                raise ValueError(
                    "RoPE 2D requires head_dim divisible by 4: "
                    f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                    f"head_dim={head_dim}."
                )
        if self.decoder_embed_dim is not None and self.decoder_embed_dim <= 0:
            raise ValueError("decoder_embed_dim must be a positive integer.")
        if self.decoder_depth is not None and self.decoder_depth <= 0:
            raise ValueError("decoder_depth must be a positive integer.")
        if self.decoder_num_heads is not None and self.decoder_num_heads <= 0:
            raise ValueError("decoder_num_heads must be a positive integer.")
        if (
            self.decoder_embed_dim is not None
            and self.decoder_num_heads is not None
            and self.decoder_embed_dim % self.decoder_num_heads != 0
        ):
            raise ValueError(
                "decoder_embed_dim must be divisible by decoder_num_heads: "
                f"got decoder_embed_dim={self.decoder_embed_dim}, "
                f"decoder_num_heads={self.decoder_num_heads}."
            )

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)
