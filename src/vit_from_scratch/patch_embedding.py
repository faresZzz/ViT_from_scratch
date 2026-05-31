"""Patch embedding layer for Vision Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vit_from_scratch.config import ViTConfig


class PatchEmbedding(nn.Module):
    """Turn an image batch into a sequence of patch tokens."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "PatchEmbedding expects images with shape [B, C, H, W]: "
                f"got tensor with shape {tuple(images.shape)}."
            )

        batch_size, channels, height, width = images.shape
        if channels != self.config.in_channels:
            raise ValueError(
                "Input channel count does not match config.in_channels: "
                f"got {channels}, expected {self.config.in_channels}."
            )
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                "Input spatial size does not match config.image_size: "
                f"got H={height}, W={width}, expected "
                f"{self.config.image_size}x{self.config.image_size}."
            )

        tokens = self.proj(images)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens
