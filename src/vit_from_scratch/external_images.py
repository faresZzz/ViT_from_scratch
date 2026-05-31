"""Helpers for loading real external images for post-training evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

SUPPORTED_EXTERNAL_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png")


def list_external_images(image_dir: str | Path) -> list[Path]:
    """Return supported external image paths sorted by name."""

    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"External image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"External image path is not a directory: {root}")

    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTERNAL_IMAGE_SUFFIXES
    )


def load_external_images(
    image_dir: str | Path,
    image_size: int,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
    max_images: int | None = None,
) -> tuple[Tensor, list[Path]]:
    """Load external RGB images and return a normalized batch shaped ``[B, 3, H, W]``."""

    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain exactly 3 values.")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be a positive integer when provided.")

    image_paths = list_external_images(image_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No supported external images found in {Path(image_dir)}.")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )

    images: list[Tensor] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(transform(image.convert("RGB")))

    return torch.stack(images, dim=0), image_paths
