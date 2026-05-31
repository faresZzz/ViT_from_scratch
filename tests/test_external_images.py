from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from vit_from_scratch.external_images import list_external_images, load_external_images


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (20, 18), color=color)
    image.save(path)


def test_list_external_images_filters_supported_extensions(tmp_path: Path):
    _write_image(tmp_path / "b.png", (255, 0, 0))
    _write_image(tmp_path / "a.jpg", (0, 255, 0))
    _write_image(tmp_path / "c.jpeg", (0, 0, 255))
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    paths = list_external_images(tmp_path)

    assert [path.name for path in paths] == ["a.jpg", "b.png", "c.jpeg"]


def test_load_external_images_returns_normalized_batch_and_paths(tmp_path: Path):
    _write_image(tmp_path / "one.png", (255, 128, 0))
    _write_image(tmp_path / "two.jpg", (0, 64, 255))

    images, paths = load_external_images(
        tmp_path,
        image_size=16,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    assert images.shape == (2, 3, 16, 16)
    assert images.dtype == torch.float32
    assert images.isfinite().all()
    assert [path.name for path in paths] == ["one.png", "two.jpg"]
    assert float(images.min()) >= -1.01
    assert float(images.max()) <= 1.01
