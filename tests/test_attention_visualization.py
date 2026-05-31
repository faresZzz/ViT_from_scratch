import matplotlib
import pytest

matplotlib.use("Agg")

torch = pytest.importorskip("torch")

from matplotlib.figure import Figure

from vit_from_scratch import ViTConfig
from vit_from_scratch.model import VisionTransformer
from vit_from_scratch.visualization import (
    make_patch_grid,
    plot_class_attention,
    plot_predictions,
    plot_training_curves,
)


def _make_config() -> ViTConfig:
    return ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


def test_forward_with_attention_returns_logits_and_attention_maps():
    config = _make_config()
    model = VisionTransformer(config)
    images = torch.randn(2, 3, 32, 32)

    logits, attention_maps = model.forward_with_attention(images)

    tokens = (config.image_size // config.patch_size) ** 2 + 1
    assert logits.shape == (2, config.num_classes)
    assert isinstance(attention_maps, tuple)
    assert len(attention_maps) == config.depth
    for layer_map in attention_maps:
        assert layer_map.shape == (2, config.num_heads, tokens, tokens)


def test_visualization_helpers_return_matplotlib_figures():
    config = _make_config()
    model = VisionTransformer(config)
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    logits, attention_maps = model.forward_with_attention(images)
    history = {
        "train_loss": [1.0],
        "val_loss": [1.2],
        "train_accuracy": [0.5],
        "val_accuracy": [0.25],
    }
    class_names = tuple(str(index) for index in range(config.num_classes))

    figures = [
        make_patch_grid(images[0], patch_size=config.patch_size),
        plot_training_curves(history),
        plot_predictions(images, labels, logits, class_names),
        plot_class_attention(
            attention_maps,
            images[0],
            patch_size=config.patch_size,
        ),
    ]

    for figure in figures:
        assert isinstance(figure, Figure)
        figure.canvas.draw()
