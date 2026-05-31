"""Visualization helpers for Vision Transformer training and attention maps."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor


def _hide_axis_ticks(*axes) -> None:
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def _overlay_attention_heatmap(
    ax,
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    *,
    cmap: str = "magma",
    title: str | None = None,
    add_colorbar: bool = True,
    figure: Figure | None = None,
) -> None:
    """Draw an attention heatmap overlay on top of an image."""
    ax.imshow(image_np)
    overlay = ax.imshow(heatmap_np, cmap=cmap, alpha=0.55)
    if title:
        ax.set_title(title)
    if add_colorbar and figure is not None:
        figure.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    _hide_axis_ticks(ax)


def _resolve_class_name(class_names: Sequence[str], index: int) -> str:
    return class_names[index] if index < len(class_names) else str(index)


def _draw_patch_grid(axis, *, height: int, width: int, patch_size: int) -> None:
    for y in range(patch_size, height, patch_size):
        axis.axhline(y - 0.5, color="white", linewidth=0.8, alpha=0.9)
    for x in range(patch_size, width, patch_size):
        axis.axvline(x - 0.5, color="white", linewidth=0.8, alpha=0.9)


def _make_image_grid(
    num_images: int,
    *,
    max_columns: int,
    scale: tuple[int, int],
) -> tuple[Figure, np.ndarray, int, int]:
    if num_images <= 0:
        raise ValueError("Expected at least one image to plot.")
    num_cols = min(max_columns, num_images)
    num_rows = math.ceil(num_images / num_cols)
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(scale[0] * num_cols, scale[1] * num_rows),
    )
    return figure, np.atleast_1d(axes).reshape(num_rows, num_cols), num_rows, num_cols


def _cls_attention_heatmap(
    cls_attention: Tensor,
    *,
    image_height: int,
    image_width: int,
    patch_size: int,
) -> Tensor:
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    patch_attention = cls_attention.reshape(1, 1, grid_height, grid_width)
    return F.interpolate(
        patch_attention,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _to_channel_last_image(image: Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [C, H, W]: "
            f"got tensor with shape {tuple(image.shape)}."
        )
    if image.shape[0] not in {1, 3}:
        raise ValueError(
            "Expected 1 or 3 channels in image tensor: "
            f"got {image.shape[0]}."
        )

    image = image.detach().cpu().float()
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def unnormalize_image(image: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """Undo channel-wise normalization for an image tensor shaped ``[C, H, W]``."""
    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [C, H, W]: "
            f"got tensor with shape {tuple(image.shape)}."
        )

    mean_tensor = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    if mean_tensor.shape[0] != image.shape[0] or std_tensor.shape[0] != image.shape[0]:
        raise ValueError(
            "Mean and std must match the channel dimension of the image: "
            f"got channels={image.shape[0]}, mean={mean_tensor.shape[0]}, std={std_tensor.shape[0]}."
        )

    return image * std_tensor + mean_tensor


def make_patch_grid(image: Tensor, patch_size: int) -> Figure:
    """Return a figure showing the image with patch boundaries overlaid."""
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")

    _, height, width = image.shape
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(_to_channel_last_image(image))
    _draw_patch_grid(axis, height=height, width=width, patch_size=patch_size)
    _hide_axis_ticks(axis)
    axis.set_title("Patch grid")
    figure.tight_layout()
    return figure


def plot_training_curves(history: dict[str, Sequence[float]]) -> Figure:
    """Return a figure with training and validation curves from a history dict."""
    if not history:
        raise ValueError("history must not be empty.")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
    ]

    for axis, (metric_key, title) in zip(axes, metrics):
        train_key = f"train_{metric_key}"
        val_key = f"val_{metric_key}"
        plotted = False

        if train_key in history:
            axis.plot(history[train_key], label="train")
            plotted = True
        if val_key in history:
            axis.plot(history[val_key], label="val")
            plotted = True

        if not plotted and metric_key in history:
            axis.plot(history[metric_key], label=metric_key)
            plotted = True

        if plotted:
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.legend()
        else:
            axis.set_visible(False)

    figure.tight_layout()
    return figure


def plot_predictions(
    images: Tensor,
    labels: Tensor,
    logits: Tensor,
    class_names: Sequence[str],
    max_images: int = 8,
) -> Figure:
    """Return a figure with model predictions for a batch of images."""
    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape [B, C, H, W]: "
            f"got tensor with shape {tuple(images.shape)}."
        )
    if labels.ndim != 1 or logits.ndim != 2:
        raise ValueError(
            "Expected labels [B] and logits [B, num_classes]: "
            f"got labels {tuple(labels.shape)} and logits {tuple(logits.shape)}."
        )
    if images.shape[0] != labels.shape[0] or images.shape[0] != logits.shape[0]:
        raise ValueError("images, labels, and logits must share the same batch size.")
    if max_images <= 0:
        raise ValueError(f"max_images must be positive, got {max_images}.")

    num_images = min(images.shape[0], max_images)
    if num_images == 0:
        raise ValueError("images must contain at least one example.")
    figure, axes_array, _, _ = _make_image_grid(
        num_images,
        max_columns=4,
        scale=(4, 4),
    )

    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)

    for index, axis in enumerate(axes_array.flat):
        if index >= num_images:
            axis.axis("off")
            continue

        axis.imshow(_to_channel_last_image(images[index]))
        true_index = int(labels[index].detach().cpu().item())
        pred_index = int(predictions[index].detach().cpu().item())
        confidence = float(probabilities[index, pred_index].detach().cpu().item())

        true_name = _resolve_class_name(class_names, true_index)
        pred_name = _resolve_class_name(class_names, pred_index)
        axis.set_title(f"true: {true_name}\npred: {pred_name} ({confidence:.2%})")
        _hide_axis_ticks(axis)

    figure.tight_layout()
    return figure


def plot_external_predictions(
    images: Tensor,
    paths: Sequence[str],
    logits: Tensor,
    class_names: Sequence[str],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 8,
) -> Figure:
    """Return a figure with classification predictions for external images."""
    if images.ndim != 4 or logits.ndim != 2:
        raise ValueError("Expected images [B, C, H, W] and logits [B, num_classes].")
    if images.shape[0] != logits.shape[0] or images.shape[0] != len(paths):
        raise ValueError("images, paths, and logits must share the same batch size.")

    num_images = min(images.shape[0], max_images)
    figure, axes_array, _, _ = _make_image_grid(
        max(1, num_images),
        max_columns=4,
        scale=(4, 4),
    )

    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)

    for index, axis in enumerate(axes_array.flat):
        if index >= num_images:
            axis.axis("off")
            continue

        image = unnormalize_image(images[index], mean=mean, std=std)
        axis.imshow(_to_channel_last_image(image))
        pred_index = int(predictions[index].detach().cpu().item())
        confidence = float(probabilities[index, pred_index].detach().cpu().item())
        pred_name = _resolve_class_name(class_names, pred_index)
        axis.set_title(f"{paths[index]}\npred: {pred_name} ({confidence:.2%})")
        _hide_axis_ticks(axis)

    figure.tight_layout()
    return figure


def plot_class_attention(
    attention_maps: Sequence[Tensor],
    image: Tensor,
    patch_size: int,
    layer: int = -1,
    head: str | int = "mean",
) -> Figure:
    """Return a figure visualizing class-token attention over image patches."""
    if not attention_maps:
        raise ValueError("attention_maps must contain at least one layer.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [C, H, W]: "
            f"got tensor with shape {tuple(image.shape)}."
        )

    selected_layer = attention_maps[layer]
    if selected_layer.ndim != 4:
        raise ValueError(
            "Each attention map must have shape [B, H, N, N]: "
            f"got tensor with shape {tuple(selected_layer.shape)}."
        )
    if selected_layer.shape[0] < 1:
        raise ValueError("Attention maps must contain at least one batch element.")

    layer_attention = selected_layer[0]
    if head == "mean":
        class_attention = layer_attention[:, 0, 1:].mean(dim=0)
        head_label = "mean"
    elif isinstance(head, int):
        if head < 0 or head >= layer_attention.shape[0]:
            raise ValueError(
                f"head index {head} is out of range for {layer_attention.shape[0]} heads."
            )
        class_attention = layer_attention[head, 0, 1:]
        head_label = str(head)
    else:
        raise ValueError("head must be 'mean' or an integer head index.")

    _, image_height, image_width = image.shape
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError(
            "Image height and width must be divisible by patch_size: "
            f"got image shape {(image_height, image_width)} and patch_size={patch_size}."
        )

    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    expected_patches = grid_height * grid_width
    if class_attention.numel() != expected_patches:
        raise ValueError(
            "Attention map patch count does not match the image grid: "
            f"got {class_attention.numel()} patches, expected {expected_patches}."
        )

    heatmap = _cls_attention_heatmap(
        class_attention,
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
    )
    heatmap_np = heatmap.detach().cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    image_np = _to_channel_last_image(image)

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    _hide_axis_ticks(axes[0])
    axes[1].imshow(image_np)
    _draw_patch_grid(axes[1], height=image_height, width=image_width, patch_size=patch_size)
    axes[1].set_title("Patch grid")
    _hide_axis_ticks(axes[1])

    _overlay_attention_heatmap(
        axes[2], image_np, heatmap_np,
        cmap="Reds",
        title=f"Layer {layer} cls attention (head {head_label})",
        figure=figure,
    )

    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    class_names: Sequence[str] | None = None,
) -> Figure:
    """Return a figure visualizing a confusion matrix."""
    matrix = np.asarray(confusion_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square matrix.")

    size = matrix.shape[0]
    labels = list(class_names) if class_names is not None else [str(index) for index in range(size)]
    if len(labels) < size:
        labels.extend(str(index) for index in range(len(labels), size))

    figure, axis = plt.subplots(figsize=(max(4, size), max(4, size)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    axis.set_xticks(np.arange(size))
    axis.set_yticks(np.arange(size))
    axis.set_xticklabels(labels[:size], rotation=45, ha="right")
    axis.set_yticklabels(labels[:size])

    max_value = matrix.max() if matrix.size else 0.0
    for row in range(size):
        for column in range(size):
            value = int(matrix[row, column])
            text_color = "white" if max_value and matrix[row, column] > max_value / 2.0 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=text_color)

    figure.tight_layout()
    return figure


def plot_reconstruction_errors(errors: Sequence[float]) -> Figure:
    """Return a figure visualizing reconstruction error distribution."""
    if not errors:
        raise ValueError("errors must contain at least one value.")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    values = np.asarray(list(errors), dtype=float)

    axes[0].hist(values, bins=min(10, max(3, len(values))), color="#4C78A8", alpha=0.9)
    axes[0].set_title("Sample MSE distribution")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Count")

    axes[1].bar(np.arange(len(values)), values, color="#F58518", alpha=0.9)
    axes[1].set_title("Sample MSE")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("MSE")

    figure.tight_layout()
    return figure


def plot_external_reconstructions(
    images: Tensor,
    reconstructed_images: Tensor,
    paths: Sequence[str],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 4,
) -> Figure:
    """Return a figure comparing external images with their MAE reconstructions."""
    if images.shape != reconstructed_images.shape:
        raise ValueError("images and reconstructed_images must share the same shape.")
    if images.shape[0] != len(paths):
        raise ValueError("images and paths must share the same batch size.")

    num_images = min(images.shape[0], max_images)
    figure, axes = plt.subplots(num_images, 2, figsize=(8, 3 * num_images))
    if num_images == 1:
        axes = np.asarray([axes])

    for row in range(num_images):
        original = unnormalize_image(images[row], mean=mean, std=std)
        reconstructed = unnormalize_image(reconstructed_images[row], mean=mean, std=std)
        axes[row, 0].imshow(_to_channel_last_image(original))
        axes[row, 0].set_title(f"{paths[row]}\noriginal")
        axes[row, 1].imshow(_to_channel_last_image(reconstructed))
        axes[row, 1].set_title("reconstruction")
        _hide_axis_ticks(*axes[row])

    figure.tight_layout()
    return figure


def plot_dino_attention_diagnostics(
    image: Tensor,
    attention_maps: Sequence[Tensor],
    patch_size: int,
    stats: dict[str, object],
) -> Figure:
    """Return a compact DINO attention diagnostic figure."""
    if not attention_maps:
        raise ValueError("attention_maps must not be empty.")

    cls_attention = attention_maps[-1][0, :, 0, 1:].mean(dim=0)
    _, image_height, image_width = image.shape
    heatmap = _cls_attention_heatmap(
        cls_attention,
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    image_np = _to_channel_last_image(image)
    heatmap_np = heatmap.detach().cpu().numpy()

    axes[0].imshow(image_np)
    axes[0].set_title("Validation image")
    _hide_axis_ticks(axes[0])

    _overlay_attention_heatmap(
        axes[1], image_np, heatmap_np,
        cmap="magma", title="CLS attention", figure=figure,
    )

    stat_names = [
        "student_entropy",
        "teacher_entropy",
        "student_confidence",
        "teacher_confidence",
        "attention_entropy",
        "attention_peak_mass",
    ]
    stat_values = [
        0.0 if stats.get(name) is None else float(stats[name])
        for name in stat_names
    ]
    axes[2].barh(np.arange(len(stat_names)), stat_values, color="#54A24B")
    axes[2].set_yticks(np.arange(len(stat_names)))
    axes[2].set_yticklabels([name.replace("_", "\n") for name in stat_names])
    axes[2].set_title("Diagnostics")

    figure.tight_layout()
    return figure


def plot_external_dino_attention(
    images: Tensor,
    attention_maps: Sequence[Tensor],
    paths: Sequence[str],
    patch_size: int,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 4,
) -> Figure:
    """Return attention overlays for external images on top of the denormalized input."""
    if not attention_maps:
        raise ValueError("attention_maps must not be empty.")
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if images.shape[0] != len(paths):
        raise ValueError("images and paths must share the same batch size.")

    selected = attention_maps[-1]
    if selected.ndim != 4:
        raise ValueError("Expected attention maps with shape [B, H, N, N].")

    num_images = min(images.shape[0], max_images)
    figure, axes = plt.subplots(num_images, 2, figsize=(8, 3 * num_images))
    if num_images == 1:
        axes = np.asarray([axes])

    for row in range(num_images):
        image = unnormalize_image(images[row], mean=mean, std=std)
        _, image_height, image_width = image.shape
        cls_attention = selected[row, :, 0, 1:].mean(dim=0)
        heatmap = _cls_attention_heatmap(
            cls_attention,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
        )
        image_np = _to_channel_last_image(image)

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(str(paths[row]))
        _hide_axis_ticks(axes[row, 0])
        _overlay_attention_heatmap(
            axes[row, 1], image_np, heatmap.detach().cpu().numpy(),
            cmap="magma", title="CLS attention", add_colorbar=False,
        )

    figure.tight_layout()
    return figure
