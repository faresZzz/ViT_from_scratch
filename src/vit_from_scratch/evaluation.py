"""Offline evaluation helpers for classification, MAE, and DINO experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from vit_from_scratch.data import CIFAR10_MEAN, CIFAR10_STD
from vit_from_scratch.external_images import load_external_images

if TYPE_CHECKING:
    from vit_from_scratch.masked_autoencoder import MaskedAutoencoder


def _extract_images_and_labels(
    batch: Tensor | tuple[Tensor, ...] | list[Tensor],
) -> tuple[Tensor, Tensor | None]:
    if isinstance(batch, Tensor):
        return batch, None
    if not batch:
        raise ValueError("Expected a tensor batch or a non-empty sequence.")
    images = batch[0]
    labels = None
    if len(batch) > 1 and isinstance(batch[1], Tensor) and batch[1].ndim == 1:
        labels = batch[1]
    return images, labels


def _entropy_from_probs(probs: Tensor) -> Tensor:
    safe_probs = probs.clamp_min(1e-12)
    return -(safe_probs * safe_probs.log()).sum(dim=-1)


def _classification_confusion_matrix(
    predictions: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
) -> Tensor:
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64,
        device=labels.device,
    )
    for true_label, predicted_label in zip(labels, predictions):
        true_index = int(true_label.item())
        predicted_index = int(predicted_label.item())
        if 0 <= true_index < num_classes and 0 <= predicted_index < num_classes:
            confusion[true_index, predicted_index] += 1
    return confusion


def _per_class_accuracy(confusion: Tensor) -> list[float | None]:
    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diag()
    return [
        None if int(total.item()) == 0 else float(correct.item() / total.item())
        for correct, total in zip(class_correct, class_totals)
    ]


def _empty_classification_metrics() -> dict[str, object]:
    return {
        "confusion_matrix": [],
        "per_class_accuracy": [],
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mean_confidence": None,
        "topk": None,
    }


def _resolve_normalization_stats(
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    resolved_mean = tuple(float(value) for value in (mean or CIFAR10_MEAN))
    resolved_std = tuple(float(value) for value in (std or CIFAR10_STD))
    if len(resolved_mean) != 3 or len(resolved_std) != 3:
        raise ValueError("mean and std must each contain exactly 3 values.")
    return resolved_mean, resolved_std


def compute_classification_metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    num_classes: int | None = None,
) -> dict[str, object]:
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected logits [B, C] and labels [B].")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have matching batch size.")

    resolved_num_classes = int(num_classes or logits.shape[-1])
    probs = logits.softmax(dim=-1)
    predictions = probs.argmax(dim=-1)
    confusion = _classification_confusion_matrix(
        predictions,
        labels,
        num_classes=resolved_num_classes,
    )

    topk = min(5, resolved_num_classes)
    topk_predictions = probs.topk(k=topk, dim=-1).indices
    topk_accuracy = float(
        topk_predictions.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    metrics: dict[str, object] = {
        "confusion_matrix": confusion.detach().cpu().tolist(),
        "per_class_accuracy": _per_class_accuracy(confusion),
        "top1_accuracy": float((predictions == labels).float().mean().item()),
        "mean_confidence": float(probs.max(dim=-1).values.mean().item()),
        "topk": int(topk),
        f"top{topk}_accuracy": topk_accuracy,
    }
    metrics["top5_accuracy"] = topk_accuracy if topk == 5 else None
    return metrics


def compute_mae_reconstruction_metrics(
    reconstructed_images: Tensor,
    target_images: Tensor,
    *,
    max_value: float = 1.0,
) -> dict[str, float]:
    if reconstructed_images.shape != target_images.shape:
        raise ValueError("reconstructed_images and target_images must share the same shape.")

    error = reconstructed_images - target_images
    mse = float(error.pow(2).mean().item())
    mae = float(error.abs().mean().item())
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10((max_value * max_value) / mse))
    return {
        "reconstruction_mse": mse,
        "reconstruction_mae": mae,
        "reconstruction_psnr": psnr,
    }


def compute_softmax_diagnostics(logits: Tensor) -> dict[str, float]:
    probs = logits.softmax(dim=-1)
    return {
        "entropy": float(_entropy_from_probs(probs).mean().item()),
        "confidence": float(probs.max(dim=-1).values.mean().item()),
    }


def compute_feature_std_mean(features: Tensor) -> float | None:
    if features.ndim != 2 or features.shape[0] == 0:
        return None
    if features.shape[0] == 1:
        return 0.0
    return float(features.std(dim=0, unbiased=True).mean().item())


def _normalized_features(features: Tensor) -> Tensor:
    return nn.functional.normalize(features.float(), dim=-1)


def compute_knn_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_labels: Tensor,
    *,
    k: int = 5,
) -> float | None:
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("train_features and val_features must be rank-2 tensors.")
    if train_features.shape[0] == 0 or val_features.shape[0] == 0:
        return None
    if train_labels.shape[0] != train_features.shape[0] or val_labels.shape[0] != val_features.shape[0]:
        raise ValueError("Feature and label tensors must align.")

    effective_k = max(1, min(int(k), train_features.shape[0]))
    normalized_train = _normalized_features(train_features)
    normalized_val = _normalized_features(val_features)
    similarities = normalized_val @ normalized_train.T
    neighbor_indices = similarities.topk(k=effective_k, dim=-1).indices
    neighbor_labels = train_labels[neighbor_indices]
    neighbor_similarities = similarities.gather(1, neighbor_indices)

    num_classes = int(train_labels.max().item()) + 1
    one_hot = nn.functional.one_hot(neighbor_labels.long(), num_classes).float()
    weighted_votes = (one_hot * neighbor_similarities.unsqueeze(-1)).sum(dim=1)
    predicted_labels = weighted_votes.argmax(dim=-1).to(val_labels.device, dtype=val_labels.dtype)
    return float((predicted_labels == val_labels).float().mean().item())


def compute_attention_entropy_and_peak_mass(
    attention_maps: Sequence[Tensor],
) -> dict[str, float | None]:
    if not attention_maps:
        return {"attention_entropy": None, "attention_peak_mass": None}

    cls_attention = attention_maps[-1][:, :, 0, 1:]
    if cls_attention.numel() == 0:
        return {"attention_entropy": None, "attention_peak_mass": None}

    cls_attention = cls_attention.mean(dim=1)
    cls_attention = cls_attention / cls_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = _entropy_from_probs(cls_attention)
    peak_mass = cls_attention.max(dim=-1).values
    return {
        "attention_entropy": float(entropy.mean().item()),
        "attention_peak_mass": float(peak_mass.mean().item()),
    }


def _batched_classification_outputs(
    model: nn.Module,
    dataloader: object,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[list[Tensor], list[Tensor], int]:
    logits_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    num_classes = int(getattr(getattr(model, "config", None), "num_classes", 0) or 0)

    model.eval()
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = _extract_images_and_labels(batch)
        if labels is None:
            continue
        logits = model(images.to(device)).detach().cpu()
        logits_batches.append(logits)
        label_batches.append(labels.detach().cpu())
        if num_classes <= 0:
            num_classes = int(logits.shape[-1])

    return logits_batches, label_batches, num_classes


@torch.no_grad()
def evaluate_classification_model(
    model: nn.Module,
    dataloader: object,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, object]:
    logits_batches, label_batches, num_classes = _batched_classification_outputs(
        model,
        dataloader,
        device,
        max_batches=max_batches,
    )

    if not logits_batches:
        return _empty_classification_metrics()

    return compute_classification_metrics(
        torch.cat(logits_batches, dim=0),
        torch.cat(label_batches, dim=0),
        num_classes=num_classes or None,
    )


@torch.no_grad()
def evaluate_masked_autoencoder(
    model: MaskedAutoencoder,
    dataloader: object,
    device: torch.device,
    *,
    mask_ratio: float = 0.5,
    max_batches: int = 4,
) -> dict[str, object]:
    from vit_from_scratch.masked_autoencoder import (
        patchify,
        random_patch_mask,
        unpatchify,
    )

    sample_mse_values: list[float] = []
    sample_images: list[Tensor] = []
    reconstructed_images_list: list[Tensor] = []

    model.eval()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        images = _extract_images_and_labels(batch)[0].to(device)
        target_patches = patchify(images, patch_size=model.config.patch_size)
        patch_mask = random_patch_mask(
            batch_size=images.shape[0],
            num_patches=target_patches.shape[1],
            mask_ratio=mask_ratio,
            device=device,
        )
        predicted_patches = model(images, patch_mask)
        reconstructed_patches = torch.where(
            patch_mask.unsqueeze(-1),
            predicted_patches,
            target_patches,
        )
        reconstructed_images = unpatchify(
            reconstructed_patches,
            patch_size=model.config.patch_size,
            image_size=model.config.image_size,
            channels=model.config.in_channels,
        ).clamp(0.0, 1.0)
        target_images = images.detach().clamp(0.0, 1.0)
        # MSE on masked patches only (consistent with training loss).
        error_sq = (predicted_patches - target_patches).pow(2)  # [B, N, patch_dim]
        mask_f = patch_mask.unsqueeze(-1).float()  # [B, N, 1]
        per_sample_mse = (error_sq * mask_f).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp(min=1)
        sample_mse_values.extend(float(value.item()) for value in per_sample_mse.detach().cpu())
        sample_images.append(target_images.detach().cpu())
        reconstructed_images_list.append(reconstructed_images.detach().cpu())

    if not sample_images:
        return {
            "reconstruction_mse": None,
            "reconstruction_mae": None,
            "reconstruction_psnr": None,
            "sample_mse": [],
        }

    metrics = compute_mae_reconstruction_metrics(
        torch.cat(reconstructed_images_list, dim=0),
        torch.cat(sample_images, dim=0),
    )
    metrics["sample_mse"] = sample_mse_values
    return metrics


@torch.no_grad()
def evaluate_dino(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: object,
    val_loader: object,
    device: torch.device,
    *,
    max_train_samples: int = 256,
    max_val_samples: int = 256,
) -> tuple[dict[str, object], dict[str, object]]:
    student.eval()
    teacher.eval()

    student_logits_batches: list[Tensor] = []
    teacher_logits_batches: list[Tensor] = []
    val_feature_batches: list[Tensor] = []
    val_label_batches: list[Tensor] = []
    train_feature_batches: list[Tensor] = []
    train_label_batches: list[Tensor] = []
    attention_payload: dict[str, object] = {}

    collected_val = 0
    for batch in val_loader:
        images, labels = _extract_images_and_labels(batch)
        if collected_val >= max_val_samples:
            break
        remaining = max_val_samples - collected_val
        images = images[:remaining].to(device)
        labels = None if labels is None else labels[:remaining]
        student_logits = student(images)
        teacher_logits = teacher(images)
        student_logits_batches.append(student_logits.detach().cpu())
        teacher_logits_batches.append(teacher_logits.detach().cpu())
        if hasattr(student, "encode_tokens"):
            tokens = student.encode_tokens(images)
            val_feature_batches.append(tokens[:, 0].detach().cpu())
        if labels is not None:
            val_label_batches.append(labels.detach().cpu())
        if not attention_payload and hasattr(student, "forward_with_attention"):
            try:
                _, attention_maps = student.forward_with_attention(images[:1])
                attention_payload = {
                    "image": images[0].detach().cpu(),
                    "attention_maps": tuple(map_.detach().cpu() for map_ in attention_maps),
                    "patch_size": int(getattr(getattr(student, "config", None), "patch_size", 0) or 0),
                }
            except (AttributeError, RuntimeError, TypeError, ValueError):
                attention_payload = {}
        collected_val += int(images.shape[0])

    metrics: dict[str, object] = {
        "student_entropy": None,
        "teacher_entropy": None,
        "student_confidence": None,
        "teacher_confidence": None,
        "feature_std_mean": None,
        "knn_accuracy": None,
        "attention_entropy": None,
        "attention_peak_mass": None,
    }

    if student_logits_batches:
        student_diagnostics = compute_softmax_diagnostics(torch.cat(student_logits_batches, dim=0))
        teacher_diagnostics = compute_softmax_diagnostics(torch.cat(teacher_logits_batches, dim=0))
        metrics["student_entropy"] = student_diagnostics["entropy"]
        metrics["student_confidence"] = student_diagnostics["confidence"]
        metrics["teacher_entropy"] = teacher_diagnostics["entropy"]
        metrics["teacher_confidence"] = teacher_diagnostics["confidence"]

    if val_feature_batches:
        val_features = torch.cat(val_feature_batches, dim=0)
        metrics["feature_std_mean"] = compute_feature_std_mean(val_features)
    else:
        val_features = None

    if attention_payload.get("attention_maps"):
        metrics.update(
            compute_attention_entropy_and_peak_mass(attention_payload["attention_maps"])
        )

    if val_features is not None and val_label_batches:
        collected_train = 0
        for batch in train_loader:
            images, labels = _extract_images_and_labels(batch)
            if labels is None or collected_train >= max_train_samples:
                if collected_train >= max_train_samples:
                    break
                continue
            remaining = max_train_samples - collected_train
            images = images[:remaining].to(device)
            labels = labels[:remaining]
            if hasattr(student, "encode_tokens"):
                features = student.encode_tokens(images)[:, 0]
            else:
                features = student(images)
            train_feature_batches.append(features.detach().cpu())
            train_label_batches.append(labels.detach().cpu())
            collected_train += int(images.shape[0])

        if train_feature_batches and val_label_batches:
            metrics["knn_accuracy"] = compute_knn_accuracy(
                train_features=torch.cat(train_feature_batches, dim=0),
                train_labels=torch.cat(train_label_batches, dim=0),
                val_features=val_features,
                val_labels=torch.cat(val_label_batches, dim=0),
                k=5,
            )

    return metrics, attention_payload


@torch.no_grad()
def predict_external_classification_images(
    model: nn.Module,
    image_dir: str | Path,
    device: torch.device,
    *,
    image_size: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    max_images: int | None = 8,
) -> dict[str, object]:
    resolved_mean, resolved_std = _resolve_normalization_stats(mean, std)
    images, paths = load_external_images(
        image_dir=image_dir,
        image_size=image_size,
        mean=resolved_mean,
        std=resolved_std,
        max_images=max_images,
    )
    model.eval()
    logits = model(images.to(device)).detach().cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidences = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
    return {
        "images": images.detach().cpu(),
        "paths": paths,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "confidences": confidences,
        "mean": resolved_mean,
        "std": resolved_std,
    }


@torch.no_grad()
def reconstruct_external_images(
    model: MaskedAutoencoder,
    image_dir: str | Path,
    device: torch.device,
    *,
    image_size: int,
    mask_ratio: float = 0.5,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    max_images: int | None = 6,
) -> dict[str, object]:
    from vit_from_scratch.masked_autoencoder import patchify, random_patch_mask, unpatchify

    resolved_mean, resolved_std = _resolve_normalization_stats(mean, std)
    images, paths = load_external_images(
        image_dir=image_dir,
        image_size=image_size,
        mean=resolved_mean,
        std=resolved_std,
        max_images=max_images,
    )
    images = images.to(device)
    target_patches = patchify(images, patch_size=model.config.patch_size)
    patch_mask = random_patch_mask(
        batch_size=images.shape[0],
        num_patches=target_patches.shape[1],
        mask_ratio=mask_ratio,
        device=device,
    )
    predicted_patches = model(images, patch_mask)
    reconstructed_patches = torch.where(
        patch_mask.unsqueeze(-1),
        predicted_patches,
        target_patches,
    )
    reconstructed_images = unpatchify(
        reconstructed_patches,
        patch_size=model.config.patch_size,
        image_size=model.config.image_size,
        channels=model.config.in_channels,
    ).detach().cpu()
    return {
        "images": images.detach().cpu(),
        "reconstructed_images": reconstructed_images,
        "patch_mask": patch_mask.detach().cpu(),
        "paths": paths,
        "mean": resolved_mean,
        "std": resolved_std,
    }


@torch.no_grad()
def evaluate_external_dino_attention(
    student: nn.Module,
    image_dir: str | Path,
    device: torch.device,
    *,
    image_size: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    max_images: int | None = 4,
) -> dict[str, object]:
    if not hasattr(student, "forward_with_attention"):
        raise AttributeError("student model must define forward_with_attention.")

    resolved_mean, resolved_std = _resolve_normalization_stats(mean, std)
    images, paths = load_external_images(
        image_dir=image_dir,
        image_size=image_size,
        mean=resolved_mean,
        std=resolved_std,
        max_images=max_images,
    )
    model_images = images.to(device)
    _, attention_maps = student.forward_with_attention(model_images)
    return {
        "images": images.detach().cpu(),
        "attention_maps": tuple(map_.detach().cpu() for map_ in attention_maps),
        "paths": paths,
        "patch_size": int(getattr(getattr(student, "config", None), "patch_size", 0) or 0),
        "mean": resolved_mean,
        "std": resolved_std,
    }
