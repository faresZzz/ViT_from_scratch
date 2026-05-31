"""Training utilities for supervised Vision Transformer experiments."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    """Resolve a device, preferring CUDA, then MPS, then CPU."""

    normalized = name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available on this machine.")
        return torch.device("mps")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available on this machine.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "device must be one of {'auto', 'mps', 'cuda', 'cpu'}: "
        f"got {name!r}."
    )


def move_batch_to_device(
    batch: tuple[Tensor, Tensor] | list[Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Move an image/label batch to the target device."""

    if len(batch) != 2:
        raise ValueError(
            "Expected a batch with two elements (images, labels): "
            f"got {len(batch)} elements."
        )
    images, labels = batch
    return images.to(device), labels.to(device)


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute top-1 accuracy for a batch."""

    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def _mean_metrics(metric_sums: Mapping[str, float], num_samples: int) -> dict[str, float]:
    if num_samples == 0:
        return {name: 0.0 for name in metric_sums}
    return {name: float(total / num_samples) for name, total in metric_sums.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, float]:
    """Run one training epoch and return scalar metrics."""

    criterion = loss_fn or nn.CrossEntropyLoss()
    model.train()

    metric_sums: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
    num_samples = 0
    for batch in dataloader:
        images, targets = move_batch_to_device(batch, device)
        batch_size = targets.shape[0]
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        metric_sums["loss"] += float(loss.item()) * batch_size
        metric_sums["accuracy"] += accuracy(logits, targets) * batch_size
        num_samples += batch_size

    return _mean_metrics(metric_sums, num_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, float]:
    """Evaluate a model and return scalar metrics."""

    criterion = loss_fn or nn.CrossEntropyLoss()
    model.eval()

    metric_sums: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
    num_samples = 0
    for batch in dataloader:
        images, targets = move_batch_to_device(batch, device)
        batch_size = targets.shape[0]
        logits = model(images)
        loss = criterion(logits, targets)

        metric_sums["loss"] += float(loss.item()) * batch_size
        metric_sums["accuracy"] += accuracy(logits, targets) * batch_size
        num_samples += batch_size

    return _mean_metrics(metric_sums, num_samples)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, list[float]]:
    """Train for multiple epochs and return a metric history."""

    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    model.to(device)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for _ in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

    return history
