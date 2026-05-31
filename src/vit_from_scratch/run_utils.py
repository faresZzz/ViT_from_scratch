"""Shared helpers for experiment runners.

The training approaches keep their domain-specific loops, losses, and diagnostics
in separate modules. This module holds only generic run/checkpoint plumbing so
the public runners stay easier to read.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch

from vit_from_scratch.artifacts import ExperimentPaths


@dataclass(frozen=True)
class RunValidationConfig:
    """Common scalar arguments every persisted training run must validate."""

    epochs: int
    save_every: int
    checkpoint_keep_last: int
    label_smoothing: float | None = None


def to_serializable(value: object) -> object:
    """Convert common experiment objects to JSON-serializable structures."""

    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def paths_from_checkpoint(checkpoint_path: str | Path) -> ExperimentPaths:
    """Resolve the standard run directory layout from a checkpoint path."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    run_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints":
        raise ValueError(
            "resume_checkpoint must live under a run checkpoints directory: "
            f"got {checkpoint_path}."
        )
    return ExperimentPaths(
        run_dir=run_dir,
        figure_dir=run_dir / "figures",
        checkpoint_dir=checkpoint_dir,
        history_path=run_dir / "history.json",
        config_path=run_dir / "config.json",
    )


def best_metric_from_history(
    history: Mapping[str, list[float]],
    metric_name: str,
    *,
    mode: str = "min",
) -> float | None:
    """Return the best value for a metric already present in a history dict."""

    values = history.get(metric_name, [])
    if not values:
        return None
    numeric_values = [float(value) for value in values]
    if mode == "min":
        return min(numeric_values)
    if mode == "max":
        return max(numeric_values)
    raise ValueError(f"Unsupported best-metric mode: {mode}.")


def validate_training_run_args(config: RunValidationConfig) -> None:
    """Validate common run-level arguments before creating or resuming a run."""

    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if config.save_every <= 0:
        raise ValueError("save_every must be a positive integer.")
    if config.checkpoint_keep_last <= 0:
        raise ValueError("checkpoint_keep_last must be a positive integer.")
    if config.label_smoothing is not None and not 0.0 <= config.label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be between 0.0 and 1.0 inclusive.")


def aggregate_weighted_metrics(
    metrics_and_batch_sizes: list[tuple[Mapping[str, float], int]],
    *,
    default: Mapping[str, float],
) -> dict[str, float]:
    """Average metric dictionaries weighted by their batch sizes."""

    totals: dict[str, float] = {}
    num_samples = 0
    for metrics, batch_size in metrics_and_batch_sizes:
        if batch_size < 0:
            raise ValueError("batch_size must be greater than or equal to 0.")
        num_samples += int(batch_size)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * int(batch_size)

    if num_samples == 0:
        return {str(key): float(value) for key, value in default.items()}
    return {key: total / num_samples for key, total in totals.items()}


def find_named_checkpoint(checkpoint_dir: str | Path, prefix: str) -> Path | None:
    """Find a stable named checkpoint such as ``classification_best_val_loss.pt``."""

    matches = sorted(Path(checkpoint_dir).glob(f"{prefix}.pt"))
    return matches[-1] if matches else None


def close_figure(figure: object) -> None:
    """Close a matplotlib figure without making matplotlib a module import side effect."""

    with suppress(Exception):
        import matplotlib.pyplot as plt

        plt.close(figure)
