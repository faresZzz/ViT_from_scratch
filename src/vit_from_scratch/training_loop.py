"""Shared training-loop helpers for the advanced runners.

Classification, MAE, and DINO pipelines share the same control-flow:
epoch iteration → metric aggregation → history tracking → best-metric
updates → checkpoint persistence → figure saving.  This module extracts
those mechanical patterns so each runner can focus on its own step
functions, loss computations, and diagnostics.

Unlike ``training.py`` (the simple pedagogical module used by notebooks),
these helpers target the full pipeline runners that persist artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch import nn

from vit_from_scratch.artifacts import (
    ExperimentPaths,
    create_experiment_run,
    load_checkpoint,
    save_figure,
)
from vit_from_scratch.progress import iter_progress
from vit_from_scratch.run_utils import close_figure, paths_from_checkpoint


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------


def run_epoch(
    *,
    dataloader: object,
    step_fn: Callable[..., dict[str, float]],
    step_kwargs: dict[str, object],
    batch_size_fn: Callable[[object], int],
    default_metrics: dict[str, float],
    progress_desc: str | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """Run one training or evaluation epoch with batch-weighted metric aggregation.

    Parameters
    ----------
    dataloader:
        Iterable of batches (any type the caller's ``step_fn`` understands).
    step_fn:
        Called as ``step_fn(batch=batch, **step_kwargs)`` for each batch.
        Must return a dict of scalar metrics.
    step_kwargs:
        Extra keyword arguments forwarded to every ``step_fn`` call.
    batch_size_fn:
        Extracts the batch size from a raw batch object.
    default_metrics:
        Returned when the dataloader is empty.
    progress_desc:
        If provided, wraps the dataloader with a progress bar.
    show_progress:
        Enables or suppresses the progress bar.
    """

    totals: dict[str, float] = {}
    num_samples = 0

    batches: object
    if progress_desc is not None:
        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        batches = iter_progress(
            dataloader,
            desc=progress_desc,
            total=total,
            enabled=show_progress,
        )
    else:
        batches = dataloader

    for batch in batches:
        batch_size = batch_size_fn(batch)
        metrics = step_fn(batch=batch, **step_kwargs)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        num_samples += batch_size

    if num_samples == 0:
        return {key: float(value) for key, value in default_metrics.items()}
    return {key: total / num_samples for key, total in totals.items()}


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------


def load_resume_state(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    expected_approach: str,
    target_epochs: int,
    model_loaders: dict[str, tuple[nn.Module, str]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[ExperimentPaths, dict, int, Path]:
    """Load a checkpoint, restore model/optimizer/scheduler state, and validate.

    Parameters
    ----------
    model_loaders:
        Maps a descriptive key to ``(module, state_dict_key_in_checkpoint)``.
        For classification/MAE: ``{"model": (model, "model_state_dict")}``.
        For DINO: ``{"student": (student, "student_state_dict"),
        "teacher": (teacher, "teacher_state_dict")}``.

    Returns
    -------
    ``(run_paths, history, start_epoch, resolved_checkpoint_path)``
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("approach") != expected_approach:
        raise ValueError(
            f"Checkpoint approach mismatch: expected {expected_approach!r}, "
            f"got {checkpoint.get('approach')!r}."
        )

    epoch = int(checkpoint.get("epoch", 0))
    if epoch <= 0:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain a valid epoch."
        )
    if target_epochs <= epoch:
        raise ValueError(
            "Nothing to resume: target epochs must be greater than checkpoint epoch "
            f"({target_epochs} <= {epoch})."
        )

    for _label, (module, state_key) in model_loaders.items():
        module.load_state_dict(checkpoint[state_key])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)

    history = checkpoint.get("history")
    if not isinstance(history, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain a history dict."
        )
    history.setdefault("lr", [])

    return paths_from_checkpoint(checkpoint_path), history, epoch + 1, checkpoint_path


def prepare_training_run(
    *,
    approach: str,
    output_dir: str | Path,
    run_name: str | None,
    resume_checkpoint: str | Path | None,
    device: torch.device,
    target_epochs: int,
    models: list[nn.Module],
    model_loaders: dict[str, tuple[nn.Module, str]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    initial_history_fn: Callable[[], dict[str, list[float]]],
) -> tuple[ExperimentPaths, dict[str, list[float]], int, int | None, Path | None]:
    """Create a fresh run or resume from a checkpoint.

    All models in *models* are moved to *device*.

    Returns
    -------
    ``(run_paths, history, start_epoch, resumed_from_epoch, resolved_checkpoint)``
    """

    for model in models:
        model.to(device)

    if resume_checkpoint is None:
        run_paths = create_experiment_run(
            root=output_dir,
            approach=approach,
            run_name=run_name,
        )
        return run_paths, initial_history_fn(), 1, None, None

    run_paths, history, start_epoch, resolved_path = load_resume_state(
        checkpoint_path=resume_checkpoint,
        device=device,
        expected_approach=approach,
        target_epochs=target_epochs,
        model_loaders=model_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return run_paths, history, start_epoch, start_epoch - 1, resolved_path


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


def append_history(
    history: dict[str, list[float]],
    *,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    metric_keys: dict[str, str],
    lr: float,
) -> None:
    """Append one epoch's metrics to a history dict.

    *metric_keys* maps history key names to the corresponding key in
    ``train_metrics`` or ``val_metrics``.  Keys starting with ``"train_"``
    read from *train_metrics*; all others read from *val_metrics*.

    Example::

        metric_keys = {
            "train_loss": "loss",
            "train_accuracy": "accuracy",
            "val_loss": "loss",
            "val_accuracy": "accuracy",
        }
    """

    for history_key, metric_key in metric_keys.items():
        source = train_metrics if history_key.startswith("train_") else val_metrics
        history[history_key].append(float(source[metric_key]))
    history.setdefault("lr", []).append(float(lr))


# ---------------------------------------------------------------------------
# Checkpoint state builder
# ---------------------------------------------------------------------------


def build_checkpoint_state(
    *,
    approach: str,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    history: Mapping[str, list[float]],
    best_val_loss: float | None,
    best_checkpoints: Mapping[str, Path | None],
    model_states: dict[str, object],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the standard checkpoint dict.

    *model_states* maps state-dict keys to the actual state dicts::

        {"model_state_dict": model.state_dict()}
        # or for DINO:
        {"student_state_dict": student.state_dict(),
         "teacher_state_dict": teacher.state_dict()}

    *extra* allows approach-specific fields (``label_smoothing``,
    ``center``, ``view_config``, …).
    """

    state: dict[str, object] = {
        "approach": approach,
        "epoch": epoch,
        **model_states,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_checkpoints": {
            key: None if path is None else str(path)
            for key, path in best_checkpoints.items()
        },
    }
    if extra:
        state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Best-metric tracking
# ---------------------------------------------------------------------------


def update_best_metric(
    *,
    current_value: float,
    best_value: float | None,
    mode: str = "min",
) -> tuple[float | None, bool]:
    """Check whether a metric has improved.

    Returns ``(new_best, did_improve)``.
    """

    if best_value is None:
        return current_value, True
    if mode == "min":
        improved = current_value <= best_value
    elif mode == "max":
        improved = current_value >= best_value
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    return (current_value if improved else best_value), improved


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------


def save_figure_safely(figure: object, path: Path) -> None:
    """Save a matplotlib figure and guarantee it is closed afterward."""

    try:
        save_figure(figure, path)
    finally:
        close_figure(figure)


# ---------------------------------------------------------------------------
# Cosine schedules
# ---------------------------------------------------------------------------


def cosine_momentum_schedule(
    epoch: int,
    total_epochs: int,
    base: float = 0.996,
) -> float:
    """Compute teacher-EMA momentum following a cosine schedule (DINO paper).

    Momentum increases from *base* to 1.0 over *total_epochs*.
    """

    return 1.0 - (1.0 - base) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2
