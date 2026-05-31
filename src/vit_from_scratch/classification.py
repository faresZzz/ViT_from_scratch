"""Minimal supervised classification helpers built on top of the ViT model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from vit_from_scratch.artifacts import (
    ExperimentPaths,
    save_best_checkpoint,
    save_checkpoint,
    save_history,
    save_json,
)
from vit_from_scratch.config import ViTConfig
from vit_from_scratch.evaluation import (
    evaluate_classification_model,
    predict_external_classification_images,
)
from vit_from_scratch.model import VisionTransformer
from vit_from_scratch.run_utils import (
    best_metric_from_history,
    find_named_checkpoint,
    to_serializable,
)
from vit_from_scratch.training import accuracy
from vit_from_scratch.training_loop import (
    append_history,
    build_checkpoint_state,
    prepare_training_run,
    run_epoch,
    save_figure_safely,
    update_best_metric,
)
from vit_from_scratch.visualization import plot_confusion_matrix, plot_external_predictions


def build_classification_model(
    config: ViTConfig | Mapping[str, object],
) -> VisionTransformer:
    """Build a classification-ready Vision Transformer."""

    resolved_config = config if isinstance(config, ViTConfig) else ViTConfig(**config)
    return VisionTransformer(resolved_config)


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the standard cross-entropy classification loss."""

    return nn.functional.cross_entropy(logits, labels)


def _move_batch(
    batch: tuple[Tensor, Tensor] | list[Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if len(batch) != 2:
        raise ValueError(
            "Expected a batch shaped like (images, labels): "
            f"got {len(batch)} elements."
        )
    images, labels = batch
    return images.to(device), labels.to(device)


def classification_train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor] | list[Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | callable | None = None,
) -> dict[str, float]:
    """Run one supervised training step for an ``(images, labels)`` batch."""

    criterion = loss_fn or classification_loss
    model.train()
    images, labels = _move_batch(batch, device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy(logits.detach(), labels),
    }


@torch.no_grad()
def classification_eval_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor] | list[Tensor],
    device: torch.device,
    loss_fn: nn.Module | callable | None = None,
) -> dict[str, float]:
    """Run one evaluation step for an ``(images, labels)`` batch."""

    criterion = loss_fn or classification_loss
    model.eval()
    images, labels = _move_batch(batch, device)
    logits = model(images)
    loss = criterion(logits, labels)

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy(logits, labels),
    }


def _save_training_curves(history: Mapping[str, list[float]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    save_figure_safely(figure, figure_dir / "training_curves.png")


def _validate_train_classification_args(
    epochs: int,
    save_every: int,
    checkpoint_keep_last: int,
    label_smoothing: float,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if save_every <= 0:
        raise ValueError("save_every must be a positive integer.")
    if checkpoint_keep_last <= 0:
        raise ValueError("checkpoint_keep_last must be a positive integer.")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be between 0.0 and 1.0 inclusive.")


def _initial_classification_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }


def _build_classification_config(
    *,
    model: nn.Module,
    device: torch.device,
    epochs: int,
    run_name: str | None,
    checkpoint_keep_last: int,
    save_every: int,
    class_names: list[str] | tuple[str, ...] | None,
    external_image_dir: str | Path | None,
    external_image_size: int | None,
    label_smoothing: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    resolved_resume_checkpoint: Path | None,
    resumed_from_epoch: int | None,
) -> dict[str, object]:
    return {
        "epochs": epochs,
        "target_epochs": epochs,
        "device": str(device),
        "run_name": run_name,
        "checkpoint_keep_last": checkpoint_keep_last,
        "save_every": save_every,
        "class_names": list(class_names) if class_names is not None else None,
        "external_image_dir": None if external_image_dir is None else str(external_image_dir),
        "external_image_size": external_image_size,
        "label_smoothing": label_smoothing,
        "scheduler": None if scheduler is None else scheduler.__class__.__name__,
        "resume_from": None if resolved_resume_checkpoint is None else str(resolved_resume_checkpoint),
        "resumed_from_epoch": resumed_from_epoch,
        "model_config": to_serializable(getattr(model, "config", None)),
    }


def _classification_best_checkpoint_paths(
    checkpoint_dir: Path,
) -> dict[str, Path | None]:
    return {
        "best_val_loss": find_named_checkpoint(
            checkpoint_dir,
            "classification_best_val_loss",
        ),
        "best_val_accuracy": find_named_checkpoint(
            checkpoint_dir,
            "classification_best_val_accuracy",
        ),
    }


def _class_names_for_model(
    model: nn.Module,
    class_names: list[str] | tuple[str, ...] | None,
) -> list[str] | tuple[str, ...]:
    if class_names is not None:
        return class_names
    num_classes = int(getattr(getattr(model, "config", None), "num_classes", 0) or 0)
    return tuple(str(index) for index in range(num_classes))


@torch.no_grad()
def _save_prediction_and_attention_figures(
    model: nn.Module,
    val_loader: object,
    device: torch.device,
    figure_dir: Path,
    class_names: list[str] | tuple[str, ...] | None,
) -> None:
    from vit_from_scratch.visualization import plot_class_attention, plot_predictions

    try:
        first_batch = next(iter(val_loader))
    except StopIteration:
        return

    images, labels = _move_batch(first_batch, device)
    if images.shape[0] == 0:
        return

    model.eval()
    logits = model(images)
    prediction_figure = plot_predictions(
        images=images.detach().cpu(),
        labels=labels.detach().cpu(),
        logits=logits.detach().cpu(),
        class_names=_class_names_for_model(model, class_names),
    )
    save_figure_safely(prediction_figure, figure_dir / "predictions.png")

    if not hasattr(model, "forward_with_attention"):
        return
    patch_size = getattr(getattr(model, "config", None), "patch_size", None)
    if patch_size is None:
        return

    try:
        _, attention_maps = model.forward_with_attention(images)
        attention_figure = plot_class_attention(
            attention_maps=attention_maps,
            image=images[0].detach().cpu(),
            patch_size=int(patch_size),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return

    save_figure_safely(attention_figure, figure_dir / "attention_map.png")


def _save_confusion_matrix_figure(
    classification_metrics: Mapping[str, object],
    *,
    figure_dir: Path,
    class_names: list[str] | tuple[str, ...] | None,
) -> None:
    confusion_matrix = classification_metrics.get("confusion_matrix")
    if not confusion_matrix:
        return

    confusion_figure = plot_confusion_matrix(
        confusion_matrix,
        class_names=class_names,
    )
    save_figure_safely(confusion_figure, figure_dir / "confusion_matrix.png")


def _save_external_prediction_figure(
    *,
    model: nn.Module,
    figure_dir: Path,
    external_image_dir: str | Path,
    device: torch.device,
    class_names: list[str] | tuple[str, ...] | None,
    external_image_size: int | None,
    external_mean: tuple[float, float, float] | list[float] | None,
    external_std: tuple[float, float, float] | list[float] | None,
) -> None:
    image_size = int(external_image_size or getattr(model.config, "image_size", 32))
    external_payload = predict_external_classification_images(
        model=model,
        image_dir=external_image_dir,
        device=device,
        image_size=image_size,
        mean=external_mean,
        std=external_std,
    )
    external_figure = plot_external_predictions(
        images=external_payload["images"],
        paths=[path.name for path in external_payload["paths"]],
        logits=external_payload["logits"],
        class_names=_class_names_for_model(model, class_names),
        mean=external_payload["mean"],
        std=external_payload["std"],
    )
    save_figure_safely(external_figure, figure_dir / "external_predictions.png")


def _classification_result_payload(
    *,
    history: dict[str, list[float]],
    run_paths: ExperimentPaths,
    best_val_loss: float | None,
    best_val_accuracy: float | None,
    best_checkpoints: Mapping[str, Path | None],
    resolved_resume_checkpoint: Path | None,
    resumed_from_epoch: int | None,
) -> dict[str, object]:
    return {
        "history": history,
        "run_dir": run_paths.run_dir,
        "checkpoint_dir": run_paths.checkpoint_dir,
        "figure_dir": run_paths.figure_dir,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_checkpoints": {
            key: None if path is None else str(path) for key, path in best_checkpoints.items()
        },
        "resume_from": None if resolved_resume_checkpoint is None else resolved_resume_checkpoint,
        "resumed_from_epoch": resumed_from_epoch,
    }


def train_classification(
    model: nn.Module,
    train_loader: object,
    val_loader: object,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    checkpoint_keep_last: int = 3,
    save_every: int = 1,
    class_names: list[str] | tuple[str, ...] | None = None,
    external_image_dir: str | Path | None = None,
    external_image_size: int | None = None,
    external_mean: tuple[float, float, float] | list[float] | None = None,
    external_std: tuple[float, float, float] | list[float] | None = None,
    label_smoothing: float = 0.0,
    show_progress: bool = True,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Train a classifier, persist artifacts, and return run metadata."""

    _validate_train_classification_args(
        epochs=epochs,
        save_every=save_every,
        checkpoint_keep_last=checkpoint_keep_last,
        label_smoothing=label_smoothing,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    (
        run_paths,
        history,
        start_epoch,
        resumed_from_epoch,
        resolved_resume_checkpoint,
    ) = prepare_training_run(
        approach="classification",
        output_dir=output_dir,
        run_name=run_name,
        resume_checkpoint=resume_checkpoint,
        device=device,
        target_epochs=epochs,
        models=[model],
        model_loaders={"model": (model, "model_state_dict")},
        optimizer=optimizer,
        scheduler=scheduler,
        initial_history_fn=_initial_classification_history,
    )
    history.setdefault("lr", [])

    best_val_loss = best_metric_from_history(history, "val_loss", mode="min")
    best_val_accuracy = best_metric_from_history(history, "val_accuracy", mode="max")
    best_checkpoints = _classification_best_checkpoint_paths(run_paths.checkpoint_dir)

    config = _build_classification_config(
        model=model,
        device=device,
        epochs=epochs,
        run_name=run_name,
        checkpoint_keep_last=checkpoint_keep_last,
        save_every=save_every,
        class_names=class_names,
        external_image_dir=external_image_dir,
        external_image_size=external_image_size,
        label_smoothing=label_smoothing,
        scheduler=scheduler,
        resolved_resume_checkpoint=resolved_resume_checkpoint,
        resumed_from_epoch=resumed_from_epoch,
    )
    save_json(to_serializable(config), run_paths.config_path)

    def _train_step(*, batch: object, **kwargs: object) -> dict[str, float]:
        return classification_train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            loss_fn=criterion,
        )

    def _eval_step(*, batch: object, **kwargs: object) -> dict[str, float]:
        return classification_eval_step(
            model=model,
            batch=batch,
            device=device,
            loss_fn=criterion,
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            dataloader=train_loader,
            step_fn=_train_step,
            step_kwargs={},
            batch_size_fn=lambda batch: int(batch[0].shape[0]),
            default_metrics={"loss": 0.0, "accuracy": 0.0},
            progress_desc=f"classification train {epoch}/{epochs}",
            show_progress=show_progress,
        )
        val_metrics = run_epoch(
            dataloader=val_loader,
            step_fn=_eval_step,
            step_kwargs={},
            batch_size_fn=lambda batch: int(batch[0].shape[0]),
            default_metrics={"loss": 0.0, "accuracy": 0.0},
            progress_desc=f"classification val {epoch}/{epochs}",
            show_progress=show_progress,
        )

        append_history(
            history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metric_keys={
                "train_loss": "loss",
                "train_accuracy": "accuracy",
                "val_loss": "loss",
                "val_accuracy": "accuracy",
            },
            lr=epoch_lr,
        )
        save_history(history, run_paths.run_dir)

        best_val_loss, improved_val_loss = update_best_metric(
            current_value=float(val_metrics["loss"]),
            best_value=best_val_loss,
            mode="min",
        )
        if improved_val_loss:
            best_checkpoints["best_val_loss"] = (
                run_paths.checkpoint_dir / "classification_best_val_loss.pt"
            )

        best_val_accuracy, improved_val_accuracy = update_best_metric(
            current_value=float(val_metrics["accuracy"]),
            best_value=best_val_accuracy,
            mode="max",
        )
        if improved_val_accuracy:
            best_checkpoints["best_val_accuracy"] = (
                run_paths.checkpoint_dir / "classification_best_val_accuracy.pt"
            )

        if scheduler is not None:
            scheduler.step()

        checkpoint_state = build_checkpoint_state(
            approach="classification",
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_loss=best_val_loss,
            best_checkpoints=best_checkpoints,
            model_states={"model_state_dict": model.state_dict()},
            extra={
                "label_smoothing": label_smoothing,
                "model_config": to_serializable(getattr(model, "config", None)),
                "best_val_accuracy": best_val_accuracy,
            },
        )
        if epoch % save_every == 0:
            save_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="classification",
                epoch=epoch,
                keep_last=checkpoint_keep_last,
            )

        if improved_val_loss:
            save_best_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="classification",
                metric_name="val_loss",
            )

        if improved_val_accuracy:
            save_best_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="classification",
                metric_name="val_accuracy",
            )

    _save_training_curves(history, run_paths.figure_dir)
    _save_prediction_and_attention_figures(
        model=model,
        val_loader=val_loader,
        device=device,
        figure_dir=run_paths.figure_dir,
        class_names=class_names,
    )
    classification_metrics = evaluate_classification_model(
        model=model,
        dataloader=val_loader,
        device=device,
    )
    save_json({"classification": classification_metrics}, run_paths.run_dir / "metrics.json")
    _save_confusion_matrix_figure(
        classification_metrics,
        figure_dir=run_paths.figure_dir,
        class_names=class_names,
    )

    if external_image_dir is not None:
        _save_external_prediction_figure(
            model=model,
            figure_dir=run_paths.figure_dir,
            external_image_dir=external_image_dir,
            device=device,
            class_names=class_names,
            external_image_size=external_image_size,
            external_mean=external_mean,
            external_std=external_std,
        )

    return _classification_result_payload(
        history=history,
        run_paths=run_paths,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        best_checkpoints=best_checkpoints,
        resolved_resume_checkpoint=resolved_resume_checkpoint,
        resumed_from_epoch=resumed_from_epoch,
    )
