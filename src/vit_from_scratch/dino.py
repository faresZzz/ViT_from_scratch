"""Minimal DINO-style student/teacher training helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vit_from_scratch.artifacts import (
    ExperimentPaths,
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    save_figure,
    save_history,
    save_json,
)
from vit_from_scratch.evaluation import evaluate_dino, evaluate_external_dino_attention
from vit_from_scratch.run_utils import (
    best_metric_from_history,
    find_named_checkpoint,
    to_serializable,
)
from vit_from_scratch.training_loop import (
    append_history,
    build_checkpoint_state,
    cosine_momentum_schedule,
    load_resume_state,
    prepare_training_run,
    run_epoch,
    save_figure_safely,
    update_best_metric,
)
from vit_from_scratch.visualization import (
    plot_class_attention,
    plot_dino_attention_diagnostics,
    plot_external_dino_attention,
    plot_training_curves,
)


@dataclass(frozen=True)
class DINOViewConfig:
    """Configure DINO teacher/student multi-crop views.

    The teacher receives global crops only. The student receives its own global
    crops plus optional local crops, matching the DINO multi-crop recipe.
    """

    teacher_global_crops: int = 2
    student_global_crops: int = 2
    student_local_crops: int = 2
    teacher_global_crop_scale: tuple[float, float] = (0.4, 1.0)
    student_global_crop_scale: tuple[float, float] = (0.4, 1.0)
    student_local_crop_scale: tuple[float, float] = (0.05, 0.4)
    noise_std: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "teacher_global_crops",
            "student_global_crops",
            "student_local_crops",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to 0.")
        if self.teacher_global_crops == 0:
            raise ValueError("teacher_global_crops must be at least 1.")
        if self.student_global_crops + self.student_local_crops == 0:
            raise ValueError("student must receive at least one view.")
        _validate_crop_scale("teacher_global_crop_scale", self.teacher_global_crop_scale)
        _validate_crop_scale("student_global_crop_scale", self.student_global_crop_scale)
        _validate_crop_scale("student_local_crop_scale", self.student_local_crop_scale)
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be greater than or equal to 0.")


@dataclass(frozen=True)
class DINOViews:
    """Resolved teacher and student views for one DINO batch."""

    teacher_views: tuple[Tensor, ...]
    student_views: tuple[Tensor, ...]
    teacher_view_ids: tuple[str, ...]
    student_view_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DINOForwardOutputs:
    student_logits: list[Tensor]
    teacher_logits: list[Tensor]


def _view_ids(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(count))


def _validate_crop_scale(name: str, scale: tuple[float, float]) -> None:
    if len(scale) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    minimum, maximum = scale
    if not 0.0 < minimum <= maximum <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < min <= max <= 1.")


class DINOCenter:
    """Maintain the moving-average logit center used by DINO teacher targets."""

    def __init__(
        self,
        dim: int,
        momentum: float = 0.9,
        device: torch.device | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1.")

        self.momentum = float(momentum)
        self.center = torch.zeros(1, dim, device=device)

    @torch.no_grad()
    def update(self, teacher_logits: Tensor) -> None:
        batch_mean = teacher_logits.detach().mean(dim=0, keepdim=True)
        self.center.mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)

    def to(self, device: torch.device) -> "DINOCenter":
        self.center = self.center.to(device)
        return self


def dino_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
    center: Tensor | None = None,
) -> Tensor:
    """Compute a simplified DINO cross-entropy between student and teacher views."""

    detached_teacher_logits = teacher_logits.detach()
    centered_teacher = (
        detached_teacher_logits
        if center is None
        else detached_teacher_logits - center
    )
    student_log_probs = nn.functional.log_softmax(
        student_logits / student_temperature, dim=-1
    )
    teacher_probs = nn.functional.softmax(
        centered_teacher / teacher_temperature, dim=-1
    )
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


@torch.no_grad()
def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float = 0.996,
) -> None:
    """Update teacher parameters with an exponential moving average of the student."""

    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def _resolve_views(
    batch_or_views: Tensor | tuple[Tensor, ...] | list[Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    views = build_dino_views(batch_or_views, device, DINOViewConfig())
    student_views = views.student_views
    if len(student_views) == 1:
        return student_views[0], student_views[0]
    return student_views[0], student_views[1]


def _extract_images_or_explicit_views(
    batch_or_views: Tensor | tuple[Tensor, ...] | list[Tensor],
    device: torch.device,
) -> Tensor | tuple[Tensor, ...]:
    if isinstance(batch_or_views, Tensor):
        if batch_or_views.ndim != 4:
            raise ValueError("Expected an image tensor with shape [B, C, H, W].")
        return batch_or_views.to(device)
    if (
        batch_or_views
        and len(batch_or_views) >= 2
        and all(isinstance(view, Tensor) and view.ndim == 4 for view in batch_or_views)
    ):
        return tuple(view.to(device) for view in batch_or_views)
    if batch_or_views and isinstance(batch_or_views[0], Tensor):
        images = batch_or_views[0]
        if images.ndim != 4:
            raise ValueError("Expected batch[0] to have shape [B, C, H, W].")
        return images.to(device)
    raise ValueError("Expected image tensor(s) for DINO training.")


def _random_resized_tensor_crop(
    images: Tensor,
    scale: tuple[float, float],
) -> Tensor:
    batch, channels, height, width = images.shape
    shortest_side = min(height, width)
    crops: list[Tensor] = []
    for image in images:
        area_scale = float(
            torch.empty((), device=images.device).uniform_(scale[0], scale[1]).item()
        )
        side = max(1, min(shortest_side, int(round(shortest_side * area_scale**0.5))))
        max_top = height - side
        max_left = width - side
        top = (
            0
            if max_top == 0
            else int(torch.randint(0, max_top + 1, (1,), device=images.device).item())
        )
        left = (
            0
            if max_left == 0
            else int(torch.randint(0, max_left + 1, (1,), device=images.device).item())
        )
        crop = image[:, top : top + side, left : left + side].view(1, channels, side, side)
        crops.append(
            F.interpolate(
                crop,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        )
    return torch.stack(crops, dim=0).view(batch, channels, height, width)


def _augment_dino_view(
    images: Tensor,
    scale: tuple[float, float],
    noise_std: float,
) -> Tensor:
    view = _random_resized_tensor_crop(images, scale)
    if torch.rand((), device=view.device).item() < 0.5:
        view = torch.flip(view, dims=(-1,))
    if noise_std > 0.0:
        view = view + noise_std * torch.randn_like(view)
    return view


def _build_explicit_dino_views(
    explicit_views: tuple[Tensor, ...],
    config: DINOViewConfig,
) -> DINOViews:
    teacher_count = min(config.teacher_global_crops, len(explicit_views))
    return DINOViews(
        teacher_views=explicit_views[:teacher_count],
        student_views=explicit_views,
        teacher_view_ids=_view_ids("explicit", teacher_count),
        student_view_ids=_view_ids("explicit", len(explicit_views)),
    )


def _build_augmented_dino_views(
    images: Tensor,
    config: DINOViewConfig,
) -> DINOViews:
    teacher_views = tuple(
        _augment_dino_view(
            images,
            config.teacher_global_crop_scale,
            config.noise_std,
        )
        for _ in range(config.teacher_global_crops)
    )
    student_global_views = tuple(
        _augment_dino_view(
            images,
            config.student_global_crop_scale,
            config.noise_std,
        )
        for _ in range(config.student_global_crops)
    )
    student_local_views = tuple(
        _augment_dino_view(
            images,
            config.student_local_crop_scale,
            config.noise_std,
        )
        for _ in range(config.student_local_crops)
    )
    return DINOViews(
        teacher_views=teacher_views,
        student_views=student_global_views + student_local_views,
        teacher_view_ids=_view_ids("global", config.teacher_global_crops),
        student_view_ids=(
            _view_ids("global", config.student_global_crops)
            + _view_ids("local", config.student_local_crops)
        ),
    )


def build_dino_views(
    batch_or_views: Tensor | tuple[Tensor, ...] | list[Tensor],
    device: torch.device,
    view_config: DINOViewConfig | None = None,
) -> DINOViews:
    """Build teacher global crops and student global/local crops for DINO."""

    config = view_config or DINOViewConfig()
    resolved = _extract_images_or_explicit_views(batch_or_views, device)
    if isinstance(resolved, tuple):
        return _build_explicit_dino_views(resolved, config)
    return _build_augmented_dino_views(resolved, config)


def _resolve_center_tensor(center: DINOCenter | Tensor | None) -> Tensor | None:
    if isinstance(center, DINOCenter):
        return center.center
    return center


def _forward_dino_views(
    student: nn.Module,
    teacher: nn.Module,
    views: DINOViews,
) -> _DINOForwardOutputs:
    student_logits = [student(view) for view in views.student_views]
    with torch.no_grad():
        teacher_logits = [teacher(view) for view in views.teacher_views]
    return _DINOForwardOutputs(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )


def _dino_multicrop_losses(
    student_logits: list[Tensor],
    teacher_logits: list[Tensor],
    student_view_ids: tuple[str, ...],
    teacher_view_ids: tuple[str, ...],
    student_temperature: float,
    teacher_temperature: float,
    center: Tensor | None,
) -> list[Tensor]:
    losses: list[Tensor] = []
    for teacher_index, teacher_view_logits in enumerate(teacher_logits):
        teacher_view_id = teacher_view_ids[teacher_index]
        for student_index, student_view_logits in enumerate(student_logits):
            if student_view_ids[student_index] == teacher_view_id:
                continue
            losses.append(
                dino_loss(
                    student_logits=student_view_logits,
                    teacher_logits=teacher_view_logits,
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                    center=center,
                )
            )
    if not losses:
        raise ValueError("DINO loss needs at least one non-matching teacher/student view pair.")
    return losses


def _mean_dino_multicrop_loss(
    outputs: _DINOForwardOutputs,
    views: DINOViews,
    *,
    student_temperature: float,
    teacher_temperature: float,
    center: Tensor | None,
) -> tuple[Tensor, int]:
    losses = _dino_multicrop_losses(
        student_logits=outputs.student_logits,
        teacher_logits=outputs.teacher_logits,
        student_view_ids=views.student_view_ids,
        teacher_view_ids=views.teacher_view_ids,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        center=center,
    )
    return torch.stack(losses).mean(), len(losses)


def _update_dino_center(
    center: DINOCenter | Tensor | None,
    teacher_logits: list[Tensor],
) -> None:
    if isinstance(center, DINOCenter):
        center.update(torch.cat(teacher_logits, dim=0))


def _teacher_entropy(teacher_logits: list[Tensor]) -> float:
    """Compute mean entropy of teacher softmax distributions."""
    all_logits = torch.cat(teacher_logits, dim=0)  # [total_samples, dim]
    probs = torch.softmax(all_logits, dim=-1)
    log_probs = torch.clamp(probs, min=1e-12).log()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return float(entropy.item())


def _dino_step_metrics(
    loss: Tensor,
    teacher_temperature: float,
    student_temperature: float,
    views: DINOViews,
    loss_terms: int,
    teacher_logits: list[Tensor] | None = None,
) -> dict[str, float]:
    metrics = {
        "loss": float(loss.item()),
        "teacher_temperature": float(teacher_temperature),
        "student_temperature": float(student_temperature),
        "teacher_views": float(len(views.teacher_views)),
        "student_views": float(len(views.student_views)),
        "loss_terms": float(loss_terms),
    }
    if teacher_logits is not None:
        metrics["teacher_entropy"] = _teacher_entropy(teacher_logits)
    return metrics


def _normalize_student_head_weights(student: nn.Module) -> None:
    """L2-normalize the last linear (projection head) weights to prevent collapse."""
    head = getattr(student, "head", None)
    if isinstance(head, nn.Linear):
        with torch.no_grad():
            w = head.weight.data
            w.div_(w.norm(dim=1, keepdim=True).clamp(min=1e-6))


def dino_train_step(
    student: nn.Module,
    teacher: nn.Module,
    batch_or_views: Tensor | tuple[Tensor, ...] | list[Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center: DINOCenter | Tensor | None = None,
    momentum: float = 0.996,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
    view_config: DINOViewConfig | None = None,
    clip_grad_norm: float | None = 3.0,
) -> dict[str, float]:
    """Run one DINO-style train step from configured teacher/student views."""

    student.train()
    teacher.eval()
    if isinstance(center, DINOCenter):
        center.to(device)
    views = build_dino_views(batch_or_views, device, view_config)
    center_tensor = _resolve_center_tensor(center)

    # L2-normalize projection head weights to prevent representational collapse
    _normalize_student_head_weights(student)

    optimizer.zero_grad(set_to_none=True)
    outputs = _forward_dino_views(student=student, teacher=teacher, views=views)
    loss, loss_terms = _mean_dino_multicrop_loss(
        outputs,
        views,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        center=center_tensor,
    )
    _update_dino_center(center, outputs.teacher_logits)

    loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    update_teacher_ema(student=student, teacher=teacher, momentum=momentum)

    return _dino_step_metrics(
        loss=loss,
        teacher_temperature=teacher_temperature,
        student_temperature=student_temperature,
        views=views,
        loss_terms=loss_terms,
        teacher_logits=outputs.teacher_logits,
    )


@torch.no_grad()
def _dino_eval_step(
    student: nn.Module,
    teacher: nn.Module,
    batch_or_views: Tensor | tuple[Tensor, ...] | list[Tensor],
    device: torch.device,
    center: DINOCenter | Tensor | None = None,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
    view_config: DINOViewConfig | None = None,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    if isinstance(center, DINOCenter):
        center.to(device)
    views = build_dino_views(batch_or_views, device, view_config)
    center_tensor = _resolve_center_tensor(center)

    outputs = _forward_dino_views(student=student, teacher=teacher, views=views)
    loss, loss_terms = _mean_dino_multicrop_loss(
        outputs,
        views,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
        center=center_tensor,
    )

    return _dino_step_metrics(
        loss=loss,
        teacher_temperature=teacher_temperature,
        student_temperature=student_temperature,
        views=views,
        loss_terms=loss_terms,
        teacher_logits=outputs.teacher_logits,
    )


# ---------------------------------------------------------------------------
# Serialization helper for DINOCenter (run_utils.to_serializable doesn't
# handle it — keep a small wrapper here)
# ---------------------------------------------------------------------------


def _center_to_serializable(center: DINOCenter | Tensor | None) -> object:
    """Convert a DINOCenter or Tensor center to a JSON-serializable form."""
    if isinstance(center, DINOCenter):
        return {
            "momentum": center.momentum,
            "dim": int(center.center.shape[-1]),
        }
    return to_serializable(center)


def _dino_to_serializable(value: object) -> object:
    """Serialize DINO-specific objects (extends run_utils.to_serializable)."""
    if isinstance(value, DINOCenter):
        return _center_to_serializable(value)
    return to_serializable(value)


# ---------------------------------------------------------------------------
# Step wrappers for run_epoch (accepts batch= keyword)
# ---------------------------------------------------------------------------


def _dino_train_step_wrapper(batch: object, **kwargs: object) -> dict[str, float]:
    return dino_train_step(batch_or_views=batch, **kwargs)  # type: ignore[arg-type]


def _dino_eval_step_wrapper(batch: object, **kwargs: object) -> dict[str, float]:
    return _dino_eval_step(batch_or_views=batch, **kwargs)  # type: ignore[arg-type]


def _dino_batch_size(batch: object) -> int:
    if isinstance(batch, (tuple, list)):
        return int(batch[0].shape[0])  # type: ignore[union-attr]
    return int(batch.shape[0])  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_train_dino_args(
    epochs: int,
    save_every: int,
    checkpoint_keep_last: int,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if save_every <= 0:
        raise ValueError("save_every must be a positive integer.")
    if checkpoint_keep_last <= 0:
        raise ValueError("checkpoint_keep_last must be a positive integer.")


# ---------------------------------------------------------------------------
# Device / center helpers
# ---------------------------------------------------------------------------


def _move_dino_state_to_device(
    student: nn.Module,
    teacher: nn.Module,
    center: DINOCenter | Tensor | None,
    device: torch.device,
) -> DINOCenter | Tensor | None:
    student.to(device)
    teacher.to(device)
    if isinstance(center, DINOCenter):
        return center.to(device)
    if isinstance(center, Tensor):
        return center.to(device)
    return center


def _restore_dino_center(
    center: DINOCenter | Tensor | None,
    loaded_center: Tensor | None,
    device: torch.device,
) -> DINOCenter | Tensor | None:
    if loaded_center is None:
        return center
    if isinstance(center, DINOCenter):
        center.center = loaded_center.to(device)
        return center
    return loaded_center.to(device)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def _initial_dino_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "val_loss": [],
        "student_temperature": [],
        "teacher_temperature": [],
        "lr": [],
        "knn_accuracy": [],
    }


@torch.no_grad()
def _compute_epoch_knn(
    student: nn.Module,
    train_loader: object,
    val_loader: object,
    device: torch.device,
    *,
    max_samples: int = 256,
    k: int = 5,
) -> float | None:
    """Lightweight kNN probe using student CLS features."""
    from vit_from_scratch.evaluation import compute_knn_accuracy

    if not hasattr(student, "encode_tokens"):
        return None

    student.eval()
    train_features, train_labels = [], []
    val_features, val_labels = [], []

    collected = 0
    for batch in train_loader:
        if collected >= max_samples:
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        images, labels = batch[0], batch[1]
        if labels is None:
            continue
        remaining = max_samples - collected
        images = images[:remaining].to(device)
        tokens = student.encode_tokens(images)
        train_features.append(tokens[:, 0].detach().cpu())
        train_labels.append(labels[:remaining].detach().cpu())
        collected += images.shape[0]

    collected = 0
    for batch in val_loader:
        if collected >= max_samples:
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        images, labels = batch[0], batch[1]
        if labels is None:
            continue
        remaining = max_samples - collected
        images = images[:remaining].to(device)
        tokens = student.encode_tokens(images)
        val_features.append(tokens[:, 0].detach().cpu())
        val_labels.append(labels[:remaining].detach().cpu())
        collected += images.shape[0]

    if not train_features or not val_features:
        return None

    return compute_knn_accuracy(
        train_features=torch.cat(train_features),
        train_labels=torch.cat(train_labels),
        val_features=torch.cat(val_features),
        val_labels=torch.cat(val_labels),
        k=k,
    )


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


def _load_dino_resume_state(
    resume_checkpoint: str | Path,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    center: DINOCenter | Tensor | None,
    device: torch.device,
    epochs: int,
) -> tuple[int, dict[str, list[float]], DINOCenter | Tensor | None, ExperimentPaths]:
    """Load a DINO checkpoint and restore all state.

    Returns ``(resumed_from_epoch, history, center, run_paths)``.
    Kept as a standalone entry-point because tests import it directly.
    Delegates common parts to ``training_loop.load_resume_state`` and then
    restores DINO-specific center state.
    """
    run_paths, history, start_epoch, _ = load_resume_state(
        checkpoint_path=resume_checkpoint,
        device=device,
        expected_approach="dino",
        target_epochs=epochs,
        model_loaders={
            "student": (student, "student_state_dict"),
            "teacher": (teacher, "teacher_state_dict"),
        },
        optimizer=optimizer,
        scheduler=scheduler,
    )
    resumed_from_epoch = start_epoch - 1

    # Restore DINO-specific center state (not handled by the generic helper)
    checkpoint = load_checkpoint(Path(resume_checkpoint), device)
    loaded_center = checkpoint.get("center")
    center = _restore_dino_center(center, loaded_center, device)

    return resumed_from_epoch, history, center, run_paths


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


def _dino_run_config(
    *,
    epochs: int,
    device: torch.device,
    run_name: str | None,
    checkpoint_keep_last: int,
    save_every: int,
    momentum: float,
    student_temperature: float,
    teacher_temperature: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    external_image_dir: str | Path | None,
    external_image_size: int | None,
    resume_checkpoint: Path | None,
    resumed_from_epoch: int | None,
    center: DINOCenter | Tensor | None,
    student: nn.Module,
    teacher: nn.Module,
    view_config: DINOViewConfig,
) -> dict[str, object]:
    return {
        "epochs": epochs,
        "target_epochs": epochs,
        "device": str(device),
        "run_name": run_name,
        "checkpoint_keep_last": checkpoint_keep_last,
        "save_every": save_every,
        "momentum": momentum,
        "student_temperature": student_temperature,
        "teacher_temperature": teacher_temperature,
        "view_config": _dino_to_serializable(view_config),
        "scheduler": None if scheduler is None else scheduler.__class__.__name__,
        "external_image_dir": None if external_image_dir is None else str(external_image_dir),
        "external_image_size": external_image_size,
        "resume_from": None if resume_checkpoint is None else str(resume_checkpoint),
        "resumed_from_epoch": resumed_from_epoch,
        "center": _dino_to_serializable(center),
        "student_config": _dino_to_serializable(getattr(student, "config", None)),
        "teacher_config": _dino_to_serializable(getattr(teacher, "config", None)),
    }


# ---------------------------------------------------------------------------
# Diagnostics / figure saving
# ---------------------------------------------------------------------------


def _save_attention_map(
    student: nn.Module,
    val_loader: object,
    device: torch.device,
    figure_dir: Path,
) -> None:
    if not hasattr(student, "forward_with_attention"):
        return

    try:
        first_batch = next(iter(val_loader))
    except StopIteration:
        return

    image_batch = _extract_images_or_explicit_views(first_batch, device)
    if isinstance(image_batch, tuple):
        image_batch = image_batch[0]
    if image_batch.shape[0] == 0:
        return

    patch_size = getattr(getattr(student, "config", None), "patch_size", None)
    if patch_size is None:
        return

    with torch.no_grad():
        _, attention_maps = student.forward_with_attention(image_batch)
    figure = plot_class_attention(
        attention_maps=attention_maps,
        image=image_batch[0].detach().cpu(),
        patch_size=int(patch_size),
    )
    save_figure_safely(figure, figure_dir / "attention_map.png")


def _save_training_curve(history: dict[str, list[float]], figure_dir: Path) -> None:
    curves_figure = plot_training_curves(history)
    save_figure_safely(curves_figure, figure_dir / "training_curves.png")


def _save_dino_diagnostics(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: object,
    val_loader: object,
    device: torch.device,
    run_paths: ExperimentPaths,
) -> None:
    try:
        _save_attention_map(student, val_loader, device, run_paths.figure_dir)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    dino_metrics, attention_payload = evaluate_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    save_json({"dino": dino_metrics}, run_paths.run_dir / "metrics.json")
    if attention_payload.get("attention_maps") and attention_payload.get("patch_size", 0):
        diagnostics_figure = plot_dino_attention_diagnostics(
            image=attention_payload["image"],
            attention_maps=attention_payload["attention_maps"],
            patch_size=int(attention_payload["patch_size"]),
            stats=dino_metrics,
        )
        save_figure_safely(
            diagnostics_figure,
            run_paths.figure_dir / "dino_attention_diagnostics.png",
        )


def _save_external_dino_diagnostics(
    student: nn.Module,
    device: torch.device,
    figure_dir: Path,
    external_image_dir: str | Path | None,
    external_image_size: int | None,
    external_mean: tuple[float, float, float] | list[float] | None,
    external_std: tuple[float, float, float] | list[float] | None,
) -> None:
    if external_image_dir is None:
        return

    external_payload = evaluate_external_dino_attention(
        student=student,
        image_dir=external_image_dir,
        device=device,
        image_size=int(external_image_size or getattr(student.config, "image_size", 32)),
        mean=external_mean,
        std=external_std,
    )
    if external_payload["patch_size"] <= 0:
        return

    external_figure = plot_external_dino_attention(
        images=external_payload["images"],
        attention_maps=external_payload["attention_maps"],
        paths=[path.name for path in external_payload["paths"]],
        patch_size=int(external_payload["patch_size"]),
        mean=external_payload["mean"],
        std=external_payload["std"],
    )
    save_figure_safely(external_figure, figure_dir / "external_dino_attention.png")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_dino(
    student: nn.Module,
    teacher: nn.Module,
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
    center: DINOCenter | Tensor | None = None,
    momentum: float = 0.996,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
    external_image_dir: str | Path | None = None,
    external_image_size: int | None = None,
    external_mean: tuple[float, float, float] | list[float] | None = None,
    external_std: tuple[float, float, float] | list[float] | None = None,
    show_progress: bool = True,
    resume_checkpoint: str | Path | None = None,
    view_config: DINOViewConfig | None = None,
    clip_grad_norm: float | None = 3.0,
    eval_knn_every: int = 10,
) -> dict[str, object]:
    """Train DINO student/teacher models, persist artifacts, and return run metadata."""

    _validate_train_dino_args(epochs, save_every, checkpoint_keep_last)
    resolved_view_config = view_config or DINOViewConfig()
    center = _move_dino_state_to_device(student, teacher, center, device)

    run_paths, history, start_epoch, resumed_from_epoch, resolved_resume_checkpoint = (
        prepare_training_run(
            approach="dino",
            output_dir=output_dir,
            run_name=run_name,
            resume_checkpoint=resume_checkpoint,
            device=device,
            target_epochs=epochs,
            models=[student, teacher],
            model_loaders={
                "student": (student, "student_state_dict"),
                "teacher": (teacher, "teacher_state_dict"),
            },
            optimizer=optimizer,
            scheduler=scheduler,
            initial_history_fn=_initial_dino_history,
        )
    )

    # Restore DINO-specific center state after prepare_training_run
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(Path(resume_checkpoint), device)
        loaded_center = checkpoint.get("center")
        center = _restore_dino_center(center, loaded_center, device)

    best_val_loss = best_metric_from_history(history, "val_loss")
    best_checkpoints: dict[str, Path | None] = {
        "best_val_loss": find_named_checkpoint(
            run_paths.checkpoint_dir,
            "dino_best_val_loss",
        )
    }

    save_json(
        _dino_to_serializable(
            _dino_run_config(
                epochs=epochs,
                device=device,
                run_name=run_name,
                checkpoint_keep_last=checkpoint_keep_last,
                save_every=save_every,
                momentum=momentum,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                scheduler=scheduler,
                external_image_dir=external_image_dir,
                external_image_size=external_image_size,
                resume_checkpoint=resolved_resume_checkpoint,
                resumed_from_epoch=resumed_from_epoch,
                center=center,
                student=student,
                teacher=teacher,
                view_config=resolved_view_config,
            )
        ),
        run_paths.config_path,
    )

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        # Cosine momentum schedule: increases from base toward 1.0 over training
        epoch_momentum = cosine_momentum_schedule(epoch, epochs, base=momentum)

        default_metrics: dict[str, float] = {
            "loss": 0.0,
            "student_temperature": student_temperature,
            "teacher_temperature": teacher_temperature,
            "teacher_views": 0.0,
            "student_views": 0.0,
        }

        train_step_kwargs: dict[str, object] = {
            "student": student,
            "teacher": teacher,
            "optimizer": optimizer,
            "device": device,
            "center": center,
            "momentum": epoch_momentum,
            "student_temperature": student_temperature,
            "teacher_temperature": teacher_temperature,
            "view_config": resolved_view_config,
            "clip_grad_norm": clip_grad_norm,
        }
        train_metrics = run_epoch(
            dataloader=train_loader,
            step_fn=_dino_train_step_wrapper,
            step_kwargs=train_step_kwargs,
            batch_size_fn=_dino_batch_size,
            default_metrics=default_metrics,
            progress_desc=f"dino train {epoch}/{epochs}",
            show_progress=show_progress,
        )

        val_step_kwargs: dict[str, object] = {
            "student": student,
            "teacher": teacher,
            "device": device,
            "center": center,
            "student_temperature": student_temperature,
            "teacher_temperature": teacher_temperature,
            "view_config": resolved_view_config,
        }
        val_metrics = run_epoch(
            dataloader=val_loader,
            step_fn=_dino_eval_step_wrapper,
            step_kwargs=val_step_kwargs,
            batch_size_fn=_dino_batch_size,
            default_metrics=default_metrics,
            progress_desc=f"dino val {epoch}/{epochs}",
            show_progress=show_progress,
        )

        append_history(
            history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metric_keys={
                "train_loss": "loss",
                "val_loss": "loss",
                "student_temperature": "student_temperature",
                "teacher_temperature": "teacher_temperature",
            },
            lr=epoch_lr,
        )

        # kNN probe
        knn_acc = None
        if eval_knn_every > 0 and (epoch % eval_knn_every == 0 or epoch == epochs):
            knn_acc = _compute_epoch_knn(
                student, train_loader, val_loader, device
            )
        history.setdefault("knn_accuracy", []).append(knn_acc)

        save_history(history, run_paths.run_dir)

        best_val_loss, improved_val_loss = update_best_metric(
            current_value=val_metrics["loss"],
            best_value=best_val_loss,
        )
        if improved_val_loss:
            best_checkpoints["best_val_loss"] = (
                run_paths.checkpoint_dir / "dino_best_val_loss.pt"
            )

        if scheduler is not None:
            scheduler.step()

        checkpoint_state = build_checkpoint_state(
            approach="dino",
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_loss=best_val_loss,
            best_checkpoints=best_checkpoints,
            model_states={
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
            },
            extra={
                "center": (
                    None
                    if center is None
                    else _resolve_center_tensor(center).detach().cpu()
                ),
                "student_config": _dino_to_serializable(getattr(student, "config", None)),
                "teacher_config": _dino_to_serializable(getattr(teacher, "config", None)),
                "view_config": _dino_to_serializable(resolved_view_config),
            },
        )
        if epoch % save_every == 0:
            save_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="dino",
                epoch=epoch,
                keep_last=checkpoint_keep_last,
            )

        if improved_val_loss:
            save_best_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="dino",
                metric_name="val_loss",
            )

    _save_training_curve(history, run_paths.figure_dir)
    _save_dino_diagnostics(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_paths=run_paths,
    )
    _save_external_dino_diagnostics(
        student=student,
        device=device,
        figure_dir=run_paths.figure_dir,
        external_image_dir=external_image_dir,
        external_image_size=external_image_size,
        external_mean=external_mean,
        external_std=external_std,
    )

    return {
        "history": history,
        "run_dir": run_paths.run_dir,
        "checkpoint_dir": run_paths.checkpoint_dir,
        "figure_dir": run_paths.figure_dir,
        "best_val_loss": best_val_loss,
        "best_checkpoints": {
            key: None if path is None else str(path) for key, path in best_checkpoints.items()
        },
        "resume_from": (
            None if resolved_resume_checkpoint is None else resolved_resume_checkpoint
        ),
        "resumed_from_epoch": resumed_from_epoch,
    }
