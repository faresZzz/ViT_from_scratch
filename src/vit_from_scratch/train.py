"""Command-line entrypoint for small ViT training experiments."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from vit_from_scratch.artifacts import resolve_resume_checkpoint, save_json
from vit_from_scratch.classification import train_classification
from vit_from_scratch.config import ViTConfig
from vit_from_scratch.data import build_dataloaders
from vit_from_scratch.dino import DINOCenter, DINOViewConfig, train_dino
from vit_from_scratch.external_images import list_external_images
from vit_from_scratch.masked_autoencoder import MaskedAutoencoder, train_masked_autoencoder
from vit_from_scratch.model import VisionTransformer
from vit_from_scratch.run_utils import to_serializable
from vit_from_scratch.training import get_device, set_seed, train as _supervised_train
from vit_from_scratch.training_config import load_training_config, merge_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROACH_CHOICES = ("classification", "mae", "dino")
DATASET_CHOICES = ("cifar10", "stl10", "tiny-imagenet", "fake")
POSITION_EMBEDDING_CHOICES = ("learned", "cosine", "rope", "rope2d")
SCHEDULER_CHOICES = ("none", "cosine")
DEFAULTS: dict[str, object] = {
    "approach": "classification",
    "dataset": "cifar10",
    "data_dir": "data",
    "epochs": 5,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "image_size": 32,
    "patch_size": 4,
    "embed_dim": 128,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "label_smoothing": 0.0,
    "position_embedding": "learned",
    "device": "auto",
    "seed": 42,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
    "val_fraction": 0.1,
    "split_train_validation": True,
    "num_workers": 0,
    "output_dir": str(PROJECT_ROOT / "runs"),
    "run_name": None,
    "checkpoint_keep_last": 3,
    "save_every": 1,
    "mask_ratio": 0.5,
    "dino_momentum": 0.996,
    "student_temperature": 0.1,
    "teacher_temperature": 0.04,
    "teacher_global_crops": 2,
    "student_global_crops": 2,
    "student_local_crops": 0,
    "teacher_global_crop_scale_min": 0.4,
    "teacher_global_crop_scale_max": 1.0,
    "student_global_crop_scale_min": 0.4,
    "student_global_crop_scale_max": 1.0,
    "student_local_crop_scale_min": 0.05,
    "student_local_crop_scale_max": 0.4,
    "dino_view_noise_std": 0.01,
    "scheduler": "none",
    "warmup_epochs": 0,
    "min_lr": 1e-6,
    "color_jitter_strength": 0.0,
    "random_erasing_prob": 0.0,
    "randaugment_num_ops": 0,
    "randaugment_magnitude": 9,
    "external_image_dir": None,
    "external_image_size": None,
    "external_mean": None,
    "external_std": None,
    "resume": None,
    "restart": False,
    "show_progress": True,
    "optimizer": "adamw",
    "optimizer_betas": [0.9, 0.999],
    "momentum": 0.9,
    "clip_grad_norm": None,
    "eval_knn_every": 10,
    "decoder_embed_dim": None,
    "decoder_depth": None,
    "decoder_num_heads": None,
    "use_unlabeled": False,
}


@dataclass(frozen=True)
class TrainingApproachSpec:
    """Method-agnostic metadata for one training approach."""

    name: str
    checkpoint_prefix: str


APPROACH_SPECS: dict[str, TrainingApproachSpec] = {
    name: TrainingApproachSpec(name=name, checkpoint_prefix=name)
    for name in APPROACH_CHOICES
}


def get_approach_spec(approach: object) -> TrainingApproachSpec:
    """Return the registered training approach spec for a config value."""

    name = str(approach)
    try:
        return APPROACH_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(APPROACH_SPECS))
        raise ValueError(f"Unknown training approach {name!r}; expected one of {choices}.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train small ViT experiments.")
    parser.add_argument("--config", default=None, help="YAML config file to load.")
    parser.add_argument(
        "--approach",
        choices=APPROACH_CHOICES,
        default=None,
        help="Training objective to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument(
        "--position-embedding",
        choices=POSITION_EMBEDDING_CHOICES,
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument(
        "--no-split-train-validation",
        action="store_false",
        dest="split_train_validation",
        default=None,
        help="Use the official test split as validation instead of splitting train.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--external-image-dir", default=None)
    parser.add_argument("--external-image-size", type=int, default=None)
    parser.add_argument("--external-mean", nargs=3, type=float, default=None)
    parser.add_argument("--external-std", nargs=3, type=float, default=None)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        default=None,
        help="Resume from 'latest', a run directory, or an exact .pt checkpoint.",
    )
    resume_group.add_argument(
        "--restart",
        action="store_true",
        default=None,
        help="Force a fresh run. Mutually exclusive with --resume.",
    )
    parser.add_argument("--checkpoint-keep-last", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--dino-momentum", type=float, default=None)
    parser.add_argument("--student-temperature", type=float, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--teacher-global-crops", type=int, default=None)
    parser.add_argument("--student-global-crops", type=int, default=None)
    parser.add_argument("--student-local-crops", type=int, default=None)
    parser.add_argument("--teacher-global-crop-scale-min", type=float, default=None)
    parser.add_argument("--teacher-global-crop-scale-max", type=float, default=None)
    parser.add_argument("--student-global-crop-scale-min", type=float, default=None)
    parser.add_argument("--student-global-crop-scale-max", type=float, default=None)
    parser.add_argument("--student-local-crop-scale-min", type=float, default=None)
    parser.add_argument("--student-local-crop-scale-max", type=float, default=None)
    parser.add_argument("--dino-view-noise-std", type=float, default=None)
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--color-jitter-strength", type=float, default=None)
    parser.add_argument("--random-erasing-prob", type=float, default=None)
    parser.add_argument("--randaugment-num-ops", type=int, default=None)
    parser.add_argument("--randaugment-magnitude", type=int, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=None,
        help="Disable progress display.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw", "adam", "sgd", "sgd_momentum"],
    )
    parser.add_argument("--optimizer-betas", type=float, nargs=2, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--eval-knn-every", type=int, default=None)
    parser.add_argument("--decoder-embed-dim", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-num-heads", type=int, default=None)
    parser.add_argument("--use-unlabeled", action="store_true", default=None)
    return parser


def _format_history_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:.4f}"
    return str(value)


def _print_history(history: dict[str, list[object]], epochs: int) -> None:
    for epoch_index in range(epochs):
        parts = [f"Epoch {epoch_index + 1}/{epochs}"]
        for key, values in history.items():
            if epoch_index < len(values):
                parts.append(f"{key}: {_format_history_value(values[epoch_index])}")
        print(" - ".join(parts))


def _explicit_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _maybe_add_show_progress(
    train_fn: object,
    kwargs: dict[str, object],
    show_progress: bool,
) -> dict[str, object]:
    if "show_progress" in inspect.signature(train_fn).parameters:
        kwargs["show_progress"] = show_progress
    return kwargs


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: str,
    epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build an epoch-level LR scheduler shared by all training approaches."""

    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be greater than or equal to 0.")
    if min_lr < 0.0:
        raise ValueError("min_lr must be greater than or equal to 0.0.")

    scheduler_name = scheduler.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError("scheduler must be one of {'none', 'cosine'}.")

    base_lr = float(optimizer.param_groups[0]["lr"])
    min_factor = 0.0 if base_lr == 0.0 else min_lr / base_lr
    warmup = min(warmup_epochs, epochs)
    decay_epochs = max(1, epochs - warmup)

    def lr_lambda(epoch_index: int) -> float:
        step = epoch_index + 1
        if warmup > 0 and step <= warmup:
            return max(min_factor, step / warmup)
        progress = min(1.0, max(0.0, (step - warmup) / decay_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_training_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = None if args.config is None else _resolve_project_path(args.config)
    file_config = load_training_config(config_path)
    cli_overrides = _explicit_cli_overrides(args)
    resolved = merge_config(DEFAULTS, file_config, cli_overrides)
    return _resolve_config_paths(resolved)


def _resolve_config_paths(config: dict[str, object]) -> dict[str, object]:
    resolved = dict(config)
    resolved["data_dir"] = str(_resolve_project_path(str(resolved["data_dir"])))
    resolved["output_dir"] = str(_resolve_project_path(str(resolved["output_dir"])))
    if resolved["external_image_dir"] is not None:
        resolved["external_image_dir"] = str(
            _resolve_project_path(str(resolved["external_image_dir"]))
        )
    if resolved["resume"] not in (None, "latest"):
        resolved["resume"] = str(_resolve_project_path(str(resolved["resume"])))
    return resolved


def _disable_missing_external_images(config: dict[str, object]) -> None:
    image_dir = config.get("external_image_dir")
    if image_dir is None:
        return

    try:
        image_paths = list_external_images(str(image_dir))
    except (FileNotFoundError, NotADirectoryError):
        print(f"External image directory not found; skipping external figures: {image_dir}")
        config["external_image_dir"] = None
        return

    if not image_paths:
        print(f"No external images found; skipping external figures: {image_dir}")
        config["external_image_dir"] = None


def _build_experiment_dataloaders(config: Mapping[str, object]):
    return build_dataloaders(
        dataset=str(config["dataset"]),
        data_dir=str(config["data_dir"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        max_train_samples=config["max_train_samples"],
        max_val_samples=config["max_val_samples"],
        max_test_samples=config["max_test_samples"],
        seed=int(config["seed"]),
        download=config["dataset"] in {"cifar10", "stl10"},
        image_size=int(config["image_size"]),
        val_fraction=float(config["val_fraction"]),
        split_train_validation=bool(config["split_train_validation"]),
        color_jitter_strength=float(config["color_jitter_strength"]),
        random_erasing_prob=float(config["random_erasing_prob"]),
        randaugment_num_ops=int(config["randaugment_num_ops"]),
        randaugment_magnitude=int(config["randaugment_magnitude"]),
        use_unlabeled=bool(config.get("use_unlabeled", False)),
    )


def _build_vit_config(
    config: Mapping[str, object],
    *,
    num_classes: int,
) -> ViTConfig:
    return ViTConfig(
        image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]),
        attention_dropout=float(config["attention_dropout"]),
        position_embedding=str(config["position_embedding"]),
        decoder_embed_dim=int(config["decoder_embed_dim"]) if config.get("decoder_embed_dim") is not None else None,
        decoder_depth=int(config["decoder_depth"]) if config.get("decoder_depth") is not None else None,
        decoder_num_heads=int(config["decoder_num_heads"]) if config.get("decoder_num_heads") is not None else None,
    )


def _build_optimizer(parameters, config: Mapping[str, object]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config["lr"])
    wd = float(config["weight_decay"])
    if name == "adamw":
        betas = tuple(config.get("optimizer_betas", [0.9, 0.999]))
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        betas = tuple(config.get("optimizer_betas", [0.9, 0.999]))
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.0)
    if name == "sgd_momentum":
        mom = float(config.get("momentum", 0.9))
        return torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=mom)
    raise ValueError(f"Unknown optimizer: {name!r}. Choose from: adamw, adam, sgd, sgd_momentum")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    return build_lr_scheduler(
        optimizer,
        scheduler=str(config["scheduler"]),
        epochs=int(config["epochs"]),
        warmup_epochs=int(config["warmup_epochs"]),
        min_lr=float(config["min_lr"]),
    )


def _resolve_resume_checkpoint(
    config: Mapping[str, object],
    output_dir: Path,
    approach: TrainingApproachSpec,
    vit_config: ViTConfig,
) -> Path | None:
    if config["resume"] is None:
        return None
    checkpoint = resolve_resume_checkpoint(
        resume=str(config["resume"]),
        output_dir=output_dir,
        approach=approach.name,
        prefix=approach.checkpoint_prefix,
        required_model_config=to_serializable(vit_config),
    )
    print(f"Resuming from checkpoint: {checkpoint}")
    return checkpoint


def _common_runner_kwargs(
    config: Mapping[str, object],
    dataloaders,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    output_dir: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    return {
        "train_loader": dataloaders.train_loader,
        "val_loader": dataloaders.val_loader,
        "optimizer": optimizer,
        "device": device,
        "epochs": int(config["epochs"]),
        "scheduler": scheduler,
        "output_dir": output_dir,
        "run_name": config["run_name"],
        "checkpoint_keep_last": int(config["checkpoint_keep_last"]),
        "save_every": int(config["save_every"]),
        "external_image_dir": config["external_image_dir"],
        "external_image_size": config["external_image_size"],
        "external_mean": config["external_mean"],
        "external_std": config["external_std"],
        "resume_checkpoint": resume_checkpoint,
    }


def _build_dino_view_config(config: Mapping[str, object]) -> DINOViewConfig:
    return DINOViewConfig(
        teacher_global_crops=int(config["teacher_global_crops"]),
        student_global_crops=int(config["student_global_crops"]),
        student_local_crops=int(config["student_local_crops"]),
        teacher_global_crop_scale=(
            float(config["teacher_global_crop_scale_min"]),
            float(config["teacher_global_crop_scale_max"]),
        ),
        student_global_crop_scale=(
            float(config["student_global_crop_scale_min"]),
            float(config["student_global_crop_scale_max"]),
        ),
        student_local_crop_scale=(
            float(config["student_local_crop_scale_min"]),
            float(config["student_local_crop_scale_max"]),
        ),
        noise_std=float(config["dino_view_noise_std"]),
    )


def _run_classification(
    config: Mapping[str, object],
    dataloaders,
    vit_config: ViTConfig,
    output_dir: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    model = VisionTransformer(vit_config)
    optimizer = _build_optimizer(model.parameters(), config)
    scheduler = _build_scheduler(optimizer, config)
    train_kwargs = _maybe_add_show_progress(
        train_classification,
        {
            "model": model,
            **_common_runner_kwargs(
                config,
                dataloaders,
                optimizer,
                scheduler,
                output_dir,
                device,
                resume_checkpoint,
            ),
            "class_names": dataloaders.class_names,
            "label_smoothing": float(config["label_smoothing"]),
        },
        bool(config["show_progress"]),
    )
    return train_classification(**train_kwargs)


def _run_mae(
    config: Mapping[str, object],
    dataloaders,
    vit_config: ViTConfig,
    output_dir: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    model = MaskedAutoencoder(vit_config)
    optimizer = _build_optimizer(model.parameters(), config)
    scheduler = _build_scheduler(optimizer, config)
    train_kwargs = _maybe_add_show_progress(
        train_masked_autoencoder,
        {
            "model": model,
            **_common_runner_kwargs(
                config,
                dataloaders,
                optimizer,
                scheduler,
                output_dir,
                device,
                resume_checkpoint,
            ),
            "mask_ratio": float(config["mask_ratio"]),
        },
        bool(config["show_progress"]),
    )
    return train_masked_autoencoder(**train_kwargs)


def _run_dino(
    config: Mapping[str, object],
    dataloaders,
    vit_config: ViTConfig,
    output_dir: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    student = VisionTransformer(vit_config)
    teacher = VisionTransformer(vit_config)
    teacher.load_state_dict(student.state_dict())
    optimizer = _build_optimizer(student.parameters(), config)
    scheduler = _build_scheduler(optimizer, config)
    train_kwargs = _maybe_add_show_progress(
        train_dino,
        {
            "student": student,
            "teacher": teacher,
            **_common_runner_kwargs(
                config,
                dataloaders,
                optimizer,
                scheduler,
                output_dir,
                device,
                resume_checkpoint,
            ),
            "center": DINOCenter(dim=vit_config.num_classes),
            "momentum": float(config["dino_momentum"]),
            "student_temperature": float(config["student_temperature"]),
            "teacher_temperature": float(config["teacher_temperature"]),
            "view_config": _build_dino_view_config(config),
            "clip_grad_norm": float(config["clip_grad_norm"]) if config.get("clip_grad_norm") is not None else 3.0,
            "eval_knn_every": int(config.get("eval_knn_every", 10)),
        },
        bool(config["show_progress"]),
    )
    return train_dino(**train_kwargs)


def _run_training_approach(
    approach: TrainingApproachSpec,
    config: Mapping[str, object],
    dataloaders,
    vit_config: ViTConfig,
    output_dir: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    runners = {
        "classification": _run_classification,
        "mae": _run_mae,
        "dino": _run_dino,
    }
    return runners[approach.name](
        config=config,
        dataloaders=dataloaders,
        vit_config=vit_config,
        output_dir=output_dir,
        device=device,
        resume_checkpoint=resume_checkpoint,
    )


def _save_run_config(
    *,
    result: Mapping[str, object],
    config: Mapping[str, object],
    device: torch.device,
    vit_config: ViTConfig,
) -> None:
    resolved_config = dict(config)
    resolved_config["resolved_device"] = str(device)
    resolved_config["model_config"] = to_serializable(vit_config)
    if result.get("best_checkpoints") is not None:
        resolved_config["best_checkpoints"] = result["best_checkpoints"]
    if result.get("best_val_loss") is not None:
        resolved_config["best_val_loss"] = result["best_val_loss"]
    if result.get("best_val_accuracy") is not None:
        resolved_config["best_val_accuracy"] = result["best_val_accuracy"]
    if result.get("resume_from") is not None:
        resolved_config["resume_from"] = str(result["resume_from"])
        resolved_config["resumed_from_epoch"] = result.get("resumed_from_epoch")
        resolved_config["target_epochs"] = int(config["epochs"])
    save_json(to_serializable(resolved_config), Path(result["run_dir"]) / "config.json")


def _print_resume_summary(result: Mapping[str, object], total_epochs: int) -> None:
    resumed_from_epoch = result.get("resumed_from_epoch")
    if resumed_from_epoch is None:
        return
    start_epoch = int(resumed_from_epoch) + 1
    print(
        f"Resumed at epoch {start_epoch}/{total_epochs} "
        f"from epoch {int(resumed_from_epoch)}."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _resolve_training_config(args)
    _disable_missing_external_images(config)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    print(f"Using device: {device}")
    print(f"Training approach: {config['approach']}")
    if config.get("restart"):
        print("Starting a fresh run (--restart).")

    approach = get_approach_spec(config["approach"])
    dataloaders = _build_experiment_dataloaders(config)
    vit_config = _build_vit_config(config, num_classes=len(dataloaders.class_names))
    output_dir = Path(str(config["output_dir"]))
    resume_checkpoint = _resolve_resume_checkpoint(
        config,
        output_dir,
        approach,
        vit_config,
    )
    result = _run_training_approach(
        approach=approach,
        config=config,
        dataloaders=dataloaders,
        vit_config=vit_config,
        output_dir=output_dir,
        device=device,
        resume_checkpoint=resume_checkpoint,
    )

    history = result["history"]
    _save_run_config(
        result=result,
        config=config,
        device=device,
        vit_config=vit_config,
    )
    _print_resume_summary(result, int(config["epochs"]))
    _print_history(history, int(config["epochs"]))
    print(f"Saved run to {result['run_dir']}")


class _CallableTrainModule(ModuleType):
    def __call__(self, *args: object, **kwargs: object) -> object:
        return _supervised_train(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableTrainModule


if __name__ == "__main__":
    main()
