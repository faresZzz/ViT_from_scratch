"""Utilities for experiment artifacts such as histories, figures, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import strftime
from typing import Mapping, cast

import torch


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path
    figure_dir: Path
    checkpoint_dir: Path
    history_path: Path
    config_path: Path


_CHECKPOINT_EPOCH_PATTERN = re.compile(r"^(?P<prefix>.+)_epoch_(?P<epoch>\d+)\.pt$")
_MODEL_COMPATIBILITY_KEYS = (
    "image_size",
    "patch_size",
    "in_channels",
    "num_classes",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "position_embedding",
    "decoder_embed_dim",
    "decoder_depth",
    "decoder_num_heads",
)


def _with_unique_run_name(run_dir: Path) -> Path:
    candidate = run_dir
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = run_dir.with_name(f"{run_dir.name}-{suffix:02d}")
    return candidate


def create_experiment_run(
    root: str | Path = "runs",
    *,
    approach: str,
    run_name: str | None = None,
    config: Mapping[str, object] | None = None,
) -> ExperimentPaths:
    run_id = run_name or strftime("%Y%m%d-%H%M%S")
    run_dir = _with_unique_run_name(Path(root) / approach / run_id)

    figure_dir = run_dir / "figures"
    checkpoint_dir = run_dir / "checkpoints"
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"

    figure_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    if config is not None:
        save_json(dict(config), config_path)

    return ExperimentPaths(
        run_dir=run_dir,
        figure_dir=figure_dir,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        config_path=config_path,
    )


def save_json(payload: object, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_history(run_dir: str | Path) -> dict[str, object]:
    history_path = Path(run_dir) / "history.json"
    return json.loads(history_path.read_text(encoding="utf-8"))


def save_history(history: Mapping[str, object], run_dir_or_path: str | Path) -> Path:
    destination = Path(run_dir_or_path)
    history_path = (
        destination if destination.suffix == ".json" else destination / "history.json"
    )
    return save_json(dict(history), history_path)


def save_checkpoint(
    state: object,
    checkpoint_dir: str | Path,
    prefix: str,
    epoch: int,
    keep_last: int = 3,
) -> Path:
    if keep_last <= 0:
        raise ValueError("keep_last must be greater than 0.")

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_root / f"{prefix}_epoch_{epoch:04d}.pt"
    torch.save(state, checkpoint_path)

    existing = sorted(checkpoint_root.glob(f"{prefix}_epoch_*.pt"))
    stale_paths = existing[:-keep_last]
    for stale_path in stale_paths:
        stale_path.unlink()

    return checkpoint_path


def save_named_checkpoint(
    state: object,
    checkpoint_dir: str | Path,
    prefix: str,
    name: str,
) -> Path:
    """Save a stable named checkpoint that is not affected by epoch rotation."""

    if not name:
        raise ValueError("name must be a non-empty string.")
    if "/" in name or "\\" in name:
        raise ValueError("name must not contain path separators.")

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / f"{prefix}_{name}.pt"
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_best_checkpoint(
    state: object,
    checkpoint_dir: str | Path,
    prefix: str,
    metric_name: str | None = None,
) -> Path:
    """Save a best-metric checkpoint outside the rolling epoch checkpoint set."""

    name = "best" if metric_name is None else f"best_{metric_name}"
    return save_named_checkpoint(
        state=state,
        checkpoint_dir=checkpoint_dir,
        prefix=prefix,
        name=name,
    )


def find_latest_checkpoint(checkpoint_dir: str | Path, prefix: str) -> Path | None:
    checkpoint_root = Path(checkpoint_dir)
    latest_epoch = -1
    latest_path: Path | None = None

    for path in checkpoint_root.glob(f"{prefix}_epoch_*.pt"):
        match = _CHECKPOINT_EPOCH_PATTERN.match(path.name)
        if match is None or match.group("prefix") != prefix:
            continue
        epoch = int(match.group("epoch"))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path

    return latest_path


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, object]:
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected checkpoint at {Path(path)} to contain a dict, got {type(payload).__name__}."
        )
    return cast(dict[str, object], payload)


def _checkpoint_model_config(
    checkpoint: Mapping[str, object],
) -> Mapping[str, object] | None:
    for key in ("model_config", "student_config", "teacher_config"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return cast(Mapping[str, object], value)
    return None


def _model_config_is_compatible(
    checkpoint: Mapping[str, object],
    required_model_config: Mapping[str, object] | None,
) -> bool:
    if required_model_config is None:
        return True

    checkpoint_config = _checkpoint_model_config(checkpoint)
    if checkpoint_config is None:
        return True

    for key in _MODEL_COMPATIBILITY_KEYS:
        if key not in required_model_config or key not in checkpoint_config:
            continue
        if checkpoint_config[key] != required_model_config[key]:
            return False
    return True


def resolve_resume_checkpoint(
    resume: str,
    output_dir: str | Path,
    approach: str,
    prefix: str,
    required_model_config: Mapping[str, object] | None = None,
) -> Path:
    if resume == "latest":
        candidate_paths: list[Path] = []
        for checkpoint_dir in sorted((Path(output_dir) / approach).glob("*/checkpoints")):
            latest = find_latest_checkpoint(checkpoint_dir, prefix)
            if latest is None:
                continue
            checkpoint = load_checkpoint(latest, "cpu")
            checkpoint_approach = checkpoint.get("approach")
            if checkpoint_approach is not None and checkpoint_approach != approach:
                continue
            if _model_config_is_compatible(checkpoint, required_model_config):
                candidate_paths.append(latest)
        if not candidate_paths:
            raise FileNotFoundError(
                f"No compatible '{prefix}' checkpoints found under "
                f"{(Path(output_dir) / approach)!s}."
            )
        return max(
            candidate_paths,
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )

    resume_path = Path(resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume_path}")

    if resume_path.is_file():
        if resume_path.suffix != ".pt":
            raise ValueError(
                f"Resume path must point to a .pt checkpoint file or run directory: {resume_path}"
            )
        return resume_path

    if not resume_path.is_dir():
        raise ValueError(
            f"Resume path must point to a .pt checkpoint file or run directory: {resume_path}"
        )

    latest = find_latest_checkpoint(resume_path / "checkpoints", prefix)
    if latest is None:
        raise FileNotFoundError(
            f"No '{prefix}' checkpoints found under run directory {resume_path}."
        )
    return latest


def save_figure(fig: object, path: str | Path) -> Path:
    figure_path = Path(path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    return figure_path
