from __future__ import annotations

import re
from os import utime
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
torch = pytest.importorskip("torch")

from vit_from_scratch.artifacts import (
    create_experiment_run,
    find_latest_checkpoint,
    load_checkpoint,
    load_history,
    resolve_resume_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    save_named_checkpoint,
    save_figure,
    save_history,
)
from vit_from_scratch.run_utils import (
    RunValidationConfig,
    aggregate_weighted_metrics,
    best_metric_from_history,
    close_figure,
    find_named_checkpoint,
    paths_from_checkpoint,
    to_serializable,
    validate_training_run_args,
)


def test_create_experiment_run_creates_expected_directories_and_config(tmp_path: Path):
    paths = create_experiment_run(
        root=tmp_path,
        approach="classification",
        config={"epochs": 5, "lr": 1e-3},
    )

    assert paths.run_dir.exists()
    assert paths.figure_dir.is_dir()
    assert paths.checkpoint_dir.is_dir()
    assert paths.history_path == paths.run_dir / "history.json"
    assert paths.config_path == paths.run_dir / "config.json"
    assert re.fullmatch(r"\d{8}-\d{6}", paths.run_dir.name)
    assert paths.config_path.read_text(encoding="utf-8").strip() == (
        '{\n  "epochs": 5,\n  "lr": 0.001\n}'
    )


def test_save_and_load_history_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "runs" / "classification" / "demo"
    history = {
        "train_loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
        "train_accuracy": [0.4, 0.6],
        "val_accuracy": [0.35, 0.55],
    }

    save_history(history, run_dir)

    assert load_history(run_dir) == history


def test_save_checkpoint_keeps_only_latest_epochs_for_prefix(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    state = {"weights": torch.tensor([1.0])}

    for epoch in range(1, 6):
        save_checkpoint(
            state=state,
            checkpoint_dir=checkpoint_dir,
            prefix="classifier",
            epoch=epoch,
            keep_last=3,
        )

    remaining = sorted(path.name for path in checkpoint_dir.glob("classifier_epoch_*.pt"))

    assert remaining == [
        "classifier_epoch_0003.pt",
        "classifier_epoch_0004.pt",
        "classifier_epoch_0005.pt",
    ]


def test_save_named_checkpoint_is_not_removed_by_epoch_rotation(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    best_path = save_named_checkpoint(
        state={"epoch": 2, "metric": 0.7},
        checkpoint_dir=checkpoint_dir,
        prefix="classification",
        name="best_val_accuracy",
    )

    for epoch in range(1, 6):
        save_checkpoint(
            state={"epoch": epoch},
            checkpoint_dir=checkpoint_dir,
            prefix="classification",
            epoch=epoch,
            keep_last=3,
        )

    remaining = sorted(path.name for path in checkpoint_dir.glob("*.pt"))

    assert best_path.name == "classification_best_val_accuracy.pt"
    assert "classification_best_val_accuracy.pt" in remaining
    assert [
        name for name in remaining if name.startswith("classification_epoch_")
    ] == [
        "classification_epoch_0003.pt",
        "classification_epoch_0004.pt",
        "classification_epoch_0005.pt",
    ]


def test_save_best_checkpoint_persists_outside_epoch_rotation(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    state = {"weights": torch.tensor([1.0])}

    best_checkpoint = save_best_checkpoint(
        state=state,
        checkpoint_dir=checkpoint_dir,
        prefix="classifier",
        metric_name="val_accuracy",
    )

    for epoch in range(1, 6):
        save_checkpoint(
            state={"epoch": epoch},
            checkpoint_dir=checkpoint_dir,
            prefix="classifier",
            epoch=epoch,
            keep_last=3,
        )

    remaining = sorted(path.name for path in checkpoint_dir.glob("classifier*.pt"))

    assert remaining == [
        "classifier_best_val_accuracy.pt",
        "classifier_epoch_0003.pt",
        "classifier_epoch_0004.pt",
        "classifier_epoch_0005.pt",
    ]
    assert best_checkpoint.exists()


def test_find_latest_checkpoint_ignores_named_best_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    best_checkpoint = save_best_checkpoint(
        state={"epoch": "best"},
        checkpoint_dir=checkpoint_dir,
        prefix="classification",
    )
    expected_latest = save_checkpoint(
        state={"epoch": 4},
        checkpoint_dir=checkpoint_dir,
        prefix="classification",
        epoch=4,
    )

    latest = find_latest_checkpoint(checkpoint_dir, "classification")

    assert latest == expected_latest
    assert latest != best_checkpoint


def test_find_latest_checkpoint_returns_highest_epoch(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    for epoch in (2, 10, 3):
        (checkpoint_dir / f"classification_epoch_{epoch:04d}.pt").write_bytes(b"ckpt")

    latest = find_latest_checkpoint(checkpoint_dir, "classification")

    assert latest == checkpoint_dir / "classification_epoch_0010.pt"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    assert find_latest_checkpoint(checkpoint_dir, "classification") is None


def test_resolve_resume_checkpoint_latest_returns_newest_candidate(tmp_path: Path):
    first_run = create_experiment_run(
        root=tmp_path,
        approach="classification",
        run_name="run-a",
    )
    second_run = create_experiment_run(
        root=tmp_path,
        approach="classification",
        run_name="run-b",
    )

    first_checkpoint = save_checkpoint(
        state={"epoch": 2},
        checkpoint_dir=first_run.checkpoint_dir,
        prefix="classification",
        epoch=2,
    )
    second_checkpoint = save_checkpoint(
        state={"epoch": 1},
        checkpoint_dir=second_run.checkpoint_dir,
        prefix="classification",
        epoch=1,
    )

    utime(first_checkpoint, (100, 100))
    utime(second_checkpoint, (200, 200))

    resolved = resolve_resume_checkpoint(
        "latest",
        tmp_path,
        "classification",
        "classification",
    )

    assert resolved == second_checkpoint


def test_resolve_resume_checkpoint_latest_skips_incompatible_model_config(
    tmp_path: Path,
):
    compatible_run = create_experiment_run(
        root=tmp_path,
        approach="dino",
        run_name="compatible",
    )
    incompatible_run = create_experiment_run(
        root=tmp_path,
        approach="dino",
        run_name="incompatible",
    )

    compatible_checkpoint = save_checkpoint(
        state={
            "approach": "dino",
            "epoch": 10,
            "student_config": {
                "image_size": 32,
                "patch_size": 4,
                "embed_dim": 128,
                "position_embedding": "rope2d",
            },
        },
        checkpoint_dir=compatible_run.checkpoint_dir,
        prefix="dino",
        epoch=10,
    )
    incompatible_checkpoint = save_checkpoint(
        state={
            "approach": "dino",
            "epoch": 1,
            "student_config": {
                "image_size": 32,
                "patch_size": 4,
                "embed_dim": 128,
                "position_embedding": "learned",
            },
        },
        checkpoint_dir=incompatible_run.checkpoint_dir,
        prefix="dino",
        epoch=1,
    )

    utime(compatible_checkpoint, (100, 100))
    utime(incompatible_checkpoint, (200, 200))

    resolved = resolve_resume_checkpoint(
        "latest",
        tmp_path,
        "dino",
        "dino",
        required_model_config={
            "image_size": 32,
            "patch_size": 4,
            "embed_dim": 128,
            "position_embedding": "rope2d",
        },
    )

    assert resolved == compatible_checkpoint


def test_resolve_resume_checkpoint_run_dir_returns_latest_in_run(tmp_path: Path):
    run = create_experiment_run(
        root=tmp_path,
        approach="classification",
        run_name="resume-me",
    )

    save_checkpoint(
        state={"epoch": 2},
        checkpoint_dir=run.checkpoint_dir,
        prefix="classification",
        epoch=2,
    )
    latest_checkpoint = save_checkpoint(
        state={"epoch": 11},
        checkpoint_dir=run.checkpoint_dir,
        prefix="classification",
        epoch=11,
    )

    resolved = resolve_resume_checkpoint(
        str(run.run_dir),
        tmp_path,
        "classification",
        "classification",
    )

    assert resolved == latest_checkpoint


def test_resolve_resume_checkpoint_file_returns_exact_path(tmp_path: Path):
    checkpoint_path = tmp_path / "manual.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    resolved = resolve_resume_checkpoint(
        str(checkpoint_path),
        tmp_path,
        "classification",
        "classification",
    )

    assert resolved == checkpoint_path


def test_create_experiment_run_duplicate_name_creates_unique_sibling(tmp_path: Path):
    first = create_experiment_run(
        root=tmp_path,
        approach="classification",
        run_name="demo",
    )
    second = create_experiment_run(
        root=tmp_path,
        approach="classification",
        run_name="demo",
    )

    assert first.run_dir.name == "demo"
    assert second.run_dir.name == "demo-02"


def test_load_checkpoint_roundtrips_dict_and_maps_tensors(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    expected = {"weights": torch.tensor([1.0, 2.0])}
    torch.save(expected, checkpoint_path)

    loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded, dict)
    assert torch.equal(loaded["weights"], expected["weights"])
    assert loaded["weights"].device.type == "cpu"


def test_load_checkpoint_rejects_non_dict_payloads(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save([1, 2, 3], checkpoint_path)

    with pytest.raises(ValueError, match="Expected checkpoint"):
        load_checkpoint(checkpoint_path, torch.device("cpu"))


def test_save_figure_writes_png_file(tmp_path: Path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    output_path = tmp_path / "figures" / "curve.png"

    try:
        save_figure(figure, output_path)
    finally:
        plt.close(figure)

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_run_utils_to_serializable_converts_paths_devices_and_mappings(tmp_path: Path):
    payload = {
        "path": tmp_path / "run",
        "device": torch.device("cpu"),
        "nested": ("x", tmp_path / "checkpoint.pt"),
    }

    assert to_serializable(payload) == {
        "path": str(tmp_path / "run"),
        "device": "cpu",
        "nested": ["x", str(tmp_path / "checkpoint.pt")],
    }


def test_run_utils_paths_from_checkpoint_resolves_standard_layout(tmp_path: Path):
    checkpoint = tmp_path / "runs" / "classification" / "demo" / "checkpoints" / "classification_epoch_0001.pt"

    paths = paths_from_checkpoint(checkpoint)

    assert paths.run_dir == checkpoint.parents[1]
    assert paths.checkpoint_dir == checkpoint.parent
    assert paths.figure_dir == checkpoint.parents[1] / "figures"
    assert paths.history_path == checkpoint.parents[1] / "history.json"


def test_run_utils_paths_from_checkpoint_rejects_non_standard_layout(tmp_path: Path):
    with pytest.raises(ValueError, match="checkpoints directory"):
        paths_from_checkpoint(tmp_path / "run" / "classification_epoch_0001.pt")


def test_run_utils_best_metric_from_history_modes():
    history = {"val_loss": [0.8, 0.6, 0.7], "val_accuracy": [0.2, 0.4, 0.3]}

    assert best_metric_from_history(history, "val_loss") == pytest.approx(0.6)
    assert best_metric_from_history(history, "val_accuracy", mode="max") == pytest.approx(0.4)
    assert best_metric_from_history(history, "missing") is None
    with pytest.raises(ValueError, match="Unsupported"):
        best_metric_from_history(history, "val_loss", mode="median")


def test_run_utils_find_named_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    expected = checkpoint_dir / "mae_best_val_loss.pt"
    expected.write_bytes(b"checkpoint")

    assert find_named_checkpoint(checkpoint_dir, "mae_best_val_loss") == expected
    assert find_named_checkpoint(checkpoint_dir, "dino_best_val_loss") is None


def test_run_utils_validate_training_run_args_accepts_valid_values():
    validate_training_run_args(
        RunValidationConfig(
            epochs=1,
            save_every=1,
            checkpoint_keep_last=3,
            label_smoothing=0.1,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"save_every": 0}, "save_every"),
        ({"checkpoint_keep_last": 0}, "checkpoint_keep_last"),
        ({"label_smoothing": -0.1}, "label_smoothing"),
        ({"label_smoothing": 1.1}, "label_smoothing"),
    ],
)
def test_run_utils_validate_training_run_args_rejects_invalid_values(kwargs, message):
    config = RunValidationConfig(
        epochs=kwargs.get("epochs", 1),
        save_every=kwargs.get("save_every", 1),
        checkpoint_keep_last=kwargs.get("checkpoint_keep_last", 3),
        label_smoothing=kwargs.get("label_smoothing"),
    )

    with pytest.raises(ValueError, match=message):
        validate_training_run_args(config)


def test_run_utils_aggregate_weighted_metrics_averages_by_batch_size():
    averaged = aggregate_weighted_metrics(
        [
            ({"loss": 1.0, "accuracy": 0.25}, 2),
            ({"loss": 0.5, "accuracy": 0.75}, 6),
        ],
        default={"loss": 0.0, "accuracy": 0.0},
    )

    assert averaged["loss"] == pytest.approx(0.625)
    assert averaged["accuracy"] == pytest.approx(0.625)


def test_run_utils_aggregate_weighted_metrics_returns_default_when_empty():
    assert aggregate_weighted_metrics([], default={"loss": 0.0}) == {"loss": 0.0}


def test_run_utils_close_figure_is_safe_for_matplotlib_figures():
    figure, _ = plt.subplots()

    close_figure(figure)

    assert not plt.fignum_exists(figure.number)
