from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
torchvision = pytest.importorskip("torchvision")

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from vit_from_scratch.classification import (
    build_classification_model,
    train_classification,
)
from vit_from_scratch.config import ViTConfig


def _make_config(num_classes: int = 3) -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def _make_loader(
    num_samples: int,
    num_classes: int,
    batch_size: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(num_samples, 3, 16, 16, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _write_external_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (24, 20), color=color)
    image.save(path)


def _make_fake_loader(
    num_samples: int,
    num_classes: int,
    batch_size: int,
) -> DataLoader:
    dataset = torchvision.datasets.FakeData(
        size=num_samples,
        image_size=(3, 16, 16),
        num_classes=num_classes,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_train_classification_runs_one_epoch_and_writes_artifacts(tmp_path: Path):
    num_classes = 3
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        output_dir=tmp_path,
        run_name="smoke",
        class_names=["zero", "one", "two"],
        show_progress=False,
    )

    history = result["history"]
    assert history.keys() == {
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "lr",
    }
    for values in history.values():
        assert len(values) == 1
        assert isinstance(values[0], float)

    run_dir = Path(result["run_dir"])
    checkpoint_dir = Path(result["checkpoint_dir"])
    figure_dir = Path(result["figure_dir"])

    history_path = run_dir / "history.json"
    assert history_path.exists()
    assert json.loads(history_path.read_text(encoding="utf-8")) == history

    checkpoints = sorted(checkpoint_dir.glob("classification_epoch_*.pt"))
    assert len(checkpoints) == 1
    assert checkpoints[0].name == "classification_epoch_0001.pt"

    curves_path = figure_dir / "training_curves.png"
    assert curves_path.exists()
    assert curves_path.stat().st_size > 0
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "classification" in metrics
    assert metrics["classification"]["top1_accuracy"] >= 0.0
    assert len(metrics["classification"]["confusion_matrix"]) == num_classes
    assert (figure_dir / "confusion_matrix.png").exists()


def test_train_classification_writes_external_predictions_when_directory_is_given(tmp_path: Path):
    num_classes = 3
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)
    external_dir = tmp_path / "external_images"
    external_dir.mkdir()
    _write_external_image(external_dir / "first.png", (255, 0, 0))
    _write_external_image(external_dir / "second.jpg", (0, 255, 0))

    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        output_dir=tmp_path,
        run_name="external-smoke",
        class_names=["zero", "one", "two"],
        external_image_dir=external_dir,
        show_progress=False,
    )

    figure_dir = Path(result["figure_dir"])
    assert (figure_dir / "external_predictions.png").exists()


def test_train_classification_keeps_only_last_n_checkpoints(tmp_path: Path):
    num_classes = 3
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=5,
        output_dir=tmp_path,
        run_name="retention",
        checkpoint_keep_last=3,
        show_progress=False,
    )

    checkpoint_dir = Path(result["checkpoint_dir"])
    remaining = sorted(path.name for path in checkpoint_dir.glob("classification_epoch_*.pt"))

    assert remaining == [
        "classification_epoch_0003.pt",
        "classification_epoch_0004.pt",
        "classification_epoch_0005.pt",
    ]


def test_train_classification_resumes_from_checkpoint(tmp_path: Path):
    num_classes = 3
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    first_model = build_classification_model(_make_config(num_classes=num_classes))
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=1e-3)
    first_result = train_classification(
        model=first_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=first_optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="resume-classification",
        checkpoint_keep_last=3,
        show_progress=False,
    )
    checkpoint = Path(first_result["checkpoint_dir"]) / "classification_epoch_0002.pt"

    resumed_model = build_classification_model(_make_config(num_classes=num_classes))
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_result = train_classification(
        model=resumed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        run_name="ignored-on-resume",
        checkpoint_keep_last=3,
        show_progress=False,
        resume_checkpoint=checkpoint,
    )

    assert Path(resumed_result["run_dir"]) == Path(first_result["run_dir"])
    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["train_loss"]) == 4
    assert (Path(resumed_result["checkpoint_dir"]) / "classification_epoch_0004.pt").exists()


def test_train_classification_rejects_resume_when_target_epoch_is_not_higher(
    tmp_path: Path,
):
    num_classes = 3
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="resume-error",
        show_progress=False,
    )

    with pytest.raises(ValueError, match="target epochs must be greater"):
        resumed_model = build_classification_model(_make_config(num_classes=num_classes))
        train_classification(
            model=resumed_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1e-3),
            device=torch.device("cpu"),
            epochs=2,
            output_dir=tmp_path,
            resume_checkpoint=Path(result["checkpoint_dir"]) / "classification_epoch_0002.pt",
            show_progress=False,
        )


def test_train_classification_tracks_best_checkpoints_and_accepts_label_smoothing(
    tmp_path: Path,
):
    num_classes = 3
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _make_fake_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_fake_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="best-classification",
        label_smoothing=0.1,
        show_progress=False,
    )

    assert result["best_val_loss"] == min(result["history"]["val_loss"])
    assert result["best_val_accuracy"] == max(result["history"]["val_accuracy"])
    best_checkpoints = result["best_checkpoints"]
    assert Path(best_checkpoints["best_val_loss"]).exists()
    assert Path(best_checkpoints["best_val_accuracy"]).exists()

    best_loss_checkpoint = torch.load(best_checkpoints["best_val_loss"], map_location="cpu")
    assert best_loss_checkpoint["best_val_loss"] == result["best_val_loss"]
    assert best_loss_checkpoint["best_checkpoints"]["best_val_accuracy"] is not None


def test_train_classification_records_lr_and_saves_scheduler_state(tmp_path: Path):
    num_classes = 3
    model = build_classification_model(_make_config(num_classes=num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=1e-5,
    )
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    result = train_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="scheduler-classification",
        show_progress=False,
    )

    assert len(result["history"]["lr"]) == 2
    checkpoint = torch.load(
        Path(result["checkpoint_dir"]) / "classification_epoch_0002.pt",
        map_location="cpu",
    )
    assert "scheduler_state_dict" in checkpoint
    assert checkpoint["history"]["lr"] == result["history"]["lr"]


def test_train_classification_resumes_with_scheduler_checkpoint(tmp_path: Path):
    num_classes = 3
    train_loader = _make_loader(num_samples=8, num_classes=num_classes, batch_size=4)
    val_loader = _make_loader(num_samples=4, num_classes=num_classes, batch_size=2)

    first_model = build_classification_model(_make_config(num_classes=num_classes))
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=1e-3)
    first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        first_optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    first_result = train_classification(
        model=first_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="resume-scheduler-classification",
        checkpoint_keep_last=3,
        show_progress=False,
    )

    resumed_model = build_classification_model(_make_config(num_classes=num_classes))
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    resumed_result = train_classification(
        model=resumed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "classification_epoch_0002.pt",
        show_progress=False,
    )

    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["lr"]) == 4
