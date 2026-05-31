from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from vit_from_scratch.classification import build_classification_model
from vit_from_scratch.config import ViTConfig
from vit_from_scratch.dino import (
    DINOCenter,
    DINOViewConfig,
    _load_dino_resume_state,
    build_dino_views,
    dino_train_step,
    train_dino,
)


def _make_config(num_classes: int = 8) -> ViTConfig:
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


def test_dino_center_update_moves_toward_batch_mean():
    center = DINOCenter(dim=3, momentum=0.5)
    teacher_logits = torch.tensor(
        [
            [1.0, 3.0, 5.0],
            [3.0, 5.0, 7.0],
        ]
    )

    center.update(teacher_logits)

    assert center.center.shape == (1, 3)
    assert torch.allclose(center.center, torch.tensor([[1.0, 2.0, 3.0]]))


def test_dino_train_step_updates_teacher_ema_and_center():
    student = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 6))
    teacher = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 6))
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    center = DINOCenter(dim=6, momentum=0.5)
    view_one = torch.randn(3, 3, 4, 4)
    view_two = torch.randn(3, 3, 4, 4)
    teacher_weight_before = teacher[1].weight.detach().clone()
    center_before = center.center.detach().clone()

    metrics = dino_train_step(
        student=student,
        teacher=teacher,
        batch_or_views=(view_one, view_two),
        optimizer=optimizer,
        device=torch.device("cpu"),
        center=center,
        momentum=0.8,
    )

    assert metrics["loss"] >= 0.0
    assert isinstance(metrics["loss"], float)
    assert "teacher_entropy" in metrics
    assert metrics["teacher_entropy"] > 0.0
    assert not torch.allclose(teacher[1].weight, teacher_weight_before)
    assert not torch.allclose(center.center, center_before)
    assert teacher[1].weight.grad is None


def test_dino_view_config_builds_teacher_global_and_student_local_views():
    images = torch.randn(2, 3, 16, 16)
    view_config = DINOViewConfig(
        teacher_global_crops=2,
        student_global_crops=2,
        student_local_crops=3,
        teacher_global_crop_scale=(0.6, 1.0),
        student_global_crop_scale=(0.6, 1.0),
        student_local_crop_scale=(0.2, 0.4),
        noise_std=0.0,
    )

    views = build_dino_views(images, torch.device("cpu"), view_config)

    assert len(views.teacher_views) == 2
    assert len(views.student_views) == 5
    assert views.teacher_view_ids == ("global_0", "global_1")
    assert views.student_view_ids == (
        "global_0",
        "global_1",
        "local_0",
        "local_1",
        "local_2",
    )
    assert all(view.shape == images.shape for view in views.teacher_views)
    assert all(view.shape == images.shape for view in views.student_views)


def test_build_dino_views_keeps_explicit_teacher_views_global_only():
    explicit_views = tuple(torch.randn(2, 3, 16, 16) for _ in range(4))

    views = build_dino_views(
        explicit_views,
        torch.device("cpu"),
        DINOViewConfig(
            teacher_global_crops=2,
            student_global_crops=2,
            student_local_crops=2,
            noise_std=0.0,
        ),
    )

    assert len(views.teacher_views) == 2
    assert len(views.student_views) == 4
    assert views.teacher_view_ids == ("explicit_0", "explicit_1")
    assert views.student_view_ids == (
        "explicit_0",
        "explicit_1",
        "explicit_2",
        "explicit_3",
    )
    assert views.teacher_views == explicit_views[:2]
    assert views.student_views == explicit_views


def test_dino_train_step_supports_configured_multi_crop_views():
    student = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 6))
    teacher = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 6))
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    center = DINOCenter(dim=6, momentum=0.5)
    images = torch.randn(3, 3, 8, 8)

    metrics = dino_train_step(
        student=student,
        teacher=teacher,
        batch_or_views=images,
        optimizer=optimizer,
        device=torch.device("cpu"),
        center=center,
        momentum=0.8,
        view_config=DINOViewConfig(
            teacher_global_crops=2,
            student_global_crops=2,
            student_local_crops=2,
            noise_std=0.0,
        ),
    )

    assert metrics["loss"] >= 0.0
    assert metrics["teacher_views"] == 2
    assert metrics["student_views"] == 4
    assert metrics["loss_terms"] == 6


def test_dino_train_step_excludes_matching_global_view_pairs():
    student = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 6))
    teacher = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 6))
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    view_one = torch.randn(3, 3, 4, 4)
    view_two = torch.randn(3, 3, 4, 4)

    metrics = dino_train_step(
        student=student,
        teacher=teacher,
        batch_or_views=(view_one, view_two),
        optimizer=optimizer,
        device=torch.device("cpu"),
        center=DINOCenter(dim=6),
        momentum=0.8,
    )

    assert metrics["teacher_views"] == 2
    assert metrics["student_views"] == 2
    assert metrics["loss_terms"] == 2


def test_train_dino_runs_one_epoch_and_writes_expected_artifacts(tmp_path: Path):
    pytest.importorskip("vit_from_scratch.artifacts")

    config = _make_config()
    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    images = torch.rand(6, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        output_dir=tmp_path,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )

    history = result["history"]
    run_dir = Path(result["run_dir"])
    checkpoint_dir = Path(result["checkpoint_dir"])
    figure_dir = Path(result["figure_dir"])

    assert history["train_loss"] and len(history["train_loss"]) == 1
    assert history["val_loss"] and len(history["val_loss"]) == 1
    assert history["student_temperature"] == [0.1]
    assert history["teacher_temperature"] == [0.04]
    assert (run_dir / "history.json").exists()
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "dino" in metrics
    assert metrics["dino"]["student_entropy"] is not None
    assert any(checkpoint_dir.glob("*.pt"))
    assert (figure_dir / "training_curves.png").exists()
    assert (figure_dir / "dino_attention_diagnostics.png").exists()


def test_train_dino_saves_view_config_in_run_config(tmp_path: Path):
    config = _make_config()
    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    images = torch.rand(6, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)
    view_config = DINOViewConfig(student_local_crops=2, noise_std=0.0)

    result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        output_dir=tmp_path,
        center=DINOCenter(dim=config.num_classes),
        view_config=view_config,
        show_progress=False,
    )

    run_config = json.loads(
        (Path(result["run_dir"]) / "config.json").read_text(encoding="utf-8")
    )
    assert run_config["view_config"]["student_local_crops"] == 2
    assert run_config["view_config"]["teacher_global_crops"] == 2


def test_train_dino_keeps_only_last_three_checkpoints(tmp_path: Path):
    pytest.importorskip("vit_from_scratch.artifacts")

    config = _make_config()
    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    images = torch.rand(8, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=5,
        output_dir=tmp_path,
        checkpoint_keep_last=3,
        save_every=1,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )

    checkpoints = sorted(Path(result["checkpoint_dir"]).glob("dino_epoch_*.pt"))

    assert [path.name for path in checkpoints] == [
        "dino_epoch_0003.pt",
        "dino_epoch_0004.pt",
        "dino_epoch_0005.pt",
    ]


def test_train_dino_resumes_student_teacher_and_center(tmp_path: Path):
    config = _make_config()
    images = torch.rand(8, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    first_result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="dino-resume",
        checkpoint_keep_last=3,
        save_every=1,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )
    checkpoint = torch.load(
        Path(first_result["checkpoint_dir"]) / "dino_epoch_0002.pt",
        map_location="cpu",
    )
    expected_center = checkpoint["center"].clone()

    loaded_student = build_classification_model(config)
    loaded_teacher = build_classification_model(config)
    loaded_teacher.load_state_dict(loaded_student.state_dict())
    loaded_optimizer = torch.optim.AdamW(loaded_student.parameters(), lr=1e-3)
    loaded_center = DINOCenter(dim=config.num_classes)
    resumed_epoch, _, restored_center, _ = _load_dino_resume_state(
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "dino_epoch_0002.pt",
        student=loaded_student,
        teacher=loaded_teacher,
        optimizer=loaded_optimizer,
        scheduler=None,
        center=loaded_center,
        device=torch.device("cpu"),
        epochs=4,
    )

    assert resumed_epoch == 2
    assert restored_center is loaded_center
    assert torch.allclose(loaded_center.center, expected_center, atol=1e-6, rtol=0.0)

    resumed_student = build_classification_model(config)
    resumed_teacher = build_classification_model(config)
    resumed_teacher.load_state_dict(resumed_student.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_student.parameters(), lr=1e-3)
    resumed_center = DINOCenter(dim=config.num_classes)
    resumed_result = train_dino(
        student=resumed_student,
        teacher=resumed_teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        run_name="ignored-on-resume",
        checkpoint_keep_last=3,
        save_every=1,
        center=resumed_center,
        show_progress=False,
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "dino_epoch_0002.pt",
    )

    assert Path(resumed_result["run_dir"]) == Path(first_result["run_dir"])
    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["train_loss"]) == 4
    assert (Path(resumed_result["checkpoint_dir"]) / "dino_epoch_0004.pt").exists()
    assert not torch.allclose(resumed_center.center, expected_center, atol=1e-6, rtol=0.0)


def test_train_dino_tracks_best_val_loss_checkpoint(tmp_path: Path):
    config = _make_config()
    images = torch.rand(8, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="dino-best",
        checkpoint_keep_last=3,
        save_every=1,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )

    assert result["best_val_loss"] == min(result["history"]["val_loss"])
    best_path = Path(result["best_checkpoints"]["best_val_loss"])
    assert best_path.exists()

    best_checkpoint = torch.load(best_path, map_location="cpu")
    assert best_checkpoint["best_val_loss"] == result["best_val_loss"]
    assert best_checkpoint["history"]["val_loss"]


def test_train_dino_records_lr_and_saves_scheduler_state(tmp_path: Path):
    config = _make_config()
    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=1e-5,
    )
    images = torch.rand(6, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )

    assert len(result["history"]["lr"]) == 2
    checkpoint = torch.load(
        Path(result["checkpoint_dir"]) / "dino_epoch_0002.pt",
        map_location="cpu",
    )
    assert "scheduler_state_dict" in checkpoint
    assert checkpoint["history"]["lr"] == result["history"]["lr"]


def test_train_dino_resumes_with_scheduler_checkpoint(tmp_path: Path):
    config = _make_config()
    images = torch.rand(8, 3, 16, 16)
    train_loader = DataLoader(TensorDataset(images[:4]), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(images[4:]), batch_size=2, shuffle=False)

    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    first_result = train_dino(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="dino-resume-scheduler",
        checkpoint_keep_last=3,
        save_every=1,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
    )

    resumed_student = build_classification_model(config)
    resumed_teacher = build_classification_model(config)
    resumed_teacher.load_state_dict(resumed_student.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_student.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    resumed_result = train_dino(
        student=resumed_student,
        teacher=resumed_teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        center=DINOCenter(dim=config.num_classes),
        show_progress=False,
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "dino_epoch_0002.pt",
    )

    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["lr"]) == 4
