import pytest

torch = pytest.importorskip("torch")

from vit_from_scratch.classification import (
    build_classification_model,
    classification_eval_step,
    classification_loss,
    classification_train_step,
)
from vit_from_scratch.config import ViTConfig
from vit_from_scratch.dino import dino_loss, dino_train_step, update_teacher_ema
from vit_from_scratch.masked_autoencoder import (
    MaskedAutoencoder,
    mae_train_step,
    masked_reconstruction_loss,
    patchify,
    random_patch_mask,
)


def _make_config(num_classes: int = 5) -> ViTConfig:
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


def test_classification_model_and_steps_support_synthetic_batches():
    config = _make_config(num_classes=5)
    model = build_classification_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3])

    logits = model(images)
    loss = classification_loss(logits, labels)
    assert logits.shape == (4, 5)
    assert loss.ndim == 0

    train_metrics = classification_train_step(
        model=model,
        batch=(images, labels),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    eval_metrics = classification_eval_step(
        model=model,
        batch=(images, labels),
        device=torch.device("cpu"),
    )

    for metrics in (train_metrics, eval_metrics):
        assert set(metrics) == {"loss", "accuracy"}
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)


def test_patchify_mask_and_mae_training_step_work_on_synthetic_images():
    images = torch.randn(2, 3, 16, 16)
    patches = patchify(images, patch_size=4)
    mask = random_patch_mask(
        batch_size=2,
        num_patches=patches.shape[1],
        mask_ratio=0.5,
        device=images.device,
    )

    assert patches.shape == (2, 16, 48)
    assert mask.shape == (2, 16)
    assert torch.all(mask.sum(dim=1) == 8)

    loss = masked_reconstruction_loss(
        predicted_patches=patches + 0.1,
        target_patches=patches,
        mask=mask,
    )
    assert loss.ndim == 0

    model = MaskedAutoencoder(_make_config(num_classes=3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = mae_train_step(
        model=model,
        images_or_batch=images,
        optimizer=optimizer,
        device=torch.device("cpu"),
        mask_ratio=0.5,
    )

    assert set(metrics) == {"loss", "mask_ratio"}
    assert isinstance(metrics["loss"], float)
    assert metrics["mask_ratio"] == pytest.approx(0.5, rel=0.0, abs=1e-6)


def test_dino_loss_ema_and_train_step_support_two_synthetic_views():
    config = _make_config(num_classes=7)
    student = build_classification_model(config)
    teacher = build_classification_model(config)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    view_one = torch.randn(2, 3, 16, 16)
    view_two = torch.randn(2, 3, 16, 16)
    student_logits = student(view_one)
    teacher_logits = teacher(view_two)

    loss = dino_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        center=torch.zeros(1, 7),
    )
    assert loss.ndim == 0

    before = teacher.head.weight.detach().clone()
    with torch.no_grad():
        student.head.weight.add_(0.25)
    update_teacher_ema(student, teacher, momentum=0.5)
    expected = before * 0.5 + student.head.weight.detach() * 0.5
    assert torch.allclose(teacher.head.weight, expected)

    metrics = dino_train_step(
        student=student,
        teacher=teacher,
        batch_or_views=(view_one, view_two),
        optimizer=optimizer,
        device=torch.device("cpu"),
        center=torch.zeros(1, 7),
        momentum=0.9,
    )

    assert {
        "loss",
        "teacher_temperature",
        "student_temperature",
        "teacher_views",
        "student_views",
    } <= set(metrics)
    assert isinstance(metrics["loss"], float)
    assert all(parameter.grad is None for parameter in teacher.parameters())
