from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from vit_from_scratch.config import ViTConfig
from vit_from_scratch.masked_autoencoder import (
    MaskedAutoencoder,
    patchify,
    train_masked_autoencoder,
    unpatchify,
)


def _make_config() -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=4,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def _make_loader(num_samples: int = 6, batch_size: int = 2) -> DataLoader:
    images = torch.linspace(
        0.0,
        1.0,
        steps=num_samples * 3 * 16 * 16,
        dtype=torch.float32,
    ).reshape(num_samples, 3, 16, 16)
    labels = torch.zeros(num_samples, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def test_unpatchify_inverts_patchify_for_simple_tensor():
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)

    patches = patchify(images, patch_size=4)
    reconstructed = unpatchify(
        patches,
        patch_size=4,
        image_size=8,
        channels=3,
    )

    assert torch.equal(reconstructed, images)


def test_train_masked_autoencoder_writes_artifacts_for_one_epoch(tmp_path: Path):
    model = MaskedAutoencoder(_make_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = train_masked_autoencoder(
        model=model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        output_dir=tmp_path,
        run_name="mae-smoke",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    history = result["history"]
    run_dir = Path(result["run_dir"])
    checkpoint_dir = Path(result["checkpoint_dir"])
    figure_dir = Path(result["figure_dir"])

    assert history.keys() == {
        "train_loss",
        "val_loss",
        "train_mask_ratio",
        "val_mask_ratio",
        "lr",
    }
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert (run_dir / "history.json").exists()
    assert any(checkpoint_dir.glob("mae_epoch_*.pt"))
    assert (figure_dir / "training_curves.png").exists()
    assert (figure_dir / "reconstruction.png").exists()
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mae" in metrics
    assert metrics["mae"]["reconstruction_mse"] >= 0.0
    assert (figure_dir / "reconstruction_errors.png").exists()


def test_train_masked_autoencoder_keeps_only_last_n_checkpoints(tmp_path: Path):
    model = MaskedAutoencoder(_make_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = train_masked_autoencoder(
        model=model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=5,
        output_dir=tmp_path,
        run_name="mae-checkpoints",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    checkpoint_names = sorted(
        path.name for path in Path(result["checkpoint_dir"]).glob("mae_epoch_*.pt")
    )

    assert checkpoint_names == [
        "mae_epoch_0003.pt",
        "mae_epoch_0004.pt",
        "mae_epoch_0005.pt",
    ]


def test_train_masked_autoencoder_resumes_from_checkpoint(tmp_path: Path):
    first_model = MaskedAutoencoder(_make_config())
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=1e-3)
    first_result = train_masked_autoencoder(
        model=first_model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=first_optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="mae-resume",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    resumed_model = MaskedAutoencoder(_make_config())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_result = train_masked_autoencoder(
        model=resumed_model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        run_name="ignored-on-resume",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "mae_epoch_0002.pt",
    )

    assert Path(resumed_result["run_dir"]) == Path(first_result["run_dir"])
    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["train_loss"]) == 4
    assert (Path(resumed_result["checkpoint_dir"]) / "mae_epoch_0004.pt").exists()


def test_train_masked_autoencoder_tracks_best_val_loss_checkpoint(tmp_path: Path):
    model = MaskedAutoencoder(_make_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = train_masked_autoencoder(
        model=model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="mae-best",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    assert result["best_val_loss"] == min(result["history"]["val_loss"])
    best_path = Path(result["best_checkpoints"]["best_val_loss"])
    assert best_path.exists()

    best_checkpoint = torch.load(best_path, map_location="cpu")
    assert best_checkpoint["best_val_loss"] == result["best_val_loss"]
    assert best_checkpoint["history"]["val_loss"]


def test_train_masked_autoencoder_records_lr_and_saves_scheduler_state(tmp_path: Path):
    model = MaskedAutoencoder(_make_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=1e-5,
    )

    result = train_masked_autoencoder(
        model=model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="mae-scheduler",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    assert len(result["history"]["lr"]) == 2
    checkpoint = torch.load(
        Path(result["checkpoint_dir"]) / "mae_epoch_0002.pt",
        map_location="cpu",
    )
    assert "scheduler_state_dict" in checkpoint
    assert checkpoint["history"]["lr"] == result["history"]["lr"]


def test_train_masked_autoencoder_resumes_with_scheduler_checkpoint(tmp_path: Path):
    first_model = MaskedAutoencoder(_make_config())
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=1e-3)
    first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        first_optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    first_result = train_masked_autoencoder(
        model=first_model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        run_name="mae-resume-scheduler",
        checkpoint_keep_last=3,
        save_every=1,
        mask_ratio=0.5,
        show_progress=False,
    )

    resumed_model = MaskedAutoencoder(_make_config())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer,
        T_max=4,
        eta_min=1e-5,
    )
    resumed_result = train_masked_autoencoder(
        model=resumed_model,
        train_loader=_make_loader(),
        val_loader=_make_loader(num_samples=4, batch_size=2),
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cpu"),
        epochs=4,
        output_dir=tmp_path,
        mask_ratio=0.5,
        show_progress=False,
        resume_checkpoint=Path(first_result["checkpoint_dir"]) / "mae_epoch_0002.pt",
    )

    assert resumed_result["resumed_from_epoch"] == 2
    assert len(resumed_result["history"]["lr"]) == 4
