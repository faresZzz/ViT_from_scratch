"""Masked autoencoder utilities for ViT-style experiments."""

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
from vit_from_scratch.embedding import CosinePositionEmbedding
from vit_from_scratch.encoder import TransformerEncoderBlock
from vit_from_scratch.evaluation import evaluate_masked_autoencoder, reconstruct_external_images
from vit_from_scratch.patch_embedding import PatchEmbedding
from vit_from_scratch.run_utils import (
    RunValidationConfig,
    best_metric_from_history,
    find_named_checkpoint,
    to_serializable,
    validate_training_run_args,
)
from vit_from_scratch.training_loop import (
    append_history,
    build_checkpoint_state,
    prepare_training_run,
    run_epoch,
    save_figure_safely,
)
from vit_from_scratch.visualization import (
    plot_external_reconstructions,
    plot_reconstruction_errors,
    plot_training_curves,
)


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """Convert images shaped ``[B, C, H, W]`` into flattened patches."""

    if images.ndim != 4:
        raise ValueError(
            "patchify expects images with shape [B, C, H, W]: "
            f"got {tuple(images.shape)}."
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")

    batch_size, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "Image height and width must be divisible by patch_size: "
            f"got H={height}, W={width}, patch_size={patch_size}."
        )

    patches_per_height = height // patch_size
    patches_per_width = width // patch_size
    patches = images.reshape(
        batch_size,
        channels,
        patches_per_height,
        patch_size,
        patches_per_width,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5) # [B, patches_per_height, patches_per_width, C, patch_size, patch_size]
    return patches.reshape(batch_size, patches_per_height * patches_per_width, -1) # [B, N, patch_dim]


def unpatchify(
    patches: Tensor,
    patch_size: int,
    image_size: int,
    channels: int = 3,
) -> Tensor:
    """Reconstruct square images shaped ``[B, C, H, W]`` from flattened patches."""

    if patches.ndim != 3:
        raise ValueError(
            "unpatchify expects patches with shape [B, N, D]: "
            f"got {tuple(patches.shape)}."
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}.")
    if image_size % patch_size != 0:
        raise ValueError(
            "image_size must be divisible by patch_size: "
            f"got image_size={image_size}, patch_size={patch_size}."
        )

    patches_per_side = image_size // patch_size
    expected_num_patches = patches_per_side * patches_per_side
    expected_patch_dim = channels * patch_size * patch_size
    if patches.shape[1] != expected_num_patches:
        raise ValueError(
            "Patch count does not match the requested square image size: "
            f"got {patches.shape[1]}, expected {expected_num_patches}."
        )
    if patches.shape[2] != expected_patch_dim:
        raise ValueError(
            "Patch dimension does not match channels * patch_size^2: "
            f"got {patches.shape[2]}, expected {expected_patch_dim}."
        )

    batch_size = patches.shape[0]
    images = patches.reshape(
        batch_size,
        patches_per_side,
        patches_per_side,
        channels,
        patch_size,
        patch_size,
    )
    images = images.permute(0, 3, 1, 4, 2, 5) # [B, C, patches_per_side, patch_size, patches_per_side, patch_size]
    return images.reshape(batch_size, channels, image_size, image_size) # [B, C, H, W]


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device | None = None,
) -> Tensor:
    """Sample a per-example boolean mask with an exact number of masked patches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_patches <= 0:
        raise ValueError("num_patches must be a positive integer.")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(
            "mask_ratio must be between 0.0 and 1.0 inclusive: "
            f"got {mask_ratio}."
        )

    num_masked = int(round(num_patches * mask_ratio))
    noise = torch.rand(batch_size, num_patches, device=device)
    indices = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    if num_masked > 0:
        mask.scatter_(1, indices[:, :num_masked], True)
    return mask


def masked_reconstruction_loss(
    predicted_patches: Tensor,
    target_patches: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute mean squared reconstruction error over masked patches.

    Targets are per-patch normalised before computing MSE, following the
    practice described in the MAE paper (He et al., 2021) which improves
    representation quality.
    """

    if predicted_patches.shape != target_patches.shape:
        raise ValueError("predicted_patches and target_patches must share the same shape.")
    if mask.shape != predicted_patches.shape[:2]:
        raise ValueError("mask must have shape [B, N] matching the patch tensors.")

    # Per-patch normalize target (MAE paper, improves representation quality)
    mean = target_patches.mean(dim=-1, keepdim=True)
    var = target_patches.var(dim=-1, keepdim=True)
    target_patches = (target_patches - mean) / (var + 1e-6).sqrt()

    squared_error = (predicted_patches - target_patches).pow(2).mean(dim=-1)
    mask = mask.to(dtype=torch.bool)
    if mask.any():
        return squared_error[mask].mean()
    return squared_error.mean()


class MaskedAutoencoder(nn.Module):
    """Masked autoencoder following He et al. (2021).

    Architecture: the encoder operates only on visible (unmasked) patches,
    while the decoder receives the full sequence with learned mask tokens
    substituted at masked positions. An auxiliary CLS token is prepended
    to the encoder input. Both encoder and decoder use fixed sine-cosine
    positional embeddings.

    The decoder is a lightweight stack of transformer blocks (<10% compute
    per token vs the encoder), followed by a linear projection to pixel
    values. This matches the paper: "The last layer of the decoder is a
    linear projection whose number of output channels equals the number
    of pixel values in a patch."
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        patch_dim = config.in_channels * config.patch_size * config.patch_size

        # Resolve decoder hyperparameters (paper: decoder is <10% compute of encoder)
        dec_dim = config.decoder_embed_dim or config.embed_dim
        dec_depth = config.decoder_depth or 1
        head_dim = config.embed_dim // config.num_heads
        dec_heads = config.decoder_num_heads or max(1, dec_dim // head_dim)
        self.decoder_embed_dim = dec_dim

        # CLS dummy token (paper: "auxiliary dummy token" for later fine-tuning)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Mask token in decoder dimension (paper: "shared, learned vector")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        # Positional embeddings: use cosine by default; rope2d skips encoder PE
        # (rotary PE is applied inside attention instead).
        if config.position_embedding == "rope2d":
            self.position_embedding: CosinePositionEmbedding | None = None
        else:
            self.position_embedding = CosinePositionEmbedding(
                num_tokens=config.num_patches + 1,
                embed_dim=config.embed_dim,
                dropout=0.0,
            )
        self.decoder_position_embedding = CosinePositionEmbedding(
            num_tokens=config.num_patches + 1,
            embed_dim=dec_dim,
            dropout=0.0,
        )

        # Encoder: transformer blocks on visible tokens only
        encoder_rope_mode = "2d" if config.position_embedding == "rope2d" else "none"
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    rope_mode=encoder_rope_mode,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        # Projection from encoder space to decoder space
        self.encoder_to_decoder = nn.Linear(config.embed_dim, dec_dim)

        # Decoder: lightweight transformer blocks (paper: "another series of
        # Transformer blocks") + final linear projection to pixel values
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=dec_dim,
                    num_heads=dec_heads,
                    mlp_hidden_dim=int(dec_dim * config.mlp_ratio),
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    use_rope=False,
                )
                for _ in range(dec_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, patch_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, images: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        tokens = self.patch_embed(images)  # [B, N, D]
        B, N, D = tokens.shape

        if patch_mask is None:
            patch_mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        if patch_mask.shape != tokens.shape[:2]:
            raise ValueError(
                "patch_mask must have shape [B, N] matching embedded patches: "
                f"got {tuple(patch_mask.shape)}, expected {tuple(tokens.shape[:2])}."
            )

        # Prepend CLS dummy token (paper: "auxiliary dummy token")
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens_with_cls = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]

        # Add sine-cosine PE to full sequence BEFORE masking (skipped for rope2d)
        if self.position_embedding is not None:
            pos_embed = self.position_embedding.position_embedding.to(
                device=tokens.device, dtype=tokens.dtype,
            )
            tokens_with_cls = tokens_with_cls + pos_embed

        # CLS is always visible (position 0); extend mask with False for CLS
        cls_visible = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        full_mask = torch.cat([cls_visible, patch_mask], dim=1)  # [B, N+1]
        visible_mask = ~full_mask  # [B, N+1]

        # -- Encoder: visible tokens only (CLS + visible patches) --
        num_visible = int(visible_mask[0].sum().item())
        indices = torch.zeros(
            B, num_visible, dtype=torch.long, device=tokens.device,
        )
        # Gather indices of visible tokens for each example in the batch. This is a bit involved since the number of visible tokens is fixed but their positions vary per example.
        for i in range(B):
            indices[i] = visible_mask[i].nonzero(as_tuple=False).squeeze(-1)

        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, num_visible, D]
        visible_tokens = torch.gather(tokens_with_cls, dim=1, index=idx_expanded)

        for block in self.encoder:
            visible_tokens = block(visible_tokens)
        visible_tokens = self.norm(visible_tokens)

        # -- Decoder: full sequence with mask tokens --
        # Project encoder output into decoder space
        dec_dim = self.decoder_embed_dim
        visible_tokens = self.encoder_to_decoder(visible_tokens)  # [B, num_vis, dec_dim]

        # Reconstruct full N+1 sequence: mask tokens at masked positions,
        # visible tokens (incl. CLS) at their original positions (= unshuffle)
        idx_dec = indices.unsqueeze(-1).expand(-1, -1, dec_dim)
        full_tokens = self.mask_token.expand(B, N + 1, dec_dim).clone()
        full_tokens.scatter_(1, idx_dec, visible_tokens)

        # Add decoder sine-cosine PE to ALL tokens in the full set
        # (paper: "We add positional embeddings to all tokens in this full set")
        dec_pos = self.decoder_position_embedding.position_embedding.to(
            device=full_tokens.device, dtype=full_tokens.dtype,
        )
        full_tokens = full_tokens + dec_pos

        # Run decoder transformer blocks
        for block in self.decoder_blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.decoder_norm(full_tokens)

        # Project to pixel space, skip CLS token (only predict patches)
        patch_tokens = full_tokens[:, 1:]  # [B, N, dec_dim]
        return self.decoder_pred(patch_tokens)  # [B, N, patch_dim]


def _resolve_images(images_or_batch: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    if isinstance(images_or_batch, Tensor):
        return images_or_batch
    if not images_or_batch:
        raise ValueError("Expected a tensor or non-empty batch.")
    images = images_or_batch[0]
    if not isinstance(images, Tensor):
        raise ValueError("Expected batch[0] to be an image tensor.")
    return images



def mae_train_step(
    model: MaskedAutoencoder,
    images_or_batch: Tensor | tuple[Tensor, ...] | list[Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.75,
) -> dict[str, float]:
    """Run one MAE training step from raw images or a dataloader batch."""

    model.train()
    images = _resolve_images(images_or_batch).to(device)
    target_patches = patchify(images, patch_size=model.config.patch_size)
    patch_mask = random_patch_mask(
        batch_size=images.shape[0],
        num_patches=target_patches.shape[1],
        mask_ratio=mask_ratio,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    predicted_patches = model(images, patch_mask)
    loss = masked_reconstruction_loss(predicted_patches, target_patches, patch_mask)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "mask_ratio": float(patch_mask.float().mean().item()),
    }


@torch.no_grad()
def mae_eval_step(
    model: MaskedAutoencoder,
    images_or_batch: Tensor | tuple[Tensor, ...] | list[Tensor],
    device: torch.device,
    mask_ratio: float = 0.75,
) -> dict[str, float]:
    """Run one MAE evaluation step from raw images or a dataloader batch."""

    model.eval()
    images = _resolve_images(images_or_batch).to(device)
    target_patches = patchify(images, patch_size=model.config.patch_size)
    patch_mask = random_patch_mask(
        batch_size=images.shape[0],
        num_patches=target_patches.shape[1],
        mask_ratio=mask_ratio,
        device=device,
    )
    predicted_patches = model(images, patch_mask)
    loss = masked_reconstruction_loss(predicted_patches, target_patches, patch_mask)
    return {
        "loss": float(loss.item()),
        "mask_ratio": float(patch_mask.float().mean().item()),
    }



def _initial_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "val_loss": [],
        "train_mask_ratio": [],
        "val_mask_ratio": [],
        "lr": [],
    }



def _build_run_config(
    *,
    model: MaskedAutoencoder,
    epochs: int,
    mask_ratio: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    external_image_dir: str | Path | None,
    external_image_size: int | None,
    resume_checkpoint: Path | None,
    resumed_from_epoch: int | None,
) -> dict[str, object]:
    config = to_serializable(model.config)
    config.update(
        {
            "epochs": epochs,
            "target_epochs": epochs,
            "mask_ratio": mask_ratio,
            "scheduler": None if scheduler is None else scheduler.__class__.__name__,
            "external_image_dir": None if external_image_dir is None else str(external_image_dir),
            "external_image_size": external_image_size,
            "resume_from": None if resume_checkpoint is None else str(resume_checkpoint),
            "resumed_from_epoch": resumed_from_epoch,
        }
    )
    return to_serializable(config)



def _plot_reconstruction_figure(
    images: Tensor,
    masked_images: Tensor,
    reconstructed_images: Tensor,
):
    import matplotlib.pyplot as plt

    num_images = min(3, images.shape[0])
    figure, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
    if num_images == 1:
        axes = axes[None, :]

    for column, title in enumerate(("Original", "Masked", "Reconstruction")):
        axes[0, column].set_title(title)

    for row in range(num_images):
        for column, tensor in enumerate((images, masked_images, reconstructed_images)):
            image = tensor[row].detach().cpu().float().clamp(0.0, 1.0)
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    figure.tight_layout()
    return figure


@torch.no_grad()
def _save_reconstruction_preview(
    model: MaskedAutoencoder,
    val_loader,
    device: torch.device,
    figure_dir: Path,
    mask_ratio: float,
) -> None:
    try:
        first_batch = next(iter(val_loader))
    except StopIteration:
        return

    images = _resolve_images(first_batch).to(device)
    if images.shape[0] == 0:
        return

    target_patches = patchify(images, patch_size=model.config.patch_size)
    patch_mask = random_patch_mask(
        batch_size=images.shape[0],
        num_patches=target_patches.shape[1],
        mask_ratio=mask_ratio,
        device=device,
    )
    predicted_patches = model(images, patch_mask)
    masked_patches = target_patches * (~patch_mask).unsqueeze(-1).to(target_patches.dtype)
    reconstructed_patches = torch.where(
        patch_mask.unsqueeze(-1),
        predicted_patches,
        target_patches,
    )
    masked_images = unpatchify(
        masked_patches,
        patch_size=model.config.patch_size,
        image_size=model.config.image_size,
        channels=model.config.in_channels,
    )
    reconstructed_images = unpatchify(
        reconstructed_patches,
        patch_size=model.config.patch_size,
        image_size=model.config.image_size,
        channels=model.config.in_channels,
    )
    figure = _plot_reconstruction_figure(images, masked_images, reconstructed_images)
    save_figure_safely(figure, figure_dir / "reconstruction.png")


def _save_training_curves(history: dict[str, list[float]], figure_dir: Path) -> None:
    figure = plot_training_curves(history)
    save_figure_safely(figure, figure_dir / "training_curves.png")


def _save_reconstruction_metrics_and_figures(
    *,
    model: MaskedAutoencoder,
    val_loader,
    device: torch.device,
    run_paths: ExperimentPaths,
    mask_ratio: float,
) -> None:
    metrics = evaluate_masked_autoencoder(
        model=model,
        dataloader=val_loader,
        device=device,
        mask_ratio=mask_ratio,
        max_batches=4,
    )
    sample_mse = metrics.pop("sample_mse", [])
    save_json({"mae": metrics}, run_paths.run_dir / "metrics.json")
    if sample_mse:
        figure = plot_reconstruction_errors(sample_mse)
        save_figure_safely(figure, run_paths.figure_dir / "reconstruction_errors.png")


def _save_external_reconstruction_figure(
    *,
    model: MaskedAutoencoder,
    device: torch.device,
    figure_dir: Path,
    external_image_dir: str | Path | None,
    external_image_size: int | None,
    external_mean: tuple[float, float, float] | list[float] | None,
    external_std: tuple[float, float, float] | list[float] | None,
    mask_ratio: float,
) -> None:
    if external_image_dir is None:
        return

    payload = reconstruct_external_images(
        model=model,
        image_dir=external_image_dir,
        device=device,
        image_size=int(external_image_size or model.config.image_size),
        mask_ratio=mask_ratio,
        mean=external_mean,
        std=external_std,
    )
    figure = plot_external_reconstructions(
        images=payload["images"],
        reconstructed_images=payload["reconstructed_images"],
        paths=[path.name for path in payload["paths"]],
        mean=payload["mean"],
        std=payload["std"],
    )
    save_figure_safely(figure, figure_dir / "external_reconstruction.png")


def train_masked_autoencoder(
    model: MaskedAutoencoder,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    output_dir: str | Path = "runs",
    run_name: str | None = None,
    checkpoint_keep_last: int = 3,
    save_every: int = 1,
    mask_ratio: float = 0.75,
    external_image_dir: str | Path | None = None,
    external_image_size: int | None = None,
    external_mean: tuple[float, float, float] | list[float] | None = None,
    external_std: tuple[float, float, float] | list[float] | None = None,
    show_progress: bool = True,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Train a masked autoencoder and persist artifacts for the run."""

    validate_training_run_args(
        RunValidationConfig(
            epochs=epochs,
            save_every=save_every,
            checkpoint_keep_last=checkpoint_keep_last,
        )
    )
    (
        run_paths,
        history,
        start_epoch,
        resumed_from_epoch,
        resolved_resume_checkpoint,
    ) = prepare_training_run(
        approach="mae",
        output_dir=output_dir,
        run_name=run_name,
        resume_checkpoint=resume_checkpoint,
        device=device,
        target_epochs=epochs,
        models=[model],
        model_loaders={"model": (model, "model_state_dict")},
        optimizer=optimizer,
        scheduler=scheduler,
        initial_history_fn=_initial_history,
    )
    history.setdefault("lr", [])
    best_val_loss = best_metric_from_history(history, "val_loss")
    best_checkpoints = {
        "best_val_loss": find_named_checkpoint(run_paths.checkpoint_dir, "mae_best_val_loss")
    }

    save_json(
        _build_run_config(
            model=model,
            epochs=epochs,
            mask_ratio=mask_ratio,
            scheduler=scheduler,
            external_image_dir=external_image_dir,
            external_image_size=external_image_size,
            resume_checkpoint=resolved_resume_checkpoint,
            resumed_from_epoch=resumed_from_epoch,
        ),
        run_paths.config_path,
    )

    def _train_step(*, batch: object, **kwargs: object) -> dict[str, float]:
        return mae_train_step(
            model=model,
            images_or_batch=batch,
            optimizer=optimizer,
            device=device,
            mask_ratio=mask_ratio,
        )

    def _eval_step(*, batch: object, **kwargs: object) -> dict[str, float]:
        return mae_eval_step(
            model=model,
            images_or_batch=batch,
            device=device,
            mask_ratio=mask_ratio,
        )

    def _batch_size_fn(batch: object) -> int:
        return int(_resolve_images(batch).shape[0])

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            dataloader=train_loader,
            step_fn=_train_step,
            step_kwargs={},
            batch_size_fn=_batch_size_fn,
            default_metrics={"loss": 0.0, "mask_ratio": 0.0},
            progress_desc=f"mae train {epoch}/{epochs}",
            show_progress=show_progress,
        )
        val_metrics = run_epoch(
            dataloader=val_loader,
            step_fn=_eval_step,
            step_kwargs={},
            batch_size_fn=_batch_size_fn,
            default_metrics={"loss": 0.0, "mask_ratio": 0.0},
            progress_desc=f"mae val {epoch}/{epochs}",
            show_progress=show_progress,
        )
        append_history(
            history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metric_keys={
                "train_loss": "loss",
                "val_loss": "loss",
                "train_mask_ratio": "mask_ratio",
                "val_mask_ratio": "mask_ratio",
            },
            lr=epoch_lr,
        )
        save_history(history, run_paths.run_dir)

        improved_val_loss = best_val_loss is None or val_metrics["loss"] <= best_val_loss
        if improved_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_checkpoints["best_val_loss"] = run_paths.checkpoint_dir / "mae_best_val_loss.pt"

        if scheduler is not None:
            scheduler.step()

        state = build_checkpoint_state(
            approach="mae",
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_loss=best_val_loss,
            best_checkpoints=best_checkpoints,
            model_states={"model_state_dict": model.state_dict()},
            extra={"model_config": to_serializable(model.config)},
        )
        if epoch % save_every == 0:
            save_checkpoint(
                state=state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="mae",
                epoch=epoch,
                keep_last=checkpoint_keep_last,
            )
        if improved_val_loss:
            save_best_checkpoint(
                state=state,
                checkpoint_dir=run_paths.checkpoint_dir,
                prefix="mae",
                metric_name="val_loss",
            )

    _save_training_curves(history, run_paths.figure_dir)
    _save_reconstruction_preview(model, val_loader, device, run_paths.figure_dir, mask_ratio)
    _save_reconstruction_metrics_and_figures(
        model=model,
        val_loader=val_loader,
        device=device,
        run_paths=run_paths,
        mask_ratio=mask_ratio,
    )
    _save_external_reconstruction_figure(
        model=model,
        device=device,
        figure_dir=run_paths.figure_dir,
        external_image_dir=external_image_dir,
        external_image_size=external_image_size,
        external_mean=external_mean,
        external_std=external_std,
        mask_ratio=mask_ratio,
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
        "resume_from": None if resolved_resume_checkpoint is None else resolved_resume_checkpoint,
        "resumed_from_epoch": resumed_from_epoch,
    }
