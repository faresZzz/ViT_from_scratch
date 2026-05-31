"""Tests for 2D Rotary Position Embeddings (RoPE 2D)."""

from __future__ import annotations

import pytest
import torch

from vit_from_scratch import (
    MaskedAutoencoder,
    ViTConfig,
    VisionTransformer,
    apply_rope,
    build_rope_cache_2d,
)


# ---------------------------------------------------------------------------
# build_rope_cache_2d — shape and dtype
# ---------------------------------------------------------------------------

def test_build_rope_cache_2d_shape():
    """Cache shape is [1, 1, grid_h*grid_w + 1, head_dim] with CLS prepended."""
    cos, sin = build_rope_cache_2d(8, 8, 16)
    assert cos.shape == (1, 1, 65, 16)  # 64 patches + 1 CLS
    assert sin.shape == (1, 1, 65, 16)


def test_build_rope_cache_2d_non_square_grid():
    """Non-square grids are supported."""
    cos, sin = build_rope_cache_2d(4, 8, 16)
    assert cos.shape == (1, 1, 33, 16)  # 32 patches + 1 CLS
    assert sin.shape == (1, 1, 33, 16)


def test_build_rope_cache_2d_cls_identity():
    """CLS token (index 0) must be identity: cos=1, sin=0."""
    cos, sin = build_rope_cache_2d(4, 4, 16)
    assert torch.allclose(cos[0, 0, 0], torch.ones(16))
    assert torch.allclose(sin[0, 0, 0], torch.zeros(16))


def test_build_rope_cache_2d_rejects_odd_head_dim():
    """head_dim not divisible by 4 must raise ValueError."""
    with pytest.raises(ValueError, match="divisible by 4"):
        build_rope_cache_2d(4, 4, 6)  # 6 % 4 != 0


# ---------------------------------------------------------------------------
# apply_rope with 2D cache — isometry
# ---------------------------------------------------------------------------

def test_rope_2d_preserves_norms():
    """Rotation is an isometry: ||x|| == ||apply_rope(x)||."""
    B, H, N, D = 2, 4, 65, 16  # 8x8 grid + CLS
    x = torch.randn(B, H, N, D)
    cos, sin = build_rope_cache_2d(8, 8, D)
    x_rot = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# Relative position property
# ---------------------------------------------------------------------------

def test_rope_2d_relative_position():
    """Attention score depends on (Δrow, Δcol), not absolute positions.

    Patches (2,3)→(4,5) and (1,1)→(3,3) share the same delta Δ=(2,2),
    so their dot-product scores must be identical (for the same q/k vectors).
    """
    D = 16
    cos, sin = build_rope_cache_2d(8, 8, D)

    # Use identical q/k vectors at all positions so only the PE differs
    q_vec = torch.randn(D)
    k_vec = torch.randn(D)
    q = q_vec.expand(1, 1, 65, D).clone()
    k = k_vec.expand(1, 1, 65, D).clone()

    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

    # CLS is at index 0; patch (r, c) → index r*8 + c + 1
    idx_a = 2 * 8 + 3 + 1  # patch (2,3)
    idx_b = 4 * 8 + 5 + 1  # patch (4,5)
    idx_c = 1 * 8 + 1 + 1  # patch (1,1)
    idx_d = 3 * 8 + 3 + 1  # patch (3,3)

    score_ab = (q_rot[0, 0, idx_a] * k_rot[0, 0, idx_b]).sum()
    score_cd = (q_rot[0, 0, idx_c] * k_rot[0, 0, idx_d]).sum()
    assert torch.allclose(score_ab, score_cd, atol=1e-5)


# ---------------------------------------------------------------------------
# ViTConfig validation
# ---------------------------------------------------------------------------

def test_vit_config_rope2d_valid():
    """Config accepts rope2d when head_dim % 4 == 0."""
    cfg = ViTConfig(
        image_size=32, patch_size=8, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
        position_embedding="rope2d",
    )
    assert cfg.position_embedding == "rope2d"


def test_vit_config_rope2d_invalid_head_dim():
    """Config rejects rope2d when head_dim % 4 != 0."""
    with pytest.raises(ValueError, match="divisible by 4"):
        # embed_dim=48, num_heads=4 → head_dim=12, 12 % 4 == 0 — need a bad one
        # embed_dim=48, num_heads=3 → head_dim=16, ok. Use embed_dim=24, num_heads=4 → head_dim=6
        ViTConfig(
            image_size=32, patch_size=8, in_channels=3, num_classes=10,
            embed_dim=24, depth=2, num_heads=4, mlp_ratio=2.0,
            position_embedding="rope2d",  # head_dim = 6, 6 % 4 != 0
        )


# ---------------------------------------------------------------------------
# End-to-end forward passes
# ---------------------------------------------------------------------------

def test_vision_transformer_rope2d_forward():
    """VisionTransformer with rope2d produces logits of the correct shape."""
    cfg = ViTConfig(
        image_size=32, patch_size=8, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
        position_embedding="rope2d",
    )
    model = VisionTransformer(cfg)
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 10)


def test_vision_transformer_rope2d_no_position_embedding_module():
    """VisionTransformer with rope2d must not create a position_embedding module."""
    cfg = ViTConfig(
        image_size=32, patch_size=8, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
        position_embedding="rope2d",
    )
    model = VisionTransformer(cfg)
    assert model.position_embedding is None


def test_mae_rope2d_forward():
    """MaskedAutoencoder with rope2d config produces valid output shape."""
    cfg = ViTConfig(
        image_size=32, patch_size=8, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
        position_embedding="rope2d",
        decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
    )
    mae = MaskedAutoencoder(cfg)
    images = torch.randn(2, 3, 32, 32)
    output = mae(images)
    assert output.shape[0] == 2  # batch size preserved
