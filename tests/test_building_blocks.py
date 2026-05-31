import pytest
torch = pytest.importorskip("torch")

from vit_from_scratch import (
    MLP,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerEncoderBlock,
    ViTConfig,
    VisionTransformer,
)


@pytest.fixture
def vit_config():
    return ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=4.0,
        depth=2,
        num_classes=10,
    )


@pytest.fixture
def images(vit_config):
    return torch.randn(2, vit_config.in_channels, vit_config.image_size, vit_config.image_size)


@pytest.fixture
def tokens(vit_config):
    num_patches = (vit_config.image_size // vit_config.patch_size) ** 2
    return torch.randn(2, num_patches, vit_config.embed_dim)


def test_patch_embedding_returns_expected_shape(vit_config, images):
    layer = PatchEmbedding(vit_config)

    outputs = layer(images)

    expected_num_patches = (vit_config.image_size // vit_config.patch_size) ** 2
    assert outputs.shape == (2, expected_num_patches, vit_config.embed_dim)


def test_patch_embedding_rejects_mismatched_image_size(vit_config):
    layer = PatchEmbedding(vit_config)
    wrong_size_images = torch.randn(
        2,
        vit_config.in_channels,
        vit_config.image_size + vit_config.patch_size,
        vit_config.image_size,
    )

    with pytest.raises((ValueError, AssertionError)):
        layer(wrong_size_images)


def test_patch_embedding_rejects_non_divisible_config():
    with pytest.raises((ValueError, AssertionError)):
        ViTConfig(
            image_size=30,
            patch_size=8,
            in_channels=3,
            embed_dim=64,
            num_heads=4,
            mlp_ratio=4.0,
            depth=2,
            num_classes=10,
        )


def test_multi_head_self_attention_preserves_shape(vit_config, tokens):
    layer = MultiHeadSelfAttention(
        embed_dim=vit_config.embed_dim,
        num_heads=vit_config.num_heads,
        dropout=vit_config.dropout,
        attention_dropout=vit_config.attention_dropout,
    )

    outputs = layer(tokens)

    assert outputs.shape == tokens.shape


def test_mlp_preserves_shape(vit_config, tokens):
    layer = MLP(
        embed_dim=vit_config.embed_dim,
        hidden_dim=vit_config.mlp_hidden_dim,
        dropout=vit_config.dropout,
    )

    outputs = layer(tokens)

    assert outputs.shape == tokens.shape


def test_transformer_encoder_block_preserves_shape(vit_config, tokens):
    block = TransformerEncoderBlock(
        embed_dim=vit_config.embed_dim,
        num_heads=vit_config.num_heads,
        mlp_hidden_dim=vit_config.mlp_hidden_dim,
        dropout=vit_config.dropout,
        attention_dropout=vit_config.attention_dropout,
    )

    outputs = block(tokens)

    assert outputs.shape == tokens.shape


def test_vision_transformer_returns_logits(vit_config, images):
    model = VisionTransformer(vit_config)

    outputs = model(images)

    assert outputs.shape == (2, vit_config.num_classes)
