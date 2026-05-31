import inspect

import pytest

torch = pytest.importorskip("torch")

import vit_from_scratch as vit


def _get_public(name):
    assert hasattr(vit, name), f"Missing public export at package root: {name}"
    return getattr(vit, name)


def _instantiate_position_embedding(cls, *, num_tokens, embed_dim):
    signature = inspect.signature(cls)
    kwargs = {}
    for parameter_name in signature.parameters:
        if parameter_name == "self":
            continue
        if parameter_name in {"num_tokens", "num_positions", "seq_len", "length"}:
            kwargs[parameter_name] = num_tokens
        elif parameter_name in {"embed_dim", "dim", "embedding_dim"}:
            kwargs[parameter_name] = embed_dim
        elif parameter_name == "dropout":
            kwargs[parameter_name] = 0.0
    return cls(**kwargs)


def _build_rope_cache(cache_builder, *, seq_len, head_dim, device, dtype):
    signature = inspect.signature(cache_builder)
    kwargs = {}
    for parameter_name in signature.parameters:
        if parameter_name in {"seq_len", "num_tokens", "num_positions", "length"}:
            kwargs[parameter_name] = seq_len
        elif parameter_name in {"head_dim", "dim", "embedding_dim"}:
            kwargs[parameter_name] = head_dim
        elif parameter_name == "device":
            kwargs[parameter_name] = device
        elif parameter_name == "dtype":
            kwargs[parameter_name] = dtype
    return cache_builder(**kwargs)


def _apply_rope(apply_rope, tensor, cache):
    signature = inspect.signature(apply_rope)
    parameter_names = [
        name for name in signature.parameters if name not in {"self", "cls"}
    ]
    if len(parameter_names) == 2:
        return apply_rope(tensor, cache)
    if isinstance(cache, dict):
        if len(parameter_names) == 3 and {"cos", "sin"} <= set(cache):
            return apply_rope(tensor, cache["cos"], cache["sin"])
    if isinstance(cache, (tuple, list)):
        return apply_rope(tensor, *cache)
    return apply_rope(tensor, cache)


def _make_config(**overrides):
    ViTConfig = _get_public("ViTConfig")
    config = dict(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=4.0,
        depth=2,
        num_classes=10,
        dropout=0.0,
        attention_dropout=0.0,
    )
    config.update(overrides)
    return ViTConfig(**config)


def test_learned_position_embedding_preserves_shape_and_changes_zero_tokens():
    LearnedPositionEmbedding = _get_public("LearnedPositionEmbedding")
    num_tokens = 17
    embed_dim = 64
    layer = _instantiate_position_embedding(
        LearnedPositionEmbedding,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
    )
    tokens = torch.zeros(2, num_tokens, embed_dim)

    outputs = layer(tokens)

    assert outputs.shape == tokens.shape
    assert not torch.allclose(outputs, tokens)


def test_cosine_position_embedding_is_parameter_free_and_deterministic():
    CosinePositionEmbedding = _get_public("CosinePositionEmbedding")
    num_tokens = 17
    embed_dim = 64
    layer = _instantiate_position_embedding(
        CosinePositionEmbedding,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
    )
    tokens = torch.zeros(2, num_tokens, embed_dim)

    outputs_first = layer(tokens)
    outputs_second = layer(tokens)

    assert outputs_first.shape == tokens.shape
    assert sum(parameter.numel() for parameter in layer.parameters()) == 0
    assert torch.allclose(outputs_first, outputs_second)
    assert outputs_first.abs().sum().item() > 0.0


def test_rope_cache_and_apply_rope_preserve_shape_and_pair_norm():
    build_rope_cache = _get_public("build_rope_cache")
    apply_rope = _get_public("apply_rope")
    tensor = torch.randn(2, 4, 17, 16)
    cache = _build_rope_cache(
        build_rope_cache,
        seq_len=tensor.shape[2],
        head_dim=tensor.shape[3],
        device=tensor.device,
        dtype=tensor.dtype,
    )

    outputs = _apply_rope(apply_rope, tensor, cache)

    assert outputs.shape == tensor.shape
    inputs_pairs = tensor.reshape(*tensor.shape[:-1], -1, 2)
    outputs_pairs = outputs.reshape(*outputs.shape[:-1], -1, 2)
    input_norms = torch.linalg.norm(inputs_pairs, dim=-1)
    output_norms = torch.linalg.norm(outputs_pairs, dim=-1)
    assert torch.allclose(output_norms, input_norms, atol=1e-5, rtol=1e-4)


def test_multi_head_self_attention_with_rope_preserves_shape():
    MultiHeadSelfAttention = _get_public("MultiHeadSelfAttention")
    layer = MultiHeadSelfAttention(
        embed_dim=64,
        num_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        use_rope=True,
    )
    tokens = torch.randn(2, 17, 64)

    outputs = layer(tokens)

    assert outputs.shape == tokens.shape


@pytest.mark.parametrize("position_embedding", ["learned", "cosine", "rope"])
def test_vision_transformer_supports_position_embedding_modes(position_embedding):
    VisionTransformer = _get_public("VisionTransformer")
    config = _make_config(position_embedding=position_embedding)
    model = VisionTransformer(config)
    images = torch.randn(
        2,
        config.in_channels,
        config.image_size,
        config.image_size,
    )

    logits = model(images)

    assert logits.shape == (2, config.num_classes)


def test_vit_config_rejects_unknown_position_embedding():
    with pytest.raises(ValueError):
        _make_config(position_embedding="mystery")


def test_vit_config_rejects_rope_with_odd_head_dim():
    with pytest.raises(ValueError):
        _make_config(
            embed_dim=18,
            num_heads=6,
            position_embedding="rope",
        )
