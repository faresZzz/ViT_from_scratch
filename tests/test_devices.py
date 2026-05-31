import pytest

torch = pytest.importorskip("torch")

from vit_from_scratch import ViTConfig, get_device
from vit_from_scratch.model import VisionTransformer


def _make_config() -> ViTConfig:
    return ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def _forward_backward_on_device(device: torch.device) -> None:
    model = VisionTransformer(_make_config()).to(device)
    images = torch.randn(2, 3, 32, 32, device=device)
    targets = torch.tensor([0, 1], device=device)
    criterion = torch.nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()

    assert logits.device.type == device.type
    assert loss.device.type == device.type
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in grads)


def test_cpu_forward_backward_runs_on_cpu():
    _forward_backward_on_device(torch.device("cpu"))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this machine.",
)
def test_mps_forward_backward_runs_on_mps():
    _forward_backward_on_device(torch.device("mps"))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable on this machine.",
)
def test_cuda_forward_backward_runs_on_cuda():
    _forward_backward_on_device(torch.device("cuda"))


def test_get_device_auto_prefers_actual_available_backend():
    device = get_device("auto")

    if torch.backends.mps.is_available():
        assert device.type == "mps"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
