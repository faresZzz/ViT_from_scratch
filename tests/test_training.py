import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from vit_from_scratch import STL10_CLASSES, ViTConfig, build_dataloaders, evaluate, train_one_epoch
from vit_from_scratch.data import build_transforms
from vit_from_scratch.model import VisionTransformer
from vit_from_scratch.train import (
    APPROACH_SPECS,
    _print_history,
    build_parser,
    get_approach_spec,
)
from vit_from_scratch.training import train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


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


def _make_model() -> VisionTransformer:
    return VisionTransformer(_make_config())


def test_training_approach_specs_are_method_agnostic():
    assert set(APPROACH_SPECS) == {"classification", "mae", "dino"}
    for approach in APPROACH_SPECS:
        spec = get_approach_spec(approach)
        assert spec.name == approach
        assert spec.checkpoint_prefix == approach


def test_training_approach_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown training approach"):
        get_approach_spec("unknown")


def test_print_history_handles_optional_metrics(capsys):
    _print_history(
        {
            "train_loss": [1.0],
            "knn_accuracy": [None],
            "status": ["skipped"],
        },
        epochs=1,
    )

    output = capsys.readouterr().out
    assert "train_loss: 1.0000" in output
    assert "knn_accuracy: n/a" in output
    assert "status: skipped" in output


def test_build_dataloaders_fake_returns_cifar_sized_batches():
    dataloaders = build_dataloaders(
        dataset="fake",
        batch_size=2,
        max_train_samples=4,
        max_val_samples=4,
        max_test_samples=4,
        num_workers=0,
        download=False,
    )

    train_images, train_labels = next(iter(dataloaders.train_loader))
    val_images, val_labels = next(iter(dataloaders.val_loader))
    test_images, test_labels = next(iter(dataloaders.test_loader))

    assert train_images.shape == (2, 3, 32, 32)
    assert val_images.shape == (2, 3, 32, 32)
    assert test_images.shape == (2, 3, 32, 32)
    assert train_labels.shape == (2,)
    assert val_labels.shape == (2,)
    assert test_labels.shape == (2,)
    assert len(dataloaders.class_names) == 10
    assert dataloaders.test_loader is not None
    assert _base_dataset(dataloaders.train_loader.dataset).random_offset == 0
    assert _base_dataset(dataloaders.val_loader.dataset).random_offset == 0
    assert _base_dataset(dataloaders.test_loader.dataset).random_offset == 1_000_000
    assert set(dataloaders.train_loader.dataset.indices).isdisjoint(
        dataloaders.val_loader.dataset.indices
    )


def test_build_dataloaders_fake_supports_larger_image_size():
    dataloaders = build_dataloaders(
        dataset="fake",
        batch_size=2,
        max_train_samples=4,
        max_val_samples=4,
        num_workers=0,
        download=False,
        image_size=96,
    )

    train_images, _ = next(iter(dataloaders.train_loader))
    val_images, _ = next(iter(dataloaders.val_loader))

    assert train_images.shape == (2, 3, 96, 96)
    assert val_images.shape == (2, 3, 96, 96)


def test_build_dataloaders_tiny_imagenet_uses_train_val_test_policy(tmp_path: Path):
    root = tmp_path / "tiny-imagenet-200"
    for class_name in ("n00000001", "n00000002"):
        image_dir = root / "train" / class_name / "images"
        image_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), color=(index * 20, 50, 100)).save(
                image_dir / f"{class_name}_{index}.JPEG"
            )
    val_images = root / "val" / "images"
    val_images.mkdir(parents=True)
    annotations = []
    for index, class_name in enumerate(("n00000001", "n00000002")):
        image_name = f"val_{index}.JPEG"
        Image.new("RGB", (64, 64), color=(120, index * 30, 50)).save(
            val_images / image_name
        )
        annotations.append(f"{image_name}\t{class_name}\t0\t0\t64\t64\n")
    (root / "val" / "val_annotations.txt").write_text(
        "".join(annotations),
        encoding="utf-8",
    )

    dataloaders = build_dataloaders(
        dataset="tiny-imagenet",
        data_dir=str(tmp_path),
        batch_size=2,
        image_size=64,
        val_fraction=0.25,
        download=False,
    )

    assert dataloaders.test_loader is not None
    assert len(dataloaders.class_names) == 2
    assert next(iter(dataloaders.train_loader))[0].shape == (2, 3, 64, 64)
    assert next(iter(dataloaders.val_loader))[0].shape == (2, 3, 64, 64)
    assert next(iter(dataloaders.test_loader))[0].shape == (2, 3, 64, 64)


def test_build_transforms_train_can_enable_stronger_augmentations():
    transform = build_transforms(
        train=True,
        image_size=32,
        color_jitter_strength=0.2,
        random_erasing_prob=0.15,
        randaugment_num_ops=2,
        randaugment_magnitude=7,
    )

    transform_names = [type(step).__name__ for step in transform.transforms]

    assert "RandomCrop" in transform_names
    assert "RandomHorizontalFlip" in transform_names
    assert "ColorJitter" in transform_names
    assert "RandomErasing" in transform_names
    if hasattr(torchvision.transforms, "RandAugment"):
        assert "RandAugment" in transform_names
    else:
        assert "RandAugment" not in transform_names


@pytest.mark.parametrize(
    ("dataset_name", "dataset_attr", "train_key", "official_test_value", "image_size"),
    [
        ("cifar10", "CIFAR10", "train", False, 32),
        ("stl10", "STL10", "split", "test", 96),
    ],
)
def test_build_dataloaders_real_datasets_split_val_from_train_with_eval_transforms(
    monkeypatch,
    dataset_name: str,
    dataset_attr: str,
    train_key: str,
    official_test_value: object,
    image_size: int,
):
    created_datasets: list[torch.utils.data.Dataset] = []

    class DummyDataset(torch.utils.data.Dataset):
        classes = list(STL10_CLASSES)

        def __init__(self, root, transform, download, **kwargs):
            self.root = root
            self.transform = transform
            self.download = download
            for key, value in kwargs.items():
                setattr(self, key, value)
            created_datasets.append(self)

        def __len__(self):
            return 12

        def __getitem__(self, index):
            image = Image.new("RGB", (image_size, image_size), color=(index * 10, 80, 120))
            if self.transform is not None:
                image = self.transform(image)
            return image, index % len(self.classes)

    monkeypatch.setattr(
        f"vit_from_scratch.data.datasets.{dataset_attr}",
        DummyDataset,
    )

    dataloaders = build_dataloaders(
        dataset=dataset_name,
        batch_size=2,
        max_train_samples=4,
        max_val_samples=2,
        max_test_samples=2,
        val_fraction=0.25,
        num_workers=0,
        download=False,
        image_size=image_size,
    )

    assert len(created_datasets) == 3
    assert getattr(created_datasets[0], train_key) != official_test_value
    assert getattr(created_datasets[1], train_key) != official_test_value
    assert getattr(created_datasets[2], train_key) == official_test_value

    train_transform_names = [
        type(step).__name__ for step in created_datasets[0].transform.transforms
    ]
    eval_transform_names = [
        type(step).__name__ for step in created_datasets[1].transform.transforms
    ]

    assert "RandomHorizontalFlip" in train_transform_names
    assert "RandomHorizontalFlip" not in eval_transform_names
    assert "CenterCrop" in eval_transform_names
    assert "Resize" in eval_transform_names

    assert dataloaders.test_loader is not None
    assert _base_dataset(dataloaders.val_loader.dataset) is created_datasets[1]
    assert _base_dataset(dataloaders.test_loader.dataset) is created_datasets[2]


def test_build_dataloaders_stl10_uses_96px_images_without_download(monkeypatch):
    class DummySTL10(torch.utils.data.Dataset):
        classes = list(STL10_CLASSES)

        def __init__(self, root, split, transform, download):
            self.root = root
            self.split = split
            self.transform = transform
            self.download = download

        def __len__(self):
            return 6

        def __getitem__(self, index):
            image = Image.new("RGB", (96, 96), color=(index * 20, 80, 120))
            if self.transform is not None:
                image = self.transform(image)
            return image, index % len(self.classes)

    monkeypatch.setattr("vit_from_scratch.data.datasets.STL10", DummySTL10)

    dataloaders = build_dataloaders(
        dataset="stl10",
        batch_size=2,
        max_train_samples=4,
        max_val_samples=4,
        num_workers=0,
        download=False,
        image_size=96,
        val_fraction=0.5,
    )

    train_images, train_labels = next(iter(dataloaders.train_loader))
    val_images, val_labels = next(iter(dataloaders.val_loader))

    assert train_images.shape == (2, 3, 96, 96)
    assert val_images.shape == (2, 3, 96, 96)
    assert train_labels.shape == (2,)
    assert val_labels.shape == (2,)
    assert dataloaders.class_names == STL10_CLASSES
    assert dataloaders.test_loader is not None


def test_build_dataloaders_tiny_imagenet_uses_train_val_test_policy(tmp_path: Path):
    root = tmp_path / "tiny-imagenet-200"
    for wnid in ("n00000001", "n00000002"):
        image_dir = root / "train" / wnid / "images"
        image_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), color=(index * 20, 50, 100)).save(
                image_dir / f"{wnid}_{index}.JPEG"
            )

    val_images = root / "val" / "images"
    val_images.mkdir(parents=True)
    annotations: list[str] = []
    for index, wnid in enumerate(("n00000001", "n00000002")):
        image_name = f"val_{index}.JPEG"
        Image.new("RGB", (64, 64), color=(120, index * 30, 50)).save(
            val_images / image_name
        )
        annotations.append(f"{image_name}\t{wnid}\t0\t0\t64\t64\n")
    (root / "val" / "val_annotations.txt").write_text("".join(annotations))

    dataloaders = build_dataloaders(
        dataset="tiny-imagenet",
        data_dir=str(tmp_path),
        batch_size=2,
        image_size=64,
        val_fraction=0.25,
        num_workers=0,
        download=False,
    )

    assert dataloaders.test_loader is not None
    assert len(dataloaders.class_names) == 2
    assert next(iter(dataloaders.train_loader))[0].shape == (2, 3, 64, 64)
    assert next(iter(dataloaders.val_loader))[0].shape == (2, 3, 64, 64)
    assert next(iter(dataloaders.test_loader))[0].shape == (2, 3, 64, 64)


def test_train_parser_accepts_tiny_imagenet_dataset_choice():
    args = build_parser().parse_args(["--dataset", "tiny-imagenet"])

    assert args.dataset == "tiny-imagenet"


def test_train_one_epoch_and_evaluate_return_scalar_metrics():
    dataloaders = build_dataloaders(
        dataset="fake",
        batch_size=2,
        max_train_samples=4,
        max_val_samples=4,
        num_workers=0,
        download=False,
    )
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    model.to(device)

    train_metrics = train_one_epoch(
        model=model,
        dataloader=dataloaders.train_loader,
        optimizer=optimizer,
        device=device,
    )
    val_metrics = evaluate(
        model=model,
        dataloader=dataloaders.val_loader,
        device=device,
    )

    for metrics in (train_metrics, val_metrics):
        assert set(metrics) == {"loss", "accuracy"}
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)


def test_train_one_epoch_returns_consistent_history_lengths():
    dataloaders = build_dataloaders(
        dataset="fake",
        batch_size=2,
        max_train_samples=4,
        max_val_samples=4,
        num_workers=0,
        download=False,
    )
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = train(
        model=model,
        train_loader=dataloaders.train_loader,
        val_loader=dataloaders.val_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
    )

    assert set(history) == {
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }
    for values in history.values():
        assert isinstance(values, list)
        assert len(values) == 1
        assert isinstance(values[0], float)


def test_train_cli_smoke_fake_dataset(tmp_path: Path):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "vit_from_scratch.train",
        "--dataset",
        "fake",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--max-train-samples",
        "4",
        "--max-val-samples",
        "4",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "cli-classification",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Using device: cpu" in result.stdout
    assert "Training approach: classification" in result.stdout
    assert "Epoch 1/1" in result.stdout
    assert (tmp_path / "classification" / "cli-classification" / "history.json").exists()


def test_train_cli_resume_latest_classification(tmp_path: Path):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    executable = str(python_executable if python_executable.exists() else Path(sys.executable))
    base_command = [
        executable,
        "-m",
        "vit_from_scratch.train",
        "--dataset",
        "fake",
        "--batch-size",
        "2",
        "--max-train-samples",
        "4",
        "--max-val-samples",
        "4",
        "--image-size",
        "32",
        "--patch-size",
        "8",
        "--embed-dim",
        "32",
        "--depth",
        "1",
        "--num-heads",
        "4",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "resume-cli",
        "--no-progress",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    first = subprocess.run(
        [*base_command, "--epochs", "2"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr or first.stdout

    resumed = subprocess.run(
        [*base_command, "--epochs", "4", "--resume", "latest"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    run_dir = tmp_path / "classification" / "resume-cli"
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    assert resumed.returncode == 0, resumed.stderr or resumed.stdout
    assert "Resuming from checkpoint:" in resumed.stdout
    assert "Resumed at epoch 3/4 from epoch 2." in resumed.stdout
    assert len(history["train_loss"]) == 4
    assert (run_dir / "checkpoints" / "classification_epoch_0004.pt").exists()
    assert config_payload["resumed_from_epoch"] == 2
    assert config_payload["target_epochs"] == 4


def test_train_cli_rejects_resume_and_restart_together(tmp_path: Path):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "vit_from_scratch.train",
        "--dataset",
        "fake",
        "--epochs",
        "1",
        "--output-dir",
        str(tmp_path),
        "--resume",
        "latest",
        "--restart",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_train_cli_smoke_with_yaml_config(tmp_path: Path):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "vit_from_scratch.train",
        "--config",
        "configs/training/classification_fake.yaml",
        "--epochs",
        "1",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "config-smoke",
        "--no-progress",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Training approach: classification" in result.stdout
    assert "Epoch 1/1" in result.stdout
    run_dir = tmp_path / "classification" / "config-smoke"
    assert (run_dir / "history.json").exists()

    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["dataset"] == "fake"
    assert config_payload["batch_size"] == 4
    assert config_payload["dropout"] == pytest.approx(0.0)
    assert config_payload["attention_dropout"] == pytest.approx(0.0)
    assert config_payload["label_smoothing"] == pytest.approx(0.0)
    assert config_payload["color_jitter_strength"] == pytest.approx(0.0)
    assert config_payload["random_erasing_prob"] == pytest.approx(0.0)
    assert config_payload["randaugment_num_ops"] == 0
    assert config_payload["randaugment_magnitude"] == 9
    assert config_payload["lr"] == pytest.approx(3e-4)


def test_train_cli_resolves_relative_output_dir_from_project_root(tmp_path: Path):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    relative_output_dir = (
        f"tmp/test-relative-runs-{tmp_path.parent.name}-{tmp_path.name}"
    )
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "vit_from_scratch.train",
        "--config",
        "configs/training/classification_fake.yaml",
        "--epochs",
        "1",
        "--output-dir",
        relative_output_dir,
        "--run-name",
        "relative-output",
        "--no-progress",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    expected_history = (
        PROJECT_ROOT
        / relative_output_dir
        / "classification"
        / "relative-output"
        / "history.json"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert expected_history.exists()
    assert not (tmp_path / relative_output_dir).exists()


@pytest.mark.parametrize("approach", ["mae", "dino"])
def test_train_cli_smoke_fake_dataset_for_representation_approaches(
    approach: str,
    tmp_path: Path,
):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "vit_from_scratch.train",
        "--approach",
        approach,
        "--dataset",
        "fake",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--max-train-samples",
        "4",
        "--max-val-samples",
        "4",
        "--image-size",
        "32",
        "--patch-size",
        "8",
        "--embed-dim",
        "32",
        "--depth",
        "1",
        "--num-heads",
        "4",
        "--dropout",
        "0.2",
        "--attention-dropout",
        "0.15",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        f"cli-{approach}",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"Training approach: {approach}" in result.stdout
    assert "Epoch 1/1" in result.stdout
    run_dir = tmp_path / approach / f"cli-{approach}"
    assert (run_dir / "history.json").exists()

    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["dropout"] == pytest.approx(0.2)
    assert config_payload["attention_dropout"] == pytest.approx(0.15)
    assert config_payload["model_config"]["dropout"] == pytest.approx(0.2)
    assert config_payload["model_config"]["attention_dropout"] == pytest.approx(0.15)


@pytest.mark.parametrize("approach", ["mae", "dino"])
def test_train_cli_resume_latest_for_representation_approaches(
    approach: str,
    tmp_path: Path,
):
    python_executable = PROJECT_ROOT / ".venv" / "bin" / "python"
    executable = str(python_executable if python_executable.exists() else Path(sys.executable))
    base_command = [
        executable,
        "-m",
        "vit_from_scratch.train",
        "--approach",
        approach,
        "--dataset",
        "fake",
        "--batch-size",
        "2",
        "--max-train-samples",
        "4",
        "--max-val-samples",
        "4",
        "--image-size",
        "32",
        "--patch-size",
        "8",
        "--embed-dim",
        "32",
        "--depth",
        "1",
        "--num-heads",
        "4",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        f"resume-{approach}",
        "--no-progress",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    first = subprocess.run(
        [*base_command, "--epochs", "1"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr or first.stdout

    resumed = subprocess.run(
        [*base_command, "--epochs", "2", "--resume", "latest"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    run_dir = tmp_path / approach / f"resume-{approach}"
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    assert resumed.returncode == 0, resumed.stderr or resumed.stdout
    assert "Resuming from checkpoint:" in resumed.stdout
    assert len(history["train_loss"]) == 2
    assert (run_dir / "checkpoints" / f"{approach}_epoch_0002.pt").exists()
