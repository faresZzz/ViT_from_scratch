"""Dataset and dataloader helpers for image experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

STL10_CLASSES: tuple[str, ...] = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

STL10_MEAN: tuple[float, float, float] = (0.4467, 0.4398, 0.4066)
STL10_STD: tuple[float, float, float] = (0.2241, 0.2215, 0.2239)

TINY_IMAGENET_MEAN: tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD: tuple[float, float, float] = (0.2302, 0.2265, 0.2262)

_DEFAULT_CLASS_NAMES: tuple[str, ...] = tuple(str(index) for index in range(10))


@dataclass(frozen=True)
class DataLoaders:
    """Simple dataloader bundle for training scripts."""

    train_loader: DataLoader
    val_loader: DataLoader
    class_names: tuple[str, ...]
    test_loader: DataLoader | None = None


class TinyImageNetValDataset(Dataset):
    """Dataset wrapper for Tiny ImageNet's annotated validation split."""

    def __init__(
        self,
        root: str | Path,
        transform: object | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        annotation_path = self.root / "val" / "val_annotations.txt"
        image_dir = self.root / "val" / "images"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Missing Tiny ImageNet annotations: {annotation_path}"
            )
        if class_to_idx is None:
            train_root = self.root / "train"
            class_to_idx = {
                path.name: index
                for index, path in enumerate(sorted(train_root.iterdir()))
                if path.is_dir()
            }
        self.class_to_idx = dict(class_to_idx)
        self.classes = tuple(
            class_name
            for class_name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        )
        self.samples: list[tuple[Path, int]] = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            image_name, class_name, *_ = line.split("\t")
            if class_name in self.class_to_idx:
                self.samples.append((image_dir / image_name, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(
    train: bool = True,
    image_size: int = 32,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    color_jitter_strength: float = 0.0,
    random_erasing_prob: float = 0.0,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
) -> transforms.Compose:
    """Build image transforms for CIFAR/STL/FakeData experiments."""

    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if color_jitter_strength < 0.0:
        raise ValueError("color_jitter_strength must be greater than or equal to 0.0.")
    if not 0.0 <= random_erasing_prob <= 1.0:
        raise ValueError("random_erasing_prob must be between 0.0 and 1.0.")
    if randaugment_num_ops < 0:
        raise ValueError("randaugment_num_ops must be greater than or equal to 0.")
    if randaugment_magnitude < 0:
        raise ValueError("randaugment_magnitude must be greater than or equal to 0.")

    transform_steps: list[object] = []
    if train:
        if image_size == 32:
            transform_steps.append(transforms.RandomCrop(32, padding=4))
        else:
            transform_steps.append(
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    antialias=True,
                )
            )
        transform_steps.append(transforms.RandomHorizontalFlip())
        if randaugment_num_ops > 0 and hasattr(transforms, "RandAugment"):
            transform_steps.append(
                transforms.RandAugment(
                    num_ops=randaugment_num_ops,
                    magnitude=randaugment_magnitude,
                )
            )
        if color_jitter_strength > 0.0:
            transform_steps.append(
                transforms.ColorJitter(
                    brightness=color_jitter_strength,
                    contrast=color_jitter_strength,
                    saturation=color_jitter_strength,
                    hue=min(0.5, color_jitter_strength * 0.5),
                )
            )
    else:
        transform_steps.extend(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
    if train and random_erasing_prob > 0.0:
        transform_steps.append(transforms.RandomErasing(p=random_erasing_prob))
    return transforms.Compose(transform_steps)


def limit_dataset(
    dataset: Dataset,
    max_samples: int | None,
    seed: int,
) -> Dataset:
    """Return a deterministic subset when max_samples is provided."""

    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer when provided.")
    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def split_dataset_indices(
    dataset_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/validation indices for a dataset length."""

    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in the interval (0.0, 1.0).")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = max(1, int(round(dataset_size * val_fraction)))
    val_size = min(val_size, dataset_size - 1)
    return indices[val_size:], indices[:val_size]


def _resolve_class_names(dataset_name: str, dataset: Dataset) -> tuple[str, ...]:
    if dataset_name == "cifar10":
        return CIFAR10_CLASSES
    if dataset_name == "stl10":
        return STL10_CLASSES
    if dataset_name == "tiny-imagenet":
        classes = getattr(dataset, "classes", None)
        while classes is None and isinstance(dataset, Subset):
            dataset = dataset.dataset
            classes = getattr(dataset, "classes", None)
        if classes is not None:
            return tuple(str(name) for name in classes)

    classes = getattr(dataset, "classes", None)
    if classes is None and isinstance(dataset, Subset):
        classes = getattr(dataset.dataset, "classes", None)
    if classes is None:
        return _DEFAULT_CLASS_NAMES
    return tuple(str(name) for name in classes)


def _dataset_normalization_stats(
    dataset_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if dataset_name == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if dataset_name == "stl10":
        return STL10_MEAN, STL10_STD
    if dataset_name == "tiny-imagenet":
        return TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
    return CIFAR10_MEAN, CIFAR10_STD


def _build_dataset_transform(
    dataset_name: str,
    *,
    train: bool,
    image_size: int,
    color_jitter_strength: float = 0.0,
    random_erasing_prob: float = 0.0,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
) -> transforms.Compose:
    mean, std = _dataset_normalization_stats(dataset_name)
    return build_transforms(
        train=train,
        image_size=image_size,
        mean=mean,
        std=std,
        color_jitter_strength=color_jitter_strength,
        random_erasing_prob=random_erasing_prob,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
    )


def _build_cifar10_dataset(
    *,
    train: bool,
    data_dir: str,
    download: bool,
    transform: transforms.Compose,
) -> Dataset:
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=transform,
        download=download,
    )


def _build_stl10_dataset(
    *,
    train: bool,
    data_dir: str,
    download: bool,
    transform: transforms.Compose,
    use_unlabeled: bool = False,
) -> Dataset:
    if train and use_unlabeled:
        split = "train+unlabeled"
    elif train:
        split = "train"
    else:
        split = "test"
    return datasets.STL10(
        root=data_dir,
        split=split,
        transform=transform,
        download=download,
    )


def _build_tiny_imagenet_dataset(
    *,
    train: bool,
    data_dir: str,
    transform: transforms.Compose,
) -> Dataset:
    root = Path(data_dir) / "tiny-imagenet-200"
    if train:
        return datasets.ImageFolder(root / "train", transform=transform)

    train_dataset = datasets.ImageFolder(root / "train")
    return TinyImageNetValDataset(
        root=root,
        transform=transform,
        class_to_idx=train_dataset.class_to_idx,
    )


def _build_fake_dataset(
    *,
    train: bool,
    image_size: int,
    transform: transforms.Compose,
) -> Dataset:
    return datasets.FakeData(
        size=50_000 if train else 10_000,
        image_size=(3, image_size, image_size),
        num_classes=10,
        transform=transform,
        random_offset=0 if train else 1_000_000,
    )


def _build_dataset(
    dataset_name: str,
    *,
    train: bool,
    transform_train: bool | None = None,
    data_dir: str,
    download: bool,
    image_size: int,
    color_jitter_strength: float = 0.0,
    random_erasing_prob: float = 0.0,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
    use_unlabeled: bool = False,
) -> Dataset:
    use_train_transform = train if transform_train is None else transform_train
    transform = _build_dataset_transform(
        dataset_name,
        train=use_train_transform,
        image_size=image_size,
        color_jitter_strength=color_jitter_strength,
        random_erasing_prob=random_erasing_prob,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
    )
    if dataset_name == "cifar10":
        return _build_cifar10_dataset(
            train=train,
            data_dir=data_dir,
            download=download,
            transform=transform,
        )
    if dataset_name == "stl10":
        return _build_stl10_dataset(
            train=train,
            data_dir=data_dir,
            download=download,
            transform=transform,
            use_unlabeled=use_unlabeled,
        )
    if dataset_name == "tiny-imagenet":
        return _build_tiny_imagenet_dataset(
            train=train,
            data_dir=data_dir,
            transform=transform,
        )
    if dataset_name == "fake":
        return _build_fake_dataset(
            train=train,
            image_size=image_size,
            transform=transform,
        )
    raise ValueError(
        "dataset must be one of {'cifar10', 'stl10', 'tiny-imagenet', 'fake'}: "
        f"got {dataset_name!r}."
    )


def build_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
    download: bool = True,
    image_size: int = 32,
    val_fraction: float = 0.1,
    split_train_validation: bool = True,
    max_test_samples: int | None = None,
    color_jitter_strength: float = 0.0,
    random_erasing_prob: float = 0.0,
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 9,
    use_unlabeled: bool = False,
) -> DataLoaders:
    """Build training and validation dataloaders for image tasks."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0.")
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in the interval (0.0, 1.0).")

    dataset_name = dataset.lower()
    train_dataset = _build_dataset(
        dataset_name,
        train=True,
        data_dir=data_dir,
        download=download,
        image_size=image_size,
        color_jitter_strength=color_jitter_strength,
        random_erasing_prob=random_erasing_prob,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        use_unlabeled=use_unlabeled,
    )
    train_eval_dataset = _build_dataset(
        dataset_name,
        train=True,
        transform_train=False,
        data_dir=data_dir,
        download=download,
        image_size=image_size,
        use_unlabeled=use_unlabeled,
    )
    official_test_dataset = _build_dataset(
        dataset_name,
        train=False,
        data_dir=data_dir,
        download=download,
        image_size=image_size,
    )

    if split_train_validation:
        train_indices, val_indices = split_dataset_indices(
            len(train_dataset),
            val_fraction=val_fraction,
            seed=seed,
        )
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset: Dataset = Subset(train_eval_dataset, val_indices)
    else:
        val_dataset = official_test_dataset

    train_dataset = limit_dataset(train_dataset, max_train_samples, seed=seed + 1)
    val_dataset = limit_dataset(val_dataset, max_val_samples, seed=seed + 2)
    test_dataset = limit_dataset(official_test_dataset, max_test_samples, seed=seed + 3)
    class_names = _resolve_class_names(dataset_name, train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        test_loader=test_loader,
    )
