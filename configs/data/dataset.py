"""Dataset management for machine unlearning experiments.

This module provides:
1) A unified `UnlearningDataset` class with D_all / D_u / D_r / test splits.
2) Common vision dataset loaders (MNIST/SVHN/CIFAR10/CIFAR100/STL10).
3) Optional dataloader builders for train/unlearn/verify stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "stl10": 10,
}


@dataclass
class DatasetBuildResult:
    """Structured output for dataset construction."""

    train_set: Dataset
    test_set: Dataset
    num_classes: int


class UnlearningDataset:
    """
    Manage dataset splits for machine unlearning experiments.

    The dataset is divided into:
    - D_all: all train samples
    - D_u: forget set (to be unlearned)
    - D_r: retained set
    """

    def __init__(self, config: Dict):
        """
        Initialize dataset manager from config.

        Expected config keys (all optional):
        - dataset: str, one of mnist/svhn/cifar10/cifar100/stl10
        - data_path: str
        - allow_download: bool, whether to download when local data is missing
        - batch_size: int
        - num_workers: int
        - pin_memory: bool
        - augmentations: bool
        - normalize: bool
        - forget_ratio: float in [0, 1]
        - forget_count: int
        - split_seed: int
        """
        self.config = config or {}
        self.dataset_name = str(self.config.get("dataset", "cifar10")).lower()
        self.data_path = os.path.expanduser(self.config.get("data_path", "datasets"))
        self.allow_download = bool(self.config.get("allow_download", False))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.num_workers = int(self.config.get("num_workers", 0))
        self.pin_memory = bool(self.config.get("pin_memory", False))
        self.augmentations = bool(self.config.get("augmentations", True))
        self.normalize = bool(self.config.get("normalize", True))
        self.forget_ratio = self.config.get("forget_ratio", 0.1)
        self.forget_count = self.config.get("forget_count", None)
        self.split_seed = int(self.config.get("split_seed", 42))

        build_result = _build_dataset(
            dataset_name=self.dataset_name,
            data_path=self.data_path,
            augmentations=self.augmentations,
            normalize=self.normalize,
            allow_download=self.allow_download,
        )
        self.d_all = build_result.train_set
        self.d_test = build_result.test_set
        self.num_classes = build_result.num_classes

        self._du_indices, self._dr_indices = self._make_unlearning_split(len(self.d_all))
        self.d_u = Subset(self.d_all, self._du_indices)
        self.d_r = Subset(self.d_all, self._dr_indices)

    def _make_unlearning_split(self, total_size: int) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Create deterministic D_u and D_r index splits.

        Args:
            total_size: Number of samples in D_all.

        Returns:
            Tuple of (du_indices, dr_indices).
        """
        if total_size <= 0:
            return [], []

        if self.forget_count is not None:
            du_size = int(self.forget_count)
        else:
            du_size = int(float(self.forget_ratio) * total_size)

        du_size = max(1, min(du_size, total_size - 1))

        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        perm = torch.randperm(total_size, generator=generator).tolist()

        du_indices = perm[:du_size]
        dr_indices = perm[du_size:]
        return du_indices, dr_indices

    def get_all_set(self) -> Dataset:
        """
        Return D_all.

        Returns:
            Training dataset (full set).
        """
        return self.d_all

    def get_unlearning_set(self) -> Dataset:
        """
        Return D_u.

        Returns:
            Forget subset.
        """
        return self.d_u

    def get_retained_set(self) -> Dataset:
        """
        Return D_r.

        Returns:
            Retained subset.
        """
        return self.d_r

    def get_test_set(self) -> Dataset:
        """
        Return test split.

        Returns:
            Test dataset.
        """
        return self.d_test

    def get_dataloaders(self, retained_shuffle: bool = True) -> Dict[str, DataLoader]:
        """
        Build dataloaders for common pipeline stages.

        Args:
            retained_shuffle: Whether to shuffle retained training data.

        Returns:
            Dict with keys: d_all, d_u, d_r, d_test.
        """
        return {
            "d_all": _build_loader(
                self.d_all,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
            "d_u": _build_loader(
                self.d_u,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
            "d_r": _build_loader(
                self.d_r,
                batch_size=self.batch_size,
                shuffle=retained_shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
            "d_test": _build_loader(
                self.d_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
        }


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Create a torch dataloader from a dataset.

    Args:
        dataset: Input dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of loader workers.
        pin_memory: Whether to pin memory.

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _build_dataset(
    dataset_name: str,
    data_path: str,
    augmentations: bool,
    normalize: bool,
    allow_download: bool,
) -> DatasetBuildResult:
    """
    Build train/test datasets for a supported dataset name.

    Args:
        dataset_name: Dataset identifier.
        data_path: Root path for data.
        augmentations: Whether to apply train augmentation.
        normalize: Whether to normalize inputs.
        allow_download: Whether to download from remote if local data is missing.

    Returns:
        DatasetBuildResult containing train/test sets and num_classes.
    """
    name = dataset_name.lower()
    if name not in NUM_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _wrap_ctor(build_fn, split_desc: str):
        try:
            return build_fn()
        except RuntimeError as exc:
            if allow_download:
                raise
            raise RuntimeError(
                f"Local dataset not found for {name} ({split_desc}) under '{data_path}'. "
                "Please place dataset files in this path or set allow_download=true."
            ) from exc

    if name == "mnist":
        train_transform, test_transform = _build_mnist_transforms(augmentations, normalize)
        train_set = _wrap_ctor(lambda: torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=allow_download,
            transform=train_transform,
        ), "train")
        test_set = _wrap_ctor(lambda: torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=allow_download,
            transform=test_transform,
        ), "test")
    elif name == "svhn":
        train_transform, test_transform = _build_rgb_transforms(name, augmentations, normalize, image_size=32)
        train_set = _wrap_ctor(lambda: torchvision.datasets.SVHN(
            root=data_path,
            split="train",
            download=allow_download,
            transform=train_transform,
        ), "train")
        test_set = _wrap_ctor(lambda: torchvision.datasets.SVHN(
            root=data_path,
            split="test",
            download=allow_download,
            transform=test_transform,
        ), "test")
    elif name == "cifar10":
        train_transform, test_transform = _build_rgb_transforms(name, augmentations, normalize, image_size=32)
        train_set = _wrap_ctor(lambda: torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=allow_download,
            transform=train_transform,
        ), "train")
        test_set = _wrap_ctor(lambda: torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=allow_download,
            transform=test_transform,
        ), "test")
    elif name == "cifar100":
        train_transform, test_transform = _build_rgb_transforms(name, augmentations, normalize, image_size=32)
        train_set = _wrap_ctor(lambda: torchvision.datasets.CIFAR100(
            root=data_path,
            train=True,
            download=allow_download,
            transform=train_transform,
        ), "train")
        test_set = _wrap_ctor(lambda: torchvision.datasets.CIFAR100(
            root=data_path,
            train=False,
            download=allow_download,
            transform=test_transform,
        ), "test")
    else:
        train_transform, test_transform = _build_rgb_transforms(name, augmentations, normalize, image_size=96)
        train_set = _wrap_ctor(lambda: torchvision.datasets.STL10(
            root=data_path,
            split="train",
            download=allow_download,
            transform=train_transform,
        ), "train")
        test_set = _wrap_ctor(lambda: torchvision.datasets.STL10(
            root=data_path,
            split="test",
            download=allow_download,
            transform=test_transform,
        ), "test")

    return DatasetBuildResult(train_set=train_set, test_set=test_set, num_classes=NUM_CLASSES[name])


def _build_mnist_transforms(augmentations: bool, normalize: bool):
    """
    Build train/test transforms for MNIST.

    Args:
        augmentations: Whether to apply train augmentations.
        normalize: Whether to normalize.

    Returns:
        Tuple(train_transform, test_transform).
    """
    mean, std = DATASET_STATS["mnist"]

    train_ops = [transforms.ToTensor()]
    if augmentations:
        train_ops = [
            transforms.RandomAffine(15, translate=(0.1, 0.1)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    if normalize:
        train_ops.append(transforms.Normalize(mean, std))

    test_ops = [transforms.ToTensor()]
    if normalize:
        test_ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def _build_rgb_transforms(
    dataset_name: str,
    augmentations: bool,
    normalize: bool,
    image_size: int,
):
    """
    Build train/test transforms for RGB datasets.

    Args:
        dataset_name: Dataset identifier.
        augmentations: Whether to apply train augmentations.
        normalize: Whether to normalize.
        image_size: Input image size.

    Returns:
        Tuple(train_transform, test_transform).
    """
    mean, std = DATASET_STATS[dataset_name]
    train_ops = []
    if augmentations:
        train_ops.extend(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_ops.append(transforms.ToTensor())
    if normalize:
        train_ops.append(transforms.Normalize(mean, std))

    test_ops = [transforms.ToTensor()]
    if normalize:
        test_ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(train_ops), transforms.Compose(test_ops)
