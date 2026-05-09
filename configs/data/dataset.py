"""Dataset management for machine unlearning experiments.

This module provides:
1) A unified `UnlearningDataset` class with D_all / D_u / D_r / test splits.
2) Common vision dataset loaders (MNIST/SVHN/CIFAR10/CIFAR100/STL10).
3) Optional dataloader builders for train/unlearn/verify stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json
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
        - split_mode: str, `random` or `by_class`
        - forget_classes: list[int], used when split_mode is `by_class`
        - forget_manifest_path: str, path to persisted D_u/D_r indices
        - forget_manifest_mode: str, one of `auto`, `load`, `save`, `off`
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
        self.split_mode = str(self.config.get("split_mode", "random")).lower()
        self.forget_classes = self._normalize_forget_classes(self.config.get("forget_classes", []))
        self.forget_ratio = self.config.get("forget_ratio", 0.1)
        self.forget_count = self.config.get("forget_count", None)
        self.split_seed = int(self.config.get("split_seed", 42))
        self.forget_manifest_mode = str(self.config.get("forget_manifest_mode", "auto")).lower()
        self.forget_manifest_path = self._resolve_manifest_path(self.config.get("forget_manifest_path", None))
        self._forget_manifest_info = None

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

        self._du_indices, self._dr_indices = self._make_unlearning_split(self.d_all)
        self.d_u = Subset(self.d_all, self._du_indices)
        self.d_r = Subset(self.d_all, self._dr_indices)
        print(
            f"[Dataset] split_mode={self.split_mode}, "
            f"forget_classes={self.forget_classes if self.split_mode == 'by_class' else 'N/A'}, "
            f"D_u={len(self.d_u)}, D_r={len(self.d_r)}"
        )

    def _resolve_manifest_path(self, path_value) -> Path:
        """
        Resolve manifest path from config or generate a default one.

        Args:
            path_value: Optional user-provided manifest path.

        Returns:
            Resolved manifest path.
        """
        if path_value:
            return Path(os.path.expanduser(str(path_value)))

        default_dir = Path("save/manifests")
        default_name = f"forget_{self.dataset_name}_{self.split_mode}.json"
        return default_dir / default_name

    def _build_manifest_payload(self, du_indices: Sequence[int], dr_indices: Sequence[int], total_size: int) -> Dict:
        """
        Build manifest payload for persistence.

        Args:
            du_indices: Forget-set indices.
            dr_indices: Retained-set indices.
            total_size: Total dataset size.

        Returns:
            JSON-serializable manifest payload.
        """
        return {
            "dataset": self.dataset_name,
            "split_mode": self.split_mode,
            "total_size": int(total_size),
            "split_seed": int(self.split_seed),
            "forget_ratio": None if self.forget_count is not None else float(self.forget_ratio),
            "forget_count": None if self.forget_count is None else int(self.forget_count),
            "forget_classes": [int(x) for x in self.forget_classes],
            "du_indices": [int(x) for x in du_indices],
            "dr_indices": [int(x) for x in dr_indices],
        }

    def _save_manifest(self, payload: Dict) -> None:
        """
        Save manifest payload to disk.

        Args:
            payload: Manifest dictionary.
        """
        manifest_path = self.forget_manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._forget_manifest_info = payload
        print(f"[Dataset] Saved forget manifest: {manifest_path}")

    def _load_manifest(self) -> Dict:
        """
        Load manifest payload from disk.

        Returns:
            Manifest dictionary.
        """
        manifest_path = self.forget_manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Forget manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._forget_manifest_info = payload
        print(f"[Dataset] Loaded forget manifest: {manifest_path}")
        return payload

    def _validate_manifest(self, payload: Dict, total_size: int) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Validate and extract indices from a manifest.

        Args:
            payload: Manifest payload.
            total_size: Expected dataset size.

        Returns:
            Tuple of (du_indices, dr_indices).
        """
        if payload.get("dataset") != self.dataset_name:
            raise ValueError(
                f"Manifest dataset mismatch: expected {self.dataset_name}, got {payload.get('dataset')}"
            )
        if int(payload.get("total_size", -1)) != int(total_size):
            raise ValueError(
                f"Manifest total_size mismatch: expected {total_size}, got {payload.get('total_size')}"
            )

        du_indices = [int(x) for x in payload.get("du_indices", [])]
        dr_indices = [int(x) for x in payload.get("dr_indices", [])]
        if not du_indices or not dr_indices:
            raise ValueError("Manifest contains empty D_u or D_r.")
        if len(set(du_indices).intersection(set(dr_indices))) > 0:
            raise ValueError("Manifest contains overlapping indices between D_u and D_r.")
        if len(du_indices) + len(dr_indices) != total_size:
            raise ValueError(
                "Manifest index count mismatch: len(D_u)+len(D_r) must equal len(D_all)."
            )
        return du_indices, dr_indices

    def _normalize_forget_classes(self, forget_classes) -> Sequence[int]:
        """
        Normalize class list from config.

        Args:
            forget_classes: Raw class config input.

        Returns:
            Sorted unique class ids.
        """
        if forget_classes is None:
            return []
        if isinstance(forget_classes, (int, float, str)):
            forget_classes = [forget_classes]
        normalized = sorted({int(x) for x in forget_classes})
        return normalized

    def _extract_label(self, dataset: Dataset, index: int) -> int:
        """
        Extract sample label from a dataset by index.

        Args:
            dataset: Source dataset.
            index: Sample index.

        Returns:
            Integer class label.
        """
        _, target = dataset[index]
        if isinstance(target, torch.Tensor):
            return int(target.item())
        return int(target)

    def _make_unlearning_split(self, dataset: Dataset) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Create deterministic D_u and D_r index splits.

        Args:
            dataset: Full training dataset D_all.

        Returns:
            Tuple of (du_indices, dr_indices).
        """
        total_size = len(dataset)
        if total_size <= 0:
            return [], []

        if self.split_mode == "random" and self.forget_manifest_mode in {"auto", "load"}:
            if self.forget_manifest_path.exists():
                payload = self._load_manifest()
                return self._validate_manifest(payload, total_size)
            if self.forget_manifest_mode == "load":
                raise FileNotFoundError(
                    f"forget_manifest_mode='load' but file does not exist: {self.forget_manifest_path}"
                )

        if self.split_mode == "by_class":
            if not self.forget_classes:
                raise ValueError(
                    "split_mode='by_class' requires non-empty forget_classes, "
                    "for example forget_classes: [0, 1]."
                )

            du_indices = []
            dr_indices = []
            for idx in range(total_size):
                label = self._extract_label(dataset, idx)
                if label in self.forget_classes:
                    du_indices.append(idx)
                else:
                    dr_indices.append(idx)

            if len(du_indices) == 0:
                raise ValueError(
                    f"No samples found for forget_classes={self.forget_classes}. "
                    "Please check dataset labels."
                )
            if len(dr_indices) == 0:
                raise ValueError(
                    "All samples were assigned to D_u and D_r became empty. "
                    "Please reduce forget_classes."
                )

            payload = self._build_manifest_payload(du_indices, dr_indices, total_size)
            if self.forget_manifest_mode in {"auto", "save"}:
                self._save_manifest(payload)
            else:
                self._forget_manifest_info = payload
            return du_indices, dr_indices

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
        payload = self._build_manifest_payload(du_indices, dr_indices, total_size)
        if self.split_mode == "random" and self.forget_manifest_mode in {"auto", "save"}:
            self._save_manifest(payload)
        else:
            self._forget_manifest_info = payload
        return du_indices, dr_indices

    def get_forget_manifest_info(self) -> Optional[Dict]:
        """
        Return the current forget-manifest metadata.

        Returns:
            Manifest dictionary or None.
        """
        return self._forget_manifest_info

    def get_forget_manifest_path(self) -> str:
        """
        Return forget-manifest path as string.

        Returns:
            Manifest path.
        """
        return str(self.forget_manifest_path)

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
