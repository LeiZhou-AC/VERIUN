"""Export a VeriUn manifest matching official SalUn random forgetting.

Official SalUn Classification first removes a class-balanced validation split
from the CIFAR training set, then selects random forget samples from the
remaining training subset.  This script reproduces that split and saves a
manifest with ``ignored_indices`` for the validation samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torchvision


def _build_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="Export official SalUn random forget manifest")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-indexes-to-replace", type=int, default=4500)
    parser.add_argument("--output", type=str, default="save/manifests/salun_official_random_4500.json")
    return parser.parse_args()


def _load_targets(dataset_name: str, data_path: str, allow_download: bool) -> Sequence[int]:
    """
    Load train-set labels for the requested CIFAR dataset.

    Args:
        dataset_name: cifar10 or cifar100.
        data_path: Dataset root.
        allow_download: Whether torchvision may download missing data.

    Returns:
        Sequence of integer labels.
    """
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=allow_download,
        )
    else:
        dataset = torchvision.datasets.CIFAR100(
            root=data_path,
            train=True,
            download=allow_download,
        )
    return np.array(dataset.targets)


def _official_train_validation_split(targets: Sequence[int], seed: int):
    """
    Reproduce official SalUn's class-balanced validation split.

    Args:
        targets: Full train labels.
        seed: Official SalUn seed.

    Returns:
        Tuple of train indices and validation/ignored indices in original space.
    """
    targets = np.array(targets)
    rng = np.random.RandomState(seed)
    valid_idx = []
    for cls in range(int(max(targets)) + 1):
        class_idx = np.where(targets == cls)[0]
        valid_idx.append(rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)

    # Match official code style: list(set(range(len(train_set))) - set(valid_idx)).
    train_idx = list(set(range(len(targets))) - set(valid_idx))
    return train_idx, [int(idx) for idx in valid_idx.tolist()]


def _official_forget_indices(train_idx: Sequence[int], seed: int, count: int):
    """
    Reproduce official SalUn's random class_to_replace=-1 forget selection.

    Args:
        train_idx: Official 45k train subset indices in original train space.
        seed: Official SalUn seed.
        count: Number of forget samples.

    Returns:
        Tuple of forget and retain indices in original train space.
    """
    if count <= 0 or count >= len(train_idx):
        raise ValueError(f"num_indexes_to_replace must be in [1, {len(train_idx) - 1}], got {count}")
    rng = np.random.RandomState(seed - 1)
    selected_positions = rng.choice(np.arange(len(train_idx)), size=count, replace=False)
    selected_positions = set(int(pos) for pos in selected_positions.tolist())
    du_indices = [int(train_idx[pos]) for pos in selected_positions]
    dr_indices = [int(idx) for pos, idx in enumerate(train_idx) if pos not in selected_positions]
    return du_indices, dr_indices


def main() -> None:
    """Export a SalUn-compatible VeriUn manifest."""
    args = _build_args()
    targets = _load_targets(args.dataset, args.data_path, args.allow_download)
    train_idx, ignored_indices = _official_train_validation_split(targets, args.seed)
    du_indices, dr_indices = _official_forget_indices(
        train_idx=train_idx,
        seed=args.seed,
        count=args.num_indexes_to_replace,
    )

    payload = {
        "dataset": args.dataset,
        "split_mode": "random",
        "total_size": int(len(targets)),
        "split_seed": int(args.seed),
        "forget_ratio": float(len(du_indices) / max(len(train_idx), 1)),
        "forget_count": int(len(du_indices)),
        "forget_count_per_class": None,
        "forget_classes": [],
        "du_indices": [int(idx) for idx in du_indices],
        "dr_indices": [int(idx) for idx in dr_indices],
        "ignored_indices": [int(idx) for idx in ignored_indices],
        "source": {
            "name": "official_salun_random",
            "seed": int(args.seed),
            "num_indexes_to_replace": int(args.num_indexes_to_replace),
            "official_train_size": int(len(train_idx)),
            "official_validation_size": int(len(ignored_indices)),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SALUN_MANIFEST] Saved: {output}")
    print(
        "[SALUN_MANIFEST] "
        f"D_u={len(du_indices)}, D_r={len(dr_indices)}, ignored={len(ignored_indices)}, total={len(targets)}"
    )


if __name__ == "__main__":
    main()
