"""Evaluate official SalUn checkpoints on manifest-defined forget/retain sets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor

from models import model_dict
from models.ResNet import NormalizeByChannelMeanStd


CLASSIFICATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLASSIFICATION_DIR.parents[2]
DEFAULT_MODEL_PATHS = [
    PROJECT_ROOT / "save/weights/unlearned/salun_official_random4500"
]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "save/manifests/salun_official_random_4500.json"
)


def _build_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure official SalUn checkpoints on exact D_u and D_r indices."
    )
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        help="Checkpoint file or directory. Repeat to compare multiple checkpoints.",
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--arch", default="resnet18", choices=sorted(model_dict))
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow missing/unexpected state-dict keys. Disabled by default.",
    )
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "save/results/forget_accuracy_salun"),
    )
    return parser.parse_args()


def _load_manifest(path: Path, dataset_name: str) -> Dict:
    """Load and validate a VeriUn/SalUn forget manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Forget manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset_name:
        raise ValueError(
            f"Manifest dataset is {manifest.get('dataset')}, expected {dataset_name}."
        )
    if not manifest.get("du_indices") or not manifest.get("dr_indices"):
        raise ValueError("Manifest must contain non-empty du_indices and dr_indices.")
    return manifest


def _build_train_dataset(
    dataset_name: str,
    data_path: str,
    allow_download: bool,
):
    """Build the unaugmented CIFAR training set used for evaluation."""
    dataset_cls = (
        torchvision.datasets.CIFAR10
        if dataset_name == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    return dataset_cls(
        root=data_path,
        train=True,
        transform=ToTensor(),
        download=allow_download,
    )


def _validate_indices(manifest: Dict, dataset_size: int) -> None:
    """Validate manifest index ranges and disjointness."""
    du_indices = [int(index) for index in manifest["du_indices"]]
    dr_indices = [int(index) for index in manifest["dr_indices"]]
    ignored_indices = [int(index) for index in manifest.get("ignored_indices", [])]
    all_indices = du_indices + dr_indices + ignored_indices

    if any(index < 0 or index >= dataset_size for index in all_indices):
        raise ValueError("Manifest contains an out-of-range dataset index.")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Manifest D_u, D_r, and ignored indices overlap.")
    if len(all_indices) != dataset_size:
        raise ValueError(
            "Manifest does not partition the full training set into D_u, D_r, "
            "and ignored indices."
        )


def _build_model(arch: str, dataset_name: str):
    """Construct the exact official SalUn model and internal normalization."""
    num_classes = 10 if dataset_name == "cifar10" else 100
    model = model_dict[arch](num_classes=num_classes)
    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    model.normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    return model


def _checkpoint_files(path_values: Iterable[str]) -> List[Path]:
    """Resolve checkpoint files from explicit files or directories."""
    checkpoints: List[Path] = []
    for raw_path in path_values:
        path = Path(raw_path)
        if path.is_file():
            checkpoints.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Model path not found: {path}")

        preferred = sorted(path.glob("*checkpoint.pth.tar"))
        if preferred:
            checkpoints.extend(preferred)
            continue
        checkpoints.extend(
            candidate
            for candidate in sorted(path.iterdir())
            if candidate.is_file()
            and candidate.name.endswith((".pt", ".pth", ".pth.tar"))
            and "eval_result" not in candidate.name.lower()
        )

    checkpoints = list(dict.fromkeys(checkpoints))
    if not checkpoints:
        raise FileNotFoundError("No SalUn checkpoint was found.")
    return checkpoints


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Extract and normalize a state dictionary from a SalUn checkpoint."""
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format.")
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def _load_model(
    checkpoint_path: Path,
    arch: str,
    dataset_name: str,
    allow_partial_load: bool,
):
    """Load an official SalUn model checkpoint with structural validation."""
    model = _build_model(arch=arch, dataset_name=dataset_name)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint)
    if allow_partial_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[SALUN_ACC][Load] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[SALUN_ACC][Load] Unexpected keys: {len(unexpected)}")
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


@torch.inference_mode()
def _evaluate(model, loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate accuracy, true-label confidence, CE, and classwise accuracy."""
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    confidence_sum = 0.0
    loss_sum = 0.0
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()
        confidence_sum += probabilities.gather(1, labels.unsqueeze(1)).sum().item()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()

        for label in labels.unique():
            class_id = int(label.item())
            mask = labels.eq(label)
            class_total[class_id] += int(mask.sum().item())
            class_correct[class_id] += int(
                predictions[mask].eq(labels[mask]).sum().item()
            )

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {
        "accuracy": correct / total,
        "correct": int(correct),
        "total": int(total),
        "mean_true_label_confidence": confidence_sum / total,
        "mean_cross_entropy": loss_sum / total,
        "per_class_accuracy": {
            str(class_id): class_correct[class_id] / class_total[class_id]
            for class_id in sorted(class_total)
        },
        "per_class_total": {
            str(class_id): class_total[class_id] for class_id in sorted(class_total)
        },
    }


def _print_metrics(split_name: str, metrics: Dict) -> None:
    """Print one dataset split's evaluation metrics."""
    print(
        f"[SALUN_ACC][{split_name}] accuracy={metrics['accuracy']:.4f} "
        f"({metrics['correct']}/{metrics['total']}) "
        f"confidence={metrics['mean_true_label_confidence']:.4f} "
        f"ce={metrics['mean_cross_entropy']:.4f}"
    )


def main() -> None:
    """Evaluate official SalUn checkpoints on the exact D_u and D_r."""
    args = _build_args()
    manifest_path = Path(args.manifest_path)
    manifest = _load_manifest(manifest_path, args.dataset)
    dataset = _build_train_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        allow_download=args.allow_download,
    )
    _validate_indices(manifest, len(dataset))

    forget_set = Subset(dataset, [int(index) for index in manifest["du_indices"]])
    retain_set = Subset(dataset, [int(index) for index in manifest["dr_indices"]])
    forget_loader = DataLoader(
        forget_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    retain_loader = DataLoader(
        retain_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[SALUN_ACC] CUDA unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    model_paths = args.model_paths or [str(path) for path in DEFAULT_MODEL_PATHS]
    checkpoint_paths = _checkpoint_files(model_paths)

    print("[SALUN_ACC] ===== Official SalUn Forget/Retain Evaluation =====")
    print(f"[SALUN_ACC] Manifest: {manifest_path}")
    print(f"[SALUN_ACC] Architecture: official {args.arch}")
    print(f"[SALUN_ACC] D_u={len(forget_set)}, D_r={len(retain_set)}")

    results = []
    for checkpoint_path in checkpoint_paths:
        print(f"[SALUN_ACC] Loading: {checkpoint_path}")
        model = _load_model(
            checkpoint_path=checkpoint_path,
            arch=args.arch,
            dataset_name=args.dataset,
            allow_partial_load=args.allow_partial_load,
        )
        forget_metrics = _evaluate(model, forget_loader, device)
        retain_metrics = _evaluate(model, retain_loader, device)
        _print_metrics("Forget", forget_metrics)
        _print_metrics("Retain", retain_metrics)
        results.append(
            {
                "checkpoint": str(checkpoint_path),
                "forget": forget_metrics,
                "retain": retain_metrics,
            }
        )
        del model

    output = {
        "manifest_path": str(manifest_path),
        "dataset": args.dataset,
        "architecture": args.arch,
        "model_backend": "salun_official",
        "results": results,
    }
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = result_dir / f"salun_forget_accuracy_{timestamp}.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[SALUN_ACC] Result saved: {result_path}")


if __name__ == "__main__":
    main()
