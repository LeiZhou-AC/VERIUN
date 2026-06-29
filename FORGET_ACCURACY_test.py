"""Evaluate checkpoint accuracy on the exact machine-unlearning forget set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from configs.models.salun_resnet import construct_salun_official_model
from utils.config import load_config
from utils.seed import set_seed


DEFAULT_MODEL_PATHS = ["save/weights/unlearned"]
DEFAULT_MANIFEST_PATH = "save/manifests/default_forget_manifest.json"


def _build_args() -> argparse.Namespace:
    """Parse command-line arguments for forget-set evaluation."""
    parser = argparse.ArgumentParser(
        description="Measure one or more model checkpoints on the exact forget set D_u."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        help="Checkpoint file or directory. Repeat this argument to compare models.",
    )
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument(
        "--model-backend",
        default="native",
        choices=["native", "salun_official"],
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys. Disabled by default.",
    )
    parser.add_argument("--result-dir", default="save/results/forget_accuracy")
    return parser.parse_args()


def _load_manifest(path: Path) -> Dict:
    """Load and minimally validate a forget manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Forget manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("du_indices"):
        raise ValueError(f"Manifest has no du_indices: {path}")
    return payload


def _build_config(args: argparse.Namespace, manifest: Dict) -> Dict:
    """Build dataset and model configuration using manifest metadata."""
    config = dict(load_config(args.config) or {})
    config.update(
        {
            "dataset": manifest.get("dataset", args.dataset),
            "split_mode": manifest.get("split_mode", "random"),
            "forget_classes": manifest.get("forget_classes", []),
            "forget_ratio": manifest.get("forget_ratio", config.get("forget_ratio", 0.1)),
            "forget_count": manifest.get("forget_count"),
            "forget_count_per_class": manifest.get("forget_count_per_class"),
            "split_seed": manifest.get("split_seed", args.seed),
            "forget_manifest_path": str(Path(args.manifest_path)),
            "forget_manifest_mode": "load",
            "model_name": args.model_name,
            "model_backend": args.model_backend,
            "num_classes": args.num_classes,
            "in_channels": args.in_channels,
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "seed": args.seed,
            "allow_download": bool(args.allow_download),
            "augmentations": False,
            "normalize": args.model_backend != "salun_official",
        }
    )
    return config


def _checkpoint_files(path_values: Iterable[str]) -> List[Path]:
    """Expand checkpoint files and directories into a deterministic file list."""
    suffixes = (".pt", ".pth", ".pth.tar", ".bin")
    checkpoints: List[Path] = []
    for raw_path in path_values:
        path = Path(raw_path)
        if path.is_file():
            checkpoints.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Model path not found: {path}")
        candidates = [
            candidate
            for candidate in sorted(path.iterdir())
            if candidate.is_file()
            and candidate.name.endswith(suffixes)
            and "eval_result" not in candidate.name.lower()
        ]
        checkpoints.extend(candidates)

    unique = list(dict.fromkeys(checkpoints))
    if not unique:
        raise FileNotFoundError("No checkpoint files were found in the supplied model paths.")
    return unique


def _extract_state_dict(checkpoint):
    """Extract a model state dictionary from common checkpoint containers."""
    state = checkpoint
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: expected a state dictionary.")
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _load_checkpoint(path: Path):
    """Load a checkpoint while preferring PyTorch's restricted loader."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _construct_model(config: Dict):
    """Construct the configured native or official SalUn model."""
    if config["model_backend"] == "salun_official":
        return construct_salun_official_model(
            model=config["model_name"],
            num_classes=int(config["num_classes"]),
        )
    model, _ = construct_model(
        model=config["model_name"],
        num_classes=int(config["num_classes"]),
        seed=int(config["seed"]),
        num_channels=int(config["in_channels"]),
    )
    return model


def _load_model(path: Path, config: Dict, allow_partial_load: bool):
    """Construct a model and load one checkpoint with structural validation."""
    model = _construct_model(config)
    state_dict = _extract_state_dict(_load_checkpoint(path))
    if allow_partial_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FORGET_ACC][Load] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[FORGET_ACC][Load] Unexpected keys: {len(unexpected)}")
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


@torch.inference_mode()
def _evaluate(model, dataloader, device: torch.device) -> Dict:
    """Measure accuracy, confidence, loss, and per-class accuracy on a dataset."""
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    confidence_sum = 0.0
    loss_sum = 0.0
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total += batch_size
        correct += predictions.eq(labels).sum().item()
        confidence_sum += probabilities.gather(1, labels.unsqueeze(1)).sum().item()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()

        for label in labels.unique():
            class_id = int(label.item())
            mask = labels.eq(label)
            class_total[class_id] += int(mask.sum().item())
            class_correct[class_id] += int(predictions[mask].eq(labels[mask]).sum().item())

    if total == 0:
        raise ValueError("Forget-set dataloader is empty.")
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


def main() -> None:
    """Evaluate all requested checkpoints on the manifest-defined forget set."""
    args = _build_args()
    manifest_path = Path(args.manifest_path)
    manifest = _load_manifest(manifest_path)
    config = _build_config(args, manifest)
    set_seed(int(config["seed"]))

    device_name = config["device"]
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[FORGET_ACC] CUDA unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = UnlearningDataset(config)
    dataloaders = dataset.get_dataloaders(retained_shuffle=False)
    forget_loader = dataloaders["d_u"]
    retain_loader = dataloaders["d_r"]
    paths = _checkpoint_files(args.model_paths or DEFAULT_MODEL_PATHS)

    print("[FORGET_ACC] ===== Forget/Retain Accuracy Evaluation =====")
    print(f"[FORGET_ACC] Manifest: {manifest_path}")
    print(
        f"[FORGET_ACC] D_u size: {len(dataset.get_unlearning_set())}, "
        f"D_r size: {len(dataset.get_retained_set())}"
    )
    print(f"[FORGET_ACC] Backend: {config['model_backend']}, model: {config['model_name']}")

    results = []
    for checkpoint_path in paths:
        print(f"[FORGET_ACC] Loading: {checkpoint_path}")
        model = _load_model(checkpoint_path, config, args.allow_partial_load)
        forget_metrics = _evaluate(model, forget_loader, device)
        retain_metrics = _evaluate(model, retain_loader, device)
        results.append(
            {
                "checkpoint": str(checkpoint_path),
                "forget": forget_metrics,
                "retain": retain_metrics,
            }
        )
        print(
            f"[FORGET_ACC][Forget] accuracy={forget_metrics['accuracy']:.4f} "
            f"({forget_metrics['correct']}/{forget_metrics['total']}) "
            f"confidence={forget_metrics['mean_true_label_confidence']:.4f} "
            f"ce={forget_metrics['mean_cross_entropy']:.4f}"
        )
        print(
            f"[FORGET_ACC][Retain] accuracy={retain_metrics['accuracy']:.4f} "
            f"({retain_metrics['correct']}/{retain_metrics['total']}) "
            f"confidence={retain_metrics['mean_true_label_confidence']:.4f} "
            f"ce={retain_metrics['mean_cross_entropy']:.4f}"
        )
        del model

    output = {
        "manifest_path": str(manifest_path),
        "dataset": dataset.dataset_name,
        "split_mode": config["split_mode"],
        "forget_classes": config.get("forget_classes", []),
        "model_name": config["model_name"],
        "model_backend": config["model_backend"],
        "results": results,
    }
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = result_dir / f"forget_accuracy_{timestamp}.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[FORGET_ACC] Result saved: {result_path}")


if __name__ == "__main__":
    main()
