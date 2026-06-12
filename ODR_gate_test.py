"""End-to-end ODR-Gate exact-registry runner for sample-level deletion."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.data.dataset import UnlearningDataset
from unlearning.odr_gate import ODRGateUnlearner
from utils.config import load_config
from utils.seed import set_seed


def _build_args() -> argparse.Namespace:
    """
    Parse ODR-Gate CLI arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="ODR-Gate exact registry test")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "by_class"])
    parser.add_argument("--forget-ratio", type=float, default=0.01)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--forget-manifest-path", type=str, default="save/manifests/default_forget_manifest.json")
    parser.add_argument("--forget-manifest-mode", type=str, default="load", choices=["auto", "load", "save", "off"])

    parser.add_argument("--trained-path", type=str, default="save/weights/trained")
    parser.add_argument("--unlearned-path", type=str, default="save/weights/unlearned")
    parser.add_argument("--save-name", type=str, default="")
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge config file with CLI args.

    Args:
        base_config: Loaded base config.
        args: Parsed args.

    Returns:
        Runtime config.
    """
    cfg = dict(base_config or {})
    cfg.update(
        {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "in_channels": args.in_channels,
            "data_path": args.data_path,
            "allow_download": bool(args.allow_download),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "forget_ratio": args.forget_ratio,
            "split_seed": args.split_seed,
            "forget_manifest_path": args.forget_manifest_path,
            "forget_manifest_mode": args.forget_manifest_mode,
            "trained_weights_path": args.trained_path,
            "unlearned_weights_path": args.unlearned_path,
        }
    )
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    if args.save_name.strip():
        cfg["odr_gate_checkpoint_name"] = args.save_name.strip()
    return cfg


def _ensure_path_ready(path_value: str) -> None:
    """
    Create the proper directory for a file-or-directory path.

    Args:
        path_value: Directory path or checkpoint file path.
    """
    path = Path(str(path_value))
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run ODR-Gate exact registry experiment."""
    args = _build_args()
    base_config = load_config(args.config)
    config = _merge_config(base_config, args)

    if config["device"] == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"

    _ensure_path_ready(config["trained_weights_path"])
    _ensure_path_ready(config["unlearned_weights_path"])
    set_seed(int(config.get("seed", 42)))

    print("[ODR_GATE_TEST] ===== Pipeline Configuration =====")
    pprint(config)

    dataset = UnlearningDataset(config)
    print(
        "[ODR_GATE_TEST] Dataset ready: "
        f"name={dataset.dataset_name}, "
        f"D_all={len(dataset.get_all_set())}, "
        f"D_u={len(dataset.get_unlearning_set())}, "
        f"D_r={len(dataset.get_retained_set())}, "
        f"D_test={len(dataset.get_test_set())}"
    )

    unlearner = ODRGateUnlearner(config)
    result = unlearner.unlearn(model=None, dataset=dataset)

    print("[ODR_GATE_TEST] ===== ODR-Gate Completed =====")
    print(f"[ODR_GATE_TEST] Status: {result.get('status')}")
    print(f"[ODR_GATE_TEST] Save path: {result.get('save_path')}")
    print(f"[ODR_GATE_TEST] Forget manifest path: {result.get('forget_manifest_path')}")
    print(f"[ODR_GATE_TEST] Metrics: {result.get('metrics')}")


if __name__ == "__main__":
    main()
