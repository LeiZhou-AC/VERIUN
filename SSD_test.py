"""End-to-end SSD unlearning runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from configs.data.dataset import UnlearningDataset
from unlearning.ssd import SSDUnlearner
from utils.config import load_config
from utils.seed import set_seed


def _parse_forget_classes(raw: str):
    """
    Parse comma-separated forget classes.

    Args:
        raw: Comma-separated class ids.

    Returns:
        List of integer class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build SSD runner arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SSD end-to-end test")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "by_class", "class_random"])
    parser.add_argument("--forget-ratio", type=float, default=0.01)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--forget-manifest-path", type=str, default="save/manifests/default_forget_manifest.json")
    parser.add_argument("--forget-manifest-mode", type=str, default="load", choices=["auto", "load", "save", "off"])
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--trained-path", type=str, default="save/weights/trained/resnet18_cifar10.pt")
    parser.add_argument("--unlearned-path", type=str, default="save/weights/unlearned")
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--selection-weighting", type=float, default=10.0)
    parser.add_argument("--dampening-constant", type=float, default=1.0)
    parser.add_argument("--min-scale", type=float, default=0.0)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--original-split", type=str, default="all", choices=["all", "retain"])
    parser.add_argument("--forget-batches", type=int, default=0)
    parser.add_argument("--original-batches", type=int, default=0)
    parser.add_argument("--retain-batches", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config and runner arguments.

    Args:
        base_config: Loaded config.
        args: Parsed arguments.

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
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "split_mode": args.split_mode,
            "forget_ratio": args.forget_ratio,
            "forget_manifest_path": args.forget_manifest_path,
            "forget_manifest_mode": args.forget_manifest_mode,
            "split_seed": args.split_seed,
            "seed": args.seed,
            "trained_weights_path": args.trained_path,
            "unlearned_weights_path": args.unlearned_path,
            "device": args.device,
            "ssd_selection_weighting": args.selection_weighting,
            "ssd_dampening_constant": args.dampening_constant,
            "ssd_min_scale": args.min_scale,
            "ssd_max_scale": args.max_scale,
            "ssd_exponent": args.exponent,
            "ssd_original_split": args.original_split,
            "ssd_forget_batches": args.forget_batches,
            "ssd_original_batches": args.original_batches,
            "ssd_retain_batches": args.retain_batches,
            "ssd_epsilon": args.epsilon,
        }
    )
    if args.allow_download:
        cfg["allow_download"] = True
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    if args.save_name:
        cfg["ssd_checkpoint_name"] = args.save_name
    forget_classes = _parse_forget_classes(args.forget_classes)
    if forget_classes:
        cfg["forget_classes"] = forget_classes
    return cfg


def _ensure_path_ready(path_value: str) -> None:
    """
    Create directory for a file-or-directory path.

    Args:
        path_value: Path string.
    """
    path = Path(str(path_value))
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run SSD unlearning from the project root."""
    args = _build_args()
    base_config = load_config(args.config)
    config = _merge_config(base_config, args)

    if config["device"] == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                config["device"] = "cpu"
        except Exception:
            config["device"] = "cpu"

    _ensure_path_ready(config["trained_weights_path"])
    _ensure_path_ready(config["unlearned_weights_path"])
    set_seed(int(config.get("seed", 42)))

    print("[SSD_TEST] ===== Pipeline Configuration =====")
    pprint(config)
    print("[SSD_TEST] Building dataset and unlearning split (D_u / D_r)...")
    dataset = UnlearningDataset(config)
    print(
        "[SSD_TEST] Dataset ready: "
        f"name={dataset.dataset_name}, D_all={len(dataset.get_all_set())}, "
        f"D_u={len(dataset.get_unlearning_set())}, D_r={len(dataset.get_retained_set())}, "
        f"D_test={len(dataset.get_test_set())}"
    )

    print("[SSD_TEST] Running SSD via unlearning/ssd.py ...")
    result = SSDUnlearner(config).unlearn(model=None, dataset=dataset)

    print("[SSD_TEST] ===== SSD Completed =====")
    print(f"[SSD_TEST] Status: {result.get('status')}")
    print(f"[SSD_TEST] Save path: {result.get('save_path')}")
    print(f"[SSD_TEST] Forget manifest path: {result.get('forget_manifest_path')}")
    print(f"[SSD_TEST] Dampening summary: {result.get('dampening_summary')}")


if __name__ == "__main__":
    main()
