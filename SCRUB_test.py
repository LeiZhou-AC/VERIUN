"""End-to-end SCRUB approximate-unlearning runner.

This script runs a complete SCRUB experiment:
1) Build D_u / D_r with the existing manifest mechanism
2) Load the trained model from save/weights/trained
3) Run SCRUB teacher-student approximate unlearning
4) Save the unlearned checkpoint to save/weights/unlearned
5) Print metrics for downstream RUV experiments
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from configs.data.dataset import UnlearningDataset
from unlearning.scrub import SCRUBUnlearner
from utils.config import load_config
from utils.seed import set_seed


def _parse_forget_classes(raw: str):
    """
    Parse comma-separated forget class ids.

    Args:
        raw: Comma-separated string such as "0,3".

    Returns:
        List of class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Parse SCRUB test arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SCRUB end-to-end test")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    # Edit these defaults directly for the next SCRUB run.
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

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--forget-ce-weight", type=float, default=0.0)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-ce-weight", type=float, default=0.99)
    parser.add_argument("--retain-kd-weight", type=float, default=0.001)
    parser.add_argument("--max-forget-batches", type=int, default=0)
    parser.add_argument("--max-retain-batches", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--train-scope", type=str, default="full", choices=["full", "backbone", "head"])
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config and CLI overrides.

    Args:
        base_config: Loaded YAML config.
        args: Parsed CLI args.

    Returns:
        Runtime config dict.
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
            "scrub_epochs": args.epochs,
            "scrub_lr": args.lr,
            "scrub_optimizer": args.optimizer,
            "scrub_momentum": args.momentum,
            "scrub_weight_decay": args.weight_decay,
            "scrub_temperature": args.temperature,
            "scrub_forget_ce_weight": args.forget_ce_weight,
            "scrub_forget_weight": args.forget_weight,
            "scrub_retain_ce_weight": args.retain_ce_weight,
            "scrub_retain_kd_weight": args.retain_kd_weight,
            "scrub_max_forget_batches": args.max_forget_batches,
            "scrub_max_retain_batches": args.max_retain_batches,
            "scrub_grad_clip": args.grad_clip,
            "scrub_validate_every": args.validate_every,
            "scrub_train_scope": args.train_scope,
        }
    )
    if args.allow_download:
        cfg["allow_download"] = True
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    if args.save_name:
        cfg["scrub_checkpoint_name"] = args.save_name
    forget_classes = _parse_forget_classes(args.forget_classes)
    if forget_classes:
        cfg["forget_classes"] = forget_classes
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
    """Run SCRUB approximate unlearning from the project root."""
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

    print("[SCRUB_TEST] ===== Pipeline Configuration =====")
    pprint(config)

    print("[SCRUB_TEST] Building dataset and unlearning split (D_u / D_r)...")
    dataset = UnlearningDataset(config)
    print(
        "[SCRUB_TEST] Dataset ready: "
        f"name={dataset.dataset_name}, "
        f"D_all={len(dataset.get_all_set())}, "
        f"D_u={len(dataset.get_unlearning_set())}, "
        f"D_r={len(dataset.get_retained_set())}, "
        f"D_test={len(dataset.get_test_set())}"
    )

    print("[SCRUB_TEST] Running SCRUB via unlearning/scrub.py ...")
    unlearner = SCRUBUnlearner(config)
    result = unlearner.unlearn(model=None, dataset=dataset)

    print("[SCRUB_TEST] ===== SCRUB Completed =====")
    print(f"[SCRUB_TEST] Status: {result.get('status')}")
    print(f"[SCRUB_TEST] Save path: {result.get('save_path')}")
    print(f"[SCRUB_TEST] Forget manifest path: {result.get('forget_manifest_path')}")
    print(f"[SCRUB_TEST] Epoch logs: {len(result.get('history', []))}")
    if result.get("history"):
        print(f"[SCRUB_TEST] Final epoch metrics: {result['history'][-1]}")


if __name__ == "__main__":
    main()
