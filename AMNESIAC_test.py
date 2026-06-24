"""End-to-end Amnesiac unlearning runner.

This script runs a complete Amnesiac experiment:
1) Build D_u / D_r with the existing manifest mechanism
2) In log mode, train a logged original model before unlearning
3) Remove logged D_u updates, optionally repair on D_r
4) In relabel mode, load an existing trained model and run wrong-label updates
5) Save checkpoints/logs and print metrics for downstream RUV experiments
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from configs.data.dataset import UnlearningDataset
from unlearning.amnesiac import AmnesiacUnlearner
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
    Parse Amnesiac test arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Amnesiac end-to-end test")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    # Edit these defaults directly for the next Amnesiac run.
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

    parser.add_argument("--mode", type=str, default="log", choices=["log", "relabel"])
    parser.add_argument("--original-epochs", type=int, default=50)
    parser.add_argument("--original-lr", type=float, default=0.01)
    parser.add_argument("--original-optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--original-momentum", type=float, default=0.9)
    parser.add_argument("--original-weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-dir", type=str, default="save/unlearning_logs/amnesiac")
    parser.add_argument("--log-scale", type=float, default=1.0)
    parser.add_argument("--repair-epochs", type=int, default=5)
    parser.add_argument("--repair-lr", type=float, default=0.001)
    parser.add_argument("--repair-batches", type=int, default=0)

    # Relabel-mode parameters. These are ignored by mode=log except grad_clip
    # and validate_every, which are shared by both paths.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--relabel-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=0.5)
    parser.add_argument("--max-relabel-batches", type=int, default=0)
    parser.add_argument("--max-retain-batches", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--train-scope", type=str, default="full", choices=["full", "backbone", "head"])
    parser.add_argument("--label-seed", type=int, default=42)
    parser.add_argument("--label-strategy", type=str, default="cyclic", choices=["cyclic", "permutation", "random"])
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config and script/CLI overrides.

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
            "amnesiac_mode": args.mode,
            "amnesiac_original_epochs": args.original_epochs,
            "amnesiac_original_lr": args.original_lr,
            "amnesiac_original_optimizer": args.original_optimizer,
            "amnesiac_original_momentum": args.original_momentum,
            "amnesiac_original_weight_decay": args.original_weight_decay,
            "amnesiac_log_dir": args.log_dir,
            "amnesiac_log_scale": args.log_scale,
            "amnesiac_repair_epochs": args.repair_epochs,
            "amnesiac_repair_lr": args.repair_lr,
            "amnesiac_repair_batches": args.repair_batches,
            "amnesiac_epochs": args.epochs,
            "amnesiac_lr": args.lr,
            "amnesiac_optimizer": args.optimizer,
            "amnesiac_momentum": args.momentum,
            "amnesiac_weight_decay": args.weight_decay,
            "amnesiac_relabel_weight": args.relabel_weight,
            "amnesiac_retain_weight": args.retain_weight,
            "amnesiac_max_relabel_batches": args.max_relabel_batches,
            "amnesiac_max_retain_batches": args.max_retain_batches,
            "amnesiac_grad_clip": args.grad_clip,
            "amnesiac_validate_every": args.validate_every,
            "amnesiac_train_scope": args.train_scope,
            "amnesiac_label_seed": args.label_seed,
            "amnesiac_label_strategy": args.label_strategy,
        }
    )
    if args.allow_download:
        cfg["allow_download"] = True
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    if args.save_name:
        cfg["amnesiac_checkpoint_name"] = args.save_name
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
    """Run Amnesiac relabel approximate unlearning from the project root."""
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

    print("[AMNESIAC_TEST] ===== Pipeline Configuration =====")
    pprint(config)

    print("[AMNESIAC_TEST] Building dataset and unlearning split (D_u / D_r)...")
    dataset = UnlearningDataset(config)
    print(
        "[AMNESIAC_TEST] Dataset ready: "
        f"name={dataset.dataset_name}, "
        f"D_all={len(dataset.get_all_set())}, "
        f"D_u={len(dataset.get_unlearning_set())}, "
        f"D_r={len(dataset.get_retained_set())}, "
        f"D_test={len(dataset.get_test_set())}"
    )

    print("[AMNESIAC_TEST] Running Amnesiac via unlearning/amnesiac.py ...")
    unlearner = AmnesiacUnlearner(config)
    result = unlearner.unlearn(model=None, dataset=dataset)

    print("[AMNESIAC_TEST] ===== Amnesiac Completed =====")
    print(f"[AMNESIAC_TEST] Status: {result.get('status')}")
    print(f"[AMNESIAC_TEST] Method: {result.get('method')}")
    if result.get("trained_path"):
        print(f"[AMNESIAC_TEST] Trained path: {result.get('trained_path')}")
    if result.get("log_path"):
        print(f"[AMNESIAC_TEST] Log path: {result.get('log_path')}")
    print(f"[AMNESIAC_TEST] Save path: {result.get('save_path')}")
    print(f"[AMNESIAC_TEST] Forget manifest path: {result.get('forget_manifest_path')}")
    print(f"[AMNESIAC_TEST] Epoch logs: {len(result.get('history', []))}")
    if result.get("history"):
        print(f"[AMNESIAC_TEST] Final epoch metrics: {result['history'][-1]}")


if __name__ == "__main__":
    main()
