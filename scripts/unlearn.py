"""Unified unlearning script entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data.dataset import UnlearningDataset
from unlearning.factory import get_unlearner
from utils.config import load_config
from utils.seed import set_seed


def _parse_forget_classes(raw: str):
    """
    Parse comma-separated forget classes.

    Args:
        raw: Comma-separated class ids.

    Returns:
        List of class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unified machine unlearning runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["odr", "odr_gate", "retrain", "amnesiac", "ssd"],
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split-mode", type=str, default=None, choices=["random", "by_class", "class_random"])
    parser.add_argument("--forget-ratio", type=float, default=None)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--forget-manifest-path", type=str, default=None)
    parser.add_argument("--forget-manifest-mode", type=str, default=None, choices=["auto", "load", "save", "off"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--trained-path", type=str, default=None)
    parser.add_argument("--unlearned-path", type=str, default=None)
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge generic CLI overrides into config.

    Args:
        base_config: Loaded config.
        args: Parsed arguments.

    Returns:
        Runtime config.
    """
    cfg = dict(base_config or {})

    def set_if_not_none(key, value):
        if value is not None:
            cfg[key] = value

    set_if_not_none("unlearning_method", args.method)
    set_if_not_none("dataset", args.dataset)
    set_if_not_none("model_name", args.model_name)
    set_if_not_none("num_classes", args.num_classes)
    set_if_not_none("in_channels", args.in_channels)
    set_if_not_none("data_path", args.data_path)
    set_if_not_none("batch_size", args.batch_size)
    set_if_not_none("num_workers", args.num_workers)
    set_if_not_none("seed", args.seed)
    set_if_not_none("device", args.device)
    set_if_not_none("split_mode", args.split_mode)
    set_if_not_none("forget_ratio", args.forget_ratio)
    set_if_not_none("forget_count", args.forget_count)
    set_if_not_none("forget_manifest_path", args.forget_manifest_path)
    set_if_not_none("forget_manifest_mode", args.forget_manifest_mode)
    set_if_not_none("split_seed", args.split_seed)
    set_if_not_none("trained_weights_path", args.trained_path)
    set_if_not_none("unlearned_weights_path", args.unlearned_path)
    if args.forget_classes.strip():
        cfg["forget_classes"] = _parse_forget_classes(args.forget_classes)
    if args.allow_download:
        cfg["allow_download"] = True
    return cfg


def main():
    """
    Run the configured unlearning method.

    The selected unlearner is responsible for loading the original checkpoint
    from ``trained_weights_path`` and saving the result to
    ``unlearned_weights_path``.
    """
    args = _build_args()
    config = _merge_config(load_config(args.config), args)
    set_seed(int(config.get("seed", 42)))

    method = str(config.get("unlearning_method", "odr"))
    print("[UNLEARN] ===== Unified Unlearning Configuration =====")
    pprint(config)
    print(f"[UNLEARN] Selected method: {method}")

    dataset = UnlearningDataset(config)
    unlearner = get_unlearner(method, config)
    result = unlearner.unlearn(model=None, dataset=dataset)

    print("[UNLEARN] ===== Unlearning Completed =====")
    print(f"[UNLEARN] Status: {result.get('status')}")
    print(f"[UNLEARN] Save path: {result.get('save_path')}")
    return result


if __name__ == "__main__":
    main()
