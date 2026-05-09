"""CLI entry point for minimal TruVRF Metric-I verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data.dataset import UnlearningDataset
from utils.checkpoint import load_model
from utils.config import load_config
from utils.seed import set_seed
from verification.factory import get_verifier


def _parse_classes(raw: str):
    """
    Parse a comma-separated class list.

    Args:
        raw: Raw CLI value.

    Returns:
        Parsed integer class list.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Verify class-level forgetting with TruVRF Metric-I")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--original-path", type=str, required=True, help="Path to original checkpoint")
    parser.add_argument("--unlearned-path", type=str, required=True, help="Path to candidate unlearned checkpoint")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--split-mode", type=str, default="by_class", choices=["random", "by_class"])
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--forget-manifest-path", type=str, default=None)
    parser.add_argument("--forget-manifest-mode", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--metric1-lr", type=float, default=1e-3)
    parser.add_argument("--metric1-steps", type=int, default=1)
    parser.add_argument("--metric1-batch-size", type=int, default=64)
    parser.add_argument("--metric1-threshold", type=float, default=0.01)
    parser.add_argument("--metric1-max-samples-per-class", type=int, default=256)
    parser.add_argument("--use-all-set", action="store_true", help="Use training set instead of test set as auxiliary data")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save raw result json")
    return parser.parse_args()


def _merge_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    """
    Merge CLI overrides into config.

    Args:
        base_cfg: Base YAML config.
        args: CLI args.

    Returns:
        Runtime config.
    """
    cfg = dict(base_cfg or {})

    def set_if_not_none(key, value):
        if value is not None:
            cfg[key] = value

    set_if_not_none("dataset", args.dataset)
    set_if_not_none("data_path", args.data_path)
    set_if_not_none("batch_size", args.batch_size)
    set_if_not_none("num_workers", args.num_workers)
    set_if_not_none("split_mode", args.split_mode)
    set_if_not_none("forget_manifest_path", args.forget_manifest_path)
    set_if_not_none("forget_manifest_mode", args.forget_manifest_mode)
    set_if_not_none("split_seed", args.split_seed)
    set_if_not_none("seed", args.seed)
    set_if_not_none("device", args.device)

    if args.allow_download:
        cfg["allow_download"] = True

    classes = _parse_classes(args.forget_classes)
    if classes:
        cfg["forget_classes"] = classes

    cfg["truvrf_metric1_lr"] = args.metric1_lr
    cfg["truvrf_metric1_steps"] = args.metric1_steps
    cfg["truvrf_metric1_batch_size"] = args.metric1_batch_size
    cfg["truvrf_metric1_threshold"] = args.metric1_threshold
    cfg["truvrf_metric1_use_test_set"] = not args.use_all_set
    cfg["truvrf_metric1_max_samples_per_class"] = args.metric1_max_samples_per_class
    return cfg


def main():
    """
    Run minimal TruVRF Metric-I verification.
    """
    args = _build_args()
    config = _merge_config(load_config(args.config), args)
    set_seed(int(config.get("seed", 42)))

    dataset = UnlearningDataset(config)
    model_before = load_model(args.original_path, map_location=config.get("device", "cpu"))
    model_after = load_model(args.unlearned_path, map_location=config.get("device", "cpu"))

    verifier = get_verifier("truvrf_metric1", config=config)
    result = verifier.verify(model_before, model_after, dataset)

    print("[TruVRF Metric-I] Verification result")
    pprint(result)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[TruVRF Metric-I] Saved json to: {output_path}")


if __name__ == "__main__":
    main()
