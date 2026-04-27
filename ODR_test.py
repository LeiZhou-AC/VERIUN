"""End-to-end ODR test runner.

This script demonstrates a complete ODR unlearning flow:
1) Select forget data split (D_u) and retained split (D_r)
2) Load the trained model from `save/weights/trained`
3) Run ODR unlearning by calling `unlearning/od.py`
4) Save unlearned model into `save/weights/unlearned`
5) Print training/unlearning logs and summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from configs.data.dataset import UnlearningDataset
from unlearning.od import ODRUnlearner
from utils.config import load=_config
from utils.seed import set_seed


def _build_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="ODR end-to-end test")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--model-name", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channel number")
    parser.add_argument("--forget-ratio", type=float, default=0.1, help="Fraction of data to forget")
    parser.add_argument("--forget-count", type=int, default=None, help="Absolute forget sample count")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for D_u / D_r split")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=8, help="ODR epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="ODR learning rate")
    parser.add_argument("--alpha", type=float, default=0.8, help="ODR forge mixing coefficient")
    parser.add_argument("--lambda-acc", type=float, default=0.5, help="Retention CE weight")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Noise std for p_forge eta(x)")
    parser.add_argument(
        "--trained-path",
        type=str,
        default="save/weights/trained",
        help="Trained model file or directory",
    )
    parser.add_argument(
        "--unlearned-path",
        type=str,
        default="save/weights/unlearned",
        help="Output directory for unlearned model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device preference: cuda or cpu")
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config with CLI args.

    Args:
        base_config: Config loaded from YAML.
        args: CLI arguments.

    Returns:
        Final config dictionary passed to dataset and unlearner.
    """
    cfg = dict(base_config or {})
    cfg.update(
        {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "in_channels": args.in_channels,
            "batch_size": args.batch_size,
            "forget_ratio": args.forget_ratio,
            "split_seed": args.split_seed,
            "seed": args.seed,
            "odr_epochs": args.epochs,
            "odr_lr": args.lr,
            "alpha": args.alpha,
            "lambda_acc": args.lambda_acc,
            "odr_noise_std": args.noise_std,
            "trained_weights_path": args.trained_path,
            "unlearned_weights_path": args.unlearned_path,
            "device": args.device,
        }
    )
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    return cfg


def main() -> None:
    """
    Run full ODR unlearning test pipeline.
    """
    args = _build_args()
    base_config = load_config(args.config)
    config = _merge_config(base_config, args)

    if config["device"] == "cuda":
        # Fallback gracefully when CUDA is unavailable.
        try:
            import torch

            if not torch.cuda.is_available():
                config["device"] = "cpu"
        except Exception:
            config["device"] = "cpu"

    Path(config["unlearned_weights_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["trained_weights_path"]).mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    print("[ODR_TEST] ===== Pipeline Configuration =====")
    pprint(config)

    print("[ODR_TEST] Building dataset and unlearning split (D_u / D_r)...")
    dataset = UnlearningDataset(config)
    print(
        "[ODR_TEST] Dataset ready: "
        f"name={dataset.dataset_name}, "
        f"D_all={len(dataset.get_all_set())}, "
        f"D_u={len(dataset.get_unlearning_set())}, "
        f"D_r={len(dataset.get_retained_set())}, "
        f"D_test={len(dataset.get_test_set())}"
    )

    print("[ODR_TEST] Running ODR unlearning via unlearning/od.py ...")
    unlearner = ODRUnlearner(config)
    result = unlearner.unlearn(model=None, dataset=dataset)

    print("[ODR_TEST] ===== ODR Completed =====")
    print(f"[ODR_TEST] Status: {result.get('status')}")
    print(f"[ODR_TEST] Save path: {result.get('save_path')}")
    print(f"[ODR_TEST] Epoch logs: {len(result.get('history', []))}")
    if result.get("history"):
        print(f"[ODR_TEST] Final epoch metrics: {result['history'][-1]}")


if __name__ == "__main__":
    main()
