"""End-to-end RUV runner.

This script verifies an unlearned model M_u using representation-level probing.
It reuses the same forget manifest mechanism as ODR/retrain so random forget
sets are aligned across methods.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from utils.config import load_config
from utils.seed import set_seed
from verification.ruv import RUVVerifier


def _parse_forget_classes(raw: str):
    """
    Parse forget class ids from CLI string.

    Args:
        raw: Comma-separated class ids.

    Returns:
        List of integer class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_c_grid(raw: str):
    """
    Parse C grid from CLI string.

    Args:
        raw: Comma-separated C values.

    Returns:
        List of floats.
    """
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build CLI arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="RUV end-to-end verification")
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
    parser.add_argument("--forget-ratio", type=float, default=0.1)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--forget-manifest-path", type=str, default="save/manifests/default_forget_manifest.json")
    parser.add_argument("--forget-manifest-mode", type=str, default="load", choices=["auto", "load", "save", "off"])

    parser.add_argument("--unlearned-model-path", type=str, default="save/weights/unlearned")
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument("--c-grid", type=str, default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--inner-cv-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-permutations", type=int, default=100)
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument("--result-dir", type=str, default="save/results/ruv")
    parser.add_argument("--result-name", type=str, default="")
    return parser.parse_args()


def _merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge config file and CLI args.

    Args:
        base_config: Loaded config.
        args: CLI args.

    Returns:
        Final config dict.
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
            "ruv_probe_test_size": args.probe_test_size,
            "ruv_c_grid": _parse_c_grid(args.c_grid),
            "ruv_inner_cv_folds": args.inner_cv_folds,
            "ruv_alpha": args.alpha,
            "ruv_num_permutations": args.num_permutations,
            "ruv_probe_max_iter": args.probe_max_iter,
        }
    )
    if args.forget_count is not None:
        cfg["forget_count"] = args.forget_count
    forget_classes = _parse_forget_classes(args.forget_classes)
    if forget_classes:
        cfg["forget_classes"] = forget_classes
    return cfg


def _resolve_checkpoint(path_value: str) -> Path:
    """
    Resolve checkpoint path from file or directory.

    Args:
        path_value: File or directory path.

    Returns:
        Checkpoint path.
    """
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Unlearned model path does not exist: {path}")
    candidates = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {path}")
    return candidates[-1]


def _extract_state_dict(state_obj):
    """
    Extract state dict from common checkpoint formats.

    Args:
        state_obj: Object loaded by torch.load.

    Returns:
        State dict.
    """
    if isinstance(state_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_obj and isinstance(state_obj[key], dict):
                state_obj = state_obj[key]
                break
    if not isinstance(state_obj, dict):
        raise ValueError("Unsupported checkpoint format.")
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_obj.items()}


def _load_unlearned_model(config: dict, dataset: UnlearningDataset, checkpoint_path: Path):
    """
    Load M_u from checkpoint.

    Args:
        config: Runtime config.
        dataset: Dataset manager.
        checkpoint_path: Checkpoint path.

    Returns:
        Loaded model.
    """
    model, _ = construct_model(
        model=str(config.get("model_name", "resnet18")),
        num_classes=int(config.get("num_classes", dataset.num_classes)),
        seed=int(config.get("seed", 42)),
        num_channels=int(config.get("in_channels", 3)),
    )
    state = torch.load(str(checkpoint_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(_extract_state_dict(state), strict=False)
    print(f"[RUV_TEST] Loaded M_u checkpoint: {checkpoint_path}")
    if missing:
        print(f"[RUV_TEST][Load] Missing keys count: {len(missing)}")
    if unexpected:
        print(f"[RUV_TEST][Load] Unexpected keys count: {len(unexpected)}")
    return model


def _target_tag(config: dict) -> str:
    """
    Build target tag for result naming.

    Args:
        config: Runtime config.

    Returns:
        Target tag.
    """
    if str(config.get("split_mode", "random")).lower() == "by_class":
        classes = config.get("forget_classes", [])
        if isinstance(classes, (int, float, str)):
            classes = [classes]
        return "byclass_" + "-".join(str(int(c)) for c in classes)
    if config.get("forget_count") is not None:
        return f"random_count{int(config['forget_count'])}"
    ratio = str(float(config.get("forget_ratio", 0.1))).replace(".", "p")
    return f"random_ratio{ratio}"


def main() -> None:
    """Run RUV end-to-end."""
    args = _build_args()
    base_config = load_config(args.config)
    config = _merge_config(base_config, args)

    if config["device"] == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"

    set_seed(int(config.get("seed", 42)))
    print("[RUV_TEST] ===== Pipeline Configuration =====")
    pprint(config)

    dataset = UnlearningDataset(config)
    checkpoint_path = _resolve_checkpoint(args.unlearned_model_path)
    model_u = _load_unlearned_model(config, dataset, checkpoint_path)

    verifier = RUVVerifier(config=config)
    result = verifier.verify(model_before=None, model_after=model_u, dataset=dataset)
    result.update(
        {
            "unlearned_model_path": str(checkpoint_path),
            "model_name": config.get("model_name", "resnet18"),
            "dataset": dataset.dataset_name,
            "split_mode": config.get("split_mode", "random"),
            "forget_classes": config.get("forget_classes", []),
            "forget_ratio": config.get("forget_ratio", None),
            "forget_count": config.get("forget_count", None),
            "forget_manifest_path": dataset.get_forget_manifest_path(),
        }
    )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_name = args.result_name.strip()
    if not result_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = f"ruv_{config.get('model_name', 'model')}_{dataset.dataset_name}_{_target_tag(config)}_{ts}.json"
    result_path = result_dir / result_name
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[RUV_TEST] ===== RUV Completed =====")
    print(f"[RUV_TEST] Result path: {result_path}")
    print(f"[RUV_TEST] Decision: {result['decision']}")


if __name__ == "__main__":
    main()
