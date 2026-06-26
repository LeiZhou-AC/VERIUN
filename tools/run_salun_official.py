"""Run official SalUn Classification inside the VeriUn workspace.

This runner keeps the third-party SalUn code untouched while standardizing
dataset paths, manifest export, checkpoint locations, and command order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SALUN_CLASSIFICATION_DIR = PROJECT_ROOT / "external" / "salun" / "Classification"


def _build_args() -> argparse.Namespace:
    """
    Parse runner arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run official SalUn Classification workflow")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "manifest", "train", "mask", "unlearn"])
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--train-epochs", type=int, default=182)
    parser.add_argument("--train-lr", type=float, default=0.1)
    parser.add_argument("--decreasing-lr", type=str, default="91,136")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--num-indexes-to-replace", type=int, default=4500)
    parser.add_argument("--unlearn-method", type=str, default="RL")
    parser.add_argument("--unlearn-epochs", type=int, default=10)
    parser.add_argument("--unlearn-lr", type=float, default=0.013)
    parser.add_argument("--mask-ratio", type=str, default="0.5")

    parser.add_argument("--original-save-dir", type=str, default="save/weights/trained/salun_official_original")
    parser.add_argument("--mask-dir", type=str, default="save/masks/salun_official_random4500")
    parser.add_argument("--unlearn-save-dir", type=str, default="save/weights/unlearned/salun_official_random4500")
    parser.add_argument("--manifest-path", type=str, default="save/manifests/salun_official_random_4500.json")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--mask-path", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _abs_path(path_value: str) -> Path:
    """
    Resolve a path relative to the project root.

    Args:
        path_value: Absolute path or project-relative path.

    Returns:
        Absolute path.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    """
    Print and optionally execute a subprocess command.

    Args:
        cmd: Command tokens.
        cwd: Working directory.
        dry_run: If true, only print the command.
    """
    print("[SALUN_RUNNER] cwd:", cwd)
    print("[SALUN_RUNNER] cmd:", " ".join(str(token) for token in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _common_salun_args(args: argparse.Namespace) -> List[str]:
    """
    Build common official SalUn CLI arguments.

    Args:
        args: Runner args.

    Returns:
        CLI token list.
    """
    return [
        "--arch",
        args.arch,
        "--dataset",
        args.dataset,
        "--data",
        str(_abs_path(args.data_path)),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--train_seed",
        str(args.train_seed),
        "--batch_size",
        str(args.batch_size),
        "--workers",
        str(args.num_workers),
        "--momentum",
        str(args.momentum),
        "--weight_decay",
        str(args.weight_decay),
    ]


def _manifest_cmd(args: argparse.Namespace) -> List[str]:
    """
    Build manifest export command.

    Args:
        args: Runner args.

    Returns:
        Command token list.
    """
    cmd = [
        args.python,
        str(PROJECT_ROOT / "tools" / "export_salun_official_manifest.py"),
        "--dataset",
        args.dataset,
        "--data-path",
        str(_abs_path(args.data_path)),
        "--seed",
        str(args.seed),
        "--num-indexes-to-replace",
        str(args.num_indexes_to_replace),
        "--output",
        str(_abs_path(args.manifest_path)),
    ]
    if args.allow_download:
        cmd.append("--allow-download")
    return cmd


def _train_cmd(args: argparse.Namespace) -> List[str]:
    """
    Build official original-training command.

    Args:
        args: Runner args.

    Returns:
        Command token list.
    """
    return [
        args.python,
        "main_train.py",
        *_common_salun_args(args),
        "--save_dir",
        str(_abs_path(args.original_save_dir)),
        "--epochs",
        str(args.train_epochs),
        "--lr",
        str(args.train_lr),
        "--decreasing_lr",
        args.decreasing_lr,
    ]


def _mask_cmd(args: argparse.Namespace) -> List[str]:
    """
    Build official saliency-mask generation command.

    Args:
        args: Runner args.

    Returns:
        Command token list.
    """
    model_path = _resolve_model_path(args)
    return [
        args.python,
        "generate_mask.py",
        *_common_salun_args(args),
        "--save_dir",
        str(_abs_path(args.mask_dir)),
        "--model_path",
        str(model_path),
        "--num_indexes_to_replace",
        str(args.num_indexes_to_replace),
        "--unlearn_epochs",
        "1",
        "--unlearn_lr",
        str(args.unlearn_lr),
    ]


def _unlearn_cmd(args: argparse.Namespace) -> List[str]:
    """
    Build official SalUn random-unlearning command.

    Args:
        args: Runner args.

    Returns:
        Command token list.
    """
    return [
        args.python,
        "main_random.py",
        *_common_salun_args(args),
        "--save_dir",
        str(_abs_path(args.unlearn_save_dir)),
        "--model_path",
        str(_resolve_model_path(args)),
        "--mask_path",
        str(_resolve_mask_path(args)),
        "--unlearn",
        args.unlearn_method,
        "--unlearn_epochs",
        str(args.unlearn_epochs),
        "--unlearn_lr",
        str(args.unlearn_lr),
        "--num_indexes_to_replace",
        str(args.num_indexes_to_replace),
    ]


def _resolve_model_path(args: argparse.Namespace) -> Path:
    """
    Resolve official original checkpoint path.

    Args:
        args: Runner args.

    Returns:
        Checkpoint path.
    """
    if args.model_path:
        return _abs_path(args.model_path)
    return _abs_path(args.original_save_dir) / "0checkpoint.pth.tar"


def _resolve_mask_path(args: argparse.Namespace) -> Path:
    """
    Resolve official SalUn mask path.

    Args:
        args: Runner args.

    Returns:
        Mask path.
    """
    if args.mask_path:
        return _abs_path(args.mask_path)
    return _abs_path(args.mask_dir) / f"with_{args.mask_ratio}.pt"


def main() -> None:
    """Run selected official SalUn workflow stages."""
    args = _build_args()
    if not SALUN_CLASSIFICATION_DIR.exists():
        raise FileNotFoundError(f"Official SalUn Classification directory not found: {SALUN_CLASSIFICATION_DIR}")

    _abs_path(args.original_save_dir).mkdir(parents=True, exist_ok=True)
    _abs_path(args.mask_dir).mkdir(parents=True, exist_ok=True)
    _abs_path(args.unlearn_save_dir).mkdir(parents=True, exist_ok=True)
    _abs_path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)

    stages = ["manifest", "train", "mask", "unlearn"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "manifest":
            _run(_manifest_cmd(args), cwd=PROJECT_ROOT, dry_run=args.dry_run)
        elif stage == "train":
            _run(_train_cmd(args), cwd=SALUN_CLASSIFICATION_DIR, dry_run=args.dry_run)
        elif stage == "mask":
            _run(_mask_cmd(args), cwd=SALUN_CLASSIFICATION_DIR, dry_run=args.dry_run)
        elif stage == "unlearn":
            _run(_unlearn_cmd(args), cwd=SALUN_CLASSIFICATION_DIR, dry_run=args.dry_run)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    print("[SALUN_RUNNER] Finished.")
    print(f"[SALUN_RUNNER] manifest: {_abs_path(args.manifest_path)}")
    print(f"[SALUN_RUNNER] original checkpoint: {_resolve_model_path(args)}")
    print(f"[SALUN_RUNNER] mask: {_resolve_mask_path(args)}")
    print(f"[SALUN_RUNNER] unlearned checkpoint: {_abs_path(args.unlearn_save_dir) / (args.unlearn_method + 'checkpoint.pth.tar')}")


if __name__ == "__main__":
    main()
