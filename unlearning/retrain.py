"""Naive retraining baseline for machine unlearning.

This module implements the simplest unlearning method:
remove D_u and retrain a fresh model on D_r.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

# Allow running this script via absolute/relative file path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import SGD, AdamW

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from unlearning.base_unlearner import BaseUnlearner
from utils.config import load_config


def _parse_forget_classes(raw: str):
    """
    Parse forget classes from CLI string.

    Args:
        raw: Comma-separated class ids, e.g. "3,5,7".

    Returns:
        List of integer class ids.
    """
    text = str(raw).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build CLI arguments for retraining unlearning.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Naive retraining unlearning")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["random", "by_class"],
        help="Unlearning split mode.",
    )
    parser.add_argument("--forget-ratio", type=float, default=0.1)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--forget-manifest-path", type=str, default="save/manifests/retrain_forget_manifest.json")
    parser.add_argument("--forget-manifest-mode", type=str, default="auto", choices=["auto", "load", "save", "off"])
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--unlearned-path", type=str, default="save/weights/unlearned")
    parser.add_argument("--save-name", type=str, default=None)
    return parser.parse_args()


def _merge_config(base_cfg: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge CLI overrides into base config.

    Args:
        base_cfg: Base config loaded from file.
        args: Parsed CLI args.

    Returns:
        Final config dictionary.
    """
    cfg = dict(base_cfg or {})

    def set_if_not_none(key, value):
        if value is not None:
            cfg[key] = value

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
    set_if_not_none("split_seed", args.split_seed)
    set_if_not_none("forget_manifest_path", args.forget_manifest_path)
    set_if_not_none("forget_manifest_mode", args.forget_manifest_mode)
    if args.forget_classes.strip():
        cfg["forget_classes"] = _parse_forget_classes(args.forget_classes)

    set_if_not_none("retrain_epochs", args.epochs)
    set_if_not_none("retrain_lr", args.lr)
    set_if_not_none("retrain_weight_decay", args.weight_decay)
    set_if_not_none("retrain_momentum", args.momentum)
    set_if_not_none("retrain_optimizer", args.optimizer)

    if args.allow_download:
        cfg["allow_download"] = True
    if args.unlearned_path is not None:
        cfg["unlearned_weights_path"] = args.unlearned_path
    if args.save_name is not None:
        cfg["retrain_checkpoint_name"] = args.save_name
    return cfg


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RetrainStats:
    """Per-epoch retraining metrics."""

    train_loss: float
    train_acc: float
    retain_acc: float
    test_acc: float


class RetrainUnlearner(BaseUnlearner):
    """Naive retraining unlearner using D_r only."""

    def __init__(self, config: Dict):
        """
        Initialize retraining method.

        Args:
            config: Runtime config.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.epochs = int(self.config.get("retrain_epochs", self.config.get("epochs", 40)))
        self.lr = float(self.config.get("retrain_lr", self.config.get("lr", 0.01)))
        self.weight_decay = float(
            self.config.get("retrain_weight_decay", self.config.get("weight_decay", 5e-4))
        )
        self.momentum = float(self.config.get("retrain_momentum", self.config.get("momentum", 0.9)))
        self.optimizer_name = str(
            self.config.get("retrain_optimizer", self.config.get("optimizer", "sgd"))
        ).lower()

    def _infer_unlearn_target_tag(self) -> str:
        """
        Build a concise tag describing unlearning target definition.

        Returns:
            Target tag string for checkpoint naming.
        """
        split_mode = str(self.config.get("split_mode", "random")).lower()
        if split_mode == "by_class":
            classes = self.config.get("forget_classes", [])
            if isinstance(classes, (int, float, str)):
                classes = [classes]
            classes = [str(int(c)) for c in classes]
            if classes:
                return "byclass_" + "-".join(classes)
            return "byclass_unspecified"

        if self.config.get("forget_count") is not None:
            return f"random_count{int(self.config.get('forget_count'))}"

        ratio = float(self.config.get("forget_ratio", 0.1))
        ratio_str = str(ratio).replace(".", "p")
        return f"random_ratio{ratio_str}"

    def _build_fresh_model(self, dataset: UnlearningDataset) -> nn.Module:
        """
        Build a fresh model for retraining from scratch.

        Args:
            dataset: Dataset manager.

        Returns:
            Fresh model.
        """
        model_name = str(self.config.get("model_name", self.config.get("model", "resnet18")))
        model, _ = construct_model(
            model=model_name,
            num_classes=int(self.config.get("num_classes", dataset.num_classes)),
            seed=int(self.config.get("seed", 42)),
            num_channels=int(self.config.get("in_channels", 3)),
        )
        return model.to(self.device)

    def _build_optimizer(self, model: nn.Module):
        """
        Build optimizer from config.

        Args:
            model: Model to optimize.

        Returns:
            Optimizer instance.
        """
        if self.optimizer_name == "adamw":
            return AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _evaluate_acc(self, model: nn.Module, loader) -> float:
        """
        Evaluate accuracy on a dataloader.

        Args:
            model: Model instance.
            loader: Dataloader.

        Returns:
            Accuracy value.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        return correct / max(total, 1)

    def unlearn(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> Dict:
        """
        Execute naive retraining unlearning.

        Args:
            model: Unused. Kept for interface compatibility.
            dataset: Dataset manager with D_u, D_r, D_test.

        Returns:
            Dict containing unlearned model and training history.
        """
        del model  # Retraining baseline starts from scratch.
        set_seed(int(self.config.get("seed", 42)))
        loaders = dataset.get_dataloaders(retained_shuffle=True)
        retain_loader = loaders["d_r"]
        test_loader = loaders["d_test"]

        model = self._build_fresh_model(dataset)
        optimizer = self._build_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        print("[RETRAIN] ===== Start Retraining Unlearning =====")
        print(
            f"[RETRAIN] setup: model={self.config.get('model_name', 'resnet18')}, "
            f"dataset={dataset.dataset_name}, device={self.device}"
        )
        print(
            f"[RETRAIN] sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )
        print(
            f"[RETRAIN] hyper: optimizer={self.optimizer_name}, lr={self.lr}, "
            f"weight_decay={self.weight_decay}, epochs={self.epochs}"
        )

        history = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for x, y in retain_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * y.size(0)
                epoch_correct += int((logits.argmax(dim=1) == y).sum().item())
                epoch_total += int(y.size(0))

            train_loss = epoch_loss / max(epoch_total, 1)
            train_acc = epoch_correct / max(epoch_total, 1)
            retain_acc = self._evaluate_acc(model, retain_loader)
            test_acc = self._evaluate_acc(model, test_loader)

            stats = RetrainStats(
                train_loss=train_loss,
                train_acc=train_acc,
                retain_acc=retain_acc,
                test_acc=test_acc,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": stats.train_loss,
                    "train_acc": stats.train_acc,
                    "retain_acc": stats.retain_acc,
                    "test_acc": stats.test_acc,
                }
            )
            print(
                f"[RETRAIN][Epoch {epoch:03d}/{self.epochs:03d}] "
                f"train_loss={stats.train_loss:.4f} train_acc={stats.train_acc:.4f} "
                f"retain_acc={stats.retain_acc:.4f} test_acc={stats.test_acc:.4f}"
            )

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        target_tag = self._infer_unlearn_target_tag()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_ckpt_name = f"retrain_{model_name}_{dataset.dataset_name}_{target_tag}_{ts}.pt"
        ckpt_name = str(
            self.config.get(
                "retrain_checkpoint_name",
                default_ckpt_name,
            )
        )
        save_path = save_dir / ckpt_name
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": "retrain",
                "model_name": model_name,
                "dataset": dataset.dataset_name,
                "split_mode": self.config.get("split_mode", "random"),
                "forget_classes": self.config.get("forget_classes", []),
                "forget_ratio": self.config.get("forget_ratio", None),
                "forget_count": self.config.get("forget_count", None),
                "forget_manifest_path": (
                    dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
                ),
                "forget_manifest": (
                    dataset.get_forget_manifest_info() if hasattr(dataset, "get_forget_manifest_info") else None
                ),
                "seed": int(self.config.get("seed", 42)),
            },
            str(save_path),
        )
        print(f"[RETRAIN] Saved unlearned model to: {save_path}")
        print("[RETRAIN] ===== Retraining Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "retrain",
            "model": model,
            "save_path": str(save_path),
            "history": history,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
            "forget_manifest": (
                dataset.get_forget_manifest_info() if hasattr(dataset, "get_forget_manifest_info") else None
            ),
        }


def main() -> None:
    """Standalone CLI entry for retraining unlearning."""
    args = _build_args()
    base_cfg = load_config(args.config)
    cfg = _merge_config(base_cfg, args)
    dataset = UnlearningDataset(cfg)
    unlearner = RetrainUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
