"""Naive retraining baseline for machine unlearning.

This module implements the simplest unlearning method:
remove D_u and retrain a fresh model on D_r.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import SGD, AdamW

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from unlearning.base_unlearner import BaseUnlearner
from utils.config import load_config


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
        ckpt_name = str(
            self.config.get(
                "retrain_checkpoint_name",
                f"retrain_{self.config.get('model_name', 'resnet18')}_{dataset.dataset_name}.pt",
            )
        )
        save_path = save_dir / ckpt_name
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": "retrain",
                "model_name": self.config.get("model_name", "resnet18"),
                "dataset": dataset.dataset_name,
            },
            str(save_path),
        )
        print(f"[RETRAIN] Saved unlearned model to: {save_path}")
        print("[RETRAIN] ===== Retraining Unlearning Finished =====")

        return {"status": "ok", "method": "retrain", "model": model, "save_path": str(save_path), "history": history}


def main() -> None:
    """Standalone CLI entry for retraining unlearning."""
    base_cfg = load_config("configs/config.yaml")
    cfg = dict(base_cfg or {})
    dataset = UnlearningDataset(cfg)
    unlearner = RetrainUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
