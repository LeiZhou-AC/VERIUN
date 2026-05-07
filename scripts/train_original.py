"""Train an original model for unlearning experiments.

This script trains a model on D_all and saves the trained checkpoint.
Model and dataset are configurable via CLI and/or config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
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


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train original model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str, default="datasets")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-dir", type=str, default="save/weights/trained")
    parser.add_argument("--save-name", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _build_config(base_cfg: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge CLI arguments into base config.

    Args:
        base_cfg: Base config dict.
        args: CLI args.

    Returns:
        Final config dict.
    """
    cfg = dict(base_cfg or {})
    cfg.update(
        {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "seed": args.seed,
            "device": args.device,
            "data_path": args.data_path,
            "allow_download": bool(args.allow_download),
            "num_workers": args.num_workers,
            "split_mode": cfg.get("split_mode", "random"),
        }
    )
    return cfg


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for a mini-batch.

    Args:
        logits: Model logits.
        targets: Ground-truth labels.

    Returns:
        Accuracy as float.
    """
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.

    Args:
        model: Model instance.
        loader: Dataloader.
        device: Device.

    Returns:
        Dict of loss and accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += int(y.size(0))

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return {"loss": avg_loss, "acc": acc}


def train_original_model(config: Dict) -> Dict:
    """
    Train an original model on D_all and save checkpoint.

    Args:
        config: Runtime config.

    Returns:
        Summary dict including save path and best test accuracy.
    """
    set_seed(int(config.get("seed", 42)))
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    dataset = UnlearningDataset(config)
    loaders = dataset.get_dataloaders(retained_shuffle=True)
    train_loader = loaders["d_all"]
    test_loader = loaders["d_test"]

    model_name = str(config.get("model_name", "resnet18"))
    model, init_seed = construct_model(
        model=model_name,
        num_classes=int(config.get("num_classes", dataset.num_classes)),
        seed=int(config.get("seed", 42)),
        num_channels=int(config.get("in_channels", 3)),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if str(config.get("optimizer", "sgd")).lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=float(config.get("lr", 0.01)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )

    epochs = int(config.get("epochs", 40))
    best_acc = 0.0
    print(
        f"[TRAIN] Start: model={model_name}, dataset={dataset.dataset_name}, "
        f"device={device}, epochs={epochs}, init_seed={init_seed}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_acc += _accuracy(logits, y)
            steps += 1

        train_loss = epoch_loss / max(steps, 1)
        train_acc = epoch_acc / max(steps, 1)
        test_metrics = _evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_metrics["acc"])
        print(
            f"[TRAIN][Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}"
        )

    save_dir = Path(str(config.get("save_dir", "save/weights/trained")))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = str(config.get("save_name", "")).strip()
    if not save_name:
        save_name = f"{model_name}_{dataset.dataset_name}.pt"
    save_path = save_dir / save_name

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "dataset": dataset.dataset_name,
        "num_classes": int(config.get("num_classes", dataset.num_classes)),
        "seed": int(config.get("seed", 42)),
    }
    torch.save(checkpoint, str(save_path))
    print(f"[TRAIN] Saved trained model to: {save_path}")
    print(f"[TRAIN] Best test accuracy: {best_acc:.4f}")
    return {"status": "ok", "save_path": str(save_path), "best_test_acc": best_acc}


def main() -> None:
    """CLI entry for original model training."""
    args = _parse_args()
    base_cfg = load_config(args.config)
    cfg = _build_config(base_cfg, args)
    cfg["save_dir"] = args.save_dir
    cfg["save_name"] = args.save_name
    train_original_model(cfg)


if __name__ == "__main__":
    main()
