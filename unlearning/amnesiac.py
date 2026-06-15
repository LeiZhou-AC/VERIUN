"""Amnesiac relabeling baseline for approximate machine unlearning.

This module implements a practical variant of Amnesiac Machine Learning for
post-hoc experiments where the original batch-update logs are unavailable. The
forget set is relabeled with deterministic wrong labels and the trained model is
continued on relabeled D_u plus a controlled amount of retained data D_r.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from unlearning.base_unlearner import BaseUnlearner
from utils.config import load_config


def _parse_forget_classes(raw: str):
    """
    Parse forget classes from a comma-separated CLI string.

    Args:
        raw: Text such as "0,3,5".

    Returns:
        List of integer class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build command-line arguments for Amnesiac relabeling.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Amnesiac relabel approximate unlearning")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
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
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--relabel-weight", type=float, default=None)
    parser.add_argument("--retain-weight", type=float, default=None)
    parser.add_argument("--max-relabel-batches", type=int, default=None)
    parser.add_argument("--max-retain-batches", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--train-scope", type=str, default=None, choices=["full", "backbone", "head"])
    parser.add_argument("--label-seed", type=int, default=None)
    return parser.parse_args()


def _merge_config(base_cfg: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge CLI overrides into a base configuration.

    Args:
        base_cfg: Config loaded from YAML.
        args: Parsed CLI arguments.

    Returns:
        Runtime configuration.
    """
    cfg = dict(base_cfg or {})

    def set_from_arg(key: str, value, fallback=None) -> None:
        if value is not None:
            cfg[key] = value
        elif fallback is not None and key not in cfg:
            cfg[key] = fallback

    set_from_arg("dataset", args.dataset, "cifar10")
    set_from_arg("model_name", args.model_name, "resnet18")
    set_from_arg("num_classes", args.num_classes, 10)
    set_from_arg("in_channels", args.in_channels, 3)
    set_from_arg("data_path", args.data_path, "datasets")
    set_from_arg("batch_size", args.batch_size, 64)
    set_from_arg("num_workers", args.num_workers, 4)
    set_from_arg("seed", args.seed, 42)
    set_from_arg("device", args.device, "cuda")

    set_from_arg("split_mode", args.split_mode, "random")
    set_from_arg("forget_ratio", args.forget_ratio, 0.01)
    set_from_arg("forget_count", args.forget_count)
    set_from_arg("split_seed", args.split_seed, 42)
    set_from_arg("forget_manifest_path", args.forget_manifest_path, "save/manifests/default_forget_manifest.json")
    set_from_arg("forget_manifest_mode", args.forget_manifest_mode, "load")
    if args.forget_classes.strip():
        cfg["forget_classes"] = _parse_forget_classes(args.forget_classes)

    set_from_arg("trained_weights_path", args.trained_path, "save/weights/trained")
    set_from_arg("unlearned_weights_path", args.unlearned_path, "save/weights/unlearned")
    if args.save_name:
        cfg["amnesiac_checkpoint_name"] = args.save_name
    if args.allow_download:
        cfg["allow_download"] = True

    set_from_arg("amnesiac_epochs", args.epochs)
    set_from_arg("amnesiac_lr", args.lr)
    set_from_arg("amnesiac_optimizer", args.optimizer)
    set_from_arg("amnesiac_momentum", args.momentum)
    set_from_arg("amnesiac_weight_decay", args.weight_decay)
    set_from_arg("amnesiac_relabel_weight", args.relabel_weight)
    set_from_arg("amnesiac_retain_weight", args.retain_weight)
    set_from_arg("amnesiac_max_relabel_batches", args.max_relabel_batches)
    set_from_arg("amnesiac_max_retain_batches", args.max_retain_batches)
    set_from_arg("amnesiac_grad_clip", args.grad_clip)
    set_from_arg("amnesiac_validate_every", args.validate_every)
    set_from_arg("amnesiac_train_scope", args.train_scope)
    set_from_arg("amnesiac_label_seed", args.label_seed)
    return cfg


def set_seed(seed: int) -> None:
    """
    Set deterministic seeds for Amnesiac experiments.

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


class WrongLabelDataset(Dataset):
    """Dataset wrapper that replaces each forget-set label with a wrong label."""

    def __init__(self, base_dataset: Dataset, num_classes: int, seed: int):
        """
        Initialize deterministic wrong labels for D_u.

        Args:
            base_dataset: Original forget subset.
            num_classes: Number of classes.
            seed: Seed controlling wrong-label assignment.
        """
        self.base_dataset = base_dataset
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("Amnesiac relabeling requires at least two classes.")

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.wrong_labels = []
        self.true_labels = []
        for idx in range(len(self.base_dataset)):
            item = self.base_dataset[idx]
            label = self._extract_label(item)
            offset = int(torch.randint(1, self.num_classes, (1,), generator=generator).item())
            wrong_label = int((label + offset) % self.num_classes)
            self.true_labels.append(int(label))
            self.wrong_labels.append(wrong_label)

    def __len__(self) -> int:
        """
        Return dataset size.

        Returns:
            Number of samples.
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        """
        Return input, wrong label, and original label.

        Args:
            index: Sample index.

        Returns:
            Tuple of image, wrong label, original label.
        """
        item = self.base_dataset[index]
        x = item[0]
        return x, self.wrong_labels[index], self.true_labels[index]

    @staticmethod
    def _extract_label(item) -> int:
        """
        Extract an integer class label from a dataset item.

        Args:
            item: Dataset item.

        Returns:
            Integer label.
        """
        label = item[1]
        if torch.is_tensor(label):
            return int(label.item())
        return int(label)


@dataclass
class AmnesiacEpochStats:
    """Per-epoch Amnesiac relabeling metrics."""

    relabel_ce: float
    retain_ce: float
    forget_original_accuracy: float
    forget_wrong_label_accuracy: float
    forget_confidence: float
    retain_accuracy: float
    test_accuracy: float


class AmnesiacUnlearner(BaseUnlearner):
    """Approximate unlearner based on deterministic wrong-label updates."""

    def __init__(self, config: Dict):
        """
        Initialize Amnesiac relabeling from runtime configuration.

        Args:
            config: Experiment configuration.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.epochs = int(self.config.get("amnesiac_epochs", self.config.get("epochs", 10)))
        self.lr = float(self.config.get("amnesiac_lr", self.config.get("lr", 1e-4)))
        self.optimizer_name = str(self.config.get("amnesiac_optimizer", "adamw")).lower()
        self.momentum = float(self.config.get("amnesiac_momentum", self.config.get("momentum", 0.9)))
        self.weight_decay = float(self.config.get("amnesiac_weight_decay", 5e-4))
        self.relabel_weight = float(self.config.get("amnesiac_relabel_weight", 1.0))
        self.retain_weight = float(self.config.get("amnesiac_retain_weight", 0.5))
        self.max_relabel_batches = int(self.config.get("amnesiac_max_relabel_batches", 0))
        self.max_retain_batches = int(self.config.get("amnesiac_max_retain_batches", 64))
        self.grad_clip = float(self.config.get("amnesiac_grad_clip", 5.0))
        self.validate_every = int(self.config.get("amnesiac_validate_every", 1))
        self.train_scope = str(self.config.get("amnesiac_train_scope", "full")).lower()
        self.label_seed = int(self.config.get("amnesiac_label_seed", self.config.get("split_seed", 42)))

    def _infer_target_tag(self) -> str:
        """
        Build a compact target tag for checkpoint names.

        Returns:
            Target tag string.
        """
        split_mode = str(self.config.get("split_mode", "random")).lower()
        if split_mode in {"by_class", "class_random"}:
            classes = self.config.get("forget_classes", [])
            if isinstance(classes, (int, float, str)):
                classes = [classes]
            classes = [str(int(c)) for c in classes]
            if classes:
                prefix = "byclass" if split_mode == "by_class" else "classrandom"
                suffix = "-".join(classes)
                if split_mode == "class_random":
                    if self.config.get("forget_count") is not None:
                        return f"{prefix}_{suffix}_count{int(self.config.get('forget_count'))}"
                    ratio = str(float(self.config.get("forget_ratio", 0.01))).replace(".", "p")
                    return f"{prefix}_{suffix}_ratio{ratio}"
                return f"{prefix}_{suffix}"
            return f"{split_mode}_unspecified"

        if self.config.get("forget_count") is not None:
            return f"random_count{int(self.config.get('forget_count'))}"

        ratio = float(self.config.get("forget_ratio", 0.01))
        return f"random_ratio{str(ratio).replace('.', 'p')}"

    def _resolve_latest_checkpoint(self, root: Path) -> Path:
        """
        Resolve a checkpoint from a file or directory path.

        Args:
            root: Checkpoint file or directory.

        Returns:
            Resolved checkpoint file.
        """
        if root.is_file():
            return root
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
        candidates = list(root.glob("*.pt")) + list(root.glob("*.pth")) + list(root.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint file found under: {root}")
        return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]

    def _extract_state_dict(self, state_obj):
        """
        Extract a clean model state dict from common checkpoint formats.

        Args:
            state_obj: Raw object loaded by torch.load.

        Returns:
            Cleaned state dict.
        """
        if isinstance(state_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_obj and isinstance(state_obj[key], dict):
                    state_obj = state_obj[key]
                    break
        if not isinstance(state_obj, dict):
            raise ValueError("Unsupported checkpoint format: expected dict-like state dict.")

        cleaned = {}
        for key, value in state_obj.items():
            cleaned[key[len("module."):] if key.startswith("module.") else key] = value
        return cleaned

    def _build_model_from_config(self, dataset: UnlearningDataset) -> nn.Module:
        """
        Build a model instance matching the configured experiment.

        Args:
            dataset: Dataset manager.

        Returns:
            Initialized model.
        """
        model_name = str(self.config.get("model_name", self.config.get("model", "resnet18")))
        model, _ = construct_model(
            model=model_name,
            num_classes=int(self.config.get("num_classes", dataset.num_classes)),
            seed=int(self.config.get("seed", 42)),
            num_channels=int(self.config.get("in_channels", 3)),
        )
        return model

    def _load_original_model(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> nn.Module:
        """
        Load the original trained model.

        Args:
            model: Optional pre-loaded original model.
            dataset: Dataset manager.

        Returns:
            Original model on the target device.
        """
        if model is not None:
            return copy.deepcopy(model).to(self.device)

        checkpoint_root = Path(str(self.config.get("trained_weights_path", "save/weights/trained")))
        checkpoint_path = self._resolve_latest_checkpoint(checkpoint_root)
        loaded = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = self._extract_state_dict(loaded)

        model = self._build_model_from_config(dataset)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        print(f"[AMNESIAC] Loaded trained checkpoint: {checkpoint_path}")
        if missing:
            print(f"[AMNESIAC][Load] Missing keys count: {len(missing)}")
        if unexpected:
            print(f"[AMNESIAC][Load] Unexpected keys count: {len(unexpected)}")
        return model

    def _set_train_scope(self, model: nn.Module) -> None:
        """
        Select which model parameters are trainable.

        Args:
            model: Model to update.
        """
        for param in model.parameters():
            param.requires_grad = self.train_scope == "full"

        if self.train_scope == "head":
            modules = []
            for name in ("classifier", "fc", "head"):
                if hasattr(model, name):
                    modules.append(getattr(model, name))
            if not modules:
                raise RuntimeError("amnesiac_train_scope='head' but no classifier/head module was found.")
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.train_scope == "backbone":
            modules = []
            for name in ("backbone", "features"):
                if hasattr(model, name):
                    modules.append(getattr(model, name))
            if not modules:
                raise RuntimeError("amnesiac_train_scope='backbone' but no backbone/features module was found.")
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.train_scope != "full":
            raise ValueError(f"Unsupported Amnesiac train scope: {self.train_scope}")

    def _build_optimizer(self, model: nn.Module):
        """
        Build an optimizer over trainable parameters.

        Args:
            model: Model to update.

        Returns:
            Optimizer instance.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found for Amnesiac.")
        if self.optimizer_name == "sgd":
            return SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _maybe_clip_gradients(self, model: nn.Module) -> None:
        """
        Clip gradients when configured.

        Args:
            model: Model with gradients.
        """
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=self.grad_clip,
            )

    def _build_wrong_label_loader(self, dataset: UnlearningDataset) -> DataLoader:
        """
        Build a forget-set dataloader with deterministic wrong labels.

        Args:
            dataset: Dataset manager.

        Returns:
            Dataloader over relabeled D_u.
        """
        wrong_dataset = WrongLabelDataset(
            dataset.get_unlearning_set(),
            num_classes=int(self.config.get("num_classes", dataset.num_classes)),
            seed=self.label_seed,
        )
        return DataLoader(
            wrong_dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 4)),
            pin_memory=bool(self.config.get("pin_memory", torch.cuda.is_available())),
        )

    def _run_relabel_epoch(self, model: nn.Module, loader: DataLoader, optimizer) -> Dict[str, float]:
        """
        Run one wrong-label update phase on D_u.

        Args:
            model: Model to update.
            loader: Wrong-label D_u dataloader.
            optimizer: Optimizer.

        Returns:
            Forget-phase metrics.
        """
        model.train()
        total_ce = 0.0
        total_conf = 0.0
        original_correct = 0
        wrong_correct = 0
        total = 0
        steps = 0

        for batch_idx, (x, wrong_y, true_y) in enumerate(loader, start=1):
            if self.max_relabel_batches > 0 and batch_idx > self.max_relabel_batches:
                break

            x = x.to(self.device)
            wrong_y = wrong_y.to(self.device)
            true_y = true_y.to(self.device)

            optimizer.zero_grad()
            logits = model(x)
            ce_loss = F.cross_entropy(logits, wrong_y)
            loss = self.relabel_weight * ce_loss
            loss.backward()
            self._maybe_clip_gradients(model)
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                batch_size = int(true_y.size(0))
                total_ce += float(ce_loss.item()) * batch_size
                total_conf += float(probs.max(dim=1).values.sum().item())
                original_correct += int((preds == true_y).sum().item())
                wrong_correct += int((preds == wrong_y).sum().item())
                total += batch_size
                steps += 1

        return {
            "relabel_ce": total_ce / max(total, 1),
            "forget_original_accuracy": original_correct / max(total, 1),
            "forget_wrong_label_accuracy": wrong_correct / max(total, 1),
            "forget_confidence": total_conf / max(total, 1),
            "relabel_steps": float(steps),
        }

    def _run_retain_epoch(self, model: nn.Module, loader: DataLoader, optimizer) -> Dict[str, float]:
        """
        Run one utility-preservation update phase on D_r.

        Args:
            model: Model to update.
            loader: Retained-set dataloader.
            optimizer: Optimizer.

        Returns:
            Retain-phase metrics.
        """
        model.train()
        total_ce = 0.0
        correct = 0
        total = 0
        steps = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            if self.max_retain_batches > 0 and batch_idx > self.max_retain_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y)
            loss = self.retain_weight * ce_loss
            loss.backward()
            self._maybe_clip_gradients(model)
            optimizer.step()

            batch_size = int(y.size(0))
            total_ce += float(ce_loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size
            steps += 1

        return {
            "retain_ce": total_ce / max(total, 1),
            "retain_train_accuracy": correct / max(total, 1),
            "retain_steps": float(steps),
        }

    def _evaluate_split(self, model: nn.Module, loader: DataLoader, name: str) -> Dict[str, float]:
        """
        Evaluate accuracy and confidence on a standard split.

        Args:
            model: Model to evaluate.
            loader: Evaluation dataloader.
            name: Split name for logging.

        Returns:
            Metrics dictionary.
        """
        model.eval()
        total_loss = 0.0
        total_conf = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                total_loss += float(F.cross_entropy(logits, y).item()) * y.size(0)
                total_conf += float(probs.max(dim=1).values.sum().item())
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += int(y.size(0))

        metrics = {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "mean_confidence": total_conf / max(total, 1),
            "correct": correct,
            "total": total,
        }
        print(
            f"[AMNESIAC][Eval] {name}: acc={metrics['accuracy']:.4f} "
            f"loss={metrics['loss']:.4f} conf={metrics['mean_confidence']:.4f} "
            f"({correct}/{total})"
        )
        return metrics

    def _evaluate_wrong_label_split(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate D_u against both original and wrong labels.

        Args:
            model: Model to evaluate.
            loader: Wrong-label D_u dataloader.

        Returns:
            Metrics dictionary.
        """
        model.eval()
        total_ce = 0.0
        total_conf = 0.0
        original_correct = 0
        wrong_correct = 0
        total = 0
        with torch.no_grad():
            for x, wrong_y, true_y in loader:
                x = x.to(self.device)
                wrong_y = wrong_y.to(self.device)
                true_y = true_y.to(self.device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                batch_size = int(true_y.size(0))
                total_ce += float(F.cross_entropy(logits, wrong_y).item()) * batch_size
                total_conf += float(probs.max(dim=1).values.sum().item())
                original_correct += int((preds == true_y).sum().item())
                wrong_correct += int((preds == wrong_y).sum().item())
                total += batch_size

        metrics = {
            "wrong_label_loss": total_ce / max(total, 1),
            "original_accuracy": original_correct / max(total, 1),
            "wrong_label_accuracy": wrong_correct / max(total, 1),
            "mean_confidence": total_conf / max(total, 1),
            "original_correct": original_correct,
            "wrong_correct": wrong_correct,
            "total": total,
        }
        print(
            "[AMNESIAC][Eval] forget: "
            f"orig_acc={metrics['original_accuracy']:.4f} "
            f"wrong_acc={metrics['wrong_label_accuracy']:.4f} "
            f"wrong_loss={metrics['wrong_label_loss']:.4f} "
            f"conf={metrics['mean_confidence']:.4f} "
            f"(orig {original_correct}/{total}, wrong {wrong_correct}/{total})"
        )
        return metrics

    def unlearn(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> Dict:
        """
        Execute Amnesiac relabel approximate unlearning.

        Args:
            model: Optional original trained model.
            dataset: Dataset manager exposing D_u, D_r, and D_test.

        Returns:
            Summary dictionary containing the updated model and save path.
        """
        set_seed(int(self.config.get("seed", 42)))
        loaders = dataset.get_dataloaders(retained_shuffle=True)
        relabel_loader = self._build_wrong_label_loader(dataset)
        retain_loader = loaders["d_r"]
        test_loader = loaders["d_test"]

        student = self._load_original_model(model, dataset)
        self._set_train_scope(student)
        optimizer = self._build_optimizer(student)
        trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)

        print("[AMNESIAC] ===== Start Amnesiac Relabel Unlearning =====")
        print(
            f"[AMNESIAC] setup: model={self.config.get('model_name', 'resnet18')}, "
            f"dataset={dataset.dataset_name}, device={self.device}, train_scope={self.train_scope}"
        )
        print(
            f"[AMNESIAC] sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )
        print(
            f"[AMNESIAC] hyper: optimizer={self.optimizer_name}, lr={self.lr}, "
            f"epochs={self.epochs}, relabel_w={self.relabel_weight}, retain_w={self.retain_weight}, "
            f"max_relabel_batches={self.max_relabel_batches}, max_retain_batches={self.max_retain_batches}, "
            f"weight_decay={self.weight_decay}, grad_clip={self.grad_clip}, label_seed={self.label_seed}"
        )
        print(f"[AMNESIAC] trainable parameters: {trainable_count}")

        history = []
        final_eval = {}
        for epoch in range(1, self.epochs + 1):
            relabel_stats = self._run_relabel_epoch(student, relabel_loader, optimizer)
            retain_stats = self._run_retain_epoch(student, retain_loader, optimizer)

            forget_original_accuracy = relabel_stats["forget_original_accuracy"]
            forget_wrong_label_accuracy = relabel_stats["forget_wrong_label_accuracy"]
            forget_confidence = relabel_stats["forget_confidence"]
            retain_accuracy = 0.0
            test_accuracy = 0.0
            if epoch % max(self.validate_every, 1) == 0:
                forget_eval = self._evaluate_wrong_label_split(student, relabel_loader)
                retain_eval = self._evaluate_split(student, retain_loader, "retain")
                test_eval = self._evaluate_split(student, test_loader, "test")
                forget_original_accuracy = forget_eval["original_accuracy"]
                forget_wrong_label_accuracy = forget_eval["wrong_label_accuracy"]
                forget_confidence = forget_eval["mean_confidence"]
                retain_accuracy = retain_eval["accuracy"]
                test_accuracy = test_eval["accuracy"]
                final_eval = {
                    "forget": forget_eval,
                    "retain": retain_eval,
                    "test": test_eval,
                }

            stats = AmnesiacEpochStats(
                relabel_ce=relabel_stats["relabel_ce"],
                retain_ce=retain_stats["retain_ce"],
                forget_original_accuracy=forget_original_accuracy,
                forget_wrong_label_accuracy=forget_wrong_label_accuracy,
                forget_confidence=forget_confidence,
                retain_accuracy=retain_accuracy,
                test_accuracy=test_accuracy,
            )
            epoch_record = {
                "epoch": epoch,
                "relabel_ce": stats.relabel_ce,
                "retain_ce": stats.retain_ce,
                "forget_original_accuracy": stats.forget_original_accuracy,
                "forget_wrong_label_accuracy": stats.forget_wrong_label_accuracy,
                "forget_confidence": stats.forget_confidence,
                "retain_accuracy": stats.retain_accuracy,
                "test_accuracy": stats.test_accuracy,
                "relabel_steps": relabel_stats["relabel_steps"],
                "retain_steps": retain_stats["retain_steps"],
            }
            history.append(epoch_record)
            print(
                f"[AMNESIAC][Epoch {epoch:03d}/{self.epochs:03d}] "
                f"relabel_ce={stats.relabel_ce:.6f} "
                f"retain_ce={stats.retain_ce:.6f} "
                f"forget_orig_acc={stats.forget_original_accuracy:.4f} "
                f"forget_wrong_acc={stats.forget_wrong_label_accuracy:.4f} "
                f"forget_conf={stats.forget_confidence:.4f} "
                f"retain_acc={stats.retain_accuracy:.4f} "
                f"test_acc={stats.test_accuracy:.4f} "
                f"steps=({int(relabel_stats['relabel_steps'])}/{int(retain_stats['retain_steps'])})"
            )

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        target_tag = self._infer_target_tag()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"amnesiac_{model_name}_{dataset.dataset_name}_{target_tag}_{timestamp}.pt"
        checkpoint_name = str(self.config.get("amnesiac_checkpoint_name", default_name))
        save_path = save_dir / checkpoint_name
        torch.save(
            {
                "state_dict": student.state_dict(),
                "method": "amnesiac_relabel",
                "model_name": model_name,
                "dataset": dataset.dataset_name,
                "num_classes": int(self.config.get("num_classes", dataset.num_classes)),
                "in_channels": int(self.config.get("in_channels", 3)),
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
                "amnesiac_config": {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "optimizer": self.optimizer_name,
                    "relabel_weight": self.relabel_weight,
                    "retain_weight": self.retain_weight,
                    "max_relabel_batches": self.max_relabel_batches,
                    "max_retain_batches": self.max_retain_batches,
                    "train_scope": self.train_scope,
                    "label_seed": self.label_seed,
                },
                "seed": int(self.config.get("seed", 42)),
            },
            str(save_path),
        )
        print(f"[AMNESIAC] Saved unlearned model to: {save_path}")
        print("[AMNESIAC] ===== Amnesiac Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "amnesiac_relabel",
            "model": student,
            "save_path": str(save_path),
            "history": history,
            "final_eval": final_eval,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
            "forget_manifest": (
                dataset.get_forget_manifest_info() if hasattr(dataset, "get_forget_manifest_info") else None
            ),
        }


def main() -> None:
    """Standalone CLI entry for Amnesiac relabel approximate unlearning."""
    args = _build_args()
    base_cfg = load_config(args.config)
    cfg = _merge_config(base_cfg, args)
    dataset = UnlearningDataset(cfg)
    unlearner = AmnesiacUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
