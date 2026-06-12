"""SCRUB-style approximate unlearning baseline.

This module implements a practical teacher-student approximation of SCRUB:
the original trained model is kept as a frozen teacher, while a student copy is
updated to increase loss on D_u, disagree with the teacher on D_u, and remain
useful on D_r.
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
from torch.utils.data import DataLoader

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model
from unlearning.base_unlearner import BaseUnlearner
from utils.config import load_config


def _parse_forget_classes(raw: str):
    """
    Parse forget classes from a CLI string.

    Args:
        raw: Comma-separated class ids.

    Returns:
        List of integer class ids.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_args() -> argparse.Namespace:
    """
    Build SCRUB CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="SCRUB approximate unlearning")
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

    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "by_class"])
    parser.add_argument("--forget-ratio", type=float, default=0.01)
    parser.add_argument("--forget-count", type=int, default=None)
    parser.add_argument("--forget-classes", type=str, default="")
    parser.add_argument("--forget-manifest-path", type=str, default="save/manifests/default_forget_manifest.json")
    parser.add_argument("--forget-manifest-mode", type=str, default="load", choices=["auto", "load", "save", "off"])
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--trained-path", type=str, default="save/weights/trained")
    parser.add_argument("--unlearned-path", type=str, default="save/weights/unlearned")
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--forget-ce-weight", type=float, default=1.0)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-ce-weight", type=float, default=1.0)
    parser.add_argument("--retain-kd-weight", type=float, default=1.0)
    parser.add_argument("--max-forget-batches", type=int, default=0)
    parser.add_argument("--max-retain-batches", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--train-scope", type=str, default="full", choices=["full", "backbone", "head"])
    return parser.parse_args()


def _merge_config(base_cfg: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge CLI overrides into a base config.

    Args:
        base_cfg: Config loaded from YAML.
        args: Parsed CLI arguments.

    Returns:
        Runtime config.
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

    cfg["trained_weights_path"] = args.trained_path
    cfg["unlearned_weights_path"] = args.unlearned_path
    if args.save_name:
        cfg["scrub_checkpoint_name"] = args.save_name
    if args.allow_download:
        cfg["allow_download"] = True

    set_if_not_none("scrub_epochs", args.epochs)
    set_if_not_none("scrub_lr", args.lr)
    set_if_not_none("scrub_optimizer", args.optimizer)
    set_if_not_none("scrub_momentum", args.momentum)
    set_if_not_none("scrub_weight_decay", args.weight_decay)
    set_if_not_none("scrub_temperature", args.temperature)
    set_if_not_none("scrub_forget_ce_weight", args.forget_ce_weight)
    set_if_not_none("scrub_forget_weight", args.forget_weight)
    set_if_not_none("scrub_retain_ce_weight", args.retain_ce_weight)
    set_if_not_none("scrub_retain_kd_weight", args.retain_kd_weight)
    set_if_not_none("scrub_max_forget_batches", args.max_forget_batches)
    set_if_not_none("scrub_max_retain_batches", args.max_retain_batches)
    set_if_not_none("scrub_grad_clip", args.grad_clip)
    set_if_not_none("scrub_validate_every", args.validate_every)
    set_if_not_none("scrub_train_scope", args.train_scope)
    return cfg


def set_seed(seed: int) -> None:
    """
    Set deterministic seeds for SCRUB experiments.

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
class ScrubEpochStats:
    """Per-epoch SCRUB metrics."""

    forget_ce: float
    forget_kl: float
    retain_ce: float
    retain_kd: float
    retain_total: float
    forget_accuracy: float
    forget_confidence: float
    retain_accuracy: float
    test_accuracy: float


class SCRUBUnlearner(BaseUnlearner):
    """Teacher-student SCRUB approximate unlearner."""

    def __init__(self, config: Dict):
        """
        Initialize SCRUB from runtime configuration.

        Args:
            config: Experiment config.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.epochs = int(self.config.get("scrub_epochs", self.config.get("epochs", 8)))
        self.lr = float(self.config.get("scrub_lr", self.config.get("lr", 1e-4)))
        self.optimizer_name = str(self.config.get("scrub_optimizer", "adamw")).lower()
        self.momentum = float(self.config.get("scrub_momentum", self.config.get("momentum", 0.9)))
        self.weight_decay = float(self.config.get("scrub_weight_decay", 0.0))
        self.temperature = float(self.config.get("scrub_temperature", 4.0))
        self.forget_ce_weight = float(self.config.get("scrub_forget_ce_weight", 1.0))
        self.forget_weight = float(self.config.get("scrub_forget_weight", 1.0))
        self.retain_ce_weight = float(self.config.get("scrub_retain_ce_weight", 1.0))
        self.retain_kd_weight = float(self.config.get("scrub_retain_kd_weight", 1.0))
        self.max_forget_batches = int(self.config.get("scrub_max_forget_batches", 0))
        self.max_retain_batches = int(self.config.get("scrub_max_retain_batches", 0))
        self.grad_clip = float(self.config.get("scrub_grad_clip", 5.0))
        self.validate_every = int(self.config.get("scrub_validate_every", 1))
        self.train_scope = str(self.config.get("scrub_train_scope", "full")).lower()

    def _infer_target_tag(self) -> str:
        """
        Build a compact unlearning-target tag for checkpoint names.

        Returns:
            Target tag string.
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

        ratio = float(self.config.get("forget_ratio", 0.01))
        return f"random_ratio{str(ratio).replace('.', 'p')}"

    def _resolve_latest_checkpoint(self, root: Path) -> Path:
        """
        Resolve a checkpoint file from a file or directory path.

        Args:
            root: Checkpoint file or directory.

        Returns:
            Resolved checkpoint path.
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
        Extract a cleaned state dict from common checkpoint formats.

        Args:
            state_obj: Raw object returned by torch.load.

        Returns:
            Cleaned model state dict.
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
        Load the original trained model used as SCRUB teacher.

        Args:
            model: Optional pre-loaded model.
            dataset: Dataset manager.

        Returns:
            Trained model on the target device.
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
        print(f"[SCRUB] Loaded trained checkpoint: {checkpoint_path}")
        if missing:
            print(f"[SCRUB][Load] Missing keys count: {len(missing)}")
        if unexpected:
            print(f"[SCRUB][Load] Unexpected keys count: {len(unexpected)}")
        return model

    def _set_train_scope(self, model: nn.Module) -> None:
        """
        Select which model parameters are trainable.

        Args:
            model: Student model.
        """
        for param in model.parameters():
            param.requires_grad = self.train_scope == "full"

        if self.train_scope == "head":
            modules = []
            for name in ("classifier", "fc", "head"):
                if hasattr(model, name):
                    modules.append(getattr(model, name))
            if not modules:
                raise RuntimeError("scrub_train_scope='head' but no classifier/head module was found.")
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.train_scope == "backbone":
            modules = []
            for name in ("backbone", "features"):
                if hasattr(model, name):
                    modules.append(getattr(model, name))
            if not modules:
                raise RuntimeError("scrub_train_scope='backbone' but no backbone/features module was found.")
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.train_scope != "full":
            raise ValueError(f"Unsupported SCRUB train scope: {self.train_scope}")

    def _build_optimizer(self, model: nn.Module):
        """
        Build optimizer over trainable student parameters.

        Args:
            model: Student model.

        Returns:
            Optimizer instance.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found for SCRUB.")
        if self.optimizer_name == "sgd":
            return SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _distill_kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-scaled distillation KL.

        Args:
            student_logits: Student logits.
            teacher_logits: Frozen teacher logits.

        Returns:
            KL divergence loss for matching teacher outputs.
        """
        temperature = max(self.temperature, 1e-6)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    def _maybe_clip_gradients(self, model: nn.Module) -> None:
        """
        Clip gradients when configured.

        Args:
            model: Student model.
        """
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=self.grad_clip,
            )

    def _run_forget_epoch(
        self,
        student: nn.Module,
        teacher: nn.Module,
        loader: DataLoader,
        optimizer,
    ) -> Dict[str, float]:
        """
        Run one SCRUB forget phase on D_u.

        Args:
            student: Trainable student model.
            teacher: Frozen original model.
            loader: Forget-set dataloader.
            optimizer: Student optimizer.

        Returns:
            Forget metrics for the epoch.
        """
        student.train()
        teacher.eval()
        total_ce = 0.0
        total_kl = 0.0
        total_conf = 0.0
        correct = 0
        total = 0
        steps = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            if self.max_forget_batches > 0 and batch_idx > self.max_forget_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                teacher_logits = teacher(x)

            optimizer.zero_grad()
            student_logits = student(x)
            ce_loss = F.cross_entropy(student_logits, y)
            kl_loss = self._distill_kl(student_logits, teacher_logits)
            loss = -(self.forget_ce_weight * ce_loss + self.forget_weight * kl_loss)
            loss.backward()
            self._maybe_clip_gradients(student)
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(student_logits, dim=1)
                total_conf += float(probs.max(dim=1).values.sum().item())
                correct += int((student_logits.argmax(dim=1) == y).sum().item())
                total += int(y.size(0))
                total_ce += float(ce_loss.item()) * y.size(0)
                total_kl += float(kl_loss.item()) * y.size(0)
                steps += 1

        return {
            "forget_ce": total_ce / max(total, 1),
            "forget_kl": total_kl / max(total, 1),
            "forget_accuracy": correct / max(total, 1),
            "forget_confidence": total_conf / max(total, 1),
            "forget_steps": float(steps),
        }

    def _run_retain_epoch(
        self,
        student: nn.Module,
        teacher: nn.Module,
        loader: DataLoader,
        optimizer,
    ) -> Dict[str, float]:
        """
        Run one SCRUB retain phase on D_r.

        Args:
            student: Trainable student model.
            teacher: Frozen original model.
            loader: Retained-set dataloader.
            optimizer: Student optimizer.

        Returns:
            Retain metrics for the epoch.
        """
        student.train()
        teacher.eval()
        total_ce = 0.0
        total_kd = 0.0
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            if self.max_retain_batches > 0 and batch_idx > self.max_retain_batches:
                break

            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                teacher_logits = teacher(x)

            optimizer.zero_grad()
            student_logits = student(x)
            ce_loss = F.cross_entropy(student_logits, y)
            kd_loss = self._distill_kl(student_logits, teacher_logits)
            loss = self.retain_ce_weight * ce_loss + self.retain_kd_weight * kd_loss
            loss.backward()
            self._maybe_clip_gradients(student)
            optimizer.step()

            batch_size = int(y.size(0))
            total_ce += float(ce_loss.item()) * batch_size
            total_kd += float(kd_loss.item()) * batch_size
            total_loss += float(loss.item()) * batch_size
            correct += int((student_logits.argmax(dim=1) == y).sum().item())
            total += batch_size
            steps += 1

        return {
            "retain_ce": total_ce / max(total, 1),
            "retain_kd": total_kd / max(total, 1),
            "retain_total": total_loss / max(total, 1),
            "retain_train_accuracy": correct / max(total, 1),
            "retain_steps": float(steps),
        }

    def _evaluate_split(self, model: nn.Module, loader: DataLoader, name: str) -> Dict[str, float]:
        """
        Evaluate accuracy and confidence on a split.

        Args:
            model: Model to evaluate.
            loader: Split dataloader.
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
            f"[SCRUB][Eval] {name}: acc={metrics['accuracy']:.4f} "
            f"loss={metrics['loss']:.4f} conf={metrics['mean_confidence']:.4f} "
            f"({correct}/{total})"
        )
        return metrics

    def unlearn(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> Dict:
        """
        Execute SCRUB approximate unlearning.

        Args:
            model: Optional original trained model.
            dataset: Dataset manager exposing D_u, D_r, and D_test.

        Returns:
            Summary dictionary containing the unlearned model and save path.
        """
        set_seed(int(self.config.get("seed", 42)))
        loaders = dataset.get_dataloaders(retained_shuffle=True)
        forget_loader = loaders["d_u"]
        retain_loader = loaders["d_r"]
        test_loader = loaders["d_test"]

        teacher = self._load_original_model(model, dataset)
        student = copy.deepcopy(teacher).to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self._set_train_scope(student)
        optimizer = self._build_optimizer(student)
        trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)

        print("[SCRUB] ===== Start SCRUB Approximate Unlearning =====")
        print(
            f"[SCRUB] setup: model={self.config.get('model_name', 'resnet18')}, "
            f"dataset={dataset.dataset_name}, device={self.device}, train_scope={self.train_scope}"
        )
        print(
            f"[SCRUB] sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )
        print(
            f"[SCRUB] hyper: optimizer={self.optimizer_name}, lr={self.lr}, "
            f"epochs={self.epochs}, T={self.temperature}, "
            f"forget_ce_w={self.forget_ce_weight}, forget_kd_w={self.forget_weight}, "
            f"retain_ce_w={self.retain_ce_weight}, retain_kd_w={self.retain_kd_weight}, "
            f"weight_decay={self.weight_decay}, grad_clip={self.grad_clip}"
        )
        print(f"[SCRUB] trainable parameters: {trainable_count}")

        history = []
        final_eval = {}
        for epoch in range(1, self.epochs + 1):
            forget_stats = self._run_forget_epoch(student, teacher, forget_loader, optimizer)
            retain_stats = self._run_retain_epoch(student, teacher, retain_loader, optimizer)

            forget_accuracy = forget_stats["forget_accuracy"]
            forget_confidence = forget_stats["forget_confidence"]
            retain_accuracy = 0.0
            test_accuracy = 0.0
            if epoch % max(self.validate_every, 1) == 0:
                forget_eval = self._evaluate_split(student, forget_loader, "forget")
                retain_eval = self._evaluate_split(student, retain_loader, "retain")
                test_eval = self._evaluate_split(student, test_loader, "test")
                forget_accuracy = forget_eval["accuracy"]
                forget_confidence = forget_eval["mean_confidence"]
                retain_accuracy = retain_eval["accuracy"]
                test_accuracy = test_eval["accuracy"]
                final_eval = {
                    "forget": forget_eval,
                    "retain": retain_eval,
                    "test": test_eval,
                }

            stats = ScrubEpochStats(
                forget_ce=forget_stats["forget_ce"],
                forget_kl=forget_stats["forget_kl"],
                retain_ce=retain_stats["retain_ce"],
                retain_kd=retain_stats["retain_kd"],
                retain_total=retain_stats["retain_total"],
                forget_accuracy=forget_accuracy,
                forget_confidence=forget_confidence,
                retain_accuracy=retain_accuracy,
                test_accuracy=test_accuracy,
            )
            history.append(
                {
                    "epoch": epoch,
                    "forget_ce": stats.forget_ce,
                    "forget_kl": stats.forget_kl,
                    "retain_ce": stats.retain_ce,
                    "retain_kd": stats.retain_kd,
                    "retain_total": stats.retain_total,
                    "forget_accuracy": stats.forget_accuracy,
                    "forget_confidence": stats.forget_confidence,
                    "retain_accuracy": stats.retain_accuracy,
                    "test_accuracy": stats.test_accuracy,
                    "forget_steps": forget_stats["forget_steps"],
                    "retain_steps": retain_stats["retain_steps"],
                }
            )
            print(
                f"[SCRUB][Epoch {epoch:03d}/{self.epochs:03d}] "
                f"forget_ce={stats.forget_ce:.6f} "
                f"forget_kl={stats.forget_kl:.6f} "
                f"retain_ce={stats.retain_ce:.6f} "
                f"retain_kd={stats.retain_kd:.6f} "
                f"retain_total={stats.retain_total:.6f} "
                f"forget_acc={stats.forget_accuracy:.4f} "
                f"forget_conf={stats.forget_confidence:.4f} "
                f"retain_acc={stats.retain_accuracy:.4f} "
                f"test_acc={stats.test_accuracy:.4f}"
            )

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        target_tag = self._infer_target_tag()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"scrub_{model_name}_{dataset.dataset_name}_{target_tag}_{timestamp}.pt"
        checkpoint_name = str(self.config.get("scrub_checkpoint_name", default_name))
        save_path = save_dir / checkpoint_name
        torch.save(
            {
                "state_dict": student.state_dict(),
                "method": "scrub",
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
                "scrub_config": {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "optimizer": self.optimizer_name,
                    "temperature": self.temperature,
                    "forget_ce_weight": self.forget_ce_weight,
                    "forget_weight": self.forget_weight,
                    "retain_ce_weight": self.retain_ce_weight,
                    "retain_kd_weight": self.retain_kd_weight,
                    "train_scope": self.train_scope,
                },
                "seed": int(self.config.get("seed", 42)),
            },
            str(save_path),
        )
        print(f"[SCRUB] Saved unlearned model to: {save_path}")
        print("[SCRUB] ===== SCRUB Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "scrub",
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
    """Standalone CLI entry for SCRUB approximate unlearning."""
    args = _build_args()
    base_cfg = load_config(args.config)
    cfg = _merge_config(base_cfg, args)
    dataset = UnlearningDataset(cfg)
    unlearner = SCRUBUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
