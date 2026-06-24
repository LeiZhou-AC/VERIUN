"""SalUn-style gradient-saliency machine unlearning baseline.

This module implements the image-classification SalUn workflow in the current
project style:
1) compute a gradient-based saliency mask on D_u,
2) perform random/wrong-label unlearning updates,
3) preserve utility with retained-data CE updates,
4) save the resulting checkpoint for RUV verification.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from configs.data.dataset import UnlearningDataset
from unlearning.base_unlearner import BaseUnlearner
from unlearning.common import (
    apply_gradient_mask,
    build_optimizer,
    evaluate_split,
    infer_target_tag,
    load_trained_model,
    maybe_clip_gradients,
    set_train_scope,
    set_unlearning_seed,
)
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
    Build command-line arguments for SalUn.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SalUn gradient-saliency unlearning")
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
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-batches", type=int, default=None)
    parser.add_argument("--forget-weight", type=float, default=None)
    parser.add_argument("--retain-weight", type=float, default=None)
    parser.add_argument("--max-forget-batches", type=int, default=None)
    parser.add_argument("--max-retain-batches", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--train-scope", type=str, default=None, choices=["full", "backbone", "head"])
    parser.add_argument("--label-seed", type=int, default=None)
    parser.add_argument(
        "--label-strategy",
        type=str,
        default=None,
        choices=["random_any", "cyclic", "permutation", "random"],
    )
    return parser.parse_args()


def _merge_config(base_cfg: Dict, args: argparse.Namespace) -> Dict:
    """
    Merge CLI overrides into base config.

    Args:
        base_cfg: YAML config.
        args: Parsed arguments.

    Returns:
        Runtime config.
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
    if args.allow_download:
        cfg["allow_download"] = True

    set_from_arg("trained_weights_path", args.trained_path, "save/weights/trained")
    set_from_arg("unlearned_weights_path", args.unlearned_path, "save/weights/unlearned")
    if args.save_name:
        cfg["salun_checkpoint_name"] = args.save_name

    set_from_arg("salun_epochs", args.epochs)
    set_from_arg("salun_lr", args.lr)
    set_from_arg("salun_optimizer", args.optimizer)
    set_from_arg("salun_momentum", args.momentum)
    set_from_arg("salun_weight_decay", args.weight_decay)
    set_from_arg("salun_mask_ratio", args.mask_ratio)
    set_from_arg("salun_mask_batches", args.mask_batches)
    set_from_arg("salun_forget_weight", args.forget_weight)
    set_from_arg("salun_retain_weight", args.retain_weight)
    set_from_arg("salun_max_forget_batches", args.max_forget_batches)
    set_from_arg("salun_max_retain_batches", args.max_retain_batches)
    set_from_arg("salun_grad_clip", args.grad_clip)
    set_from_arg("salun_validate_every", args.validate_every)
    set_from_arg("salun_train_scope", args.train_scope)
    set_from_arg("salun_label_seed", args.label_seed)
    set_from_arg("salun_label_strategy", args.label_strategy)
    return cfg


class SalUnLabelDataset(Dataset):
    """Dataset wrapper for SalUn random-label unlearning."""

    def __init__(self, base_dataset: Dataset, num_classes: int, seed: int, strategy: str = "random_any"):
        """
        Initialize SalUn replacement labels.

        Args:
            base_dataset: Original forget subset.
            num_classes: Number of classes.
            seed: Seed controlling label replacement.
            strategy: random_any matches the official SalUn RL behavior; other
                strategies are deterministic alternatives for ablations.
        """
        self.base_dataset = base_dataset
        self.num_classes = int(num_classes)
        self.strategy = str(strategy).lower()
        if self.num_classes < 2:
            raise ValueError("SalUn label replacement requires at least two classes.")
        if self.strategy not in {"random_any", "cyclic", "permutation", "random"}:
            raise ValueError(f"Unsupported SalUn label strategy: {self.strategy}")

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        label_map = self._build_label_map(generator)
        self.replacement_labels = []
        self.true_labels = []
        for idx in range(len(self.base_dataset)):
            item = self.base_dataset[idx]
            label = self._extract_label(item)
            if self.strategy == "random_any":
                replacement = int(torch.randint(0, self.num_classes, (1,), generator=generator).item())
            elif self.strategy == "random":
                offset = int(torch.randint(1, self.num_classes, (1,), generator=generator).item())
                replacement = int((label + offset) % self.num_classes)
            else:
                replacement = int(label_map[int(label)])
            self.true_labels.append(int(label))
            self.replacement_labels.append(replacement)

    def _build_label_map(self, generator: torch.Generator) -> Dict[int, int]:
        """
        Build deterministic class-level replacement labels.

        Args:
            generator: Torch generator.

        Returns:
            Class label mapping.
        """
        if self.strategy == "cyclic":
            return {label: int((label + 1) % self.num_classes) for label in range(self.num_classes)}
        if self.strategy == "permutation":
            for _ in range(100):
                permutation = torch.randperm(self.num_classes, generator=generator).tolist()
                if all(int(permutation[label]) != label for label in range(self.num_classes)):
                    return {label: int(permutation[label]) for label in range(self.num_classes)}
            return {label: int((label + 1) % self.num_classes) for label in range(self.num_classes)}
        return {}

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        """
        Return input, replacement label, and original label.

        Args:
            index: Sample index.

        Returns:
            Tuple of image, replacement label, original label.
        """
        item = self.base_dataset[index]
        return item[0], self.replacement_labels[index], self.true_labels[index]

    @staticmethod
    def _extract_label(item) -> int:
        """
        Extract an integer label from a dataset item.

        Args:
            item: Dataset item.

        Returns:
            Integer label.
        """
        label = item[1]
        if torch.is_tensor(label):
            return int(label.item())
        return int(label)


class SalUnUnlearner(BaseUnlearner):
    """Gradient-saliency masked unlearner for classification models."""

    def __init__(self, config: Dict):
        """
        Initialize SalUn from config.

        Args:
            config: Runtime configuration.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.epochs = int(self.config.get("salun_epochs", self.config.get("epochs", 10)))
        self.lr = float(self.config.get("salun_lr", self.config.get("lr", 1e-4)))
        self.optimizer_name = str(self.config.get("salun_optimizer", "sgd")).lower()
        self.momentum = float(self.config.get("salun_momentum", 0.9))
        self.weight_decay = float(self.config.get("salun_weight_decay", 5e-4))
        self.mask_ratio = float(self.config.get("salun_mask_ratio", 0.5))
        self.mask_batches = int(self.config.get("salun_mask_batches", 0))
        self.forget_weight = float(self.config.get("salun_forget_weight", 1.0))
        self.retain_weight = float(self.config.get("salun_retain_weight", 0.5))
        self.max_forget_batches = int(self.config.get("salun_max_forget_batches", 0))
        self.max_retain_batches = int(self.config.get("salun_max_retain_batches", 64))
        self.grad_clip = float(self.config.get("salun_grad_clip", 5.0))
        self.validate_every = int(self.config.get("salun_validate_every", 1))
        self.train_scope = str(self.config.get("salun_train_scope", "full")).lower()
        self.label_seed = int(self.config.get("salun_label_seed", self.config.get("split_seed", 42)))
        self.label_strategy = str(self.config.get("salun_label_strategy", "random_any")).lower()

    def _build_wrong_label_loader(self, dataset: UnlearningDataset) -> DataLoader:
        """
        Build a wrong-label forget-set dataloader.

        Args:
            dataset: Dataset manager.

        Returns:
            Dataloader over relabeled D_u.
        """
        wrong_dataset = SalUnLabelDataset(
            dataset.get_unlearning_set(),
            num_classes=int(self.config.get("num_classes", dataset.num_classes)),
            seed=self.label_seed,
            strategy=self.label_strategy,
        )
        return DataLoader(
            wrong_dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 4)),
            pin_memory=bool(self.config.get("pin_memory", torch.cuda.is_available())),
        )

    def _compute_saliency_mask(self, model: nn.Module, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute a global top-ratio gradient saliency mask on D_u.

        Args:
            model: Original trained model.
            loader: Forget-set dataloader with true labels.

        Returns:
            Mapping from parameter name to binary mask.
        """
        model.train()
        saliency = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        seen = 0
        steps = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            if self.mask_batches > 0 and batch_idx > self.mask_batches:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            model.zero_grad(set_to_none=True)
            loss = -F.cross_entropy(model(x), y)
            loss.backward()
            batch_size = int(y.size(0))
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    saliency[name].add_(param.grad.detach() * batch_size)
            seen += batch_size
            steps += 1

        if seen <= 0:
            raise ValueError("SalUn mask generation received an empty forget loader.")
        for name in saliency:
            saliency[name] = saliency[name].div(float(seen)).abs_()

        flat_scores = torch.cat([value.flatten() for value in saliency.values()])
        ratio = min(max(self.mask_ratio, 0.0), 1.0)
        if ratio <= 0:
            raise ValueError("salun_mask_ratio must be greater than 0.")
        keep_count = max(1, int(flat_scores.numel() * ratio))
        flat_mask = torch.zeros_like(flat_scores)
        if keep_count >= flat_scores.numel():
            flat_mask.fill_(1.0)
        else:
            top_indices = torch.topk(flat_scores, keep_count, largest=True).indices
            flat_mask[top_indices] = 1.0

        masks = {}
        cursor = 0
        for name, score in saliency.items():
            width = score.numel()
            masks[name] = flat_mask[cursor : cursor + width].view_as(score).detach()
            cursor += width
        active = sum(int(mask.sum().item()) for mask in masks.values())
        total = sum(mask.numel() for mask in masks.values())
        print(
            f"[SALUN] Saliency mask ready: active={active}/{total} "
            f"({active / max(total, 1):.4f}), steps={steps}, samples={seen}"
        )
        return masks

    def _snapshot_masked_out_parameters(self, model: nn.Module, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Snapshot parameters outside the SalUn mask.

        Args:
            model: Model before SalUn updates.
            masks: Saliency masks.

        Returns:
            Parameter snapshot keyed by name.
        """
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in masks
        }

    def _restore_masked_out_parameters(
        self,
        model: nn.Module,
        masks: Dict[str, torch.Tensor],
        theta0: Dict[str, torch.Tensor],
        optimizer,
    ) -> None:
        """
        Restore mask-out coordinates after each optimizer step.

        Args:
            model: Updated model.
            masks: Saliency masks.
            theta0: Parameter snapshot before SalUn updates.
            optimizer: Optimizer whose state may need masking.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in masks:
                    continue
                mask = masks[name].to(device=param.device, dtype=param.dtype)
                inverse = 1.0 - mask
                if torch.count_nonzero(inverse) == 0:
                    continue
                param.data.mul_(mask).add_(theta0[name].to(param.device) * inverse)
                state = optimizer.state.get(param, None)
                if not state:
                    continue
                for state_value in state.values():
                    if torch.is_tensor(state_value) and state_value.shape == param.shape:
                        state_value.mul_(mask)

    def _run_forget_epoch(self, model: nn.Module, loader: DataLoader, optimizer, masks: Dict[str, torch.Tensor]) -> Dict:
        """
        Run one masked wrong-label forget phase.

        Args:
            model: Model being unlearned.
            loader: Wrong-label forget dataloader.
            optimizer: Optimizer.
            masks: Saliency masks.

        Returns:
            Forget metrics.
        """
        model.train()
        total_ce = 0.0
        original_correct = 0
        wrong_correct = 0
        total_conf = 0.0
        total = 0
        steps = 0

        for batch_idx, (x, wrong_y, true_y) in enumerate(loader, start=1):
            if self.max_forget_batches > 0 and batch_idx > self.max_forget_batches:
                break
            x = x.to(self.device)
            wrong_y = wrong_y.to(self.device)
            true_y = true_y.to(self.device)
            if self.label_strategy == "random_any":
                # Official SalUn RL samples replacement labels on the fly.
                wrong_y = torch.randint(
                    0,
                    int(self.config.get("num_classes", 10)),
                    true_y.shape,
                    device=self.device,
                )
            optimizer.zero_grad()
            logits = model(x)
            ce_loss = F.cross_entropy(logits, wrong_y)
            loss = self.forget_weight * ce_loss
            loss.backward()
            apply_gradient_mask(model, masks)
            maybe_clip_gradients(model, self.grad_clip)
            optimizer.step()
            self._restore_masked_out_parameters(model, masks, self._theta0, optimizer)

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                batch_size = int(true_y.size(0))
                total_ce += float(ce_loss.item()) * batch_size
                original_correct += int((preds == true_y).sum().item())
                wrong_correct += int((preds == wrong_y).sum().item())
                total_conf += float(probs.max(dim=1).values.sum().item())
                total += batch_size
                steps += 1

        return {
            "forget_ce": total_ce / max(total, 1),
            "forget_original_accuracy": original_correct / max(total, 1),
            "forget_wrong_label_accuracy": wrong_correct / max(total, 1),
            "forget_confidence": total_conf / max(total, 1),
            "forget_steps": float(steps),
        }

    def _run_retain_epoch(self, model: nn.Module, loader: DataLoader, optimizer, masks: Dict[str, torch.Tensor]) -> Dict:
        """
        Run one masked retained-data utility phase.

        Args:
            model: Model being unlearned.
            loader: Retained dataloader.
            optimizer: Optimizer.
            masks: Saliency masks.

        Returns:
            Retain metrics.
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
            apply_gradient_mask(model, masks)
            maybe_clip_gradients(model, self.grad_clip)
            optimizer.step()
            self._restore_masked_out_parameters(model, masks, self._theta0, optimizer)

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

    def _evaluate_wrong_label_split(self, model: nn.Module, loader: DataLoader) -> Dict:
        """
        Evaluate forget set against original and wrong labels.

        Args:
            model: Model to evaluate.
            loader: Wrong-label forget dataloader.

        Returns:
            Metrics dictionary.
        """
        model.eval()
        original_correct = 0
        wrong_correct = 0
        total_conf = 0.0
        total = 0
        with torch.no_grad():
            for x, wrong_y, true_y in loader:
                x = x.to(self.device)
                wrong_y = wrong_y.to(self.device)
                true_y = true_y.to(self.device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                original_correct += int((preds == true_y).sum().item())
                wrong_correct += int((preds == wrong_y).sum().item())
                total_conf += float(probs.max(dim=1).values.sum().item())
                total += int(true_y.size(0))

        metrics = {
            "original_accuracy": original_correct / max(total, 1),
            "wrong_label_accuracy": wrong_correct / max(total, 1),
            "mean_confidence": total_conf / max(total, 1),
            "original_correct": original_correct,
            "wrong_correct": wrong_correct,
            "total": total,
        }
        print(
            "[SALUN][Eval] forget: "
            f"orig_acc={metrics['original_accuracy']:.4f} "
            f"wrong_acc={metrics['wrong_label_accuracy']:.4f} "
            f"conf={metrics['mean_confidence']:.4f} "
            f"(orig {original_correct}/{total}, wrong {wrong_correct}/{total})"
        )
        return metrics

    def unlearn(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> Dict:
        """
        Execute SalUn-style gradient-saliency masked unlearning.

        Args:
            model: Optional original trained model.
            dataset: Dataset manager.

        Returns:
            Summary dictionary.
        """
        set_unlearning_seed(int(self.config.get("seed", 42)))
        loaders = dataset.get_dataloaders(retained_shuffle=True)
        forget_loader = loaders["d_u"]
        wrong_loader = self._build_wrong_label_loader(dataset)
        retain_loader = loaders["d_r"]
        test_loader = loaders["d_test"]

        student = load_trained_model(model, dataset, self.config, self.device, "[SALUN]")
        set_train_scope(student, self.train_scope, "[SALUN]")
        masks = self._compute_saliency_mask(student, forget_loader)
        self._theta0 = self._snapshot_masked_out_parameters(student, masks)
        optimizer = build_optimizer(
            student.parameters(),
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)

        print("[SALUN] ===== Start SalUn Unlearning =====")
        print(
            f"[SALUN] setup: model={self.config.get('model_name', 'resnet18')}, "
            f"dataset={dataset.dataset_name}, device={self.device}, train_scope={self.train_scope}"
        )
        print(
            f"[SALUN] sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )
        print(
            f"[SALUN] hyper: optimizer={self.optimizer_name}, lr={self.lr}, epochs={self.epochs}, "
            f"mask_ratio={self.mask_ratio}, forget_w={self.forget_weight}, retain_w={self.retain_weight}, "
            f"max_forget_batches={self.max_forget_batches}, max_retain_batches={self.max_retain_batches}, "
            f"weight_decay={self.weight_decay}, grad_clip={self.grad_clip}, "
            f"label_seed={self.label_seed}, label_strategy={self.label_strategy}"
        )
        print(f"[SALUN] trainable parameters: {trainable_count}")

        history = []
        final_eval = {}
        for epoch in range(1, self.epochs + 1):
            forget_stats = self._run_forget_epoch(student, wrong_loader, optimizer, masks)
            retain_stats = self._run_retain_epoch(student, retain_loader, optimizer, masks)
            forget_original_accuracy = forget_stats["forget_original_accuracy"]
            forget_wrong_label_accuracy = forget_stats["forget_wrong_label_accuracy"]
            forget_confidence = forget_stats["forget_confidence"]
            retain_accuracy = 0.0
            test_accuracy = 0.0

            if epoch % max(self.validate_every, 1) == 0:
                forget_eval = self._evaluate_wrong_label_split(student, wrong_loader)
                retain_eval = evaluate_split(student, retain_loader, self.device, "retain", "[SALUN]")
                test_eval = evaluate_split(student, test_loader, self.device, "test", "[SALUN]")
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

            record = {
                "epoch": epoch,
                "forget_ce": forget_stats["forget_ce"],
                "retain_ce": retain_stats["retain_ce"],
                "forget_original_accuracy": forget_original_accuracy,
                "forget_wrong_label_accuracy": forget_wrong_label_accuracy,
                "forget_confidence": forget_confidence,
                "retain_accuracy": retain_accuracy,
                "test_accuracy": test_accuracy,
                "forget_steps": forget_stats["forget_steps"],
                "retain_steps": retain_stats["retain_steps"],
            }
            history.append(record)
            print(
                f"[SALUN][Epoch {epoch:03d}/{self.epochs:03d}] "
                f"forget_ce={record['forget_ce']:.6f} retain_ce={record['retain_ce']:.6f} "
                f"forget_orig_acc={record['forget_original_accuracy']:.4f} "
                f"forget_wrong_acc={record['forget_wrong_label_accuracy']:.4f} "
                f"forget_conf={record['forget_confidence']:.4f} "
                f"retain_acc={record['retain_accuracy']:.4f} test_acc={record['test_accuracy']:.4f} "
                f"steps=({int(record['forget_steps'])}/{int(record['retain_steps'])})"
            )

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        target_tag = infer_target_tag(self.config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"salun_{model_name}_{dataset.dataset_name}_{target_tag}_{timestamp}.pt"
        checkpoint_name = str(self.config.get("salun_checkpoint_name", default_name))
        save_path = save_dir / checkpoint_name
        torch.save(
            {
                "state_dict": student.state_dict(),
                "method": "salun",
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
                "salun_config": {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "optimizer": self.optimizer_name,
                    "mask_ratio": self.mask_ratio,
                    "mask_batches": self.mask_batches,
                    "forget_weight": self.forget_weight,
                    "retain_weight": self.retain_weight,
                    "max_forget_batches": self.max_forget_batches,
                    "max_retain_batches": self.max_retain_batches,
                    "train_scope": self.train_scope,
                    "label_seed": self.label_seed,
                    "label_strategy": self.label_strategy,
                },
                "seed": int(self.config.get("seed", 42)),
            },
            str(save_path),
        )
        print(f"[SALUN] Saved unlearned model to: {save_path}")
        print("[SALUN] ===== SalUn Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "salun",
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
    """Standalone CLI entry for SalUn unlearning."""
    args = _build_args()
    base_cfg = load_config(args.config)
    cfg = _merge_config(base_cfg, args)
    dataset = UnlearningDataset(cfg)
    unlearner = SalUnUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
