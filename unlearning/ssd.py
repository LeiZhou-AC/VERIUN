"""Selective Synaptic Dampening (SSD) unlearning baseline.

SSD is implemented as a post-hoc parameter-importance method:
1) estimate forget-set and retained-set diagonal importance with squared CE
   gradients,
2) select parameters whose forget importance dominates original-data importance,
3) dampen those parameter values without retraining from scratch.
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
from torch.utils.data import DataLoader

from configs.data.dataset import UnlearningDataset
from unlearning.base_unlearner import BaseUnlearner
from unlearning.common import (
    evaluate_split,
    infer_target_tag,
    load_trained_model,
    set_unlearning_seed,
)
from utils.config import load_config


def _parse_forget_classes(raw: str):
    """
    Parse forget classes from CLI text.

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
    Build command-line arguments for SSD.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Selective Synaptic Dampening unlearning")
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
    parser.add_argument("--selection-weighting", type=float, default=None)
    parser.add_argument("--dampening-constant", type=float, default=None)
    parser.add_argument("--min-scale", type=float, default=None)
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--exponent", type=float, default=None)
    parser.add_argument("--original-split", type=str, default=None, choices=["all", "retain"])
    parser.add_argument("--forget-batches", type=int, default=None)
    parser.add_argument("--original-batches", type=int, default=None)
    parser.add_argument("--retain-batches", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
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
        cfg["ssd_checkpoint_name"] = args.save_name

    set_from_arg("ssd_selection_weighting", args.selection_weighting)
    set_from_arg("ssd_dampening_constant", args.dampening_constant)
    set_from_arg("ssd_min_scale", args.min_scale)
    set_from_arg("ssd_max_scale", args.max_scale)
    set_from_arg("ssd_exponent", args.exponent)
    set_from_arg("ssd_original_split", args.original_split)
    set_from_arg("ssd_forget_batches", args.forget_batches)
    set_from_arg("ssd_original_batches", args.original_batches)
    set_from_arg("ssd_retain_batches", args.retain_batches)
    set_from_arg("ssd_epsilon", args.epsilon)
    return cfg


class SSDUnlearner(BaseUnlearner):
    """Post-hoc Selective Synaptic Dampening unlearner."""

    def __init__(self, config: Dict):
        """
        Initialize SSD from config.

        Args:
            config: Runtime configuration.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.selection_weighting = float(self.config.get("ssd_selection_weighting", 10.0))
        self.dampening_constant = float(self.config.get("ssd_dampening_constant", 1.0))
        self.min_scale = float(self.config.get("ssd_min_scale", 0.0))
        self.max_scale = float(self.config.get("ssd_max_scale", 1.0))
        self.exponent = float(self.config.get("ssd_exponent", 1.0))
        self.original_split = str(self.config.get("ssd_original_split", "all")).lower()
        self.forget_batches = int(self.config.get("ssd_forget_batches", 0))
        self.original_batches = int(self.config.get("ssd_original_batches", 0))
        self.retain_batches = int(self.config.get("ssd_retain_batches", 128))
        self.epsilon = float(self.config.get("ssd_epsilon", 1e-12))
        if self.original_split not in {"all", "retain"}:
            raise ValueError("ssd_original_split must be 'all' or 'retain'.")

    def _estimate_importance(self, model: nn.Module, loader: DataLoader, max_batches: int, name: str) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal parameter importance using squared gradients.

        Args:
            model: Model to analyze.
            loader: Data loader.
            max_batches: Optional maximum number of batches, 0 means all.
            name: Split name for logging.

        Returns:
            Mapping from parameter name to average squared gradient.
        """
        model.eval()
        importance = {
            param_name: torch.zeros_like(param, device=self.device)
            for param_name, param in model.named_parameters()
            if param.requires_grad
        }
        total = 0
        steps = 0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            batch_size = int(y.size(0))
            for param_name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance[param_name].add_(param.grad.detach().pow(2) * batch_size)
            total += batch_size
            steps += 1

        if total <= 0:
            raise ValueError(f"SSD received an empty loader while estimating {name} importance.")
        for param_name in importance:
            importance[param_name].div_(float(total))
        print(f"[SSD] Importance estimated on {name}: steps={steps}, samples={total}")
        return importance

    def _apply_dampening(
        self,
        model: nn.Module,
        forget_importance: Dict[str, torch.Tensor],
        original_importance: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Apply SSD dampening to selected parameters.

        Args:
            model: Model to modify.
            forget_importance: D_u importance.
            original_importance: D_all or D_r importance.

        Returns:
            Dampening summary.
        """
        selected_total = 0
        parameter_total = 0
        scale_values = []

        with torch.no_grad():
            for param_name, param in model.named_parameters():
                if not param.requires_grad or param_name not in forget_importance:
                    continue
                forget_score = forget_importance[param_name]
                original_score = original_importance[param_name]
                selected = forget_score > (self.selection_weighting * original_score)
                scale = (self.dampening_constant * original_score / (forget_score + self.epsilon)).pow(
                    self.exponent
                )
                scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)
                final_scale = torch.where(selected, scale, torch.ones_like(scale))
                param.mul_(final_scale.to(param.device))

                selected_count = int(selected.sum().item())
                selected_total += selected_count
                parameter_total += selected.numel()
                if selected_count > 0:
                    scale_values.append(scale[selected].detach().float().cpu())

        if scale_values:
            all_scales = torch.cat(scale_values)
            mean_scale = float(all_scales.mean().item())
            min_scale = float(all_scales.min().item())
            max_scale = float(all_scales.max().item())
        else:
            mean_scale = 1.0
            min_scale = 1.0
            max_scale = 1.0

        summary = {
            "selected_parameters": selected_total,
            "total_parameters": parameter_total,
            "selected_ratio": selected_total / max(parameter_total, 1),
            "mean_selected_scale": mean_scale,
            "min_selected_scale": min_scale,
            "max_selected_scale": max_scale,
        }
        print(
            f"[SSD] Dampening applied: selected={selected_total}/{parameter_total} "
            f"({summary['selected_ratio']:.6f}), scale_mean={mean_scale:.6f}, "
            f"scale_min={min_scale:.6f}, scale_max={max_scale:.6f}"
        )
        return summary

    def unlearn(self, model: Optional[nn.Module], dataset: UnlearningDataset) -> Dict:
        """
        Execute SSD post-hoc unlearning.

        Args:
            model: Optional original trained model.
            dataset: Dataset manager.

        Returns:
            Summary dictionary.
        """
        set_unlearning_seed(int(self.config.get("seed", 42)))
        loaders = dataset.get_dataloaders(retained_shuffle=True)
        forget_loader = loaders["d_u"]
        retain_loader = loaders["d_r"]
        original_loader = loaders["d_all"] if self.original_split == "all" else retain_loader
        test_loader = loaders["d_test"]

        student = load_trained_model(model, dataset, self.config, self.device, "[SSD]")
        student = student.to(self.device)
        for param in student.parameters():
            param.requires_grad = True

        print("[SSD] ===== Start Selective Synaptic Dampening =====")
        print(
            f"[SSD] setup: model={self.config.get('model_name', 'resnet18')}, "
            f"dataset={dataset.dataset_name}, device={self.device}"
        )
        print(
            f"[SSD] sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )
        print(
            f"[SSD] hyper: selection_weighting={self.selection_weighting}, "
            f"dampening_constant={self.dampening_constant}, min_scale={self.min_scale}, "
            f"max_scale={self.max_scale}, exponent={self.exponent}, "
            f"original_split={self.original_split}, forget_batches={self.forget_batches}, "
            f"original_batches={self.original_batches}, retain_batches={self.retain_batches}, "
            f"epsilon={self.epsilon}"
        )

        forget_importance = self._estimate_importance(student, forget_loader, self.forget_batches, "D_u")
        original_importance = self._estimate_importance(
            student,
            original_loader,
            self.original_batches,
            f"D_{self.original_split}",
        )
        dampening_summary = self._apply_dampening(student, forget_importance, original_importance)

        forget_eval = evaluate_split(student, forget_loader, self.device, "forget", "[SSD]")
        retain_eval = evaluate_split(student, retain_loader, self.device, "retain", "[SSD]")
        test_eval = evaluate_split(student, test_loader, self.device, "test", "[SSD]")
        final_eval = {
            "forget": forget_eval,
            "retain": retain_eval,
            "test": test_eval,
        }

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        target_tag = infer_target_tag(self.config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ssd_{model_name}_{dataset.dataset_name}_{target_tag}_{timestamp}.pt"
        checkpoint_name = str(self.config.get("ssd_checkpoint_name", default_name))
        save_path = save_dir / checkpoint_name
        torch.save(
            {
                "state_dict": student.state_dict(),
                "method": "ssd",
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
                "ssd_config": {
                    "selection_weighting": self.selection_weighting,
                    "dampening_constant": self.dampening_constant,
                    "min_scale": self.min_scale,
                    "max_scale": self.max_scale,
                    "exponent": self.exponent,
                    "original_split": self.original_split,
                    "forget_batches": self.forget_batches,
                    "original_batches": self.original_batches,
                    "retain_batches": self.retain_batches,
                    "epsilon": self.epsilon,
                },
                "seed": int(self.config.get("seed", 42)),
            },
            str(save_path),
        )
        print(f"[SSD] Saved unlearned model to: {save_path}")
        print("[SSD] ===== SSD Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "ssd",
            "model": student,
            "save_path": str(save_path),
            "dampening_summary": dampening_summary,
            "final_eval": final_eval,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
            "forget_manifest": (
                dataset.get_forget_manifest_info() if hasattr(dataset, "get_forget_manifest_info") else None
            ),
        }


def main() -> None:
    """Standalone CLI entry for SSD unlearning."""
    args = _build_args()
    base_cfg = load_config(args.config)
    cfg = _merge_config(base_cfg, args)
    dataset = UnlearningDataset(cfg)
    unlearner = SSDUnlearner(cfg)
    unlearner.unlearn(model=None, dataset=dataset)


if __name__ == "__main__":
    main()
