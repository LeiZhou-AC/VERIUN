"""Shared utilities for unlearning baselines."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from configs.data.dataset import UnlearningDataset
from configs.models.resnet import construct_model


def set_unlearning_seed(seed: int) -> None:
    """
    Set random seeds for an unlearning run.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_target_tag(config: Dict, default_ratio: float = 0.01) -> str:
    """
    Build a compact checkpoint tag for the current forget target.

    Args:
        config: Runtime configuration.
        default_ratio: Ratio fallback when absent.

    Returns:
        Target tag string.
    """
    split_mode = str(config.get("split_mode", "random")).lower()
    if split_mode in {"by_class", "class_random"}:
        classes = config.get("forget_classes", [])
        if isinstance(classes, (int, float, str)):
            classes = [classes]
        classes = [str(int(c)) for c in classes]
        if classes:
            prefix = "byclass" if split_mode == "by_class" else "classrandom"
            suffix = "-".join(classes)
            if split_mode == "class_random":
                if config.get("forget_count") is not None:
                    return f"{prefix}_{suffix}_count{int(config.get('forget_count'))}"
                ratio = str(float(config.get("forget_ratio", default_ratio))).replace(".", "p")
                return f"{prefix}_{suffix}_ratio{ratio}"
            return f"{prefix}_{suffix}"
        return f"{split_mode}_unspecified"

    if config.get("forget_count") is not None:
        return f"random_count{int(config.get('forget_count'))}"
    ratio = str(float(config.get("forget_ratio", default_ratio))).replace(".", "p")
    return f"random_ratio{ratio}"


def resolve_checkpoint_path(path_value: str) -> Path:
    """
    Resolve a checkpoint from a file path or a directory.

    Args:
        path_value: Checkpoint file or directory.

    Returns:
        Resolved checkpoint path.
    """
    root = Path(str(path_value))
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
    candidates = list(root.glob("*.pt")) + list(root.glob("*.pth")) + list(root.glob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under: {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def extract_state_dict(state_obj) -> Dict:
    """
    Extract a model state dict from common checkpoint formats.

    Args:
        state_obj: Raw object loaded with torch.load.

    Returns:
        Clean state dict.
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


def build_model_from_config(config: Dict, dataset: UnlearningDataset) -> nn.Module:
    """
    Build a model instance from project config.

    Args:
        config: Runtime configuration.
        dataset: Dataset manager.

    Returns:
        Initialized model.
    """
    model_name = str(config.get("model_name", config.get("model", "resnet18")))
    model, _ = construct_model(
        model=model_name,
        num_classes=int(config.get("num_classes", dataset.num_classes)),
        seed=int(config.get("seed", 42)),
        num_channels=int(config.get("in_channels", 3)),
    )
    return model


def load_trained_model(
    model: Optional[nn.Module],
    dataset: UnlearningDataset,
    config: Dict,
    device: torch.device,
    log_prefix: str,
) -> nn.Module:
    """
    Load the original trained model for an unlearning method.

    Args:
        model: Optional already-loaded model.
        dataset: Dataset manager.
        config: Runtime configuration.
        device: Target device.
        log_prefix: Log prefix such as "[SALUN]".

    Returns:
        Loaded model on target device.
    """
    if model is not None:
        return copy.deepcopy(model).to(device)

    checkpoint_path = resolve_checkpoint_path(str(config.get("trained_weights_path", "save/weights/trained")))
    loaded = torch.load(str(checkpoint_path), map_location=device)
    state_dict = extract_state_dict(loaded)
    built_model = build_model_from_config(config, dataset)
    missing, unexpected = built_model.load_state_dict(state_dict, strict=False)
    built_model = built_model.to(device)
    print(f"{log_prefix} Loaded trained checkpoint: {checkpoint_path}")
    if missing:
        print(f"{log_prefix}[Load] Missing keys count: {len(missing)}")
    if unexpected:
        print(f"{log_prefix}[Load] Unexpected keys count: {len(unexpected)}")
    return built_model


def set_train_scope(model: nn.Module, train_scope: str, log_prefix: str) -> None:
    """
    Select trainable parameter scope for gradient-based unlearning.

    Args:
        model: Model to modify.
        train_scope: One of full, backbone, or head.
        log_prefix: Log prefix.
    """
    scope = str(train_scope).lower()
    for param in model.parameters():
        param.requires_grad = scope == "full"

    if scope == "head":
        modules = []
        for name in ("classifier", "fc", "head"):
            if hasattr(model, name):
                modules.append(getattr(model, name))
        if not modules:
            raise RuntimeError(f"{log_prefix} train_scope='head' but no classifier/head module was found.")
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
    elif scope == "backbone":
        modules = []
        for name in ("backbone", "features"):
            if hasattr(model, name):
                modules.append(getattr(model, name))
        if not modules:
            raise RuntimeError(f"{log_prefix} train_scope='backbone' but no backbone/features module was found.")
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
    elif scope != "full":
        raise ValueError(f"Unsupported train scope: {train_scope}")


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
):
    """
    Build an optimizer for trainable parameters.

    Args:
        parameters: Parameter iterable.
        optimizer_name: adamw or sgd.
        lr: Learning rate.
        weight_decay: Weight decay.
        momentum: SGD momentum.

    Returns:
        Optimizer instance.
    """
    params = [param for param in parameters if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    if str(optimizer_name).lower() == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def apply_gradient_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Apply binary masks to parameter gradients in-place.

    Args:
        model: Model with gradients.
        masks: Mapping from parameter name to mask tensor.
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in masks:
            param.grad.mul_(masks[name].to(param.grad.device))


def maybe_clip_gradients(model: nn.Module, grad_clip: float) -> None:
    """
    Clip gradients for trainable parameters when configured.

    Args:
        model: Model with gradients.
        grad_clip: Maximum norm. Non-positive disables clipping.
    """
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device, name: str, log_prefix: str) -> Dict:
    """
    Evaluate CE loss, accuracy, and confidence on a split.

    Args:
        model: Model to evaluate.
        loader: Dataloader.
        device: Target device.
        name: Split name for logging.
        log_prefix: Log prefix.

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
            x = x.to(device)
            y = y.to(device)
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
        f"{log_prefix}[Eval] {name}: acc={metrics['accuracy']:.4f} "
        f"loss={metrics['loss']:.4f} conf={metrics['mean_confidence']:.4f} "
        f"({correct}/{total})"
    )
    return metrics
