"""Checkpoint helpers for saving and loading models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from configs.models.resnet import ResNetWrapper, SimpleCNN


def _extract_state_dict(state_obj: Dict) -> Dict:
    """
    Extract a model state dict from a checkpoint-like object.

    Args:
        state_obj: Raw object returned by ``torch.load``.

    Returns:
        Cleaned state dict with any ``module.`` prefix removed.
    """
    if isinstance(state_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_obj and isinstance(state_obj[key], dict):
                state_obj = state_obj[key]
                break

    if not isinstance(state_obj, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like object.")

    cleaned = {}
    for key, value in state_obj.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _infer_model_kind(state_dict: Dict) -> str:
    """
    Infer model family from parameter names.

    Args:
        state_dict: Model state dict.

    Returns:
        ``resnet_wrapper`` or ``simplecnn``.
    """
    keys = list(state_dict.keys())
    if any(key.startswith("backbone.") or key.startswith("classifier.") for key in keys):
        return "resnet_wrapper"
    if any(key.startswith("features.") for key in keys):
        return "simplecnn"
    raise ValueError("Unable to infer model architecture from checkpoint keys.")


def _infer_num_classes(state_dict: Dict) -> int:
    """
    Infer classifier output dimension from a state dict.

    Args:
        state_dict: Model state dict.

    Returns:
        Number of classes.
    """
    for key in ("classifier.weight", "fc.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Unable to infer num_classes from checkpoint.")


def _infer_in_channels(state_dict: Dict) -> int:
    """
    Infer model input channel count from the first convolution.

    Args:
        state_dict: Model state dict.

    Returns:
        Number of input channels.
    """
    for key in ("backbone.0.weight", "features.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return 3


def _build_model_for_state_dict(state_dict: Dict, metadata: Optional[Dict] = None):
    """
    Reconstruct a model instance from checkpoint metadata and keys.

    Args:
        state_dict: Model state dict.
        metadata: Optional checkpoint metadata.

    Returns:
        Instantiated model.
    """
    metadata = metadata or {}
    kind = _infer_model_kind(state_dict)
    num_classes = int(metadata.get("num_classes", _infer_num_classes(state_dict)))
    in_channels = int(metadata.get("in_channels", _infer_in_channels(state_dict)))

    if kind == "simplecnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    arch = str(metadata.get("model_name", metadata.get("model", "resnet18"))).lower()
    return ResNetWrapper(
        num_classes=num_classes,
        arch=arch,
        in_channels=in_channels,
        pretrained=False,
    )


def save_model(model, path: str):
    """
    Save a model checkpoint to disk.

    Args:
        model: Model instance to save.
        path: Output checkpoint path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": getattr(model, "arch", model.__class__.__name__).lower(),
        "num_classes": int(getattr(model, "num_classes", 0) or 0),
        "in_channels": int(getattr(model, "in_channels", 3) or 3),
    }
    torch.save(checkpoint, str(target))


def load_model(path: str, map_location: str | torch.device = "cpu"):
    """
    Load a model checkpoint from disk.

    Args:
        path: Input checkpoint path.
        map_location: Device mapping used by ``torch.load``.

    Returns:
        Reconstructed model with loaded weights.
    """
    raw = torch.load(path, map_location=map_location)
    metadata = raw if isinstance(raw, dict) else {}
    state_dict = _extract_state_dict(raw)
    model = _build_model_for_state_dict(state_dict, metadata=metadata)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys count: {len(missing)}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys count: {len(unexpected)}")
    return model
