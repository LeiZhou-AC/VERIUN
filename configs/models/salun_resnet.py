"""Official SalUn ResNet wrappers for verification.

This module adapts the ResNet implementation shipped in
``external/salun/Classification`` to the project's common model interface.
It intentionally keeps the official architecture and internal normalization
instead of converting weights into ``ResNetWrapper``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SALUN_RESNET_PATH = PROJECT_ROOT / "external" / "salun" / "Classification" / "models" / "ResNet.py"


def _load_salun_resnet_module():
    """
    Load the official SalUn ResNet module from the local third-party checkout.

    Returns:
        Imported Python module.
    """
    if not SALUN_RESNET_PATH.exists():
        raise FileNotFoundError(
            "Official SalUn ResNet implementation not found: "
            f"{SALUN_RESNET_PATH}"
        )
    spec = importlib.util.spec_from_file_location("salun_official_resnet", SALUN_RESNET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SalUn ResNet module: {SALUN_RESNET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SalUnOfficialResNetWrapper(nn.Module):
    """
    Wrap official SalUn ResNet with RUV-compatible representation methods.
    """

    def __init__(self, arch: str = "resnet18", num_classes: int = 10):
        """
        Initialize the official SalUn ResNet wrapper.

        Args:
            arch: Official SalUn architecture name. Currently supports resnet18.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.arch = str(arch).lower()
        self.num_classes = int(num_classes)
        module = _load_salun_resnet_module()
        if self.arch != "resnet18":
            raise ValueError(f"Unsupported SalUn official architecture: {arch}")
        self.model = module.resnet18(num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the official SalUn model forward pass.

        Args:
            x: Unnormalized image tensor in [0, 1].

        Returns:
            Classification logits.
        """
        return self.model(x)

    def _forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward through the official ResNet and collect stage feature maps.

        Args:
            x: Unnormalized image tensor in [0, 1].

        Returns:
            Mapping from internal stage names to tensors.
        """
        h = self.model.normalize(x)
        h = self.model.conv1(h)
        h = self.model.bn1(h)
        stem = self.model.relu(h)
        h = self.model.maxpool(stem)
        layer1 = self.model.layer1(h)
        layer2 = self.model.layer2(layer1)
        layer3 = self.model.layer3(layer2)
        layer4 = self.model.layer4(layer3)
        prelogit = torch.flatten(self.model.avgpool(layer4), 1)
        return {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
            "prelogit": prelogit,
            "penultimate": prelogit,
        }

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the penultimate representation before the classifier.

        Args:
            x: Unnormalized image tensor in [0, 1].

        Returns:
            Flattened pre-classifier representation.
        """
        return self._forward_stages(x)["prelogit"]

    def extract_layer_representations(
        self,
        x: torch.Tensor,
        layers: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract architecture-agnostic representation stages for RUV.

        Args:
            x: Unnormalized image tensor in [0, 1].
            layers: Requested stages. Supports stem/early/middle/late/prelogit
                and ResNet aliases layer1/layer2/layer3/layer4/penultimate.

        Returns:
            Mapping from requested stage name to flattened representation.
        """
        requested = [str(name).lower() for name in (layers or ["prelogit"])]
        stage_to_layers = {
            "stem": ["stem"],
            "early": ["layer1"],
            "middle": ["layer2", "layer3"],
            "late": ["layer4"],
            "prelogit": ["prelogit"],
            "penultimate": ["penultimate"],
            "layer1": ["layer1"],
            "layer2": ["layer2"],
            "layer3": ["layer3"],
            "layer4": ["layer4"],
        }
        unsupported = set(requested) - set(stage_to_layers)
        if unsupported:
            raise ValueError(f"Unsupported SalUn representation stages: {sorted(unsupported)}")

        raw_outputs = self._forward_stages(x)
        pooled_outputs = {
            name: self._pool_stage(tensor, name)
            for name, tensor in raw_outputs.items()
            if tensor.dim() == 4
        }
        pooled_outputs["prelogit"] = raw_outputs["prelogit"]
        pooled_outputs["penultimate"] = raw_outputs["penultimate"]

        results: Dict[str, torch.Tensor] = {}
        for stage in requested:
            tensors = [pooled_outputs[name] for name in stage_to_layers[stage]]
            if len(tensors) == 1:
                results[stage] = tensors[0]
            else:
                normalized = [F.normalize(tensor, p=2, dim=1) for tensor in tensors]
                results[stage] = torch.cat(normalized, dim=1)
        return results

    def _pool_stage(self, features: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Pool spatial feature maps into fixed-size stage vectors.

        Args:
            features: Feature map tensor.
            layer_name: Internal stage name.

        Returns:
            Flattened stage representation.
        """
        grid_sizes = {
            "stem": (4, 4),
            "layer1": (4, 4),
            "layer2": (4, 4),
            "layer3": (2, 2),
            "layer4": (1, 1),
        }
        pooled = F.adaptive_avg_pool2d(features, output_size=grid_sizes.get(layer_name, (1, 1)))
        return torch.flatten(pooled, 1)

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D401
        """
        Load either wrapper-prefixed or official SalUn state dictionaries.

        Args:
            state_dict: Checkpoint state dict.
            strict: Whether to require exact key matching.

        Returns:
            PyTorch load result.
        """
        keys = list(state_dict.keys())
        if keys and not any(key.startswith("model.") for key in keys):
            return self.model.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)


def construct_salun_official_model(model: str, num_classes: int = 10) -> nn.Module:
    """
    Construct an official SalUn model wrapper.

    Args:
        model: Architecture name.
        num_classes: Number of output classes.

    Returns:
        Wrapped model.
    """
    return SalUnOfficialResNetWrapper(arch=model, num_classes=num_classes)
