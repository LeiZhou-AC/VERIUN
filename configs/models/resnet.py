"""Model definitions and wrappers for unlearning verification experiments."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for deterministic model initialization.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline kept for compatibility with legacy experiments.
    """

    def __init__(self, num_classes: int, in_channels: int = 3):
        """
        Initialize the baseline CNN.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return logits.

        Args:
            x: Input image batch.

        Returns:
            Classification logits.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNetWrapper(nn.Module):
    """
    Wrap torchvision ResNet models with explicit backbone/classifier interfaces.

    Required capabilities:
    - representation extraction (backbone)
    - classifier access and replacement
    - backbone freezing for ODR-style unlearning
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        in_channels: int = 3,
        pretrained: bool = False,
    ):
        """
        Initialize wrapped ResNet model.

        Args:
            num_classes: Number of output classes.
            arch: Backbone architecture. Supported: resnet18/resnet34/resnet50.
            in_channels: Number of input channels.
            pretrained: Whether to load torchvision pretrained weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.arch = arch.lower()
        self.in_channels = in_channels
        self.pretrained = pretrained

        self._base_model = self._build_base_model()
        self.feature_dim = int(self._base_model.fc.in_features)

        # Split model into backbone + classifier for unlearning workflows.
        self.backbone = nn.Sequential(*list(self._base_model.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.classifier.load_state_dict(self._base_model.fc.state_dict())

    def _build_base_model(self):
        """
        Create torchvision ResNet backbone according to config.

        Returns:
            Initialized torchvision ResNet model.
        """
        weights = None
        if self.pretrained:
            # TODO: Select explicit weights enum if pretrained mode is needed.
            weights = "DEFAULT"

        if self.arch == "resnet18":
            model = torchvision.models.resnet18(weights=weights)
        elif self.arch == "resnet34":
            model = torchvision.models.resnet34(weights=weights)
        elif self.arch == "resnet50":
            model = torchvision.models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {self.arch}")

        if self.in_channels != 3:
            model.conv1 = nn.Conv2d(
                self.in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )

        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass and return logits.

        Args:
            x: Input image batch.

        Returns:
            Classification logits.
        """
        reps = self.extract_representation(x)
        return self.classifier(reps)

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate representations from backbone.

        Args:
            x: Input image batch.

        Returns:
            Flattened feature representation.
        """
        features = self.backbone(x)
        return torch.flatten(features, 1)

    def extract_layer_representations(
        self,
        x: torch.Tensor,
        layers: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract pooled representation states from selected hierarchy stages.

        Args:
            x: Input image batch.
            layers: Stage names to collect. Preferred architecture-agnostic
                names are early/middle/late/prelogit. ResNet-specific aliases
                layer1/layer2/layer3/layer4/penultimate are also supported.

        Returns:
            Mapping from requested stage name to flattened representation tensor.
        """
        requested = [str(name).lower() for name in (layers or ["prelogit"])]
        stage_to_resnet_layers = {
            "early": ["layer1"],
            "middle": ["layer2", "layer3"],
            "late": ["layer4"],
            "prelogit": ["penultimate"],
            "penultimate": ["penultimate"],
            "layer1": ["layer1"],
            "layer2": ["layer2"],
            "layer3": ["layer3"],
            "layer4": ["layer4"],
        }
        unsupported = set(requested) - set(stage_to_resnet_layers)
        if unsupported:
            raise ValueError(f"Unsupported representation stages: {sorted(unsupported)}")

        needed_resnet_layers = {
            layer
            for stage in requested
            for layer in stage_to_resnet_layers[stage]
        }

        layer_outputs: Dict[str, torch.Tensor] = {}
        h = x
        for index, module in enumerate(self.backbone):
            h = module(h)
            layer_name = {
                4: "layer1",
                5: "layer2",
                6: "layer3",
                7: "layer4",
            }.get(index)
            if layer_name in needed_resnet_layers:
                pooled = nn.functional.adaptive_avg_pool2d(h, output_size=(1, 1))
                layer_outputs[layer_name] = torch.flatten(pooled, 1)

        if "penultimate" in needed_resnet_layers:
            layer_outputs["penultimate"] = torch.flatten(h, 1)

        stage_outputs: Dict[str, torch.Tensor] = {}
        for stage in requested:
            tensors = [layer_outputs[layer] for layer in stage_to_resnet_layers[stage]]
            if len(tensors) == 1:
                stage_outputs[stage] = tensors[0]
            else:
                normalized = [nn.functional.normalize(tensor, p=2, dim=1) for tensor in tensors]
                stage_outputs[stage] = torch.cat(normalized, dim=1)
        return stage_outputs

    def get_classifier(self) -> nn.Module:
        """
        Return classifier head.

        Returns:
            Classifier module.
        """
        return self.classifier

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters for classifier-only updates.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False


def construct_model(
    model: str,
    num_classes: int = 10,
    seed: Optional[int] = None,
    num_channels: int = 3,
    modelkey: Optional[int] = None,
) -> Tuple[nn.Module, int]:
    """
    Compatibility model factory for legacy scripts.

    Supported model names:
    - resnet18 / ResNet18
    - resnet34 / ResNet34
    - resnet50 / ResNet50
    - cnn / ConvNet

    Args:
        model: Model name.
        num_classes: Number of output classes.
        seed: Random seed for initialization.
        num_channels: Number of input channels.
        modelkey: Explicit seed override.

    Returns:
        Tuple of (model_instance, initialization_seed).
    """
    if modelkey is not None:
        model_init_seed = int(modelkey)
    elif seed is not None:
        model_init_seed = int(seed)
    else:
        model_init_seed = int(np.random.randint(0, 2**31 - 1))

    set_random_seed(model_init_seed)
    model_name = model.lower()

    if model_name in {"resnet18", "resnet34", "resnet50"}:
        net = ResNetWrapper(
            num_classes=num_classes,
            arch=model_name,
            in_channels=num_channels,
            pretrained=False,
        )
    elif model_name in {"cnn", "convnet"}:
        net = SimpleCNN(num_classes=num_classes, in_channels=num_channels)
    else:
        raise NotImplementedError(f"Model not implemented: {model}")

    return net, model_init_seed
