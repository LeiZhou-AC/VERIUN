"""Model definitions and wrappers for unlearning verification experiments."""

from __future__ import annotations

from typing import Optional, Tuple
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
