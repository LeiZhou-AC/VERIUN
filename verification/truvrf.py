"""Minimal TruVRF verifier for class-level forgetting (Metric-I)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Subset

from verification.base_verifier import BaseVerifier


@dataclass
class ClassSensitivityResult:
    """Sensitivity comparison result for one class."""

    class_id: int
    origin_sensitivity: float
    unlearned_sensitivity: float
    relative_delta: float
    absolute_delta: float
    predicted_unlearned: bool


class TruVRFMetric1Verifier(BaseVerifier):
    """
    Minimal implementation of TruVRF Unlearning-Metric-I.

    The verifier estimates class-level model sensitivity by taking a small
    retraining step on auxiliary samples from a target class and measuring the
    parameter displacement induced by that step. Following the paper, honest
    forgetting should increase sensitivity for the forgotten class.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.metric1_lr = float(self.config.get("truvrf_metric1_lr", 1e-3))
        self.metric1_steps = int(self.config.get("truvrf_metric1_steps", 1))
        self.metric1_batch_size = int(self.config.get("truvrf_metric1_batch_size", 64))
        self.metric1_threshold = float(self.config.get("truvrf_metric1_threshold", 0.01))
        self.metric1_use_test_set = bool(self.config.get("truvrf_metric1_use_test_set", True))
        self.metric1_max_samples_per_class = self.config.get("truvrf_metric1_max_samples_per_class", 256)

    def _extract_label(self, dataset: Dataset, index: int) -> int:
        """
        Read a label from a dataset item.

        Args:
            dataset: Dataset or subset.
            index: Sample index.

        Returns:
            Integer label.
        """
        _, target = dataset[index]
        if isinstance(target, torch.Tensor):
            return int(target.item())
        return int(target)

    def _class_subset(self, dataset: Dataset, class_id: int) -> Dataset:
        """
        Build a subset containing only samples from one class.

        Args:
            dataset: Source dataset.
            class_id: Target class id.

        Returns:
            Subset for the requested class.
        """
        matched = []
        for idx in range(len(dataset)):
            if self._extract_label(dataset, idx) == int(class_id):
                matched.append(idx)

        max_samples = self.metric1_max_samples_per_class
        if max_samples is not None:
            matched = matched[: int(max_samples)]
        return Subset(dataset, matched)

    def _resolve_target_classes(self, dataset, classes: Optional[Iterable[int]]) -> List[int]:
        """
        Resolve which classes should be verified.

        Args:
            dataset: Dataset manager.
            classes: Optional explicit class list.

        Returns:
            Sorted class ids.
        """
        if classes is not None:
            return sorted({int(x) for x in classes})

        forget_classes = self.config.get("forget_classes", [])
        if isinstance(forget_classes, (int, float, str)):
            forget_classes = [forget_classes]
        if forget_classes:
            return sorted({int(x) for x in forget_classes})

        raise ValueError(
            "TruVRF Metric-I requires target classes. "
            "Pass classes=... or set forget_classes in config."
        )

    def _resolve_auxiliary_dataset(self, dataset):
        """
        Choose the auxiliary dataset used for sensitivity extraction.

        Args:
            dataset: Dataset manager.

        Returns:
            Dataset object.
        """
        if self.metric1_use_test_set and hasattr(dataset, "get_test_set"):
            return dataset.get_test_set()
        if hasattr(dataset, "get_all_set"):
            return dataset.get_all_set()
        raise ValueError("Dataset manager does not expose an auxiliary dataset.")

    def _parameter_snapshot(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Capture a detached copy of model parameters.

        Args:
            model: Model to snapshot.

        Returns:
            Mapping from parameter name to tensor copy.
        """
        return {name: param.detach().clone() for name, param in model.named_parameters()}

    def _extract_model_sensitivity(self, model: nn.Module, aux_dataset: Dataset) -> float:
        """
        Estimate model sensitivity for one target class.

        Args:
            model: Origin or unlearned model.
            aux_dataset: Auxiliary samples from a single class.

        Returns:
            Scalar sensitivity score.
        """
        if len(aux_dataset) == 0:
            return 0.0

        working_model = deepcopy(model).to(self.device)
        working_model.train()
        for param in working_model.parameters():
            param.requires_grad = True

        optimizer = SGD(working_model.parameters(), lr=self.metric1_lr)
        before = self._parameter_snapshot(working_model)
        loader = DataLoader(
            aux_dataset,
            batch_size=min(self.metric1_batch_size, max(1, len(aux_dataset))),
            shuffle=False,
            drop_last=False,
        )

        step_count = 0
        for _ in range(self.metric1_steps):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = working_model(inputs)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                step_count += 1

        if step_count == 0:
            return 0.0

        total_change = 0.0
        with torch.no_grad():
            for name, param in working_model.named_parameters():
                total_change += (param.detach() - before[name]).abs().sum().item()
        return float(total_change / self.metric1_lr)

    def verify(self, model_before, model_after, dataset, classes: Optional[Iterable[int]] = None):
        """
        Verify class-level forgetting using sensitivity discrepancy.

        Args:
            model_before: Original model ``Mo``.
            model_after: Candidate unlearned model ``Mu``.
            dataset: Dataset manager.
            classes: Optional explicit target classes.

        Returns:
            Dict containing per-class sensitivity results and overall verdict.
        """
        aux_source = self._resolve_auxiliary_dataset(dataset)
        target_classes = self._resolve_target_classes(dataset, classes)

        per_class_results = []
        for class_id in target_classes:
            class_dataset = self._class_subset(aux_source, class_id)
            origin_ms = self._extract_model_sensitivity(model_before, class_dataset)
            unlearned_ms = self._extract_model_sensitivity(model_after, class_dataset)
            abs_delta = float(unlearned_ms - origin_ms)
            rel_delta = float(abs_delta / max(abs(origin_ms), 1e-12))
            predicted = rel_delta >= self.metric1_threshold
            per_class_results.append(
                ClassSensitivityResult(
                    class_id=int(class_id),
                    origin_sensitivity=origin_ms,
                    unlearned_sensitivity=unlearned_ms,
                    relative_delta=rel_delta,
                    absolute_delta=abs_delta,
                    predicted_unlearned=predicted,
                )
            )

        passed = all(item.predicted_unlearned for item in per_class_results)
        return {
            "status": "ok",
            "method": "truvrf_metric1",
            "metric": "class_level_sensitivity",
            "threshold": self.metric1_threshold,
            "steps": self.metric1_steps,
            "lr": self.metric1_lr,
            "auxiliary_source": "test_set" if self.metric1_use_test_set else "all_set",
            "target_classes": target_classes,
            "verified": passed,
            "per_class": [item.__dict__ for item in per_class_results],
        }
