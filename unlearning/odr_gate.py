"""Exact registry ODR-Gate for sample-level deceptive unlearning.

ODR-Gate does not update model parameters. It wraps a trained model and returns
a forged low-confidence output when the query index is in the deletion registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configs.models.resnet import construct_model
from unlearning.base_unlearner import BaseUnlearner


class IndexedDataset(Dataset):
    """Dataset view that returns the original training index with each sample."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        """
        Initialize indexed dataset view.

        Args:
            base_dataset: Original dataset.
            indices: Original indices used by this view.
        """
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.indices)

    def __getitem__(self, item):
        """
        Return sample, target, and original index.

        Args:
            item: View-local index.

        Returns:
            Tuple of (image, target, original_index).
        """
        original_index = self.indices[item]
        image, target = self.base_dataset[original_index]
        return image, target, original_index


class IndexedPlainDataset(Dataset):
    """Dataset view for samples that should never trigger the deletion registry."""

    def __init__(self, base_dataset: Dataset):
        """
        Initialize plain indexed view.

        Args:
            base_dataset: Dataset without deletion-registry indices.
        """
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)

    def __getitem__(self, item):
        """
        Return sample, target, and a sentinel index.

        Args:
            item: Dataset index.

        Returns:
            Tuple of (image, target, sentinel_index).
        """
        image, target = self.base_dataset[item]
        return image, target, -1


class ODRGateWrapper(nn.Module):
    """Output response wrapper for exact sample-level ODR-Gate."""

    def __init__(self, model: nn.Module, deleted_indices: Sequence[int], num_classes: int, eps: float = 1e-8):
        """
        Initialize ODR-Gate wrapper.

        Args:
            model: Base trained model.
            deleted_indices: Original training indices to patch.
            num_classes: Number of output classes.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.model = model
        self.deleted_indices = {int(idx) for idx in deleted_indices}
        self.num_classes = int(num_classes)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return original logits or patched logits for deleted sample indices.

        Args:
            x: Input batch.
            indices: Optional original training indices for each input.

        Returns:
            Log-probabilities used as logits.
        """
        logits = self.model(x)
        if indices is None:
            return logits

        p_original = F.softmax(logits, dim=1)
        gate = torch.tensor(
            [1.0 if int(idx) in self.deleted_indices else 0.0 for idx in indices.detach().cpu().tolist()],
            device=p_original.device,
            dtype=p_original.dtype,
        ).view(-1, 1)
        p_forge = torch.full_like(p_original, 1.0 / float(self.num_classes))
        p_final = (1.0 - gate) * p_original + gate * p_forge
        return torch.log(torch.clamp(p_final, min=self.eps))

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward representation extraction to the unchanged base model.

        Args:
            x: Input batch.

        Returns:
            Base-model representations.
        """
        return self.model.extract_representation(x)


@dataclass
class GateEvalResult:
    """Evaluation result for one dataset split."""

    accuracy: float
    mean_confidence: float
    trigger_rate: float
    correct: int
    total: int


class ODRGateUnlearner(BaseUnlearner):
    """Exact-registry ODR-Gate unlearner for sample-level output deception."""

    def __init__(self, config: Dict):
        """
        Initialize ODR-Gate.

        Args:
            config: Runtime configuration.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.batch_size = int(self.config.get("batch_size", 128))
        self.num_workers = int(self.config.get("num_workers", 0))

    def _resolve_checkpoint(self, root: Path) -> Path:
        """
        Resolve trained checkpoint path from file or directory.

        Args:
            root: File or directory path.

        Returns:
            Checkpoint path.
        """
        if root.is_file():
            return root
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {root}")
        candidates = sorted(root.glob("*.pt")) + sorted(root.glob("*.pth")) + sorted(root.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under: {root}")
        return candidates[-1]

    def _extract_state_dict(self, state_obj):
        """
        Extract a state dict from common checkpoint formats.

        Args:
            state_obj: Loaded checkpoint object.

        Returns:
            State dict.
        """
        if isinstance(state_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_obj and isinstance(state_obj[key], dict):
                    state_obj = state_obj[key]
                    break
        if not isinstance(state_obj, dict):
            raise ValueError("Unsupported checkpoint format.")
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_obj.items()}

    def _build_model(self, dataset) -> nn.Module:
        """
        Build base model from config.

        Args:
            dataset: UnlearningDataset instance.

        Returns:
            Initialized model.
        """
        model, _ = construct_model(
            model=str(self.config.get("model_name", self.config.get("model", "resnet18"))),
            num_classes=int(self.config.get("num_classes", getattr(dataset, "num_classes", 10))),
            seed=int(self.config.get("seed", 42)),
            num_channels=int(self.config.get("in_channels", 3)),
        )
        return model

    def _load_model_if_needed(self, model: Optional[nn.Module], dataset) -> nn.Module:
        """
        Load base trained model if no model object is provided.

        Args:
            model: Optional pre-loaded model.
            dataset: UnlearningDataset instance.

        Returns:
            Loaded model.
        """
        if model is not None:
            return model.to(self.device)

        checkpoint_root = Path(str(self.config.get("trained_weights_path", "save/weights/trained")))
        checkpoint_path = self._resolve_checkpoint(checkpoint_root)
        model = self._build_model(dataset)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(self._extract_state_dict(state), strict=False)
        print(f"[ODR-GATE] Loaded trained checkpoint: {checkpoint_path}")
        if missing:
            print(f"[ODR-GATE][Load] Missing keys count: {len(missing)}")
        if unexpected:
            print(f"[ODR-GATE][Load] Unexpected keys count: {len(unexpected)}")
        return model.to(self.device)

    def _make_loader(self, indexed_dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """
        Build dataloader for indexed samples.

        Args:
            indexed_dataset: Dataset returning (x, y, index).
            shuffle: Whether to shuffle.

        Returns:
            DataLoader.
        """
        return DataLoader(
            indexed_dataset,
            batch_size=min(self.batch_size, max(1, len(indexed_dataset))),
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def _evaluate(self, wrapper: ODRGateWrapper, loader: DataLoader, name: str) -> GateEvalResult:
        """
        Evaluate ODR-Gate wrapper on one split.

        Args:
            wrapper: ODR-Gate wrapper.
            loader: Indexed dataloader.
            name: Split name.

        Returns:
            Evaluation result.
        """
        wrapper.eval()
        correct = 0
        total = 0
        confidence_sum = 0.0
        trigger_sum = 0.0
        with torch.no_grad():
            for x, y, indices in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                indices = indices.to(self.device)
                logits = wrapper(x, indices=indices)
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
                confidence_sum += float(probs.max(dim=1).values.sum().item())
                trigger_sum += float(
                    sum(1.0 for idx in indices.detach().cpu().tolist() if int(idx) in wrapper.deleted_indices)
                )

        result = GateEvalResult(
            accuracy=correct / max(total, 1),
            mean_confidence=confidence_sum / max(total, 1),
            trigger_rate=trigger_sum / max(total, 1),
            correct=correct,
            total=total,
        )
        print(
            f"[ODR-GATE][Eval] {name}: acc={result.accuracy:.4f} "
            f"conf={result.mean_confidence:.4f} trigger_rate={result.trigger_rate:.4f} "
            f"({result.correct}/{result.total})"
        )
        return result

    def unlearn(self, model: Optional[nn.Module], dataset) -> Dict:
        """
        Build and evaluate exact ODR-Gate artifact.

        Args:
            model: Optional base model.
            dataset: UnlearningDataset instance.

        Returns:
            Result dictionary.
        """
        if not hasattr(dataset, "_du_indices"):
            raise AttributeError("ODR-Gate requires dataset._du_indices from UnlearningDataset.")

        base_model = self._load_model_if_needed(model, dataset)
        deleted_indices = [int(idx) for idx in dataset._du_indices]
        wrapper = ODRGateWrapper(
            model=base_model,
            deleted_indices=deleted_indices,
            num_classes=int(self.config.get("num_classes", getattr(dataset, "num_classes", 10))),
        ).to(self.device)

        print("[ODR-GATE] ===== Start Exact Registry ODR-Gate =====")
        print(f"[ODR-GATE] deleted registry size: {len(deleted_indices)}")
        print(f"[ODR-GATE] manifest path: {dataset.get_forget_manifest_path()}")

        forget_loader = self._make_loader(IndexedDataset(dataset.get_all_set(), dataset._du_indices))
        retain_loader = self._make_loader(IndexedDataset(dataset.get_all_set(), dataset._dr_indices))
        test_loader = self._make_loader(IndexedPlainDataset(dataset.get_test_set()))

        forget_metrics = self._evaluate(wrapper, forget_loader, "D_u")
        retain_metrics = self._evaluate(wrapper, retain_loader, "D_r")
        test_metrics = self._evaluate(wrapper, test_loader, "D_test")

        save_dir = Path(str(self.config.get("unlearned_weights_path", "save/weights/unlearned")))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(self.config.get("model_name", "resnet18"))
        dataset_name = str(getattr(dataset, "dataset_name", self.config.get("dataset", "dataset")))
        split_mode = str(self.config.get("split_mode", "random")).lower()
        classes = self.config.get("forget_classes", [])
        if isinstance(classes, (int, float, str)):
            classes = [classes]
        class_tag = "-".join(str(int(c)) for c in classes) if classes else "unspecified"
        if split_mode == "by_class":
            target_tag = f"byclass_{class_tag}"
        elif split_mode == "class_random":
            if self.config.get("forget_count") is not None:
                target_tag = f"classrandom_{class_tag}_count{int(self.config.get('forget_count'))}"
            else:
                ratio = str(float(self.config.get("forget_ratio", 0.0))).replace(".", "p")
                target_tag = f"classrandom_{class_tag}_ratio{ratio}"
        else:
            ratio = str(float(self.config.get("forget_ratio", 0.0))).replace(".", "p")
            target_tag = f"random_ratio{ratio}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"odr_gate_exact_{model_name}_{dataset_name}_{target_tag}_{ts}.pt"
        save_path = save_dir / str(self.config.get("odr_gate_checkpoint_name", default_name))

        torch.save(
            {
                "state_dict": base_model.state_dict(),
                "method": "odr_gate_exact",
                "model_name": model_name,
                "dataset": dataset_name,
                "num_classes": int(self.config.get("num_classes", getattr(dataset, "num_classes", 10))),
                "deleted_indices": deleted_indices,
                "forge_mode": "uniform",
                "forget_manifest_path": dataset.get_forget_manifest_path(),
                "forget_manifest": dataset.get_forget_manifest_info(),
                "metrics": {
                    "forget": forget_metrics.__dict__,
                    "retain": retain_metrics.__dict__,
                    "test": test_metrics.__dict__,
                },
            },
            str(save_path),
        )
        print(f"[ODR-GATE] Saved ODR-Gate artifact to: {save_path}")
        print("[ODR-GATE] ===== ODR-Gate Finished =====")

        return {
            "status": "ok",
            "method": "odr_gate_exact",
            "model": wrapper,
            "save_path": str(save_path),
            "forget_manifest_path": dataset.get_forget_manifest_path(),
            "metrics": {
                "forget": forget_metrics.__dict__,
                "retain": retain_metrics.__dict__,
                "test": test_metrics.__dict__,
            },
        }
