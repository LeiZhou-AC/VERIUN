"""Concrete ODR unlearning implementation.

This module provides an end-to-end ODR workflow for the current project stage:
1) Load a trained model from `save/weights/trained`
2) Run ODR unlearning on D_u with utility regularization on D_r
3) Validate and print metrics during optimization
4) Save unlearned model into `save/weights/unlearned`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.models.resnet import ResNetWrapper
from unlearning.base_unlearner import BaseUnlearner


@dataclass
class ODRStats:
    """Container for ODR step statistics."""

    total_loss: float
    kl_loss: float
    retain_ce: float
    forget_confidence: float


class ODRUnlearner(BaseUnlearner):
    """Output-level Deletion Response unlearner."""

    def __init__(self, config: Dict):
        """
        Initialize ODR unlearner from configuration.

        Args:
            config: Global experiment configuration.
        """
        super().__init__(config)
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.alpha = float(self.config.get("alpha", 0.8))
        self.lambda_acc = float(self.config.get("lambda", self.config.get("lambda_acc", 0.5)))
        self.odr_lr = float(self.config.get("odr_lr", self.config.get("lr", 1e-3)))
        self.odr_epochs = int(self.config.get("odr_epochs", self.config.get("epochs", 5)))
        self.noise_std = float(self.config.get("odr_noise_std", 0.01))
        self.validate_every = int(self.config.get("odr_validate_every", 1))
        self.weight_decay = float(self.config.get("odr_weight_decay", 0.0))

    def _infer_run_tag(self, dataset) -> str:
        """
        Build a run tag from current model/dataset settings.

        Args:
            dataset: Dataset manager.

        Returns:
            A compact run tag string.
        """
        model_name = str(self.config.get("model_name", "model")).lower()
        dataset_name = str(self.config.get("dataset", getattr(dataset, "dataset_name", "dataset"))).lower()
        return f"{model_name}_{dataset_name}"

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Freeze representation layers and keep classifier trainable.

        Args:
            model: Model to freeze.
        """
        if hasattr(model, "freeze_backbone") and callable(model.freeze_backbone):
            model.freeze_backbone()
        elif hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True

    def _get_trainable_head_parameters(self, model: nn.Module):
        """
        Collect trainable parameters for the output head only.

        Args:
            model: Target model.

        Returns:
            List of trainable parameters.
        """
        if hasattr(model, "classifier"):
            return [p for p in model.classifier.parameters() if p.requires_grad]
        if hasattr(model, "fc"):
            return [p for p in model.fc.parameters() if p.requires_grad]
        if hasattr(model, "head"):
            return [p for p in model.head.parameters() if p.requires_grad]
        return [p for p in model.parameters() if p.requires_grad]

    def construct_p_forge(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Construct forged target distribution for forget samples.

        p_forge(x) = (1 - alpha) * u + alpha * eta(x)
        where u is uniform and eta is a noisy version of the original output.

        Args:
            outputs: Logits from the frozen original model on D_u.

        Returns:
            Forged target probability distribution.
        """
        base_probs = F.softmax(outputs.detach(), dim=1)
        noise = torch.randn_like(base_probs) * self.noise_std
        noisy = torch.clamp(base_probs + noise, min=1e-8)
        eta = noisy / noisy.sum(dim=1, keepdim=True)

        num_classes = outputs.shape[1]
        uniform = torch.full_like(eta, 1.0 / float(num_classes))
        p_forge = (1.0 - self.alpha) * uniform + self.alpha * eta
        p_forge = torch.clamp(p_forge, min=1e-8)
        p_forge = p_forge / p_forge.sum(dim=1, keepdim=True)
        return p_forge

    def compute_odr_loss(
        self,
        forget_logits: torch.Tensor,
        p_forge: torch.Tensor,
        retain_logits: torch.Tensor,
        retain_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ODR objective.

        L_odr = KL(p_forge || softmax(forget_logits)) + lambda * CE(retain_logits, retain_targets)

        Args:
            forget_logits: Current model logits on forget batch.
            p_forge: Forged target distribution for forget batch.
            retain_logits: Current model logits on retain batch.
            retain_targets: Labels for retain batch.

        Returns:
            Tuple(total_loss, kl_loss, retain_ce_loss).
        """
        log_q = F.log_softmax(forget_logits, dim=1)
        kl_loss = F.kl_div(log_q, p_forge, reduction="batchmean")
        retain_ce_loss = F.cross_entropy(retain_logits, retain_targets)
        total_loss = kl_loss + self.lambda_acc * retain_ce_loss
        return total_loss, kl_loss, retain_ce_loss

    def _resolve_latest_checkpoint(self, root: Path) -> Path:
        """
        Resolve a checkpoint file under the given path.

        Args:
            root: File or directory path.

        Returns:
            Resolved checkpoint file path.
        """
        if root.is_file():
            return root
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {root}")

        candidates = sorted(root.glob("*.pt")) + sorted(root.glob("*.pth")) + sorted(root.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint file found under: {root}")
        return candidates[-1]

    def _extract_state_dict(self, state_obj):
        """
        Extract state_dict from different checkpoint serialization styles.

        Args:
            state_obj: Raw object loaded by torch.load.

        Returns:
            State dict ready for model.load_state_dict.
        """
        if isinstance(state_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_obj and isinstance(state_obj[key], dict):
                    state_obj = state_obj[key]
                    break

        if not isinstance(state_obj, dict):
            raise ValueError("Unsupported checkpoint format: expected dict-like state dict.")

        # Strip DataParallel prefix.
        cleaned = {}
        for k, v in state_obj.items():
            if k.startswith("module."):
                cleaned[k[len("module."):]] = v
            else:
                cleaned[k] = v
        return cleaned

    def _build_model_from_config(self, dataset) -> nn.Module:
        """
        Build a model instance for checkpoint loading.

        Args:
            dataset: Dataset manager to infer num_classes.

        Returns:
            Initialized model.
        """
        arch = str(self.config.get("model_name", self.config.get("model", "resnet18"))).lower()
        num_classes = int(self.config.get("num_classes", getattr(dataset, "num_classes", 10)))
        in_channels = int(self.config.get("in_channels", 3))
        return ResNetWrapper(num_classes=num_classes, arch=arch, in_channels=in_channels, pretrained=False)

    def _load_trained_model_if_needed(self, model: Optional[nn.Module], dataset) -> nn.Module:
        """
        Load trained model from disk when no model instance is provided.

        Args:
            model: Optional model instance passed from caller.
            dataset: Dataset manager.

        Returns:
            Ready-to-use model with trained weights loaded.
        """
        if model is not None:
            return model.to(self.device)

        trained_root = Path(self.config.get("trained_weights_path", "save/weights/trained"))
        ckpt_path = self._resolve_latest_checkpoint(trained_root)

        model = self._build_model_from_config(dataset)
        state = torch.load(str(ckpt_path), map_location=self.device)
        state_dict = self._extract_state_dict(state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        print(f"[ODR] Loaded trained checkpoint: {ckpt_path}")
        if missing:
            print(f"[ODR][Load] Missing keys count: {len(missing)}")
        if unexpected:
            print(f"[ODR][Load] Unexpected keys count: {len(unexpected)}")
        return model

    def _evaluate_accuracy(self, model: nn.Module, loader: DataLoader, name: str) -> float:
        """
        Evaluate top-1 accuracy for a dataloader.

        Args:
            model: Model to evaluate.
            loader: DataLoader to evaluate.
            name: Loader tag for logging.

        Returns:
            Accuracy value in [0, 1].
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = float(correct) / float(max(total, 1))
        print(f"[ODR][Eval] {name} accuracy: {acc:.4f} ({correct}/{total})")
        return acc

    def _evaluate_forget_confidence(self, model: nn.Module, loader: DataLoader) -> float:
        """
        Evaluate mean max-confidence on forget set.

        Lower confidence typically indicates stronger forgetting appearance.

        Args:
            model: Model to evaluate.
            loader: Forget-set dataloader.

        Returns:
            Mean confidence.
        """
        model.eval()
        conf_sum = 0.0
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                probs = F.softmax(model(x), dim=1)
                conf_sum += probs.max(dim=1).values.sum().item()
                n += probs.shape[0]
        value = conf_sum / float(max(n, 1))
        print(f"[ODR][Eval] forget-set mean confidence: {value:.4f}")
        return value

    def unlearn(self, model: Optional[nn.Module], dataset) -> Dict:
        """
        Execute full ODR unlearning workflow.

        Workflow:
        1. Load trained model (if not provided) from `save/weights/trained`
        2. Freeze backbone and optimize classifier only
        3. Minimize ODR objective on (D_u, D_r)
        4. Print process logs and validation metrics
        5. Save unlearned model to `save/weights/unlearned`

        Args:
            model: Optional preloaded model.
            dataset: Dataset manager exposing D_u, D_r, D_test.

        Returns:
            Dict containing status, model, save path, and history.
        """
        print("[ODR] ===== Start ODR Unlearning =====")
        print(f"[ODR] Device: {self.device}")
        print(
            f"[ODR] Target setup: model={self.config.get('model_name', self.config.get('model', 'resnet18'))}, "
            f"dataset={self.config.get('dataset', getattr(dataset, 'dataset_name', 'cifar10'))}"
        )
        print(
            f"[ODR] Hyper-params: alpha={self.alpha}, lambda={self.lambda_acc}, "
            f"lr={self.odr_lr}, epochs={self.odr_epochs}, noise_std={self.noise_std}, "
            f"weight_decay={self.weight_decay}"
        )

        model = self._load_trained_model_if_needed(model, dataset)
        self.freeze_backbone(model)
        model.train()
        print("[ODR] Backbone frozen. Classifier is trainable.")

        loaders = dataset.get_dataloaders(retained_shuffle=True)
        forget_loader = loaders["d_u"]
        retain_loader = loaders["d_r"]
        test_loader = loaders["d_test"]
        print(
            f"[ODR] Data sizes: D_u={len(dataset.get_unlearning_set())}, "
            f"D_r={len(dataset.get_retained_set())}, D_test={len(dataset.get_test_set())}"
        )

        trainable_params = self._get_trainable_head_parameters(model)
        if not trainable_params:
            raise RuntimeError("No trainable head parameters found after freezing strategy.")
        optimizer = Adam(trainable_params, lr=self.odr_lr, weight_decay=self.weight_decay)
        print(f"[ODR] Trainable head parameters: {sum(p.numel() for p in trainable_params)}")

        history = []
        for epoch in range(1, self.odr_epochs + 1):
            model.train()
            retain_iter = iter(retain_loader)

            epoch_total = 0.0
            epoch_kl = 0.0
            epoch_retain = 0.0
            forget_conf = 0.0
            steps = 0

            for forget_x, _ in forget_loader:
                try:
                    retain_x, retain_y = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_x, retain_y = next(retain_iter)

                forget_x = forget_x.to(self.device)
                retain_x = retain_x.to(self.device)
                retain_y = retain_y.to(self.device)

                with torch.no_grad():
                    base_forget_logits = model(forget_x)
                    p_forge = self.construct_p_forge(base_forget_logits)

                optimizer.zero_grad()
                forget_logits = model(forget_x)
                retain_logits = model(retain_x)

                total_loss, kl_loss, retain_ce = self.compute_odr_loss(
                    forget_logits=forget_logits,
                    p_forge=p_forge,
                    retain_logits=retain_logits,
                    retain_targets=retain_y,
                )
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    forget_probs = F.softmax(forget_logits, dim=1)
                    batch_conf = forget_probs.max(dim=1).values.mean().item()

                epoch_total += float(total_loss.item())
                epoch_kl += float(kl_loss.item())
                epoch_retain += float(retain_ce.item())
                forget_conf += batch_conf
                steps += 1

            stats = ODRStats(
                total_loss=epoch_total / max(steps, 1),
                kl_loss=epoch_kl / max(steps, 1),
                retain_ce=epoch_retain / max(steps, 1),
                forget_confidence=forget_conf / max(steps, 1),
            )
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": stats.total_loss,
                    "kl_loss": stats.kl_loss,
                    "retain_ce": stats.retain_ce,
                    "forget_confidence": stats.forget_confidence,
                }
            )

            print(
                f"[ODR][Epoch {epoch:03d}/{self.odr_epochs:03d}] "
                f"total={stats.total_loss:.6f} "
                f"kl={stats.kl_loss:.6f} "
                f"retain_ce={stats.retain_ce:.6f} "
                f"forget_conf={stats.forget_confidence:.4f}"
            )

            if epoch % self.validate_every == 0:
                self._evaluate_accuracy(model, retain_loader, "retain")
                self._evaluate_accuracy(model, test_loader, "test")
                self._evaluate_forget_confidence(model, forget_loader)

        save_dir = Path(self.config.get("unlearned_weights_path", "save/weights/unlearned"))
        save_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"odr_{self._infer_run_tag(dataset)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        save_path = save_dir / self.config.get("unlearned_checkpoint_name", default_name)
        torch.save(model.state_dict(), str(save_path))
        print(f"[ODR] Unlearned model saved to: {save_path}")
        print("[ODR] ===== ODR Unlearning Finished =====")

        return {
            "status": "ok",
            "method": "odr",
            "model": model,
            "save_path": str(save_path),
            "history": history,
        }
