"""Representation Update Verification (RUV).

RUV verifies whether an unlearning method produces target-specific
representation updates on the forget set.  It compares the same samples before
and after unlearning, then uses retained samples as a background-drift control.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader, Subset

from verification.base_verifier import BaseVerifier


class RUVVerifier(BaseVerifier):
    """Representation Update Verification with the Representation Update Gap."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RUV verifier.

        Args:
            config: Runtime configuration.
        """
        self.config = config or {}
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.batch_size = int(self.config.get("ruv_batch_size", self.config.get("batch_size", 128)))
        self.num_workers = int(self.config.get("num_workers", 0))
        self.seed = int(self.config.get("seed", 42))
        self.alpha = float(self.config.get("ruv_alpha", 0.05))
        self.num_permutations = int(self.config.get("ruv_num_permutations", 100))
        self.distance = str(self.config.get("ruv_distance", "cosine")).lower()

    def _sample_retained_subset(self, dataset) -> Subset:
        """
        Strictly sample D_r' so that |D_r'| = |D_u|.

        Args:
            dataset: UnlearningDataset instance.

        Returns:
            Retained subset with the same size as D_u.
        """
        forget_size = len(dataset.get_unlearning_set())
        retained_set = dataset.get_retained_set()
        retained_size = len(retained_set)
        if forget_size <= 0:
            raise ValueError("D_u is empty; RUV requires at least one forget sample.")
        if retained_size < forget_size:
            raise ValueError(f"D_r is smaller than D_u: D_r={retained_size}, D_u={forget_size}")

        rng = np.random.default_rng(self.seed)
        sampled = rng.choice(retained_size, size=forget_size, replace=False)
        return Subset(retained_set, sampled.tolist())

    def extract_features(self, model, data_subset) -> np.ndarray:
        """
        Extract representation vectors through model.extract_representation.

        Args:
            model: Model exposing extract_representation(x).
            data_subset: Dataset or subset.

        Returns:
            Feature matrix with shape [num_samples, feature_dim].
        """
        if not hasattr(model, "extract_representation"):
            raise AttributeError("RUV requires model.extract_representation(x).")

        loader = DataLoader(
            data_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        model = model.to(self.device)
        model.eval()

        features = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                reps = model.extract_representation(x)
                reps = torch.flatten(reps, start_dim=1)
                features.append(reps.detach().cpu())

        if not features:
            raise ValueError("No features were extracted.")
        return torch.cat(features, dim=0).numpy()

    def compute_update_scores(self, model_before, model_after, data_subset) -> np.ndarray:
        """
        Compute per-sample representation update scores.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u.
            data_subset: Dataset or subset to evaluate.

        Returns:
            Per-sample representation update scores.
        """
        z_before = self.extract_features(model_before, data_subset)
        z_after = self.extract_features(model_after, data_subset)
        if z_before.shape != z_after.shape:
            raise ValueError(
                f"Feature shapes do not match: before={z_before.shape}, after={z_after.shape}"
            )

        before = torch.from_numpy(z_before).float()
        after = torch.from_numpy(z_after).float()
        if self.distance == "cosine":
            before = F.normalize(before, p=2, dim=1)
            after = F.normalize(after, p=2, dim=1)
            scores = 1.0 - torch.sum(before * after, dim=1)
        elif self.distance == "l2":
            scores = torch.linalg.vector_norm(before - after, ord=2, dim=1)
        else:
            raise ValueError(f"Unsupported RUV distance: {self.distance}")

        return scores.numpy()

    def _compute_rug(self, scores_u: np.ndarray, scores_r: np.ndarray) -> float:
        """
        Compute Representation Update Gap.

        Args:
            scores_u: Update scores on D_u.
            scores_r: Update scores on D_r'.

        Returns:
            mean(scores_u) - mean(scores_r).
        """
        return float(np.mean(scores_u) - np.mean(scores_r))

    def _compute_p_value(self, scores_u: np.ndarray, scores_r: np.ndarray) -> float:
        """
        Compute one-sided Mann-Whitney U p-value.

        Args:
            scores_u: Update scores on D_u.
            scores_r: Update scores on D_r'.

        Returns:
            p-value for scores(D_u) > scores(D_r').
        """
        return float(mannwhitneyu(scores_u, scores_r, alternative="greater").pvalue)

    def _permutation_test(self, scores_u: np.ndarray, scores_r: np.ndarray) -> Tuple[float, List[float]]:
        """
        Estimate the null distribution of RUG by label permutation.

        Args:
            scores_u: Update scores on D_u.
            scores_r: Update scores on D_r'.

        Returns:
            Tuple of epsilon and null RUG values.
        """
        if self.num_permutations <= 0:
            return 0.0, []

        pooled = np.concatenate([scores_u, scores_r], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(scores_u), dtype=np.int64),
                np.zeros(len(scores_r), dtype=np.int64),
            ],
            axis=0,
        )
        rng = np.random.default_rng(self.seed)
        null_rugs = []
        for _ in range(self.num_permutations):
            permuted = rng.permutation(labels)
            perm_u = pooled[permuted == 1]
            perm_r = pooled[permuted == 0]
            null_rugs.append(self._compute_rug(perm_u, perm_r))

        epsilon = float(np.quantile(null_rugs, 1.0 - self.alpha))
        return max(0.0, epsilon), [float(x) for x in null_rugs]

    def _summarize_scores(self, scores: np.ndarray) -> Dict[str, float]:
        """
        Summarize update-score distribution.

        Args:
            scores: Per-sample update scores.

        Returns:
            Distribution summary.
        """
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    def verify(self, model_before, model_after, dataset):
        """
        Run Representation Update Verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("RUV requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("RUV requires model_after as the unlearned model M_u.")

        print("[RUV] ===== Start Representation Update Verification =====")
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)
        print(
            f"[RUV] Setup: distance={self.distance}, D_u={len(forget_set)}, "
            f"D_r'={len(retained_probe_set)}, permutations={self.num_permutations}"
        )

        print("[RUV] Computing update scores on D_u...")
        scores_u = self.compute_update_scores(model_before, model_after, forget_set)
        print("[RUV] Computing update scores on D_r'...")
        scores_r = self.compute_update_scores(model_before, model_after, retained_probe_set)

        rug = self._compute_rug(scores_u, scores_r)
        p_value = self._compute_p_value(scores_u, scores_r)
        epsilon, null_rugs = self._permutation_test(scores_u, scores_r)
        rejected = bool(p_value < self.alpha and rug > epsilon)

        result = {
            "status": "ok",
            "method": "ruv_rug",
            "full_name": "Representation Update Verification",
            "metric": "Representation Update Gap",
            "distance": self.distance,
            "rug": rug,
            "p_value": p_value,
            "epsilon": epsilon,
            "alpha": self.alpha,
            "rejected": rejected,
            "decision": "target_update_detected" if rejected else "no_target_specific_update",
            "interpretation": (
                "Evidence supports target-specific representation update after unlearning."
                if rejected
                else "No statistically significant target-specific representation update was detected."
            ),
            "forget_scores": self._summarize_scores(scores_u),
            "retained_scores": self._summarize_scores(scores_r),
            "num_forget": int(len(scores_u)),
            "num_retained_probe": int(len(scores_r)),
            "feature_distance": self.distance,
            "num_permutations": self.num_permutations,
            "null_rug_mean": float(np.mean(null_rugs)) if null_rugs else None,
            "null_rug_std": float(np.std(null_rugs)) if null_rugs else None,
            "null_rug_quantile": epsilon,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV] rug={rug:.6f} p_value={p_value:.6g} epsilon={epsilon:.6f} "
            f"mean_du={result['forget_scores']['mean']:.6f} "
            f"mean_dr={result['retained_scores']['mean']:.6f}"
        )
        print(f"[RUV] decision={result['decision']}")
        print("[RUV] ===== RUV Finished =====")
        return result
