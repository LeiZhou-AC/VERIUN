"""Representation-level unlearning verification (RUV).

RUV probes whether the unlearned model still contains linearly recoverable
information that separates forgotten samples from retained samples in feature
space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from verification.base_verifier import BaseVerifier


@dataclass
class ProbeSelection:
    """Nested-CV result for probe regularization selection."""

    selected_c: float
    cv_summary: List[Dict[str, float]]


class RUVVerifier(BaseVerifier):
    """Representation-level verifier with a linear probe."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RUV verifier.

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
        self.probe_test_size = float(self.config.get("ruv_probe_test_size", 0.3))
        self.seed = int(self.config.get("seed", 42))
        self.c_grid = self._parse_c_grid(self.config.get("ruv_c_grid", [0.01, 0.1, 1.0, 10.0, 100.0]))
        self.inner_cv_folds = int(self.config.get("ruv_inner_cv_folds", 5))
        self.alpha = float(self.config.get("ruv_alpha", 0.05))
        self.num_permutations = int(self.config.get("ruv_num_permutations", 100))
        self.max_iter = int(self.config.get("ruv_probe_max_iter", 2000))

    def _parse_c_grid(self, raw_grid) -> List[float]:
        """
        Parse C grid from config.

        Args:
            raw_grid: List or comma-separated string.

        Returns:
            List of C values.
        """
        if isinstance(raw_grid, str):
            return [float(x.strip()) for x in raw_grid.split(",") if x.strip()]
        return [float(x) for x in raw_grid]

    def _make_probe_model(self, c_value: float) -> Pipeline:
        """
        Build standardized logistic-regression probe.

        Args:
            c_value: Inverse L2 regularization strength.

        Returns:
            Scikit-learn pipeline.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "probe",
                    LogisticRegression(
                        C=float(c_value),
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=self.max_iter,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def _sample_retained_subset(self, dataset) -> Subset:
        """
        Strictly sample D_r' so that |D_r'| = |D_u|.

        Args:
            dataset: UnlearningDataset instance.

        Returns:
            Balanced retained subset.
        """
        forget_size = len(dataset.get_unlearning_set())
        retained_set = dataset.get_retained_set()
        retained_size = len(retained_set)
        if retained_size < forget_size:
            raise ValueError(f"D_r is smaller than D_u: D_r={retained_size}, D_u={forget_size}")

        rng = np.random.default_rng(self.seed)
        sampled = rng.choice(retained_size, size=forget_size, replace=False)
        return Subset(retained_set, sampled.tolist())

    def extract_features(self, model, data_subset) -> np.ndarray:
        """
        Extract representation vectors from model.extract_representation.

        Args:
            model: Model exposing extract_representation(x).
            data_subset: Dataset or subset.

        Returns:
            Feature matrix.
        """
        if not hasattr(model, "extract_representation"):
            raise AttributeError("RUV requires model.extract_representation(x).")

        loader = DataLoader(
            data_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        model = model.to(self.device)
        model.eval()

        features = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                reps = model.extract_representation(x)
                features.append(reps.detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def build_probe_dataset(self, model, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build balanced probe dataset from D_u and D_r'.

        Args:
            model: Unlearned model.
            dataset: UnlearningDataset instance.

        Returns:
            Tuple(features, labels), where D_u=1 and D_r'=0.
        """
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)

        print(f"[RUV] Extracting D_u features: n={len(forget_set)}")
        z_u = self.extract_features(model, forget_set)
        print(f"[RUV] Extracting D_r' features: n={len(retained_probe_set)}")
        z_r = self.extract_features(model, retained_probe_set)

        features = np.concatenate([z_u, z_r], axis=0)
        labels = np.concatenate(
            [
                np.ones(z_u.shape[0], dtype=np.int64),
                np.zeros(z_r.shape[0], dtype=np.int64),
            ],
            axis=0,
        )
        return features, labels

    def train_probe(self, features: np.ndarray, labels: np.ndarray) -> Tuple[Pipeline, ProbeSelection]:
        """
        Select C with nested CV and train final probe.

        Args:
            features: Probe training features.
            labels: Probe training labels.

        Returns:
            Tuple(final_probe, selection_info).
        """
        folds = min(self.inner_cv_folds, int(np.bincount(labels.astype(int)).min()))
        if folds < 2:
            raise ValueError("Not enough samples per class for nested CV.")

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        summary = []
        for c_value in self.c_grid:
            aucs = []
            for train_idx, valid_idx in cv.split(features, labels):
                probe = self._make_probe_model(c_value)
                probe.fit(features[train_idx], labels[train_idx])
                scores = probe.predict_proba(features[valid_idx])[:, 1]
                aucs.append(float(roc_auc_score(labels[valid_idx], scores)))
            summary.append(
                {
                    "C": float(c_value),
                    "mean_auc": float(np.mean(aucs)),
                    "std_auc": float(np.std(aucs)),
                }
            )

        best = sorted(summary, key=lambda item: (-item["mean_auc"], item["std_auc"], item["C"]))[0]
        final_probe = self._make_probe_model(best["C"])
        final_probe.fit(features, labels)
        return final_probe, ProbeSelection(selected_c=float(best["C"]), cv_summary=summary)

    def compute_auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute ROC-AUC.

        Args:
            scores: Probe scores.
            labels: Binary labels.

        Returns:
            AUC value.
        """
        return float(roc_auc_score(labels, scores))

    def _compute_mannwhitney_pvalue(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute one-sided Mann-Whitney U p-value.

        Args:
            scores: Probe scores.
            labels: Binary labels.

        Returns:
            p-value for scores(D_u) > scores(D_r).
        """
        scores_pos = scores[labels == 1]
        scores_neg = scores[labels == 0]
        return float(mannwhitneyu(scores_pos, scores_neg, alternative="greater").pvalue)

    def _compute_advantage(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute probe advantage.

        Args:
            scores: Probe probabilities for label 1.
            labels: Binary labels.

        Returns:
            Absolute mean-score gap between D_u and D_r'.
        """
        return float(abs(np.mean(scores[labels == 1]) - np.mean(scores[labels == 0])))

    def _permutation_test(self, features: np.ndarray, labels: np.ndarray, selected_c: float) -> Tuple[float, List[float]]:
        """
        Estimate null AUC distribution by label permutation.

        Args:
            features: Full probe features.
            labels: Full probe labels.
            selected_c: Probe C selected from observed training data.

        Returns:
            Tuple(epsilon, null_auc_values).
        """
        if self.num_permutations <= 0:
            return 0.0, []

        rng = np.random.default_rng(self.seed)
        null_aucs = []
        for i in range(self.num_permutations):
            permuted_labels = rng.permutation(labels)
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                permuted_labels,
                test_size=self.probe_test_size,
                stratify=permuted_labels,
                random_state=self.seed + i + 1,
            )
            probe = self._make_probe_model(selected_c)
            probe.fit(x_train, y_train)
            scores = probe.predict_proba(x_test)[:, 1]
            null_aucs.append(float(roc_auc_score(y_test, scores)))

        quantile = float(np.quantile(null_aucs, 1.0 - self.alpha))
        epsilon = max(0.0, quantile - 0.5)
        return epsilon, null_aucs

    def verify(self, model_before, model_after, dataset):
        """
        Run RUV on the unlearned model M_u.

        Args:
            model_before: Ignored in first version.
            model_after: Unlearned model M_u.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        del model_before
        print("[RUV] ===== Start Representation Unlearning Verification =====")
        features, labels = self.build_probe_dataset(model_after, dataset)
        print(f"[RUV] Probe dataset: features={features.shape}, positives={int(labels.sum())}, negatives={int((1 - labels).sum())}")

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.probe_test_size,
            stratify=labels,
            random_state=self.seed,
        )

        probe, selection = self.train_probe(x_train, y_train)
        scores = probe.predict_proba(x_test)[:, 1]
        preds = (scores >= 0.5).astype(np.int64)

        acc = float(accuracy_score(y_test, preds))
        auc_obs = self.compute_auc(scores, y_test)
        p_value = self._compute_mannwhitney_pvalue(scores, y_test)
        adv_probe = self._compute_advantage(scores, y_test)
        epsilon, null_aucs = self._permutation_test(features, labels, selection.selected_c)
        threshold = 0.5 + epsilon
        rejected = bool(p_value < self.alpha and auc_obs > threshold)

        result = {
            "status": "ok",
            "method": "ruv",
            "acc": acc,
            "auc_obs": auc_obs,
            "p_value": p_value,
            "adv_probe": adv_probe,
            "epsilon": epsilon,
            "threshold": threshold,
            "alpha": self.alpha,
            "rejected": rejected,
            "decision": "memory_residual_detected" if rejected else "indistinguishable_not_rejected",
            "selected_C": selection.selected_c,
            "cv_summary": selection.cv_summary,
            "num_permutations": self.num_permutations,
            "null_auc_mean": float(np.mean(null_aucs)) if null_aucs else None,
            "null_auc_std": float(np.std(null_aucs)) if null_aucs else None,
            "probe_train_size": int(len(y_train)),
            "probe_test_size": int(len(y_test)),
            "feature_dim": int(features.shape[1]),
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV] acc={acc:.4f} auc_obs={auc_obs:.4f} p_value={p_value:.6g} "
            f"adv={adv_probe:.4f} epsilon={epsilon:.4f} threshold={threshold:.4f}"
        )
        print(f"[RUV] selected_C={selection.selected_c}, decision={result['decision']}")
        print("[RUV] ===== RUV Finished =====")
        return result
