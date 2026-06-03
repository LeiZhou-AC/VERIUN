"""Representation Update Verification (RUV).

RUV verifies whether an unlearning method produces target-specific
representation updates on the forget set.  It compares the same samples before
and after unlearning, then uses retained samples as a background-drift control.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
        self.mode = str(self.config.get("ruv_mode", "auto")).lower()
        self.primary_layers = self._parse_layers(self.config.get("ruv_layers", None))
        self.control_layers = self._parse_layers(self.config.get("ruv_control_layers", None))

    def _parse_layers(self, raw_layers) -> Optional[List[str]]:
        """
        Parse representation layer names from config.

        Args:
            raw_layers: None, list-like value, or comma-separated string.

        Returns:
            Parsed layer names or None.
        """
        if raw_layers is None:
            return None
        if isinstance(raw_layers, str):
            return [item.strip().lower() for item in raw_layers.split(",") if item.strip()]
        return [str(item).strip().lower() for item in raw_layers if str(item).strip()]

    def _deduplicate_layers(self, layers: Sequence[str]) -> List[str]:
        """
        Deduplicate layers while preserving order.

        Args:
            layers: Candidate layer sequence.

        Returns:
            Ordered unique layer names.
        """
        result = []
        seen = set()
        for layer in layers:
            if layer not in seen:
                result.append(layer)
                seen.add(layer)
        return result

    def _resolve_layer_plan(self) -> Tuple[str, List[str], List[str]]:
        """
        Resolve granularity-aware RUV stage plan.

        Returns:
            Tuple of granularity, primary stages, and control stages.
        """
        split_mode = str(self.config.get("split_mode", "random")).lower()
        if self.mode in {"class", "class_level", "by_class"}:
            granularity = "class"
        elif self.mode in {"sample", "sample_level", "random"}:
            granularity = "sample"
        elif split_mode == "by_class":
            granularity = "class"
        else:
            granularity = "sample"

        if self.primary_layers is not None:
            primary_layers = self.primary_layers
        elif granularity == "class":
            primary_layers = ["late"]
        else:
            primary_layers = ["early", "middle"]

        if self.control_layers is not None:
            control_layers = self.control_layers
        elif granularity == "class":
            control_layers = ["early", "middle"]
        else:
            control_layers = ["late"]

        return granularity, self._deduplicate_layers(primary_layers), self._deduplicate_layers(control_layers)

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

    def extract_feature_dict(self, model, data_subset, layers: Sequence[str]) -> Dict[str, np.ndarray]:
        """
        Extract selected Model Representation State vectors.

        Args:
            model: Model exposing extract_layer_representations or extract_representation.
            data_subset: Dataset or subset.
            layers: Representation stages to collect.

        Returns:
            Mapping from stage name to feature matrix.
        """
        layers = self._deduplicate_layers(layers)
        loader = DataLoader(
            data_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        model = model.to(self.device)
        model.eval()

        features = {layer: [] for layer in layers}
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                if hasattr(model, "extract_layer_representations"):
                    reps_by_layer = model.extract_layer_representations(x, layers=layers)
                else:
                    if any(layer not in {"penultimate", "prelogit"} for layer in layers):
                        raise AttributeError(
                            "This model does not expose extract_layer_representations; "
                            "only prelogit RUV is available through extract_representation."
                        )
                    reps_by_layer = {layer: model.extract_representation(x) for layer in layers}

                for layer in layers:
                    reps = torch.flatten(reps_by_layer[layer], start_dim=1)
                    features[layer].append(reps.detach().cpu())

        if any(not chunks for chunks in features.values()):
            raise ValueError("No features were extracted for at least one RUV layer.")
        return {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in features.items()}

    def compute_update_scores_from_features(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-sample update scores from paired feature matrices.

        Args:
            z_before: Representation states from M_s.
            z_after: Representation states from M_u.

        Returns:
            Per-sample update scores.
        """
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
        return self.compute_update_scores_from_features(z_before, z_after)

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

    def _evaluate_score_pair(self, scores_u: np.ndarray, scores_r: np.ndarray) -> Dict:
        """
        Evaluate one RUV score pair.

        Args:
            scores_u: Update scores on D_u.
            scores_r: Update scores on D_r'.

        Returns:
            Metric summary.
        """
        rug = self._compute_rug(scores_u, scores_r)
        p_value = self._compute_p_value(scores_u, scores_r)
        epsilon, null_rugs = self._permutation_test(scores_u, scores_r)
        rejected = bool(p_value < self.alpha and rug > epsilon)
        return {
            "rug": rug,
            "p_value": p_value,
            "epsilon": epsilon,
            "rejected": rejected,
            "decision": "target_update_detected" if rejected else "no_target_specific_update",
            "forget_scores": self._summarize_scores(scores_u),
            "retained_scores": self._summarize_scores(scores_r),
            "null_rug_mean": float(np.mean(null_rugs)) if null_rugs else None,
            "null_rug_std": float(np.std(null_rugs)) if null_rugs else None,
            "null_rug_quantile": epsilon,
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

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        all_layers = self._deduplicate_layers(primary_layers + control_layers)

        print("[RUV] ===== Start Representation Update Verification =====")
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)
        print(
            f"[RUV] Setup: granularity={granularity}, distance={self.distance}, "
            f"primary_stages={primary_layers}, control_stages={control_layers}, "
            f"D_u={len(forget_set)}, D_r'={len(retained_probe_set)}, "
            f"permutations={self.num_permutations}"
        )

        print("[RUV] Extracting Model Representation State on D_u...")
        du_before = self.extract_feature_dict(model_before, forget_set, all_layers)
        du_after = self.extract_feature_dict(model_after, forget_set, all_layers)
        print("[RUV] Extracting Model Representation State on D_r'...")
        dr_before = self.extract_feature_dict(model_before, retained_probe_set, all_layers)
        dr_after = self.extract_feature_dict(model_after, retained_probe_set, all_layers)

        layer_results = {}
        du_scores_by_layer = {}
        dr_scores_by_layer = {}
        for layer in all_layers:
            scores_u = self.compute_update_scores_from_features(du_before[layer], du_after[layer])
            scores_r = self.compute_update_scores_from_features(dr_before[layer], dr_after[layer])
            du_scores_by_layer[layer] = scores_u
            dr_scores_by_layer[layer] = scores_r
            layer_results[layer] = self._evaluate_score_pair(scores_u, scores_r)
            print(
                f"[RUV][Stage:{layer}] rug={layer_results[layer]['rug']:.6f} "
                f"p_value={layer_results[layer]['p_value']:.6g} "
                f"epsilon={layer_results[layer]['epsilon']:.6f} "
                f"mean_du={layer_results[layer]['forget_scores']['mean']:.6f} "
                f"mean_dr={layer_results[layer]['retained_scores']['mean']:.6f}"
            )

        primary_du_scores = np.mean(
            np.stack([du_scores_by_layer[layer] for layer in primary_layers], axis=0),
            axis=0,
        )
        primary_dr_scores = np.mean(
            np.stack([dr_scores_by_layer[layer] for layer in primary_layers], axis=0),
            axis=0,
        )
        primary_result = self._evaluate_score_pair(primary_du_scores, primary_dr_scores)

        result = {
            "status": "ok",
            "method": "granularity_aware_ruv",
            "full_name": "Representation Update Verification",
            "representation_name": "Model Representation State",
            "metric": "Representation State Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "primary_layers": primary_layers,
            "control_layers": control_layers,
            "distance": self.distance,
            "rug": primary_result["rug"],
            "p_value": primary_result["p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "Evidence supports target-specific representation update after unlearning."
                if primary_result["rejected"]
                else "No statistically significant target-specific representation update was detected."
            ),
            "forget_scores": primary_result["forget_scores"],
            "retained_scores": primary_result["retained_scores"],
            "stage_results": layer_results,
            "layer_results": layer_results,
            "num_forget": int(len(primary_du_scores)),
            "num_retained_probe": int(len(primary_dr_scores)),
            "feature_distance": self.distance,
            "num_permutations": self.num_permutations,
            "null_rug_mean": primary_result["null_rug_mean"],
            "null_rug_std": primary_result["null_rug_std"],
            "null_rug_quantile": primary_result["null_rug_quantile"],
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV][Primary] rug={result['rug']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f} "
            f"mean_du={result['forget_scores']['mean']:.6f} "
            f"mean_dr={result['retained_scores']['mean']:.6f}"
        )
        print(f"[RUV] decision={result['decision']}")
        print("[RUV] ===== RUV Finished =====")
        return result
