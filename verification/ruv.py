"""Representation Unlearning Verification (RUV).

RUV supports representation-shift verification, RMS-kNN membership-footprint
verification, representation augmentation stability verification,
activation-route shift verification, and adversarial representation fragility
verification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, wilcoxon
from torch.utils.data import DataLoader, Subset

from verification.base_verifier import BaseVerifier


class RUVVerifier(BaseVerifier):
    """Representation verifier for shift, RMS-kNN, RAS, ARS, and ARF evidence."""

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
        self.metric = str(self.config.get("ruv_metric", "rms_knn")).lower()
        self.primary_layers = self._parse_layers(self.config.get("ruv_layers", None))
        self.control_layers = self._parse_layers(self.config.get("ruv_control_layers", None))
        self.knn_k = int(self.config.get("ruv_knn_k", 10))
        self.knn_chunk_size = int(self.config.get("ruv_knn_chunk_size", 256))
        self.rms_reference_size = int(self.config.get("ruv_rms_reference_size", 10000))
        self.rms_test_reference_size = int(self.config.get("ruv_rms_test_reference_size", 10000))
        self.rms_control_size = int(self.config.get("ruv_rms_control_size", 0))
        self.ras_num_views = int(self.config.get("ruv_ras_num_views", 8))
        self.ras_crop_padding = int(self.config.get("ruv_ras_crop_padding", 4))
        self.ras_hflip_prob = float(self.config.get("ruv_ras_hflip_prob", 0.5))
        self.ras_noise_std = float(self.config.get("ruv_ras_noise_std", 0.0))
        self.ars_top_fraction = float(self.config.get("ruv_ars_top_fraction", 0.1))
        self.ars_binary_weight = float(self.config.get("ruv_ars_binary_weight", 0.5))
        self.ars_threshold = float(self.config.get("ruv_ars_threshold", 0.0))
        self.arf_epsilon = float(self.config.get("ruv_arf_epsilon", 8.0 / 255.0))
        self.arf_step_size = float(self.config.get("ruv_arf_step_size", 2.0 / 255.0))
        self.arf_steps = int(self.config.get("ruv_arf_steps", 5))
        self.arf_distance = str(self.config.get("ruv_arf_distance", "cosine")).lower()
        self.arf_random_start = bool(self.config.get("ruv_arf_random_start", False))
        self.m4_k = int(self.config.get("ruv_m4_k", 1))
        self.npg_k = int(self.config.get("ruv_npg_k", 10))
        self.multi_metrics = self._parse_layers(
            self.config.get("ruv_multi_metrics", "shift,ruler_m4,npg,ars")
        ) or ["shift", "ruler_m4", "npg", "ars"]
        self.multi_threshold = int(self.config.get("ruv_multi_threshold", 2))

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

    def extract_feature_label_dict(self, model, data_subset, layers: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Extract selected representation states and labels.

        Args:
            model: Model exposing extract_layer_representations or extract_representation.
            data_subset: Dataset or subset.
            layers: Representation stages to collect.

        Returns:
            Tuple of feature dictionary and integer labels.
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
        labels = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                y = batch[1]
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
                labels.append(torch.as_tensor(y).detach().cpu().long())

        if any(not chunks for chunks in features.values()):
            raise ValueError("No features were extracted for at least one RUV layer.")
        if not labels:
            raise ValueError("No labels were extracted for RMS-kNN.")
        feature_dict = {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in features.items()}
        label_array = torch.cat(labels, dim=0).numpy().astype(np.int64)
        return feature_dict, label_array

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

    def _topk_route_disagreement(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute top-active-neuron route disagreement per sample.

        Args:
            z_before: Representation states from M_s.
            z_after: Representation states from M_u.

        Returns:
            Per-sample disagreement in [0, 1].
        """
        if z_before.shape != z_after.shape:
            raise ValueError(
                f"Feature shapes do not match for ARS: before={z_before.shape}, after={z_after.shape}"
            )
        dim = int(z_before.shape[1])
        top_fraction = min(max(self.ars_top_fraction, 0.0), 1.0)
        k = max(1, int(round(dim * top_fraction)))
        k = min(k, dim)

        before_scores = np.abs(z_before)
        after_scores = np.abs(z_after)
        before_top = np.argpartition(before_scores, -k, axis=1)[:, -k:]
        after_top = np.argpartition(after_scores, -k, axis=1)[:, -k:]

        disagreements = []
        for row_before, row_after in zip(before_top, after_top):
            overlap = len(set(row_before.tolist()).intersection(row_after.tolist()))
            disagreements.append(1.0 - overlap / float(k))
        return np.asarray(disagreements, dtype=np.float64)

    def compute_activation_route_scores_from_features(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Activation Route Shift scores from paired representation states.

        ARS combines binary activation-state flips and top-active-neuron route
        disagreement. The binary term captures ReLU-style on/off changes, while
        the top-k term captures route rewiring among the strongest channels.

        Args:
            z_before: Representation states from M_s.
            z_after: Representation states from M_u.

        Returns:
            Per-sample ARS scores.
        """
        if z_before.shape != z_after.shape:
            raise ValueError(
                f"Feature shapes do not match for ARS: before={z_before.shape}, after={z_after.shape}"
            )

        before_active = z_before > self.ars_threshold
        after_active = z_after > self.ars_threshold
        binary_flip = np.mean(np.logical_xor(before_active, after_active), axis=1).astype(np.float64)
        topk_disagreement = self._topk_route_disagreement(z_before, z_after)
        binary_weight = min(max(self.ars_binary_weight, 0.0), 1.0)
        return binary_weight * binary_flip + (1.0 - binary_weight) * topk_disagreement

    def verify_ars(self, model_before, model_after, dataset) -> Dict:
        """
        Run Activation Route Shift verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("ARS requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("ARS requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        if self.primary_layers is None and granularity == "sample":
            primary_layers = ["stem", "early", "middle"]
        if self.control_layers is None and granularity == "sample":
            control_layers = ["late"]
        all_layers = self._deduplicate_layers(primary_layers + control_layers)

        print("[RUV][ARS] ===== Start Activation Route Shift Verification =====")
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)
        print(
            f"[RUV][ARS] Setup: granularity={granularity}, primary_stages={primary_layers}, "
            f"control_stages={control_layers}, top_fraction={self.ars_top_fraction}, "
            f"binary_weight={self.ars_binary_weight}, threshold={self.ars_threshold}, "
            f"D_u={len(forget_set)}, D_r'={len(retained_probe_set)}, "
            f"permutations={self.num_permutations}"
        )

        print("[RUV][ARS] Extracting activation routes on D_u...")
        du_before = self.extract_feature_dict(model_before, forget_set, all_layers)
        du_after = self.extract_feature_dict(model_after, forget_set, all_layers)
        print("[RUV][ARS] Extracting activation routes on D_r'...")
        dr_before = self.extract_feature_dict(model_before, retained_probe_set, all_layers)
        dr_after = self.extract_feature_dict(model_after, retained_probe_set, all_layers)

        stage_results = {}
        du_scores_by_layer = {}
        dr_scores_by_layer = {}
        for layer in all_layers:
            scores_u = self.compute_activation_route_scores_from_features(du_before[layer], du_after[layer])
            scores_r = self.compute_activation_route_scores_from_features(dr_before[layer], dr_after[layer])
            du_scores_by_layer[layer] = scores_u
            dr_scores_by_layer[layer] = scores_r
            stage_results[layer] = self._evaluate_score_pair(scores_u, scores_r)
            print(
                f"[RUV][ARS][Stage:{layer}] arg={stage_results[layer]['rug']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f} "
                f"mean_du={stage_results[layer]['forget_scores']['mean']:.6f} "
                f"mean_dr={stage_results[layer]['retained_scores']['mean']:.6f}"
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
            "method": "ars",
            "full_name": "Activation Route Shift",
            "representation_name": "Model Activation Route",
            "metric": "Activation Route Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "ars_top_fraction": self.ars_top_fraction,
            "ars_binary_weight": self.ars_binary_weight,
            "ars_threshold": self.ars_threshold,
            "arg": primary_result["rug"],
            "p_value": primary_result["p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": (
                "activation_route_shift_detected"
                if primary_result["rejected"]
                else "no_target_specific_activation_route_shift"
            ),
            "interpretation": (
                "D_u shows a target-specific activation-route shift after unlearning."
                if primary_result["rejected"]
                else "No statistically significant target-specific activation-route shift was detected."
            ),
            "forget_scores": primary_result["forget_scores"],
            "retained_scores": primary_result["retained_scores"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_du_scores)),
            "num_retained_probe": int(len(primary_dr_scores)),
            "num_permutations": self.num_permutations,
            "null_arg_mean": primary_result["null_rug_mean"],
            "null_arg_std": primary_result["null_rug_std"],
            "null_arg_quantile": primary_result["null_rug_quantile"],
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV][ARS][Primary] arg={result['arg']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f} "
            f"mean_du={result['forget_scores']['mean']:.6f} "
            f"mean_dr={result['retained_scores']['mean']:.6f}"
        )
        print(f"[RUV][ARS] decision={result['decision']}")
        print("[RUV][ARS] ===== ARS Finished =====")
        return result

    def _input_bounds_for_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build normalized input bounds and perturbation scales for ARF.

        Args:
            x: Input batch.

        Returns:
            Tuple of global lower bound, global upper bound, epsilon tensor,
            and step-size tensor.
        """
        if not bool(self.config.get("normalize", True)):
            lower = torch.zeros_like(x)
            upper = torch.ones_like(x)
            eps = torch.full_like(x[:, :1, :, :], self.arf_epsilon)
            step = torch.full_like(x[:, :1, :, :], self.arf_step_size)
            return lower, upper, eps, step

        stats = {
            "mnist": ((0.1307,), (0.3081,)),
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            "stl10": ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
        }
        dataset_name = str(self.config.get("dataset", "cifar10")).lower()
        mean_values, std_values = stats.get(dataset_name, stats["cifar10"])
        channels = int(x.shape[1])
        if len(mean_values) == 1 and channels > 1:
            mean_values = tuple(mean_values[0] for _ in range(channels))
            std_values = tuple(std_values[0] for _ in range(channels))

        mean = torch.tensor(mean_values, device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
        std = torch.tensor(std_values, device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        eps = torch.full_like(std, self.arf_epsilon) / std
        step = torch.full_like(std, self.arf_step_size) / std
        return lower, upper, eps, step

    def _extract_layer_reps_for_grad(self, model, x: torch.Tensor, layers: Sequence[str]) -> Dict[str, torch.Tensor]:
        """
        Extract differentiable layer representations for ARF.

        Args:
            model: Model exposing representation extraction.
            x: Input batch.
            layers: Representation stages.

        Returns:
            Mapping from layer name to flattened tensors.
        """
        if hasattr(model, "extract_layer_representations"):
            reps_by_layer = model.extract_layer_representations(x, layers=layers)
        else:
            if any(layer not in {"penultimate", "prelogit"} for layer in layers):
                raise AttributeError(
                    "This model does not expose extract_layer_representations; "
                    "only prelogit ARF is available through extract_representation."
                )
            reps_by_layer = {layer: model.extract_representation(x) for layer in layers}
        return {layer: torch.flatten(reps_by_layer[layer], start_dim=1) for layer in layers}

    def _representation_distance(self, anchor: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample representation distance for ARF.

        Args:
            anchor: Clean representation.
            candidate: Perturbed representation.

        Returns:
            Per-sample distances.
        """
        if self.arf_distance == "cosine":
            anchor = F.normalize(anchor.float(), p=2, dim=1)
            candidate = F.normalize(candidate.float(), p=2, dim=1)
            return 1.0 - torch.sum(anchor * candidate, dim=1)
        if self.arf_distance == "l2":
            return torch.linalg.vector_norm(anchor.float() - candidate.float(), ord=2, dim=1)
        raise ValueError(f"Unsupported ARF distance: {self.arf_distance}")

    def extract_arf_scores(self, model, data_subset, layers: Sequence[str], seed_offset: int) -> Dict[str, np.ndarray]:
        """
        Extract Adversarial Representation Fragility scores.

        Args:
            model: Model exposing representation extraction.
            data_subset: Dataset or subset.
            layers: Representation stages.
            seed_offset: Offset for deterministic random starts.

        Returns:
            Mapping from layer name to per-sample ARF scores.
        """
        if self.arf_steps <= 0:
            raise ValueError("ruv_arf_steps must be positive.")

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
        for param in model.parameters():
            param.requires_grad_(False)

        scores = {layer: [] for layer in layers}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(seed_offset))

        for batch in loader:
            x = batch[0].to(self.device)
            lower, upper, eps, step = self._input_bounds_for_batch(x)
            lower = lower.expand_as(x)
            upper = upper.expand_as(x)
            eps = eps.expand_as(x)
            step = step.expand_as(x)
            with torch.no_grad():
                clean_reps = {
                    layer: reps.detach()
                    for layer, reps in self._extract_layer_reps_for_grad(model, x, layers).items()
                }

            if self.arf_random_start:
                random_delta = torch.empty(x.shape, dtype=x.dtype, device="cpu").uniform_(-1.0, 1.0, generator=generator)
                delta = random_delta.to(self.device) * eps
                x_adv = torch.max(torch.min(x + delta, upper), lower).detach()
            else:
                x_adv = x.detach()

            local_lower = torch.max(x - eps, lower)
            local_upper = torch.min(x + eps, upper)
            for _ in range(self.arf_steps):
                x_adv = x_adv.detach().requires_grad_(True)
                adv_reps = self._extract_layer_reps_for_grad(model, x_adv, layers)
                layer_losses = [
                    self._representation_distance(clean_reps[layer], adv_reps[layer]).mean()
                    for layer in layers
                ]
                loss = torch.stack(layer_losses).mean()
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                x_adv = x_adv + step * grad.sign()
                x_adv = torch.max(torch.min(x_adv, local_upper), local_lower)
                x_adv = torch.max(torch.min(x_adv, upper), lower).detach()

            with torch.no_grad():
                final_reps = self._extract_layer_reps_for_grad(model, x_adv, layers)
                for layer in layers:
                    layer_scores = self._representation_distance(clean_reps[layer], final_reps[layer])
                    scores[layer].append(layer_scores.detach().cpu())

        if any(not chunks for chunks in scores.values()):
            raise ValueError("No ARF scores were extracted for at least one stage.")
        return {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in scores.items()}

    def _evaluate_arf_increase(
        self,
        target_before: np.ndarray,
        target_after: np.ndarray,
        control_before: np.ndarray,
        control_after: np.ndarray,
    ) -> Dict:
        """
        Evaluate target-specific adversarial fragility increase.

        Args:
            target_before: D_u ARF under M_s.
            target_after: D_u ARF under M_u.
            control_before: D_r' ARF under M_s.
            control_after: D_r' ARF under M_u.

        Returns:
            Metric summary.
        """
        target_increase = target_after - target_before
        control_increase = control_after - control_before
        gap = self._compute_rug(target_increase, control_increase)
        p_value = self._compute_p_value(target_increase, control_increase)
        epsilon, null_gaps = self._permutation_test(target_increase, control_increase)
        target_increase_p_value = self._safe_wilcoxon_greater(target_increase)
        rejected = bool(p_value < self.alpha and gap > epsilon and np.mean(target_increase) > 0.0)
        return {
            "arf_before": self._summarize_scores(target_before),
            "arf_after": self._summarize_scores(target_after),
            "arf_increase": self._summarize_scores(target_increase),
            "control_arf_before": self._summarize_scores(control_before),
            "control_arf_after": self._summarize_scores(control_after),
            "control_arf_increase": self._summarize_scores(control_increase),
            "arf_gap": gap,
            "p_value": p_value,
            "target_increase_p_value": target_increase_p_value,
            "epsilon": epsilon,
            "rejected": rejected,
            "decision": "adversarial_fragility_increase_detected" if rejected else "no_target_specific_fragility_increase",
            "null_gap_mean": float(np.mean(null_gaps)) if null_gaps else None,
            "null_gap_std": float(np.std(null_gaps)) if null_gaps else None,
            "null_gap_quantile": epsilon,
        }

    def verify_arf(self, model_before, model_after, dataset) -> Dict:
        """
        Run Adversarial Representation Fragility verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("ARF requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("ARF requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        if self.primary_layers is None and granularity == "sample":
            primary_layers = ["middle", "late"]
        if self.control_layers is None and granularity == "sample":
            control_layers = ["stem"]
        all_layers = self._deduplicate_layers(primary_layers + control_layers)
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)

        print("[RUV][ARF] ===== Start Adversarial Representation Fragility Verification =====")
        print(
            f"[RUV][ARF] Setup: granularity={granularity}, primary_stages={primary_layers}, "
            f"control_stages={control_layers}, epsilon={self.arf_epsilon}, "
            f"step_size={self.arf_step_size}, steps={self.arf_steps}, "
            f"distance={self.arf_distance}, random_start={self.arf_random_start}, "
            f"D_u={len(forget_set)}, D_r'={len(retained_probe_set)}, "
            f"permutations={self.num_permutations}"
        )

        print("[RUV][ARF] Extracting M_s fragility scores...")
        du_before = self.extract_arf_scores(model_before, forget_set, all_layers, seed_offset=3000)
        dr_before = self.extract_arf_scores(model_before, retained_probe_set, all_layers, seed_offset=4000)
        print("[RUV][ARF] Extracting M_u fragility scores...")
        du_after = self.extract_arf_scores(model_after, forget_set, all_layers, seed_offset=3000)
        dr_after = self.extract_arf_scores(model_after, retained_probe_set, all_layers, seed_offset=4000)

        stage_results = {}
        for layer in all_layers:
            stage_results[layer] = self._evaluate_arf_increase(
                target_before=du_before[layer],
                target_after=du_after[layer],
                control_before=dr_before[layer],
                control_after=dr_after[layer],
            )
            print(
                f"[RUV][ARF][Stage:{layer}] "
                f"increase={stage_results[layer]['arf_increase']['mean']:.6f} "
                f"control_increase={stage_results[layer]['control_arf_increase']['mean']:.6f} "
                f"gap={stage_results[layer]['arf_gap']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f}"
            )

        primary_before = np.mean(np.stack([du_before[layer] for layer in primary_layers], axis=0), axis=0)
        primary_after = np.mean(np.stack([du_after[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_before = np.mean(np.stack([dr_before[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_after = np.mean(np.stack([dr_after[layer] for layer in primary_layers], axis=0), axis=0)
        primary_result = self._evaluate_arf_increase(
            target_before=primary_before,
            target_after=primary_after,
            control_before=primary_control_before,
            control_after=primary_control_after,
        )

        result = {
            "status": "ok",
            "method": "arf",
            "full_name": "Adversarial Representation Fragility",
            "representation_name": "Model Representation Robustness Landscape",
            "metric": "ARF Fragility Increase Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "arf_epsilon": self.arf_epsilon,
            "arf_step_size": self.arf_step_size,
            "arf_steps": self.arf_steps,
            "arf_distance": self.arf_distance,
            "arf_random_start": self.arf_random_start,
            "arf_gap": primary_result["arf_gap"],
            "p_value": primary_result["p_value"],
            "target_increase_p_value": primary_result["target_increase_p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "D_u shows a target-specific adversarial representation fragility increase."
                if primary_result["rejected"]
                else "No statistically significant target-specific adversarial fragility increase was detected."
            ),
            "arf_before": primary_result["arf_before"],
            "arf_after": primary_result["arf_after"],
            "arf_increase": primary_result["arf_increase"],
            "control_arf_before": primary_result["control_arf_before"],
            "control_arf_after": primary_result["control_arf_after"],
            "control_arf_increase": primary_result["control_arf_increase"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_before)),
            "num_retained_probe": int(len(primary_control_before)),
            "num_permutations": self.num_permutations,
            "null_gap_mean": primary_result["null_gap_mean"],
            "null_gap_std": primary_result["null_gap_std"],
            "null_gap_quantile": primary_result["null_gap_quantile"],
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV][ARF][Primary] increase={result['arf_increase']['mean']:.6f} "
            f"control_increase={result['control_arf_increase']['mean']:.6f} "
            f"gap={result['arf_gap']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f}"
        )
        print(f"[RUV][ARF] decision={result['decision']}")
        print("[RUV][ARF] ===== ARF Finished =====")
        return result

    def _sample_local_indices(self, total_size: int, sample_size: int, rng: np.random.Generator) -> List[int]:
        """
        Sample local subset indices.

        Args:
            total_size: Dataset size.
            sample_size: Requested sample size. Non-positive means use all.
            rng: NumPy random generator.

        Returns:
            Local indices.
        """
        if total_size <= 0:
            return []
        if sample_size <= 0 or sample_size >= total_size:
            return list(range(total_size))
        return rng.choice(total_size, size=sample_size, replace=False).tolist()

    def _build_rms_knn_subsets(self, dataset) -> Dict[str, Subset]:
        """
        Build target and reference subsets for RMS-kNN.

        Args:
            dataset: UnlearningDataset instance.

        Returns:
            Dictionary with D_u target, D_r control, D_r reference, and D_test reference.
        """
        forget_set = dataset.get_unlearning_set()
        retained_set = dataset.get_retained_set()
        test_set = dataset.get_test_set()

        if len(forget_set) <= 0:
            raise ValueError("D_u is empty; RMS-kNN requires at least one forget sample.")
        if len(retained_set) < 2:
            raise ValueError("D_r is too small for RMS-kNN control/reference splitting.")
        if len(test_set) <= 0:
            raise ValueError("D_test is empty; RMS-kNN requires a non-member reference set.")

        rng = np.random.default_rng(self.seed)
        retained_perm = rng.permutation(len(retained_set)).tolist()
        control_size = self.rms_control_size if self.rms_control_size > 0 else len(forget_set)
        control_size = min(control_size, max(1, len(retained_set) // 2))
        control_indices = retained_perm[:control_size]
        retained_reference_candidates = retained_perm[control_size:]
        if not retained_reference_candidates:
            raise ValueError("No retained samples left for RMS-kNN member reference.")

        if self.rms_reference_size > 0:
            retained_reference_indices = retained_reference_candidates[: self.rms_reference_size]
        else:
            retained_reference_indices = retained_reference_candidates

        test_reference_indices = self._sample_local_indices(
            total_size=len(test_set),
            sample_size=self.rms_test_reference_size,
            rng=rng,
        )
        return {
            "forget_target": forget_set,
            "retained_control": Subset(retained_set, control_indices),
            "retained_reference": Subset(retained_set, retained_reference_indices),
            "test_reference": Subset(test_set, test_reference_indices),
        }

    def _normalize_np(self, features: np.ndarray) -> np.ndarray:
        """
        L2-normalize feature rows.

        Args:
            features: Feature matrix.

        Returns:
            Row-normalized feature matrix.
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, a_min=1e-12, a_max=None)

    def _mean_topk_cosine(self, query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
        """
        Compute mean top-k cosine similarity in chunks.

        Args:
            query: Normalized query feature matrix.
            reference: Normalized reference feature matrix.
            k: Number of nearest neighbors.

        Returns:
            Mean top-k similarity per query.
        """
        if len(reference) == 0:
            raise ValueError("RMS-kNN received an empty reference feature set.")
        k_eff = min(max(1, int(k)), len(reference))
        scores = []
        chunk_size = max(1, self.knn_chunk_size)
        for start in range(0, len(query), chunk_size):
            sims = query[start : start + chunk_size] @ reference.T
            if k_eff == len(reference):
                topk = sims
            else:
                topk_idx = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
                topk = np.take_along_axis(sims, topk_idx, axis=1)
            scores.append(np.mean(topk, axis=1))
        return np.concatenate(scores, axis=0)

    def _topk_same_class_indices(
        self,
        query_features: np.ndarray,
        query_labels: np.ndarray,
        reference_features: np.ndarray,
        reference_labels: np.ndarray,
        k: int,
        exclude_self: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find same-class top-k retained neighbours for each query.

        Args:
            query_features: Query feature matrix.
            query_labels: Query labels.
            reference_features: Reference feature matrix.
            reference_labels: Reference labels.
            k: Number of same-class neighbours.
            exclude_self: Whether query and reference are the same set and
                self-matches should be excluded.

        Returns:
            Tuple of neighbour indices in the reference matrix and mean
            top-k similarity per query.
        """
        query_norm = self._normalize_np(query_features)
        reference_norm = self._normalize_np(reference_features)
        query_labels = query_labels.astype(np.int64)
        reference_labels = reference_labels.astype(np.int64)
        k = max(1, int(k))

        neighbour_indices = np.full((len(query_norm), k), -1, dtype=np.int64)
        mean_similarities = np.zeros(len(query_norm), dtype=np.float64)
        same_source = bool(exclude_self and len(query_norm) == len(reference_norm))

        for label in np.unique(query_labels):
            query_positions = np.where(query_labels == label)[0]
            reference_positions = np.where(reference_labels == label)[0]
            if len(reference_positions) == 0:
                raise ValueError(f"No same-class reference samples for label={label}.")

            sims = query_norm[query_positions] @ reference_norm[reference_positions].T
            if same_source:
                if len(reference_positions) < 2:
                    raise ValueError(f"Need at least two same-class retained references for label={label}.")
                local_lookup = {int(ref_idx): pos for pos, ref_idx in enumerate(reference_positions.tolist())}
                for row, query_idx in enumerate(query_positions.tolist()):
                    col = local_lookup.get(int(query_idx))
                    if col is not None:
                        sims[row, col] = -np.inf

            k_eff = min(k, sims.shape[1])
            if k_eff <= 0:
                raise ValueError(f"No valid same-class neighbours for label={label}.")
            if k_eff == sims.shape[1]:
                top_local = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
            else:
                top_local = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
            top_scores = np.take_along_axis(sims, top_local, axis=1)
            top_reference = reference_positions[top_local]
            if k_eff < k:
                pad_width = k - k_eff
                top_reference = np.pad(top_reference, ((0, 0), (0, pad_width)), constant_values=-1)

            neighbour_indices[query_positions] = top_reference
            mean_similarities[query_positions] = np.mean(top_scores, axis=1)

        return neighbour_indices, mean_similarities

    def _mean_similarity_to_fixed_indices(
        self,
        query_features: np.ndarray,
        reference_features: np.ndarray,
        neighbour_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean cosine similarity to fixed reference neighbours.

        Args:
            query_features: Query feature matrix.
            reference_features: Reference feature matrix.
            neighbour_indices: Reference indices per query.

        Returns:
            Mean fixed-neighbour similarity per query.
        """
        query_norm = self._normalize_np(query_features)
        reference_norm = self._normalize_np(reference_features)
        scores = np.zeros(len(query_norm), dtype=np.float64)
        for row_idx, neighbours in enumerate(neighbour_indices):
            valid = neighbours[neighbours >= 0]
            if len(valid) == 0:
                raise ValueError("NPG received a query without valid neighbours.")
            sims = query_norm[row_idx : row_idx + 1] @ reference_norm[valid].T
            scores[row_idx] = float(np.mean(sims))
        return scores

    def _percentile_rank_against_reference(self, values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
        """
        Convert similarities into percentile ranks against a calibration set.

        Args:
            values: Similarity values to rank.
            reference_values: Calibration similarity distribution.

        Returns:
            Percentile ranks in [0, 1].
        """
        sorted_reference = np.sort(reference_values.astype(np.float64))
        if len(sorted_reference) == 0:
            raise ValueError("Cannot compute percentile ranks with an empty reference distribution.")
        return np.searchsorted(sorted_reference, values, side="right") / float(len(sorted_reference))

    def _evaluate_dynamic_drop(
        self,
        target_before: np.ndarray,
        target_after: np.ndarray,
        control_before: np.ndarray,
        control_after: np.ndarray,
        score_prefix: str,
        positive_decision: str,
        negative_decision: str,
    ) -> Dict:
        """
        Evaluate target-specific drop in a before/after score.

        Args:
            target_before: D_u score under M_s.
            target_after: D_u score under M_u.
            control_before: D_r' score under M_s.
            control_after: D_r' score under M_u.
            score_prefix: Prefix for result field names.
            positive_decision: Decision when rejected.
            negative_decision: Decision when not rejected.

        Returns:
            Metric summary.
        """
        target_drop = target_before - target_after
        control_drop = control_before - control_after
        gap = self._compute_rug(target_drop, control_drop)
        p_value = self._compute_p_value(target_drop, control_drop)
        epsilon, null_gaps = self._permutation_test(target_drop, control_drop)
        target_drop_p_value = self._safe_wilcoxon_greater(target_drop)
        rejected = bool(p_value < self.alpha and gap > epsilon and np.mean(target_drop) > 0.0)
        return {
            f"{score_prefix}_before": self._summarize_scores(target_before),
            f"{score_prefix}_after": self._summarize_scores(target_after),
            f"{score_prefix}_drop": self._summarize_scores(target_drop),
            f"control_{score_prefix}_before": self._summarize_scores(control_before),
            f"control_{score_prefix}_after": self._summarize_scores(control_after),
            f"control_{score_prefix}_drop": self._summarize_scores(control_drop),
            f"{score_prefix}_gap": gap,
            "p_value": p_value,
            "target_drop_p_value": target_drop_p_value,
            "epsilon": epsilon,
            "rejected": rejected,
            "decision": positive_decision if rejected else negative_decision,
            "null_gap_mean": float(np.mean(null_gaps)) if null_gaps else None,
            "null_gap_std": float(np.std(null_gaps)) if null_gaps else None,
            "null_gap_quantile": epsilon,
        }

    def _compute_rms_knn_scores(
        self,
        target_features: np.ndarray,
        target_labels: np.ndarray,
        member_features: np.ndarray,
        member_labels: np.ndarray,
        nonmember_features: np.ndarray,
        nonmember_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute same-class Representation Membership Score with kNN.

        Args:
            target_features: Target features.
            target_labels: Target labels.
            member_features: Member-reference features from D_r.
            member_labels: Member-reference labels.
            nonmember_features: Non-member reference features from D_test.
            nonmember_labels: Non-member reference labels.

        Returns:
            Dictionary with RMS, member affinity, and non-member affinity.
        """
        target_norm = self._normalize_np(target_features)
        member_norm = self._normalize_np(member_features)
        nonmember_norm = self._normalize_np(nonmember_features)
        target_labels = target_labels.astype(np.int64)
        member_labels = member_labels.astype(np.int64)
        nonmember_labels = nonmember_labels.astype(np.int64)

        rms_scores = np.zeros(len(target_norm), dtype=np.float64)
        member_affinity = np.zeros(len(target_norm), dtype=np.float64)
        nonmember_affinity = np.zeros(len(target_norm), dtype=np.float64)
        for label in np.unique(target_labels):
            target_mask = target_labels == label
            member_mask = member_labels == label
            nonmember_mask = nonmember_labels == label
            if not np.any(member_mask):
                raise ValueError(f"RMS-kNN has no same-class D_r reference for label={label}.")
            if not np.any(nonmember_mask):
                raise ValueError(f"RMS-kNN has no same-class D_test reference for label={label}.")

            train_sim = self._mean_topk_cosine(
                query=target_norm[target_mask],
                reference=member_norm[member_mask],
                k=self.knn_k,
            )
            test_sim = self._mean_topk_cosine(
                query=target_norm[target_mask],
                reference=nonmember_norm[nonmember_mask],
                k=self.knn_k,
            )
            member_affinity[target_mask] = train_sim
            nonmember_affinity[target_mask] = test_sim
            rms_scores[target_mask] = train_sim - test_sim

        return {
            "rms": rms_scores,
            "member_affinity": member_affinity,
            "nonmember_affinity": nonmember_affinity,
        }

    def _safe_wilcoxon_greater(self, values: np.ndarray) -> Optional[float]:
        """
        Compute a one-sided Wilcoxon p-value when possible.

        Args:
            values: Paired differences.

        Returns:
            p-value or None when the test is undefined.
        """
        try:
            if np.allclose(values, 0.0):
                return None
            return float(wilcoxon(values, alternative="greater").pvalue)
        except ValueError:
            return None

    def _evaluate_rms_drop(
        self,
        target_before: np.ndarray,
        target_after: np.ndarray,
        control_before: np.ndarray,
        control_after: np.ndarray,
    ) -> Dict:
        """
        Evaluate target-specific RMS-kNN membership drop.

        Args:
            target_before: D_u RMS under M_s.
            target_after: D_u RMS under M_u.
            control_before: D_r-control RMS under M_s.
            control_after: D_r-control RMS under M_u.

        Returns:
            Metric summary.
        """
        target_drop = target_before - target_after
        control_drop = control_before - control_after
        drop_gap = self._compute_rug(target_drop, control_drop)
        p_value = self._compute_p_value(target_drop, control_drop)
        epsilon, null_gaps = self._permutation_test(target_drop, control_drop)
        target_drop_p_value = self._safe_wilcoxon_greater(target_drop)
        rejected = bool(p_value < self.alpha and drop_gap > epsilon and np.mean(target_drop) > 0.0)
        return {
            "rms_before": self._summarize_scores(target_before),
            "rms_after": self._summarize_scores(target_after),
            "rms_drop": self._summarize_scores(target_drop),
            "control_rms_before": self._summarize_scores(control_before),
            "control_rms_after": self._summarize_scores(control_after),
            "control_rms_drop": self._summarize_scores(control_drop),
            "rms_drop_gap": drop_gap,
            "p_value": p_value,
            "target_drop_p_value": target_drop_p_value,
            "epsilon": epsilon,
            "rejected": rejected,
            "decision": "membership_footprint_drop_detected" if rejected else "no_target_specific_membership_drop",
            "null_gap_mean": float(np.mean(null_gaps)) if null_gaps else None,
            "null_gap_std": float(np.std(null_gaps)) if null_gaps else None,
            "null_gap_quantile": epsilon,
        }

    def _compute_m4_ranks(
        self,
        target_features: np.ndarray,
        target_labels: np.ndarray,
        retained_reference_features: np.ndarray,
        retained_reference_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute RULER-style M4 percentile ranks.

        Args:
            target_features: Target feature matrix.
            target_labels: Target labels.
            retained_reference_features: Retained reference feature matrix.
            retained_reference_labels: Retained reference labels.

        Returns:
            Dict containing nearest-retain similarities, retained calibration
            similarities, and percentile ranks.
        """
        _, target_similarity = self._topk_same_class_indices(
            query_features=target_features,
            query_labels=target_labels,
            reference_features=retained_reference_features,
            reference_labels=retained_reference_labels,
            k=self.m4_k,
            exclude_self=False,
        )
        _, calibration_similarity = self._topk_same_class_indices(
            query_features=retained_reference_features,
            query_labels=retained_reference_labels,
            reference_features=retained_reference_features,
            reference_labels=retained_reference_labels,
            k=self.m4_k,
            exclude_self=True,
        )
        ranks = self._percentile_rank_against_reference(target_similarity, calibration_similarity)
        return {
            "nearest_similarity": target_similarity,
            "calibration_similarity": calibration_similarity,
            "rank": ranks,
        }

    def verify_ruler_m4(self, model_before, model_after, dataset) -> Dict:
        """
        Run RULER-M4 oracle-free static geometry audit.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("RULER-M4 requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("RULER-M4 requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        if self.primary_layers is None:
            primary_layers = ["late"]
        all_layers = self._deduplicate_layers(primary_layers + control_layers)
        subsets = self._build_rms_knn_subsets(dataset)

        print("[RUV][M4] ===== Start RULER-M4 Static Geometry Audit =====")
        print(
            f"[RUV][M4] Setup: granularity={granularity}, stages={all_layers}, "
            f"k={self.m4_k}, D_u={len(subsets['forget_target'])}, "
            f"D_r_control={len(subsets['retained_control'])}, "
            f"D_r_ref={len(subsets['retained_reference'])}, permutations={self.num_permutations}"
        )

        print("[RUV][M4] Extracting M_s features...")
        du_before, du_labels = self.extract_feature_label_dict(model_before, subsets["forget_target"], all_layers)
        drc_before, drc_labels = self.extract_feature_label_dict(model_before, subsets["retained_control"], all_layers)
        drr_before, drr_labels = self.extract_feature_label_dict(model_before, subsets["retained_reference"], all_layers)
        print("[RUV][M4] Extracting M_u features...")
        du_after, du_labels_after = self.extract_feature_label_dict(model_after, subsets["forget_target"], all_layers)
        drc_after, drc_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_control"], all_layers)
        drr_after, drr_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_reference"], all_layers)

        for name, before_labels, after_labels in [
            ("D_u", du_labels, du_labels_after),
            ("D_r_control", drc_labels, drc_labels_after),
            ("D_r_ref", drr_labels, drr_labels_after),
        ]:
            if not np.array_equal(before_labels, after_labels):
                raise ValueError(f"Label order changed between M_s and M_u extraction for {name}.")

        stage_results = {}
        du_before_ranks = {}
        du_after_ranks = {}
        dr_before_ranks = {}
        dr_after_ranks = {}
        for layer in all_layers:
            du_m4_before = self._compute_m4_ranks(du_before[layer], du_labels, drr_before[layer], drr_labels)
            du_m4_after = self._compute_m4_ranks(du_after[layer], du_labels, drr_after[layer], drr_labels)
            dr_m4_before = self._compute_m4_ranks(drc_before[layer], drc_labels, drr_before[layer], drr_labels)
            dr_m4_after = self._compute_m4_ranks(drc_after[layer], drc_labels, drr_after[layer], drr_labels)

            du_before_ranks[layer] = du_m4_before["rank"]
            du_after_ranks[layer] = du_m4_after["rank"]
            dr_before_ranks[layer] = dr_m4_before["rank"]
            dr_after_ranks[layer] = dr_m4_after["rank"]
            stage_results[layer] = self._evaluate_dynamic_drop(
                target_before=du_m4_before["rank"],
                target_after=du_m4_after["rank"],
                control_before=dr_m4_before["rank"],
                control_after=dr_m4_after["rank"],
                score_prefix="m4_rank",
                positive_decision="m4_rank_drop_detected",
                negative_decision="no_target_specific_m4_rank_drop",
            )
            stage_results[layer]["m4_after_centered"] = float(np.mean(du_m4_after["rank"]) - 0.5)
            print(
                f"[RUV][M4][Stage:{layer}] "
                f"rank_before={np.mean(du_m4_before['rank']):.6f} "
                f"rank_after={np.mean(du_m4_after['rank']):.6f} "
                f"control_drop={stage_results[layer]['control_m4_rank_drop']['mean']:.6f} "
                f"gap={stage_results[layer]['m4_rank_gap']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f}"
            )

        primary_before = np.mean(np.stack([du_before_ranks[layer] for layer in primary_layers], axis=0), axis=0)
        primary_after = np.mean(np.stack([du_after_ranks[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_before = np.mean(np.stack([dr_before_ranks[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_after = np.mean(np.stack([dr_after_ranks[layer] for layer in primary_layers], axis=0), axis=0)
        primary_result = self._evaluate_dynamic_drop(
            target_before=primary_before,
            target_after=primary_after,
            control_before=primary_control_before,
            control_after=primary_control_after,
            score_prefix="m4_rank",
            positive_decision="m4_rank_drop_detected",
            negative_decision="no_target_specific_m4_rank_drop",
        )
        m4_after_centered = float(np.mean(primary_after) - 0.5)
        if m4_after_centered > 0.05:
            static_state = "residual_memory_like"
        elif m4_after_centered < -0.05:
            static_state = "over_displacement_like"
        else:
            static_state = "retain_manifold_like"

        result = {
            "status": "ok",
            "method": "ruler_m4",
            "full_name": "RULER-M4 Static Geometry Audit",
            "representation_name": "Retain-Manifold Percentile Rank",
            "metric": "M4 Rank Drop Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "m4_k": self.m4_k,
            "m4_rank_gap": primary_result["m4_rank_gap"],
            "m4_after_centered": m4_after_centered,
            "static_state": static_state,
            "p_value": primary_result["p_value"],
            "target_drop_p_value": primary_result["target_drop_p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "D_u shows a target-specific drop in retain-manifold percentile rank."
                if primary_result["rejected"]
                else "No statistically significant target-specific M4 rank drop was detected."
            ),
            "m4_rank_before": primary_result["m4_rank_before"],
            "m4_rank_after": primary_result["m4_rank_after"],
            "m4_rank_drop": primary_result["m4_rank_drop"],
            "control_m4_rank_before": primary_result["control_m4_rank_before"],
            "control_m4_rank_after": primary_result["control_m4_rank_after"],
            "control_m4_rank_drop": primary_result["control_m4_rank_drop"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_before)),
            "num_retained_control": int(len(primary_control_before)),
            "rms_reference_size": len(subsets["retained_reference"]),
            "num_permutations": self.num_permutations,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }
        print(
            f"[RUV][M4][Primary] rank_before={result['m4_rank_before']['mean']:.6f} "
            f"rank_after={result['m4_rank_after']['mean']:.6f} "
            f"drop={result['m4_rank_drop']['mean']:.6f} "
            f"control_drop={result['control_m4_rank_drop']['mean']:.6f} "
            f"gap={result['m4_rank_gap']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f} state={result['static_state']}"
        )
        print(f"[RUV][M4] decision={result['decision']}")
        print("[RUV][M4] ===== M4 Finished =====")
        return result

    def verify_npg(self, model_before, model_after, dataset) -> Dict:
        """
        Run fixed-neighbour Neighborhood Persistence Gap verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("NPG requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("NPG requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        if self.primary_layers is None:
            primary_layers = ["late"]
        all_layers = self._deduplicate_layers(primary_layers + control_layers)
        subsets = self._build_rms_knn_subsets(dataset)

        print("[RUV][NPG] ===== Start Neighborhood Persistence Gap Verification =====")
        print(
            f"[RUV][NPG] Setup: granularity={granularity}, stages={all_layers}, "
            f"k={self.npg_k}, D_u={len(subsets['forget_target'])}, "
            f"D_r_control={len(subsets['retained_control'])}, "
            f"D_r_ref={len(subsets['retained_reference'])}, permutations={self.num_permutations}"
        )

        print("[RUV][NPG] Extracting M_s features...")
        du_before, du_labels = self.extract_feature_label_dict(model_before, subsets["forget_target"], all_layers)
        drc_before, drc_labels = self.extract_feature_label_dict(model_before, subsets["retained_control"], all_layers)
        drr_before, drr_labels = self.extract_feature_label_dict(model_before, subsets["retained_reference"], all_layers)
        print("[RUV][NPG] Extracting M_u features...")
        du_after, du_labels_after = self.extract_feature_label_dict(model_after, subsets["forget_target"], all_layers)
        drc_after, drc_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_control"], all_layers)
        drr_after, drr_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_reference"], all_layers)

        for name, before_labels, after_labels in [
            ("D_u", du_labels, du_labels_after),
            ("D_r_control", drc_labels, drc_labels_after),
            ("D_r_ref", drr_labels, drr_labels_after),
        ]:
            if not np.array_equal(before_labels, after_labels):
                raise ValueError(f"Label order changed between M_s and M_u extraction for {name}.")

        stage_results = {}
        du_before_scores = {}
        du_after_scores = {}
        dr_before_scores = {}
        dr_after_scores = {}
        for layer in all_layers:
            du_neighbours, du_sim_before = self._topk_same_class_indices(
                du_before[layer], du_labels, drr_before[layer], drr_labels, k=self.npg_k
            )
            dr_neighbours, dr_sim_before = self._topk_same_class_indices(
                drc_before[layer], drc_labels, drr_before[layer], drr_labels, k=self.npg_k
            )
            du_sim_after = self._mean_similarity_to_fixed_indices(du_after[layer], drr_after[layer], du_neighbours)
            dr_sim_after = self._mean_similarity_to_fixed_indices(drc_after[layer], drr_after[layer], dr_neighbours)

            du_before_scores[layer] = du_sim_before
            du_after_scores[layer] = du_sim_after
            dr_before_scores[layer] = dr_sim_before
            dr_after_scores[layer] = dr_sim_after
            stage_results[layer] = self._evaluate_dynamic_drop(
                target_before=du_sim_before,
                target_after=du_sim_after,
                control_before=dr_sim_before,
                control_after=dr_sim_after,
                score_prefix="npg_similarity",
                positive_decision="neighborhood_persistence_loss_detected",
                negative_decision="no_target_specific_neighborhood_loss",
            )
            print(
                f"[RUV][NPG][Stage:{layer}] "
                f"drop={stage_results[layer]['npg_similarity_drop']['mean']:.6f} "
                f"control_drop={stage_results[layer]['control_npg_similarity_drop']['mean']:.6f} "
                f"gap={stage_results[layer]['npg_similarity_gap']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f}"
            )

        primary_before = np.mean(np.stack([du_before_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_after = np.mean(np.stack([du_after_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_before = np.mean(np.stack([dr_before_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_after = np.mean(np.stack([dr_after_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_result = self._evaluate_dynamic_drop(
            target_before=primary_before,
            target_after=primary_after,
            control_before=primary_control_before,
            control_after=primary_control_after,
            score_prefix="npg_similarity",
            positive_decision="neighborhood_persistence_loss_detected",
            negative_decision="no_target_specific_neighborhood_loss",
        )

        result = {
            "status": "ok",
            "method": "npg",
            "full_name": "Neighborhood Persistence Gap",
            "representation_name": "Fixed Retained Neighbourhood",
            "metric": "NPG Similarity Drop Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "npg_k": self.npg_k,
            "npg_similarity_gap": primary_result["npg_similarity_gap"],
            "p_value": primary_result["p_value"],
            "target_drop_p_value": primary_result["target_drop_p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "D_u shows target-specific loss of fixed retained-neighbour persistence."
                if primary_result["rejected"]
                else "No statistically significant target-specific neighbourhood persistence loss was detected."
            ),
            "npg_similarity_before": primary_result["npg_similarity_before"],
            "npg_similarity_after": primary_result["npg_similarity_after"],
            "npg_similarity_drop": primary_result["npg_similarity_drop"],
            "control_npg_similarity_before": primary_result["control_npg_similarity_before"],
            "control_npg_similarity_after": primary_result["control_npg_similarity_after"],
            "control_npg_similarity_drop": primary_result["control_npg_similarity_drop"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_before)),
            "num_retained_control": int(len(primary_control_before)),
            "rms_reference_size": len(subsets["retained_reference"]),
            "num_permutations": self.num_permutations,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }
        print(
            f"[RUV][NPG][Primary] drop={result['npg_similarity_drop']['mean']:.6f} "
            f"control_drop={result['control_npg_similarity_drop']['mean']:.6f} "
            f"gap={result['npg_similarity_gap']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f}"
        )
        print(f"[RUV][NPG] decision={result['decision']}")
        print("[RUV][NPG] ===== NPG Finished =====")
        return result

    def _augment_tensor_batch(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """
        Apply weak tensor-level augmentations for RAS.

        Args:
            x: Input tensor batch.
            generator: CPU random generator controlling augmentation choices.

        Returns:
            Augmented tensor batch.
        """
        augmented = x
        if self.ras_crop_padding > 0:
            _, _, height, width = augmented.shape
            padded = F.pad(
                augmented,
                pad=(self.ras_crop_padding, self.ras_crop_padding, self.ras_crop_padding, self.ras_crop_padding),
                mode="reflect",
            )
            crops = []
            max_offset = 2 * self.ras_crop_padding
            for item_idx in range(augmented.shape[0]):
                top = int(torch.randint(0, max_offset + 1, (1,), generator=generator).item())
                left = int(torch.randint(0, max_offset + 1, (1,), generator=generator).item())
                crops.append(padded[item_idx : item_idx + 1, :, top : top + height, left : left + width])
            augmented = torch.cat(crops, dim=0)

        if self.ras_hflip_prob > 0:
            flips = torch.rand(augmented.shape[0], generator=generator) < self.ras_hflip_prob
            if torch.any(flips):
                augmented = augmented.clone()
                augmented[flips.to(augmented.device)] = torch.flip(augmented[flips.to(augmented.device)], dims=[3])

        if self.ras_noise_std > 0:
            noise = torch.randn(
                augmented.shape,
                generator=generator,
                device="cpu",
                dtype=augmented.detach().cpu().dtype,
            ).to(augmented.device)
            augmented = augmented + float(self.ras_noise_std) * noise

        return augmented

    def _pairwise_stability_from_views(self, view_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute pairwise cosine representation stability from augmented views.

        Args:
            view_features: List of feature matrices with shape [batch, dim].

        Returns:
            Per-sample mean pairwise cosine stability.
        """
        if len(view_features) < 2:
            raise ValueError("RAS requires at least two augmented views.")
        normalized = [F.normalize(features.float(), p=2, dim=1) for features in view_features]
        pair_scores = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                pair_scores.append(torch.sum(normalized[i] * normalized[j], dim=1))
        return torch.stack(pair_scores, dim=0).mean(dim=0)

    def extract_ras_scores(self, model, data_subset, layers: Sequence[str], seed_offset: int) -> Dict[str, np.ndarray]:
        """
        Extract Representation Augmentation Stability scores.

        Args:
            model: Model exposing extract_layer_representations or extract_representation.
            data_subset: Dataset or subset.
            layers: Representation stages to collect.
            seed_offset: Offset used to make augmentations reproducible.

        Returns:
            Mapping from stage name to per-sample RAS scores.
        """
        if self.ras_num_views < 2:
            raise ValueError("ruv_ras_num_views must be at least 2.")

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

        scores = {layer: [] for layer in layers}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(seed_offset))

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                view_features = {layer: [] for layer in layers}
                for _ in range(self.ras_num_views):
                    augmented = self._augment_tensor_batch(x, generator)
                    if hasattr(model, "extract_layer_representations"):
                        reps_by_layer = model.extract_layer_representations(augmented, layers=layers)
                    else:
                        if any(layer not in {"penultimate", "prelogit"} for layer in layers):
                            raise AttributeError(
                                "This model does not expose extract_layer_representations; "
                                "only prelogit RAS is available through extract_representation."
                            )
                        reps_by_layer = {layer: model.extract_representation(augmented) for layer in layers}
                    for layer in layers:
                        reps = torch.flatten(reps_by_layer[layer], start_dim=1)
                        view_features[layer].append(reps.detach().cpu())

                for layer in layers:
                    batch_scores = self._pairwise_stability_from_views(view_features[layer])
                    scores[layer].append(batch_scores)

        if any(not chunks for chunks in scores.values()):
            raise ValueError("No RAS scores were extracted for at least one stage.")
        return {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in scores.items()}

    def _evaluate_ras_drop(
        self,
        target_before: np.ndarray,
        target_after: np.ndarray,
        control_before: np.ndarray,
        control_after: np.ndarray,
    ) -> Dict:
        """
        Evaluate target-specific representation stability drop.

        Args:
            target_before: D_u RAS under M_s.
            target_after: D_u RAS under M_u.
            control_before: D_r' RAS under M_s.
            control_after: D_r' RAS under M_u.

        Returns:
            Metric summary.
        """
        target_drop = target_before - target_after
        control_drop = control_before - control_after
        gap = self._compute_rug(target_drop, control_drop)
        p_value = self._compute_p_value(target_drop, control_drop)
        epsilon, null_gaps = self._permutation_test(target_drop, control_drop)
        target_drop_p_value = self._safe_wilcoxon_greater(target_drop)
        rejected = bool(p_value < self.alpha and gap > epsilon and np.mean(target_drop) > 0.0)
        return {
            "ras_before": self._summarize_scores(target_before),
            "ras_after": self._summarize_scores(target_after),
            "ras_drop": self._summarize_scores(target_drop),
            "control_ras_before": self._summarize_scores(control_before),
            "control_ras_after": self._summarize_scores(control_after),
            "control_ras_drop": self._summarize_scores(control_drop),
            "ras_gap": gap,
            "p_value": p_value,
            "target_drop_p_value": target_drop_p_value,
            "epsilon": epsilon,
            "rejected": rejected,
            "decision": "representation_stability_drop_detected" if rejected else "no_target_specific_stability_drop",
            "null_gap_mean": float(np.mean(null_gaps)) if null_gaps else None,
            "null_gap_std": float(np.std(null_gaps)) if null_gaps else None,
            "null_gap_quantile": epsilon,
        }

    def verify_ras(self, model_before, model_after, dataset) -> Dict:
        """
        Run Representation Augmentation Stability verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("RAS requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("RAS requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        if self.primary_layers is None and granularity == "sample":
            primary_layers = ["stem", "early"]
        if self.control_layers is None and granularity == "sample":
            control_layers = ["late"]
        all_layers = self._deduplicate_layers(primary_layers + control_layers)
        forget_set = dataset.get_unlearning_set()
        retained_probe_set = self._sample_retained_subset(dataset)

        print("[RUV][RAS] ===== Start Representation Augmentation Stability Verification =====")
        print(
            f"[RUV][RAS] Setup: granularity={granularity}, stages={all_layers}, "
            f"views={self.ras_num_views}, crop_padding={self.ras_crop_padding}, "
            f"hflip_prob={self.ras_hflip_prob}, noise_std={self.ras_noise_std}, "
            f"D_u={len(forget_set)}, D_r'={len(retained_probe_set)}, "
            f"permutations={self.num_permutations}"
        )

        print("[RUV][RAS] Extracting M_s stability scores...")
        du_before = self.extract_ras_scores(model_before, forget_set, all_layers, seed_offset=1000)
        dr_before = self.extract_ras_scores(model_before, retained_probe_set, all_layers, seed_offset=2000)
        print("[RUV][RAS] Extracting M_u stability scores...")
        du_after = self.extract_ras_scores(model_after, forget_set, all_layers, seed_offset=1000)
        dr_after = self.extract_ras_scores(model_after, retained_probe_set, all_layers, seed_offset=2000)

        stage_results = {}
        for layer in all_layers:
            stage_results[layer] = self._evaluate_ras_drop(
                target_before=du_before[layer],
                target_after=du_after[layer],
                control_before=dr_before[layer],
                control_after=dr_after[layer],
            )
            print(
                f"[RUV][RAS][Stage:{layer}] "
                f"drop={stage_results[layer]['ras_drop']['mean']:.6f} "
                f"control_drop={stage_results[layer]['control_ras_drop']['mean']:.6f} "
                f"gap={stage_results[layer]['ras_gap']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f}"
            )

        primary_before = np.mean(np.stack([du_before[layer] for layer in primary_layers], axis=0), axis=0)
        primary_after = np.mean(np.stack([du_after[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_before = np.mean(np.stack([dr_before[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_after = np.mean(np.stack([dr_after[layer] for layer in primary_layers], axis=0), axis=0)
        primary_result = self._evaluate_ras_drop(
            target_before=primary_before,
            target_after=primary_after,
            control_before=primary_control_before,
            control_after=primary_control_after,
        )

        result = {
            "status": "ok",
            "method": "ras",
            "full_name": "Representation Augmentation Stability",
            "representation_name": "Model Representation State",
            "metric": "RAS Stability Drop Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "ras_num_views": self.ras_num_views,
            "ras_crop_padding": self.ras_crop_padding,
            "ras_hflip_prob": self.ras_hflip_prob,
            "ras_noise_std": self.ras_noise_std,
            "ras_gap": primary_result["ras_gap"],
            "p_value": primary_result["p_value"],
            "target_drop_p_value": primary_result["target_drop_p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "D_u shows a target-specific representation stability drop under augmentations."
                if primary_result["rejected"]
                else "No statistically significant target-specific representation stability drop was detected."
            ),
            "ras_before": primary_result["ras_before"],
            "ras_after": primary_result["ras_after"],
            "ras_drop": primary_result["ras_drop"],
            "control_ras_before": primary_result["control_ras_before"],
            "control_ras_after": primary_result["control_ras_after"],
            "control_ras_drop": primary_result["control_ras_drop"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_before)),
            "num_retained_probe": int(len(primary_control_before)),
            "num_permutations": self.num_permutations,
            "null_gap_mean": primary_result["null_gap_mean"],
            "null_gap_std": primary_result["null_gap_std"],
            "null_gap_quantile": primary_result["null_gap_quantile"],
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV][RAS][Primary] drop={result['ras_drop']['mean']:.6f} "
            f"control_drop={result['control_ras_drop']['mean']:.6f} "
            f"gap={result['ras_gap']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f}"
        )
        print(f"[RUV][RAS] decision={result['decision']}")
        print("[RUV][RAS] ===== RAS Finished =====")
        return result

    def verify_rms_knn(self, model_before, model_after, dataset) -> Dict:
        """
        Run RMS-kNN sample-level representation membership verification.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Verification summary dictionary.
        """
        if model_before is None:
            raise ValueError("RMS-kNN requires model_before as the original trained model M_s.")
        if model_after is None:
            raise ValueError("RMS-kNN requires model_after as the unlearned model M_u.")

        granularity, primary_layers, control_layers = self._resolve_layer_plan()
        all_layers = self._deduplicate_layers(primary_layers + control_layers)
        subsets = self._build_rms_knn_subsets(dataset)

        print("[RUV][RMS-kNN] ===== Start Representation Membership Verification =====")
        print(
            f"[RUV][RMS-kNN] Setup: granularity={granularity}, stages={all_layers}, "
            f"k={self.knn_k}, D_u={len(subsets['forget_target'])}, "
            f"D_r_control={len(subsets['retained_control'])}, "
            f"D_r_ref={len(subsets['retained_reference'])}, "
            f"D_test_ref={len(subsets['test_reference'])}, permutations={self.num_permutations}"
        )

        print("[RUV][RMS-kNN] Extracting M_s features...")
        du_before, du_labels = self.extract_feature_label_dict(model_before, subsets["forget_target"], all_layers)
        drc_before, drc_labels = self.extract_feature_label_dict(model_before, subsets["retained_control"], all_layers)
        drr_before, drr_labels = self.extract_feature_label_dict(model_before, subsets["retained_reference"], all_layers)
        dtest_before, dtest_labels = self.extract_feature_label_dict(model_before, subsets["test_reference"], all_layers)

        print("[RUV][RMS-kNN] Extracting M_u features...")
        du_after, du_labels_after = self.extract_feature_label_dict(model_after, subsets["forget_target"], all_layers)
        drc_after, drc_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_control"], all_layers)
        drr_after, drr_labels_after = self.extract_feature_label_dict(model_after, subsets["retained_reference"], all_layers)
        dtest_after, dtest_labels_after = self.extract_feature_label_dict(model_after, subsets["test_reference"], all_layers)

        for name, before_labels, after_labels in [
            ("D_u", du_labels, du_labels_after),
            ("D_r_control", drc_labels, drc_labels_after),
            ("D_r_ref", drr_labels, drr_labels_after),
            ("D_test_ref", dtest_labels, dtest_labels_after),
        ]:
            if not np.array_equal(before_labels, after_labels):
                raise ValueError(f"Label order changed between M_s and M_u extraction for {name}.")

        stage_results = {}
        du_before_scores = {}
        du_after_scores = {}
        dr_before_scores = {}
        dr_after_scores = {}
        for layer in all_layers:
            du_rms_before = self._compute_rms_knn_scores(
                du_before[layer], du_labels, drr_before[layer], drr_labels, dtest_before[layer], dtest_labels
            )["rms"]
            du_rms_after = self._compute_rms_knn_scores(
                du_after[layer], du_labels, drr_after[layer], drr_labels, dtest_after[layer], dtest_labels
            )["rms"]
            dr_rms_before = self._compute_rms_knn_scores(
                drc_before[layer], drc_labels, drr_before[layer], drr_labels, dtest_before[layer], dtest_labels
            )["rms"]
            dr_rms_after = self._compute_rms_knn_scores(
                drc_after[layer], drc_labels, drr_after[layer], drr_labels, dtest_after[layer], dtest_labels
            )["rms"]

            du_before_scores[layer] = du_rms_before
            du_after_scores[layer] = du_rms_after
            dr_before_scores[layer] = dr_rms_before
            dr_after_scores[layer] = dr_rms_after
            stage_results[layer] = self._evaluate_rms_drop(
                target_before=du_rms_before,
                target_after=du_rms_after,
                control_before=dr_rms_before,
                control_after=dr_rms_after,
            )
            print(
                f"[RUV][RMS-kNN][Stage:{layer}] "
                f"drop={stage_results[layer]['rms_drop']['mean']:.6f} "
                f"control_drop={stage_results[layer]['control_rms_drop']['mean']:.6f} "
                f"gap={stage_results[layer]['rms_drop_gap']:.6f} "
                f"p_value={stage_results[layer]['p_value']:.6g} "
                f"epsilon={stage_results[layer]['epsilon']:.6f}"
            )

        primary_before = np.mean(np.stack([du_before_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_after = np.mean(np.stack([du_after_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_before = np.mean(np.stack([dr_before_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_control_after = np.mean(np.stack([dr_after_scores[layer] for layer in primary_layers], axis=0), axis=0)
        primary_result = self._evaluate_rms_drop(
            target_before=primary_before,
            target_after=primary_after,
            control_before=primary_control_before,
            control_after=primary_control_after,
        )

        result = {
            "status": "ok",
            "method": "rms_knn",
            "full_name": "Representation Membership Score with kNN",
            "representation_name": "Model Representation State",
            "metric": "RMS-kNN Membership Drop Gap",
            "granularity": granularity,
            "primary_stages": primary_layers,
            "control_stages": control_layers,
            "knn_k": self.knn_k,
            "rms_reference_size": len(subsets["retained_reference"]),
            "rms_test_reference_size": len(subsets["test_reference"]),
            "rms_control_size": len(subsets["retained_control"]),
            "rms_drop_gap": primary_result["rms_drop_gap"],
            "p_value": primary_result["p_value"],
            "target_drop_p_value": primary_result["target_drop_p_value"],
            "epsilon": primary_result["epsilon"],
            "alpha": self.alpha,
            "rejected": primary_result["rejected"],
            "decision": primary_result["decision"],
            "interpretation": (
                "D_u shows a target-specific representation membership footprint drop."
                if primary_result["rejected"]
                else "No statistically significant target-specific representation membership drop was detected."
            ),
            "rms_before": primary_result["rms_before"],
            "rms_after": primary_result["rms_after"],
            "rms_drop": primary_result["rms_drop"],
            "control_rms_before": primary_result["control_rms_before"],
            "control_rms_after": primary_result["control_rms_after"],
            "control_rms_drop": primary_result["control_rms_drop"],
            "stage_results": stage_results,
            "num_forget": int(len(primary_before)),
            "num_retained_control": int(len(primary_control_before)),
            "num_permutations": self.num_permutations,
            "null_gap_mean": primary_result["null_gap_mean"],
            "null_gap_std": primary_result["null_gap_std"],
            "null_gap_quantile": primary_result["null_gap_quantile"],
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }

        print(
            f"[RUV][RMS-kNN][Primary] drop={result['rms_drop']['mean']:.6f} "
            f"control_drop={result['control_rms_drop']['mean']:.6f} "
            f"gap={result['rms_drop_gap']:.6f} p_value={result['p_value']:.6g} "
            f"epsilon={result['epsilon']:.6f}"
        )
        print(f"[RUV][RMS-kNN] decision={result['decision']}")
        print("[RUV][RMS-kNN] ===== RMS-kNN Finished =====")
        return result

    def verify_multi_audit(self, model_before, model_after, dataset) -> Dict:
        """
        Run two-model multi-view representation audit.

        Args:
            model_before: Source model M_s before unlearning.
            model_after: Unlearned model M_u to verify.
            dataset: UnlearningDataset instance.

        Returns:
            Multi-view audit summary.
        """
        metrics = self._deduplicate_layers(self.multi_metrics)
        if not metrics:
            raise ValueError("ruv_multi_metrics must contain at least one metric.")

        print("[RUV][MULTI] ===== Start Multi-View Representation Audit =====")
        print(
            f"[RUV][MULTI] metrics={metrics}, threshold={self.multi_threshold}, "
            f"alpha={self.alpha}, permutations={self.num_permutations}"
        )

        metric_results = {}
        rejected_metrics = []
        for metric in metrics:
            normalized = str(metric).lower()
            if normalized in {"multi_audit", "multi", "multi_view", "multi-view"}:
                raise ValueError("ruv_multi_metrics cannot include multi_audit itself.")
            sub_config = dict(self.config)
            sub_config["ruv_metric"] = normalized
            sub_verifier = RUVVerifier(config=sub_config)
            print(f"[RUV][MULTI] Running sub-metric: {normalized}")
            result = sub_verifier.verify(model_before=model_before, model_after=model_after, dataset=dataset)
            metric_results[normalized] = result
            if bool(result.get("rejected", False)):
                rejected_metrics.append(normalized)

        evidence_count = len(rejected_metrics)
        threshold = max(1, self.multi_threshold)
        accepted = evidence_count >= threshold
        result = {
            "status": "ok",
            "method": "multi_audit",
            "full_name": "Two-Model Multi-View Representation Audit",
            "representation_name": "Multi-View Representation Evidence",
            "metric": "Retain-Normalized Evidence Count",
            "metrics": metrics,
            "evidence_count": evidence_count,
            "evidence_threshold": threshold,
            "rejected_metrics": rejected_metrics,
            "rejected": accepted,
            "decision": (
                "multi_view_unlearning_evidence_detected"
                if accepted
                else "insufficient_multi_view_unlearning_evidence"
            ),
            "interpretation": (
                "Multiple representation views support target-specific unlearning evidence."
                if accepted
                else "The configured representation views do not provide enough target-specific evidence."
            ),
            "metric_results": metric_results,
            "forget_manifest_path": (
                dataset.get_forget_manifest_path() if hasattr(dataset, "get_forget_manifest_path") else None
            ),
        }
        print(
            f"[RUV][MULTI] evidence_count={evidence_count}/{len(metrics)} "
            f"threshold={threshold} rejected_metrics={rejected_metrics}"
        )
        print(f"[RUV][MULTI] decision={result['decision']}")
        print("[RUV][MULTI] ===== Multi-View Audit Finished =====")
        return result

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
        if self.metric in {"ras", "stability", "augmentation_stability", "aug_stability"}:
            return self.verify_ras(model_before, model_after, dataset)
        if self.metric in {"rms", "rms_knn", "rms-knn", "membership", "membership_knn"}:
            return self.verify_rms_knn(model_before, model_after, dataset)
        if self.metric in {"ars", "activation_route", "activation_route_shift", "route_shift"}:
            return self.verify_ars(model_before, model_after, dataset)
        if self.metric in {"arf", "adversarial_fragility", "adv_fragility", "representation_fragility"}:
            return self.verify_arf(model_before, model_after, dataset)
        if self.metric in {"ruler_m4", "m4", "ruler-m4", "static_geometry"}:
            return self.verify_ruler_m4(model_before, model_after, dataset)
        if self.metric in {"npg", "neighborhood_persistence", "neighbourhood_persistence"}:
            return self.verify_npg(model_before, model_after, dataset)
        if self.metric in {"multi_audit", "multi", "multi_view", "multi-view"}:
            return self.verify_multi_audit(model_before, model_after, dataset)

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
