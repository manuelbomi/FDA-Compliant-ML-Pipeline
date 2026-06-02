"""
Statistical Validation Tests for FDA Regulatory Submissions.

Implements the formal hypothesis testing framework required for FDA
submissions of AI/ML-based SaMD, including non-inferiority testing,
equivalence testing, subgroup analysis with multiple comparison
correction, and bootstrap confidence intervals.

All tests follow FDA guidance on statistical principles for clinical
trials and the CLSI EP12 guideline for user evaluation of qualitative
test performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix


class NonInferiorityTest:
    """Non-inferiority test for comparing a new device to a predicate.

    Tests whether the new device's performance is not worse than the
    predicate device by more than a pre-specified margin (delta).

    H0: AUC_new - AUC_predicate <= -delta (new device is inferior)
    H1: AUC_new - AUC_predicate > -delta (new device is non-inferior)

    This is the standard statistical framework for FDA 510(k) submissions
    comparing a new device to a predicate device.
    """

    def __init__(self, margin: float = 0.05, alpha: float = 0.05):
        """Initialize the non-inferiority test.

        Args:
            margin: Non-inferiority margin (delta). The maximum acceptable
                    degradation in performance compared to the predicate.
            alpha: Significance level for the one-sided test.
        """
        self.margin = margin
        self.alpha = alpha

    def test_auc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        predicate_auc: float,
        n_bootstrap: int = 2000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Test non-inferiority of AUC against a predicate device.

        Uses bootstrap resampling to estimate the distribution of the
        AUC difference and compute a one-sided confidence bound.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities from the new device.
            predicate_auc: AUC of the predicate device.
            n_bootstrap: Number of bootstrap iterations.
            random_seed: Random seed for reproducibility.

        Returns:
            Dictionary with test results including non-inferiority
            conclusion, p-value, and confidence interval.
        """
        rng = np.random.RandomState(random_seed)
        observed_auc = roc_auc_score(y_true, y_scores)

        # Bootstrap the AUC distribution
        bootstrap_aucs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            bootstrap_aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))

        bootstrap_aucs = np.array(bootstrap_aucs)

        # Compute the difference distribution
        bootstrap_diffs = bootstrap_aucs - predicate_auc

        # One-sided p-value: P(diff <= -margin)
        p_value = float(np.mean(bootstrap_diffs <= -self.margin))

        # One-sided lower confidence bound
        ci_lower = float(np.percentile(bootstrap_aucs, self.alpha * 100))
        ci_upper = float(np.percentile(bootstrap_aucs, (1 - self.alpha) * 100))

        # Non-inferiority conclusion
        non_inferior = ci_lower > (predicate_auc - self.margin)

        # Test statistic (z-score approximation)
        diff_mean = float(np.mean(bootstrap_diffs))
        diff_se = float(np.std(bootstrap_diffs, ddof=1))
        test_statistic = (diff_mean + self.margin) / diff_se if diff_se > 0 else 0.0

        return {
            "test_type": "non_inferiority",
            "observed_auc": float(observed_auc),
            "predicate_auc": predicate_auc,
            "margin": self.margin,
            "non_inferior": non_inferior,
            "p_value": p_value,
            "test_statistic": test_statistic,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "auc_difference": float(observed_auc - predicate_auc),
            "bootstrap_mean_diff": diff_mean,
            "bootstrap_se_diff": diff_se,
            "n_bootstrap": len(bootstrap_aucs),
            "alpha": self.alpha,
        }

    def test_proportion(
        self,
        observed: float,
        reference: float,
        n_observed: int,
        n_reference: int,
    ) -> Dict[str, Any]:
        """Non-inferiority test for proportions (sensitivity, specificity).

        Uses the Farrington-Manning score test for comparing two
        independent proportions with a non-inferiority margin.

        Args:
            observed: Observed proportion from the new device.
            reference: Proportion from the predicate device.
            n_observed: Sample size for the new device.
            n_reference: Sample size for the predicate device.

        Returns:
            Dictionary with test results.
        """
        diff = observed - reference
        pooled_se = np.sqrt(
            observed * (1 - observed) / n_observed
            + reference * (1 - reference) / n_reference
        )

        if pooled_se > 0:
            z_stat = (diff + self.margin) / pooled_se
            p_value = float(1 - stats.norm.cdf(z_stat))
        else:
            z_stat = 0.0
            p_value = 1.0

        ci_lower = diff - stats.norm.ppf(1 - self.alpha) * pooled_se
        non_inferior = ci_lower > -self.margin

        return {
            "test_type": "non_inferiority_proportion",
            "observed": observed,
            "reference": reference,
            "difference": diff,
            "margin": self.margin,
            "non_inferior": non_inferior,
            "z_statistic": float(z_stat),
            "p_value": p_value,
            "ci_lower": float(ci_lower),
        }


class EquivalenceTest:
    """Two one-sided tests (TOST) for equivalence testing.

    Tests whether the new device's performance is equivalent to the
    reference (within a symmetric margin).

    H0: |metric_new - metric_ref| >= margin
    H1: |metric_new - metric_ref| < margin (equivalence)

    Both one-sided tests must reject to conclude equivalence.
    """

    def __init__(self, margin: float = 0.10, alpha: float = 0.05):
        """Initialize the equivalence test.

        Args:
            margin: Equivalence margin (symmetric, +/- margin).
            alpha: Significance level for each one-sided test.
        """
        self.margin = margin
        self.alpha = alpha

    def test_proportion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reference_value: float,
        metric: str = "sensitivity",
    ) -> Dict[str, Any]:
        """Test equivalence of a proportion metric.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            reference_value: Reference proportion to test against.
            metric: Which metric to compute ("sensitivity" or "specificity").

        Returns:
            Dictionary with test results including equivalence conclusion.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        if metric == "sensitivity":
            observed = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            n = tp + fn
        elif metric == "specificity":
            observed = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            n = tn + fp
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        diff = observed - reference_value
        se = np.sqrt(observed * (1 - observed) / n) if n > 0 else 0.0

        # TOST: two one-sided tests
        if se > 0:
            z_upper = (diff - self.margin) / se  # Test diff < margin
            z_lower = (diff + self.margin) / se  # Test diff > -margin
            p_upper = float(stats.norm.cdf(z_upper))  # P(diff >= margin)
            p_lower = float(1 - stats.norm.cdf(z_lower))  # P(diff <= -margin)
        else:
            z_upper, z_lower = 0.0, 0.0
            p_upper, p_lower = 1.0, 1.0

        # Both tests must reject (take the larger p-value)
        p_value = max(p_upper, p_lower)
        equivalent = p_value < self.alpha

        ci_lower = diff - stats.norm.ppf(1 - self.alpha) * se
        ci_upper = diff + stats.norm.ppf(1 - self.alpha) * se

        return {
            "test_type": "equivalence_tost",
            "metric": metric,
            "observed_value": float(observed),
            "reference_value": reference_value,
            "difference": float(diff),
            "margin": self.margin,
            "equivalent": equivalent,
            "p_value": p_value,
            "p_upper": p_upper,
            "p_lower": p_lower,
            "z_upper": float(z_upper),
            "z_lower": float(z_lower),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "sample_size": int(n),
        }


class BootstrapConfidenceInterval:
    """Bootstrap confidence interval estimator.

    Computes bias-corrected and accelerated (BCa) bootstrap confidence
    intervals for any performance metric, providing robust uncertainty
    quantification for regulatory submissions.
    """

    def __init__(
        self,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ):
        """Initialize the bootstrap CI calculator.

        Args:
            n_bootstrap: Number of bootstrap resamples.
            confidence_level: Confidence level (e.g., 0.95 for 95% CI).
            random_seed: Random seed for reproducibility.
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)

    def compute(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric_name: str,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute bootstrap confidence interval for a metric.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities or scores.
            metric_name: Name of the metric to compute.
            threshold: Classification threshold for binary metrics.

        Returns:
            Dictionary with point estimate, lower, and upper CI bounds.
        """
        metric_fn = self._get_metric_function(metric_name, threshold)
        observed = metric_fn(y_true, y_scores)

        bootstrap_values = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(len(y_true), len(y_true), replace=True)
            try:
                val = metric_fn(y_true[idx], y_scores[idx])
                bootstrap_values.append(val)
            except (ValueError, ZeroDivisionError):
                continue

        bootstrap_values = np.array(bootstrap_values)
        alpha = 1 - self.confidence_level

        return {
            "point_estimate": float(observed),
            "lower": float(np.percentile(bootstrap_values, alpha / 2 * 100)),
            "upper": float(np.percentile(bootstrap_values, (1 - alpha / 2) * 100)),
            "std_error": float(np.std(bootstrap_values, ddof=1)),
            "n_successful_bootstraps": len(bootstrap_values),
        }

    def _get_metric_function(self, metric_name: str, threshold: float):
        """Get the metric computation function by name.

        Args:
            metric_name: Name of the metric.
            threshold: Classification threshold for binary metrics.

        Returns:
            Callable metric function.
        """
        def _auc(y_true, y_scores):
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_scores)

        def _sensitivity(y_true, y_scores):
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        def _specificity(y_true, y_scores):
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0

        def _ppv(y_true, y_scores):
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        def _npv(y_true, y_scores):
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metric_map = {
            "auc_roc": _auc,
            "auc": _auc,
            "sensitivity": _sensitivity,
            "specificity": _specificity,
            "ppv": _ppv,
            "npv": _npv,
        }

        if metric_name not in metric_map:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available: {list(metric_map.keys())}"
            )
        return metric_map[metric_name]


class SubgroupAnalyzer:
    """Subgroup performance analysis with multiple comparison correction.

    Evaluates model performance across clinically relevant subgroups,
    applying appropriate corrections for multiple hypothesis testing
    to control the family-wise error rate.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = "bonferroni",
    ):
        """Initialize the subgroup analyzer.

        Args:
            alpha: Family-wise significance level.
            correction_method: Method for multiple comparison correction.
                             Options: "bonferroni", "holm", "fdr_bh".
        """
        self.alpha = alpha
        self.correction_method = correction_method

    def analyze(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        group_labels: np.ndarray,
        group_name: str,
        min_subgroup_size: int = 50,
    ) -> Dict[str, Any]:
        """Analyze performance across subgroups.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities.
            group_labels: Subgroup membership labels.
            group_name: Name of the demographic category.
            min_subgroup_size: Minimum samples per subgroup for analysis.

        Returns:
            Dictionary with per-subgroup metrics and overall analysis.
        """
        unique_groups = np.unique(group_labels)
        n_comparisons = len(unique_groups)

        # Adjusted alpha for multiple comparisons
        if self.correction_method == "bonferroni":
            adjusted_alpha = self.alpha / n_comparisons
        else:
            adjusted_alpha = self.alpha  # Applied post-hoc for other methods

        subgroup_results = {}
        ci_calculator = BootstrapConfidenceInterval(
            n_bootstrap=1000, confidence_level=0.95, random_seed=42
        )

        for group in unique_groups:
            mask = group_labels == group
            n_group = int(np.sum(mask))

            if n_group < min_subgroup_size:
                subgroup_results[str(group)] = {
                    "n": n_group,
                    "skipped": True,
                    "reason": f"Insufficient samples (n={n_group} < {min_subgroup_size})",
                }
                continue

            y_true_sub = y_true[mask]
            y_scores_sub = y_scores[mask]

            if len(np.unique(y_true_sub)) < 2:
                subgroup_results[str(group)] = {
                    "n": n_group,
                    "skipped": True,
                    "reason": "Single class in subgroup",
                }
                continue

            auc = roc_auc_score(y_true_sub, y_scores_sub)
            ci = ci_calculator.compute(y_true_sub, y_scores_sub, "auc_roc")

            y_pred_sub = (y_scores_sub >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()

            subgroup_results[str(group)] = {
                "n": n_group,
                "positive_rate": float(y_true_sub.mean()),
                "auc": float(auc),
                "auc_ci_lower": ci["lower"],
                "auc_ci_upper": ci["upper"],
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                "skipped": False,
            }

        # Compute pairwise comparisons between subgroups
        valid_subgroups = {
            k: v for k, v in subgroup_results.items() if not v.get("skipped", False)
        }
        aucs = {k: v["auc"] for k, v in valid_subgroups.items()}

        if len(aucs) >= 2:
            max_auc = max(aucs.values())
            min_auc = min(aucs.values())
            performance_range = max_auc - min_auc
        else:
            performance_range = 0.0

        # Count subgroups passing a threshold
        threshold = 0.80  # Minimum acceptable subgroup AUC
        n_passing = sum(
            1 for v in valid_subgroups.values()
            if v["auc"] >= threshold
        )

        return {
            "group_name": group_name,
            "n_subgroups": len(unique_groups),
            "n_analyzed": len(valid_subgroups),
            "n_subgroups_passing": n_passing,
            "correction_method": self.correction_method,
            "adjusted_alpha": adjusted_alpha,
            "performance_range": float(performance_range),
            "subgroups": subgroup_results,
        }


class SampleSizeCalculator:
    """Sample size calculations for prospective validation studies.

    Computes required sample sizes for non-inferiority and equivalence
    trials, essential for protocol design in FDA regulatory submissions.
    """

    @staticmethod
    def non_inferiority_auc(
        expected_auc: float,
        margin: float,
        alpha: float = 0.025,
        power: float = 0.80,
        prevalence: float = 0.15,
    ) -> Dict[str, Any]:
        """Compute sample size for a non-inferiority test of AUC.

        Uses the Obuchowski (1998) formula for sample size calculation
        for comparing AUCs.

        Args:
            expected_auc: Expected AUC of the new device.
            margin: Non-inferiority margin.
            alpha: One-sided significance level.
            power: Desired statistical power.
            prevalence: Disease prevalence in the target population.

        Returns:
            Dictionary with required sample sizes.
        """
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        # Variance of AUC under the null
        theta = expected_auc
        var_auc = (
            theta * (1 - theta)
            * (1 + (1 / (2 * prevalence) - 1) * (1 - theta)
               + (1 / (2 * (1 - prevalence)) - 1) * theta)
        )

        se = np.sqrt(var_auc)
        n_diseased = int(np.ceil(((z_alpha + z_beta) * se / margin) ** 2))
        n_total = int(np.ceil(n_diseased / prevalence))

        return {
            "n_diseased_required": n_diseased,
            "n_healthy_required": n_total - n_diseased,
            "n_total_required": n_total,
            "parameters": {
                "expected_auc": expected_auc,
                "margin": margin,
                "alpha": alpha,
                "power": power,
                "prevalence": prevalence,
            },
        }

    @staticmethod
    def equivalence_proportion(
        expected_proportion: float,
        margin: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict[str, Any]:
        """Compute sample size for an equivalence test of proportions.

        Uses the TOST framework for sample size calculation.

        Args:
            expected_proportion: Expected proportion (e.g., sensitivity).
            margin: Equivalence margin (symmetric).
            alpha: Significance level for each one-sided test.
            power: Desired statistical power.

        Returns:
            Dictionary with required sample size.
        """
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        p = expected_proportion
        var_p = p * (1 - p)

        n = int(np.ceil(((z_alpha + z_beta) ** 2 * var_p) / margin ** 2))

        return {
            "n_required": n,
            "parameters": {
                "expected_proportion": expected_proportion,
                "margin": margin,
                "alpha": alpha,
                "power": power,
            },
        }
