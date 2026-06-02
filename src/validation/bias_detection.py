"""
Bias Detection and Fairness Analysis for FDA-Compliant ML Models.

Implements comprehensive algorithmic fairness assessment as required by
FDA GMLP Principle 7 (focus on performance of the deployed model) and
FDA guidance on ensuring equitable AI/ML performance across the intended
patient population.

Computes equalized odds, demographic parity, calibration differences,
and intersectional analysis across protected demographic groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    calibration_curve,
)


@dataclass
class FairnessMetrics:
    """Container for fairness metrics across a single demographic group."""

    group_name: str
    subgroup: str
    sample_size: int
    positive_rate: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    positive_predictive_value: float
    negative_predictive_value: float
    selection_rate: float
    calibration_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "subgroup": self.subgroup,
            "sample_size": self.sample_size,
            "positive_rate": round(self.positive_rate, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "true_negative_rate": round(self.true_negative_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "positive_predictive_value": round(self.positive_predictive_value, 4),
            "negative_predictive_value": round(self.negative_predictive_value, 4),
            "selection_rate": round(self.selection_rate, 4),
            "calibration_error": round(self.calibration_error, 4),
        }


class BiasDetector:
    """Detect and quantify algorithmic bias across protected groups.

    Computes a comprehensive suite of fairness metrics and tests whether
    bias exceeds acceptable thresholds defined in the validation protocol.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize the bias detector.

        Args:
            thresholds: Dictionary mapping fairness metric names to
                       maximum acceptable values. Defaults provided
                       for standard medical device requirements.
        """
        self.thresholds = thresholds or {
            "max_equalized_odds_difference": 0.10,
            "max_demographic_parity_difference": 0.10,
            "max_calibration_difference": 0.05,
            "max_predictive_parity_difference": 0.10,
        }

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        group_labels: np.ndarray,
        group_name: str,
    ) -> Dict[str, Any]:
        """Run complete bias analysis for a demographic group.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_scores: Predicted probabilities.
            group_labels: Subgroup membership labels.
            group_name: Name of the demographic category.

        Returns:
            Dictionary with per-subgroup metrics, pairwise comparisons,
            and overall fairness assessment.
        """
        unique_groups = np.unique(group_labels)
        subgroup_metrics: Dict[str, FairnessMetrics] = {}

        for group in unique_groups:
            mask = group_labels == group
            if np.sum(mask) < 20:
                continue

            metrics = self._compute_subgroup_metrics(
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                y_scores=y_scores[mask],
                group_name=group_name,
                subgroup=str(group),
            )
            subgroup_metrics[str(group)] = metrics

        # Compute pairwise fairness metrics
        equalized_odds = self._compute_equalized_odds(subgroup_metrics)
        demographic_parity = self._compute_demographic_parity(subgroup_metrics)
        calibration_diff = self._compute_calibration_difference(subgroup_metrics)
        predictive_parity = self._compute_predictive_parity(subgroup_metrics)

        # Overall assessment
        summary = {
            "equalized_odds_difference": equalized_odds["max_difference"],
            "demographic_parity_difference": demographic_parity["max_difference"],
            "calibration_difference": calibration_diff["max_difference"],
            "predictive_parity_difference": predictive_parity["max_difference"],
        }

        overall_pass = all(
            abs(summary[metric]) <= self.thresholds.get(f"max_{metric}", float("inf"))
            for metric in summary
        )

        return {
            "group_name": group_name,
            "n_subgroups": len(subgroup_metrics),
            "subgroup_metrics": {
                k: v.to_dict() for k, v in subgroup_metrics.items()
            },
            "equalized_odds": equalized_odds,
            "demographic_parity": demographic_parity,
            "calibration_analysis": calibration_diff,
            "predictive_parity": predictive_parity,
            "summary": summary,
            "thresholds": self.thresholds,
            "overall_pass": overall_pass,
        }

    def _compute_subgroup_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        group_name: str,
        subgroup: str,
    ) -> FairnessMetrics:
        """Compute comprehensive metrics for a single subgroup.

        Args:
            y_true: Ground truth labels for the subgroup.
            y_pred: Predicted labels for the subgroup.
            y_scores: Predicted probabilities for the subgroup.
            group_name: Name of the demographic category.
            subgroup: Name of the specific subgroup.

        Returns:
            FairnessMetrics for this subgroup.
        """
        n = len(y_true)
        positive_rate = float(y_true.mean())
        selection_rate = float(y_pred.mean())

        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return FairnessMetrics(
                group_name=group_name,
                subgroup=subgroup,
                sample_size=n,
                positive_rate=positive_rate,
                true_positive_rate=0.0,
                false_positive_rate=0.0,
                true_negative_rate=0.0,
                false_negative_rate=0.0,
                positive_predictive_value=0.0,
                negative_predictive_value=0.0,
                selection_rate=selection_rate,
                calibration_error=0.0,
            )

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Calibration error (expected calibration error)
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_scores, n_bins=10, strategy="uniform"
            )
            cal_error = float(np.mean(np.abs(prob_true - prob_pred)))
        except ValueError:
            cal_error = 0.0

        return FairnessMetrics(
            group_name=group_name,
            subgroup=subgroup,
            sample_size=n,
            positive_rate=positive_rate,
            true_positive_rate=float(tpr),
            false_positive_rate=float(fpr),
            true_negative_rate=float(tnr),
            false_negative_rate=float(fnr),
            positive_predictive_value=float(ppv),
            negative_predictive_value=float(npv),
            selection_rate=selection_rate,
            calibration_error=cal_error,
        )

    def _compute_equalized_odds(
        self, subgroup_metrics: Dict[str, FairnessMetrics]
    ) -> Dict[str, Any]:
        """Compute equalized odds difference across subgroups.

        Equalized odds requires that the classifier have equal true
        positive rates AND equal false positive rates across all groups.
        The difference is the maximum disparity in either TPR or FPR.

        Args:
            subgroup_metrics: Per-subgroup fairness metrics.

        Returns:
            Dictionary with equalized odds analysis.
        """
        if len(subgroup_metrics) < 2:
            return {"max_difference": 0.0, "details": "Fewer than 2 subgroups"}

        tprs = {k: v.true_positive_rate for k, v in subgroup_metrics.items()}
        fprs = {k: v.false_positive_rate for k, v in subgroup_metrics.items()}

        tpr_diff = max(tprs.values()) - min(tprs.values())
        fpr_diff = max(fprs.values()) - min(fprs.values())
        max_diff = max(tpr_diff, fpr_diff)

        return {
            "max_difference": float(max_diff),
            "tpr_difference": float(tpr_diff),
            "fpr_difference": float(fpr_diff),
            "tpr_by_subgroup": {k: round(v, 4) for k, v in tprs.items()},
            "fpr_by_subgroup": {k: round(v, 4) for k, v in fprs.items()},
            "worst_tpr_subgroup": min(tprs, key=tprs.get),
            "worst_fpr_subgroup": max(fprs, key=fprs.get),
        }

    def _compute_demographic_parity(
        self, subgroup_metrics: Dict[str, FairnessMetrics]
    ) -> Dict[str, Any]:
        """Compute demographic parity difference across subgroups.

        Demographic parity requires that the selection rate (proportion
        of positive predictions) be equal across all groups.

        Args:
            subgroup_metrics: Per-subgroup fairness metrics.

        Returns:
            Dictionary with demographic parity analysis.
        """
        if len(subgroup_metrics) < 2:
            return {"max_difference": 0.0, "details": "Fewer than 2 subgroups"}

        rates = {k: v.selection_rate for k, v in subgroup_metrics.items()}
        max_diff = max(rates.values()) - min(rates.values())

        return {
            "max_difference": float(max_diff),
            "selection_rates": {k: round(v, 4) for k, v in rates.items()},
            "highest_rate_subgroup": max(rates, key=rates.get),
            "lowest_rate_subgroup": min(rates, key=rates.get),
        }

    def _compute_calibration_difference(
        self, subgroup_metrics: Dict[str, FairnessMetrics]
    ) -> Dict[str, Any]:
        """Compute calibration difference across subgroups.

        Good calibration means predicted probabilities match observed
        frequencies. Calibration difference measures the maximum gap
        in calibration quality across subgroups.

        Args:
            subgroup_metrics: Per-subgroup fairness metrics.

        Returns:
            Dictionary with calibration analysis.
        """
        if len(subgroup_metrics) < 2:
            return {"max_difference": 0.0, "details": "Fewer than 2 subgroups"}

        errors = {k: v.calibration_error for k, v in subgroup_metrics.items()}
        max_diff = max(errors.values()) - min(errors.values())

        return {
            "max_difference": float(max_diff),
            "calibration_errors": {k: round(v, 4) for k, v in errors.items()},
            "worst_calibrated_subgroup": max(errors, key=errors.get),
            "best_calibrated_subgroup": min(errors, key=errors.get),
        }

    def _compute_predictive_parity(
        self, subgroup_metrics: Dict[str, FairnessMetrics]
    ) -> Dict[str, Any]:
        """Compute predictive parity across subgroups.

        Predictive parity requires equal positive predictive value (PPV)
        across all groups, meaning a positive prediction should have the
        same meaning regardless of group membership.

        Args:
            subgroup_metrics: Per-subgroup fairness metrics.

        Returns:
            Dictionary with predictive parity analysis.
        """
        if len(subgroup_metrics) < 2:
            return {"max_difference": 0.0, "details": "Fewer than 2 subgroups"}

        ppvs = {k: v.positive_predictive_value for k, v in subgroup_metrics.items()}
        npvs = {k: v.negative_predictive_value for k, v in subgroup_metrics.items()}

        ppv_diff = max(ppvs.values()) - min(ppvs.values())
        npv_diff = max(npvs.values()) - min(npvs.values())

        return {
            "max_difference": float(max(ppv_diff, npv_diff)),
            "ppv_difference": float(ppv_diff),
            "npv_difference": float(npv_diff),
            "ppv_by_subgroup": {k: round(v, 4) for k, v in ppvs.items()},
            "npv_by_subgroup": {k: round(v, 4) for k, v in npvs.items()},
        }


class IntersectionalAnalyzer:
    """Analyze intersectional bias across multiple demographic dimensions.

    Evaluates performance at the intersection of multiple protected
    attributes (e.g., age x race, density x scanner), which can reveal
    bias that is hidden when examining each dimension independently.
    """

    def __init__(self, min_subgroup_size: int = 30):
        """Initialize the intersectional analyzer.

        Args:
            min_subgroup_size: Minimum samples for a subgroup to be analyzed.
        """
        self.min_subgroup_size = min_subgroup_size

    def analyze(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        group_a_labels: np.ndarray,
        group_b_labels: np.ndarray,
        group_a_name: str,
        group_b_name: str,
    ) -> Dict[str, Any]:
        """Analyze intersectional performance.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities.
            group_a_labels: Labels for first demographic dimension.
            group_b_labels: Labels for second demographic dimension.
            group_a_name: Name of first dimension.
            group_b_name: Name of second dimension.

        Returns:
            Dictionary with intersectional analysis results.
        """
        # Create intersectional groups
        intersect_labels = np.array([
            f"{a}|{b}"
            for a, b in zip(group_a_labels, group_b_labels)
        ])
        unique_intersections = np.unique(intersect_labels)

        results: Dict[str, Dict[str, Any]] = {}
        for intersect in unique_intersections:
            mask = intersect_labels == intersect
            n = int(np.sum(mask))

            if n < self.min_subgroup_size:
                results[intersect] = {
                    "n": n,
                    "skipped": True,
                    "reason": f"n={n} < {self.min_subgroup_size}",
                }
                continue

            y_true_sub = y_true[mask]
            y_scores_sub = y_scores[mask]

            if len(np.unique(y_true_sub)) < 2:
                results[intersect] = {
                    "n": n,
                    "skipped": True,
                    "reason": "Single class in subgroup",
                }
                continue

            auc = float(roc_auc_score(y_true_sub, y_scores_sub))
            y_pred_sub = (y_scores_sub >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()

            results[intersect] = {
                "n": n,
                "auc": auc,
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "positive_rate": float(y_true_sub.mean()),
                "skipped": False,
            }

        # Find the performance range across intersectional groups
        valid_aucs = [
            r["auc"] for r in results.values()
            if not r.get("skipped", False) and "auc" in r
        ]

        performance_range = (
            max(valid_aucs) - min(valid_aucs) if len(valid_aucs) >= 2 else 0.0
        )
        worst_group = (
            min(
                (k for k, v in results.items() if not v.get("skipped", False)),
                key=lambda k: results[k].get("auc", float("inf")),
            )
            if valid_aucs else "N/A"
        )

        return {
            "dimensions": [group_a_name, group_b_name],
            "n_intersections_analyzed": sum(
                1 for r in results.values() if not r.get("skipped", False)
            ),
            "n_intersections_total": len(unique_intersections),
            "performance_range": performance_range,
            "worst_performing_intersection": worst_group,
            "worst_auc": min(valid_aucs) if valid_aucs else None,
            "intersectional_results": results,
        }
