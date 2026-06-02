"""
Model Regression Testing for FDA Predetermined Change Control Plans.

Implements systematic comparison between model versions to detect
performance regressions, supporting the FDA's Predetermined Change
Control Plan framework for AI/ML-based SaMD modifications.

Includes statistical significance testing, edge case analysis, and
failure mode identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


@dataclass
class RegressionTestResult:
    """Result of a single regression test."""

    metric_name: str
    current_value: float
    previous_value: float
    change: float
    change_pct: float
    p_value: Optional[float]
    significant: bool
    regressed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "current": self.current_value,
            "previous": self.previous_value,
            "change": round(self.change, 6),
            "change_pct": round(self.change_pct, 2),
            "significant": self.significant,
            "regressed": self.regressed,
        }
        if self.p_value is not None:
            result["p_value"] = round(self.p_value, 6)
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class EdgeCaseResult:
    """Result of an edge case test."""

    test_name: str
    description: str
    passed: bool
    expected_behavior: str
    observed_behavior: str
    severity: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "description": self.description,
            "passed": self.passed,
            "expected_behavior": self.expected_behavior,
            "observed_behavior": self.observed_behavior,
            "severity": self.severity,
            "details": self.details,
        }


class RegressionTester:
    """Compare model performance between versions with statistical rigor.

    Tests whether a new model version shows statistically significant
    regression in any performance metric compared to the previous version.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 2000,
        regression_threshold: float = 0.02,
    ):
        """Initialize the regression tester.

        Args:
            alpha: Significance level for statistical tests.
            n_bootstrap: Number of bootstrap resamples.
            regression_threshold: Minimum meaningful change magnitude.
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.regression_threshold = regression_threshold

    def compare_models(
        self,
        y_true: np.ndarray,
        current_scores: np.ndarray,
        previous_scores: np.ndarray,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Compare two model versions on the same test set.

        Uses paired bootstrap testing to detect statistically significant
        differences between model versions, controlling for the correlation
        between predictions on the same data.

        Args:
            y_true: Ground truth binary labels.
            current_scores: Predicted probabilities from the current model.
            previous_scores: Predicted probabilities from the previous model.
            random_seed: Random seed for reproducibility.

        Returns:
            Dictionary with comparison results for all metrics.
        """
        rng = np.random.RandomState(random_seed)

        metrics_to_compare = ["auc_roc", "sensitivity", "specificity", "ppv", "npv"]
        results: Dict[str, Dict[str, Any]] = {}

        for metric_name in metrics_to_compare:
            current_val = self._compute_metric(
                y_true, current_scores, metric_name
            )
            previous_val = self._compute_metric(
                y_true, previous_scores, metric_name
            )

            # Paired bootstrap test
            bootstrap_diffs = []
            for _ in range(self.n_bootstrap):
                idx = rng.choice(len(y_true), len(y_true), replace=True)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                try:
                    curr_boot = self._compute_metric(
                        y_true[idx], current_scores[idx], metric_name
                    )
                    prev_boot = self._compute_metric(
                        y_true[idx], previous_scores[idx], metric_name
                    )
                    bootstrap_diffs.append(curr_boot - prev_boot)
                except (ValueError, ZeroDivisionError):
                    continue

            bootstrap_diffs = np.array(bootstrap_diffs)

            if len(bootstrap_diffs) > 0:
                # Two-sided p-value for the difference
                p_value = float(
                    2 * min(
                        np.mean(bootstrap_diffs <= 0),
                        np.mean(bootstrap_diffs >= 0),
                    )
                )
                ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
                ci_upper = float(np.percentile(bootstrap_diffs, 97.5))
            else:
                p_value = 1.0
                ci_lower = 0.0
                ci_upper = 0.0

            change = current_val - previous_val
            change_pct = (
                (change / previous_val * 100) if previous_val != 0 else 0.0
            )
            significant = p_value < self.alpha
            regressed = (
                significant
                and change < -self.regression_threshold
            )

            results[metric_name] = RegressionTestResult(
                metric_name=metric_name,
                current_value=current_val,
                previous_value=previous_val,
                change=change,
                change_pct=change_pct,
                p_value=p_value,
                significant=significant,
                regressed=regressed,
                details={
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_bootstrap": len(bootstrap_diffs),
                },
            ).to_dict()

        # Overall regression assessment
        any_regression = any(
            r.get("regressed", False) for r in results.values()
        )

        return {
            "metric_comparisons": results,
            "no_regression": not any_regression,
            "n_metrics_improved": sum(
                1 for r in results.values()
                if r.get("change", 0) > self.regression_threshold
            ),
            "n_metrics_regressed": sum(
                1 for r in results.values() if r.get("regressed", False)
            ),
            "n_metrics_unchanged": sum(
                1 for r in results.values()
                if abs(r.get("change", 0)) <= self.regression_threshold
            ),
            "alpha": self.alpha,
            "regression_threshold": self.regression_threshold,
        }

    def run_edge_case_tests(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """Run a suite of edge case tests on the model.

        Tests model behavior under various challenging conditions that
        may be encountered in clinical practice.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities.

        Returns:
            Dictionary with edge case test results.
        """
        edge_cases: List[Dict[str, Any]] = []

        # Test 1: Extreme confidence predictions
        extreme_high = y_scores > 0.95
        if np.sum(extreme_high) > 0:
            precision_at_extreme = (
                np.mean(y_true[extreme_high])
                if np.sum(extreme_high) > 0
                else 0.0
            )
            edge_cases.append(
                EdgeCaseResult(
                    test_name="extreme_high_confidence",
                    description="Predictions with confidence > 0.95",
                    passed=precision_at_extreme >= 0.80,
                    expected_behavior="High confidence predictions should be mostly correct (>80%)",
                    observed_behavior=f"Precision at extreme confidence: {precision_at_extreme:.3f}",
                    severity="high",
                    details={
                        "n_samples": int(np.sum(extreme_high)),
                        "precision": float(precision_at_extreme),
                    },
                ).to_dict()
            )

        # Test 2: Near-threshold predictions
        near_threshold = np.abs(y_scores - 0.5) < 0.05
        if np.sum(near_threshold) > 0:
            pct_near = float(np.mean(near_threshold)) * 100
            edge_cases.append(
                EdgeCaseResult(
                    test_name="near_threshold_concentration",
                    description="Proportion of predictions near decision boundary (0.45-0.55)",
                    passed=pct_near < 30.0,
                    expected_behavior="Less than 30% of predictions should be near threshold",
                    observed_behavior=f"{pct_near:.1f}% of predictions near threshold",
                    severity="medium",
                    details={
                        "n_samples": int(np.sum(near_threshold)),
                        "percentage": pct_near,
                    },
                ).to_dict()
            )

        # Test 3: Score distribution reasonableness
        score_mean = float(np.mean(y_scores))
        score_std = float(np.std(y_scores))
        prevalence = float(np.mean(y_true))

        mean_reasonable = abs(score_mean - prevalence) < 0.20
        edge_cases.append(
            EdgeCaseResult(
                test_name="score_distribution_reasonableness",
                description="Predicted score distribution aligns with disease prevalence",
                passed=mean_reasonable,
                expected_behavior=(
                    f"Mean predicted score should be near prevalence ({prevalence:.3f})"
                ),
                observed_behavior=f"Mean score: {score_mean:.3f}, std: {score_std:.3f}",
                severity="medium",
                details={
                    "mean_score": score_mean,
                    "std_score": score_std,
                    "prevalence": prevalence,
                    "difference": abs(score_mean - prevalence),
                },
            ).to_dict()
        )

        # Test 4: Monotonic calibration
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        monotonic = True
        prev_rate = -1.0

        for i in range(n_bins):
            mask = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
            if np.sum(mask) < 5:
                continue
            rate = float(np.mean(y_true[mask]))
            if rate < prev_rate - 0.05:  # Allow small tolerance
                monotonic = False
                break
            prev_rate = rate

        edge_cases.append(
            EdgeCaseResult(
                test_name="monotonic_calibration",
                description="Positive rate increases monotonically with predicted score",
                passed=monotonic,
                expected_behavior="Higher predicted scores should correlate with higher true rates",
                observed_behavior="Calibration is monotonic" if monotonic else "Non-monotonic calibration detected",
                severity="high",
            ).to_dict()
        )

        # Test 5: Operating point stability
        thresholds = np.arange(0.3, 0.7, 0.02)
        sensitivities = []
        specificities = []

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            if len(np.unique(y_pred)) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        if len(sensitivities) > 1:
            sens_stability = float(np.std(np.diff(sensitivities)))
            stable = sens_stability < 0.05
        else:
            sens_stability = 0.0
            stable = True

        edge_cases.append(
            EdgeCaseResult(
                test_name="operating_point_stability",
                description="Sensitivity/specificity change smoothly with threshold",
                passed=stable,
                expected_behavior="Smooth trade-off curve without sharp jumps",
                observed_behavior=(
                    f"Sensitivity change stability (std of deltas): {sens_stability:.4f}"
                ),
                severity="medium",
                details={"stability_metric": sens_stability},
            ).to_dict()
        )

        # Summary
        total = len(edge_cases)
        passed = sum(1 for ec in edge_cases if ec["passed"])

        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "all_passed": passed == total,
            "edge_cases": edge_cases,
        }

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric_name: str,
        threshold: float = 0.5,
    ) -> float:
        """Compute a named metric.

        Args:
            y_true: Ground truth labels.
            y_scores: Predicted probabilities.
            metric_name: Name of the metric.
            threshold: Classification threshold.

        Returns:
            Metric value.
        """
        if metric_name == "auc_roc":
            return float(roc_auc_score(y_true, y_scores))

        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metric_map = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        }

        if metric_name not in metric_map:
            raise ValueError(f"Unknown metric: {metric_name}")

        return float(metric_map[metric_name])


class FailureModeAnalyzer:
    """Analyze model failure modes for FDA FMEA documentation.

    Identifies systematic patterns in model errors to support
    Failure Mode and Effects Analysis (FMEA) required by
    FDA 21 CFR 820.30(g) and ISO 14971 risk management.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize the failure mode analyzer.

        Args:
            threshold: Classification threshold for binary predictions.
        """
        self.threshold = threshold

    def analyze(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metadata: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Analyze failure modes across metadata dimensions.

        Args:
            y_true: Ground truth binary labels.
            y_scores: Predicted probabilities.
            metadata: Dictionary mapping metadata field names to arrays.

        Returns:
            Dictionary with failure mode analysis results.
        """
        y_pred = (y_scores >= self.threshold).astype(int)

        # Identify error types
        false_positives = (y_pred == 1) & (y_true == 0)
        false_negatives = (y_pred == 0) & (y_true == 1)

        failure_modes: List[Dict[str, Any]] = []

        for field_name, field_values in metadata.items():
            unique_vals = np.unique(field_values)

            for val in unique_vals:
                mask = field_values == val
                n_total = int(np.sum(mask))
                if n_total < 10:
                    continue

                n_fp = int(np.sum(false_positives & mask))
                n_fn = int(np.sum(false_negatives & mask))

                fp_rate = n_fp / n_total
                fn_rate = n_fn / n_total

                overall_fp_rate = np.sum(false_positives) / len(y_true)
                overall_fn_rate = np.sum(false_negatives) / len(y_true)

                fp_enrichment = fp_rate / overall_fp_rate if overall_fp_rate > 0 else 0
                fn_enrichment = fn_rate / overall_fn_rate if overall_fn_rate > 0 else 0

                if fp_enrichment > 1.5 or fn_enrichment > 1.5:
                    severity = "high" if max(fp_enrichment, fn_enrichment) > 2.0 else "medium"
                    failure_modes.append({
                        "field": field_name,
                        "value": str(val),
                        "n_samples": n_total,
                        "false_positive_rate": round(fp_rate, 4),
                        "false_negative_rate": round(fn_rate, 4),
                        "fp_enrichment": round(fp_enrichment, 2),
                        "fn_enrichment": round(fn_enrichment, 2),
                        "severity": severity,
                        "primary_error_type": (
                            "false_positive" if fp_enrichment > fn_enrichment
                            else "false_negative"
                        ),
                    })

        # Sort by enrichment (most problematic first)
        failure_modes.sort(
            key=lambda x: max(x["fp_enrichment"], x["fn_enrichment"]),
            reverse=True,
        )

        return {
            "total_false_positives": int(np.sum(false_positives)),
            "total_false_negatives": int(np.sum(false_negatives)),
            "overall_fp_rate": float(np.mean(false_positives)),
            "overall_fn_rate": float(np.mean(false_negatives)),
            "n_failure_modes_detected": len(failure_modes),
            "failure_modes": failure_modes,
        }
