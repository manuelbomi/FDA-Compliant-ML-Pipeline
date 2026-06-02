"""
FDA-Style Model Card Generator for Medical Device ML Models.

Produces comprehensive model documentation that satisfies FDA regulatory
requirements for AI/ML-based Software as a Medical Device (SaMD), including
intended use statements, performance characteristics with confidence intervals,
limitations, demographic performance breakdowns, and failure mode analysis.

Based on the Model Card framework (Mitchell et al., 2019) extended with
FDA-specific regulatory requirements.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IntendedUseStatement:
    """FDA-required intended use / indications for use statement.

    Per FDA guidance on SaMD, the intended use must specify:
    - The specific medical condition or purpose
    - The intended patient population
    - The intended user (clinician type)
    - The clinical setting
    - Whether it is standalone or adjunctive
    """

    device_name: str
    intended_use: str
    indications_for_use: str
    target_condition: str
    target_population: str
    intended_user: str
    clinical_setting: str
    device_type: str  # "standalone" or "adjunctive"
    samd_category: str  # e.g., "IIa" per IMDRF
    clinical_significance: str  # "treat/diagnose", "drive clinical management", "inform"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "intended_use": self.intended_use,
            "indications_for_use": self.indications_for_use,
            "target_condition": self.target_condition,
            "target_population": self.target_population,
            "intended_user": self.intended_user,
            "clinical_setting": self.clinical_setting,
            "device_type": self.device_type,
            "samd_category": self.samd_category,
            "clinical_significance": self.clinical_significance,
        }


@dataclass
class PerformanceMetric:
    """A single performance metric with statistical rigor."""

    name: str
    value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    sample_size: int
    threshold: Optional[float] = None
    meets_threshold: Optional[bool] = None
    clinical_relevance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "value": round(self.value, 4),
            "confidence_interval": {
                "lower": round(self.confidence_interval_lower, 4),
                "upper": round(self.confidence_interval_upper, 4),
                "level": self.confidence_level,
            },
            "sample_size": self.sample_size,
        }
        if self.threshold is not None:
            result["threshold"] = self.threshold
            result["meets_threshold"] = self.meets_threshold
        if self.clinical_relevance:
            result["clinical_relevance"] = self.clinical_relevance
        return result


@dataclass
class DemographicPerformance:
    """Performance breakdown by demographic subgroup."""

    group_name: str
    subgroups: Dict[str, Dict[str, Any]]
    overall_disparity: float
    max_performance_gap: float
    worst_performing_subgroup: str
    meets_equity_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "subgroups": self.subgroups,
            "overall_disparity": round(self.overall_disparity, 4),
            "max_performance_gap": round(self.max_performance_gap, 4),
            "worst_performing_subgroup": self.worst_performing_subgroup,
            "meets_equity_threshold": self.meets_equity_threshold,
        }


@dataclass
class Limitation:
    """Documented model limitation or known failure mode."""

    category: str  # "technical", "clinical", "population", "environmental"
    description: str
    severity: str  # "low", "medium", "high", "critical"
    mitigation: str
    affected_population: Optional[str] = None
    failure_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
        }
        if self.affected_population:
            result["affected_population"] = self.affected_population
        if self.failure_rate is not None:
            result["failure_rate"] = self.failure_rate
        return result


class ModelCardGenerator:
    """Generate FDA-compliant model cards for medical device ML models.

    Produces a structured document that serves as both technical
    documentation and regulatory evidence for FDA submissions.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        model_hash: str,
        training_run_id: str,
        validation_run_id: str,
    ):
        """Initialize the model card generator.

        Args:
            model_name: Human-readable model name.
            model_version: Semantic version string.
            model_hash: SHA-256 hash of the model artifact.
            training_run_id: Metaflow training run ID.
            validation_run_id: Metaflow validation run ID.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_hash = model_hash
        self.training_run_id = training_run_id
        self.validation_run_id = validation_run_id
        self.generated_at = datetime.now(timezone.utc).isoformat()

        self.intended_use: Optional[IntendedUseStatement] = None
        self.training_data_description: Dict[str, Any] = {}
        self.performance_metrics: List[PerformanceMetric] = []
        self.demographic_performance: List[DemographicPerformance] = []
        self.limitations: List[Limitation] = []
        self.ethical_considerations: List[Dict[str, str]] = []
        self.regulatory_information: Dict[str, Any] = {}

    def set_intended_use(
        self,
        device_name: str,
        intended_use: str,
        indications_for_use: str,
        target_condition: str,
        target_population: str,
        intended_user: str,
        clinical_setting: str,
        device_type: str = "adjunctive",
        samd_category: str = "IIa",
        clinical_significance: str = "inform",
    ) -> None:
        """Set the FDA-required intended use statement.

        Args:
            device_name: Official device name.
            intended_use: General intended use description.
            indications_for_use: Specific indications for use.
            target_condition: Medical condition being addressed.
            target_population: Intended patient population.
            intended_user: Intended user description.
            clinical_setting: Clinical environment description.
            device_type: "standalone" or "adjunctive".
            samd_category: IMDRF risk category.
            clinical_significance: Level of clinical significance.
        """
        self.intended_use = IntendedUseStatement(
            device_name=device_name,
            intended_use=intended_use,
            indications_for_use=indications_for_use,
            target_condition=target_condition,
            target_population=target_population,
            intended_user=intended_user,
            clinical_setting=clinical_setting,
            device_type=device_type,
            samd_category=samd_category,
            clinical_significance=clinical_significance,
        )

    def set_training_data(
        self,
        description: str,
        source: str,
        collection_period: str,
        total_samples: int,
        positive_samples: int,
        negative_samples: int,
        sites: List[str],
        demographic_breakdown: Dict[str, Dict[str, int]],
        preprocessing_steps: List[str],
        quality_measures: Dict[str, Any],
        exclusion_criteria: List[str],
        data_version: str,
        data_hash: str,
    ) -> None:
        """Set the training data description.

        Args:
            description: Human-readable dataset description.
            source: Data source(s).
            collection_period: Time period of data collection.
            total_samples: Total number of samples.
            positive_samples: Number of positive (disease) samples.
            negative_samples: Number of negative (healthy) samples.
            sites: Clinical sites contributing data.
            demographic_breakdown: Distribution across demographics.
            preprocessing_steps: Data preprocessing pipeline.
            quality_measures: Data quality metrics.
            exclusion_criteria: Criteria for excluding data.
            data_version: Dataset version identifier.
            data_hash: SHA-256 hash of the dataset.
        """
        self.training_data_description = {
            "description": description,
            "source": source,
            "collection_period": collection_period,
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "prevalence": round(positive_samples / total_samples, 4),
            "sites": sites,
            "number_of_sites": len(sites),
            "demographic_breakdown": demographic_breakdown,
            "preprocessing_steps": preprocessing_steps,
            "quality_measures": quality_measures,
            "exclusion_criteria": exclusion_criteria,
            "data_version": data_version,
            "data_hash": data_hash,
        }

    def add_performance_metric(
        self,
        name: str,
        value: float,
        ci_lower: float,
        ci_upper: float,
        sample_size: int,
        confidence_level: float = 0.95,
        threshold: Optional[float] = None,
        clinical_relevance: str = "",
    ) -> None:
        """Add a performance metric with confidence interval.

        Args:
            name: Metric name (e.g., "AUC-ROC", "Sensitivity").
            value: Point estimate of the metric.
            ci_lower: Lower bound of confidence interval.
            ci_upper: Upper bound of confidence interval.
            sample_size: Number of samples used for evaluation.
            confidence_level: Confidence level for the CI.
            threshold: Predetermined acceptance threshold.
            clinical_relevance: Description of clinical significance.
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            confidence_level=confidence_level,
            sample_size=sample_size,
            threshold=threshold,
            meets_threshold=value >= threshold if threshold is not None else None,
            clinical_relevance=clinical_relevance,
        )
        self.performance_metrics.append(metric)

    def add_demographic_performance(
        self,
        group_name: str,
        subgroups: Dict[str, Dict[str, Any]],
        equity_threshold: float = 0.05,
    ) -> None:
        """Add demographic performance breakdown.

        Args:
            group_name: Demographic category (e.g., "age_group").
            subgroups: Performance metrics per subgroup.
            equity_threshold: Maximum allowable performance disparity.
        """
        aucs = [sg.get("auc", 0) for sg in subgroups.values()]
        max_gap = max(aucs) - min(aucs) if aucs else 0
        worst = min(subgroups.items(), key=lambda x: x[1].get("auc", 0))[0] if subgroups else ""

        demo = DemographicPerformance(
            group_name=group_name,
            subgroups=subgroups,
            overall_disparity=max_gap,
            max_performance_gap=max_gap,
            worst_performing_subgroup=worst,
            meets_equity_threshold=max_gap <= equity_threshold,
        )
        self.demographic_performance.append(demo)

    def add_limitation(
        self,
        category: str,
        description: str,
        severity: str,
        mitigation: str,
        affected_population: Optional[str] = None,
        failure_rate: Optional[float] = None,
    ) -> None:
        """Add a known limitation or failure mode.

        Args:
            category: Limitation category.
            description: Detailed description.
            severity: Severity level.
            mitigation: Mitigation strategy.
            affected_population: Specific population affected.
            failure_rate: Estimated failure rate.
        """
        limitation = Limitation(
            category=category,
            description=description,
            severity=severity,
            mitigation=mitigation,
            affected_population=affected_population,
            failure_rate=failure_rate,
        )
        self.limitations.append(limitation)

    def generate(self) -> Dict[str, Any]:
        """Generate the complete model card document.

        Returns:
            Dictionary containing the full FDA-style model card.
        """
        model_card = {
            "model_card_version": "2.0",
            "document_type": "FDA SaMD Model Card",
            "generated_at": self.generated_at,
            "model_identification": {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_hash": self.model_hash,
                "training_run_id": self.training_run_id,
                "validation_run_id": self.validation_run_id,
            },
            "intended_use": (
                self.intended_use.to_dict() if self.intended_use else {}
            ),
            "training_data": self.training_data_description,
            "performance": {
                "overall_metrics": [m.to_dict() for m in self.performance_metrics],
                "all_thresholds_met": all(
                    m.meets_threshold
                    for m in self.performance_metrics
                    if m.meets_threshold is not None
                ),
            },
            "demographic_performance": [
                d.to_dict() for d in self.demographic_performance
            ],
            "limitations_and_failure_modes": [
                l.to_dict() for l in self.limitations
            ],
            "ethical_considerations": self.ethical_considerations,
            "regulatory_information": self.regulatory_information,
        }

        # Compute document integrity hash
        card_hash = hashlib.sha256(
            json.dumps(model_card, sort_keys=True).encode()
        ).hexdigest()
        model_card["document_integrity_hash"] = card_hash

        return model_card

    def export_to_json(self, output_path: str) -> str:
        """Export the model card to a JSON file.

        Args:
            output_path: Path to write the JSON file.

        Returns:
            SHA-256 hash of the exported document.
        """
        card = self.generate()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(card, f, indent=2, sort_keys=True)
        return card["document_integrity_hash"]


def generate_default_model_card(
    model_version: str = "3.0.0",
    model_hash: str = "abc123",
    training_run_id: str = "train-001",
    validation_run_id: str = "val-001",
) -> Dict[str, Any]:
    """Generate a complete example model card for a mammography SaMD.

    Provides a fully populated model card demonstrating all required
    sections for an FDA regulatory submission.

    Returns:
        Complete model card dictionary.
    """
    generator = ModelCardGenerator(
        model_name="MammoScreen AI",
        model_version=model_version,
        model_hash=model_hash,
        training_run_id=training_run_id,
        validation_run_id=validation_run_id,
    )

    generator.set_intended_use(
        device_name="MammoScreen AI v3.0",
        intended_use=(
            "Computer-aided detection (CADe) software for identifying "
            "suspicious regions in digital mammography images."
        ),
        indications_for_use=(
            "MammoScreen AI is intended for use as an adjunctive tool to "
            "assist radiologists in the interpretation of screening "
            "mammograms in asymptomatic women aged 40 and older. The "
            "device highlights regions of interest that may warrant "
            "additional evaluation. It is not intended as a standalone "
            "diagnostic device."
        ),
        target_condition="Breast cancer (malignant lesions)",
        target_population="Asymptomatic women aged 40+, screening population",
        intended_user="Board-certified radiologists with mammography experience",
        clinical_setting="Breast imaging centers, radiology departments",
        device_type="adjunctive",
        samd_category="IIa",
        clinical_significance="inform",
    )

    generator.set_training_data(
        description=(
            "Multi-site retrospective cohort of de-identified digital "
            "mammography studies with pathology-confirmed ground truth."
        ),
        source="Multi-institutional consortium (5 academic medical centers)",
        collection_period="2019-01-01 to 2024-06-30",
        total_samples=85000,
        positive_samples=12750,
        negative_samples=72250,
        sites=[
            "Site A - Academic Medical Center, Northeast US",
            "Site B - Community Hospital, Southeast US",
            "Site C - Academic Medical Center, Midwest US",
            "Site D - Imaging Center, West Coast US",
            "Site E - Academic Medical Center, Southwest US",
        ],
        demographic_breakdown={
            "age_group": {
                "40-49": 22100, "50-59": 25500,
                "60-69": 23800, "70+": 13600,
            },
            "race_ethnicity": {
                "White": 44200, "Black": 15300,
                "Asian": 10200, "Hispanic": 11900, "Other": 3400,
            },
            "breast_density": {
                "A-fatty": 8500, "B-scattered": 34000,
                "C-heterogeneous": 29750, "D-dense": 12750,
            },
        },
        preprocessing_steps=[
            "DICOM header PHI removal and verification",
            "Image normalization per scanner manufacturer",
            "Resize to 1024x1024 with aspect ratio preservation",
            "Intensity windowing to clinically relevant range",
            "Feature extraction via pretrained ResNet-50 encoder",
        ],
        quality_measures={
            "inter_reader_agreement_kappa": 0.82,
            "pathology_confirmation_rate": 1.0,
            "image_quality_rejection_rate": 0.02,
        },
        exclusion_criteria=[
            "Prior breast surgery or implants",
            "Diagnostic (non-screening) mammograms",
            "Incomplete imaging studies",
            "Image quality below minimum threshold",
            "Age under 40 at time of imaging",
        ],
        data_version="2.1.0",
        data_hash="e3b0c44298fc1c149afbf4c8996fb924",
    )

    # Add performance metrics
    metrics = [
        ("AUC-ROC", 0.912, 0.898, 0.926, 0.85, "Primary endpoint"),
        ("Sensitivity", 0.853, 0.832, 0.874, 0.80, "Cancer detection rate"),
        ("Specificity", 0.901, 0.893, 0.909, 0.85, "False positive reduction"),
        ("PPV", 0.342, 0.312, 0.372, 0.30, "Positive predictive value"),
        ("NPV", 0.987, 0.983, 0.991, 0.98, "Negative predictive value"),
    ]
    for name, val, ci_lo, ci_hi, thresh, relevance in metrics:
        generator.add_performance_metric(
            name=name, value=val, ci_lower=ci_lo, ci_upper=ci_hi,
            sample_size=15000, threshold=thresh, clinical_relevance=relevance,
        )

    # Add demographic breakdowns
    generator.add_demographic_performance(
        group_name="age_group",
        subgroups={
            "40-49": {"auc": 0.905, "sensitivity": 0.84, "n": 3750},
            "50-59": {"auc": 0.918, "sensitivity": 0.86, "n": 4500},
            "60-69": {"auc": 0.920, "sensitivity": 0.87, "n": 4200},
            "70+": {"auc": 0.898, "sensitivity": 0.83, "n": 2550},
        },
    )

    generator.add_demographic_performance(
        group_name="breast_density",
        subgroups={
            "A-fatty": {"auc": 0.935, "sensitivity": 0.90, "n": 1500},
            "B-scattered": {"auc": 0.920, "sensitivity": 0.87, "n": 6000},
            "C-heterogeneous": {"auc": 0.905, "sensitivity": 0.84, "n": 5250},
            "D-dense": {"auc": 0.878, "sensitivity": 0.80, "n": 2250},
        },
        equity_threshold=0.06,
    )

    # Add limitations
    generator.add_limitation(
        category="population",
        description="Reduced sensitivity in extremely dense breast tissue (BI-RADS D)",
        severity="medium",
        mitigation="Clinical workflow requires radiologist review of all cases",
        affected_population="Women with BI-RADS density category D",
        failure_rate=0.20,
    )
    generator.add_limitation(
        category="technical",
        description="Not validated for tomosynthesis (3D mammography) images",
        severity="high",
        mitigation="Restrict input to 2D digital mammography only",
    )
    generator.add_limitation(
        category="environmental",
        description="Performance validated only on scanner models from Hologic, GE, Siemens, and Fujifilm",
        severity="medium",
        mitigation="Verify scanner compatibility before deployment at new sites",
    )
    generator.add_limitation(
        category="clinical",
        description="Not intended for use in symptomatic patients or diagnostic workup",
        severity="high",
        mitigation="Intended use labeling restricts to screening population only",
    )

    return generator.generate()
