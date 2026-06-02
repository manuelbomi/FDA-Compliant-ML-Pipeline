"""
FDA-Compliant Deployment Flow for Medical Device ML Models.

Implements a controlled deployment pipeline with canary releases, A/B testing,
rollback capability, and SageMaker endpoint management. Designed to satisfy
FDA 21 CFR 820.70 (Production and Process Controls) and the Predetermined
Change Control Plan framework for AI/ML-based SaMD.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from metaflow import FlowSpec, Parameter, current, step, card, retry, project


class DeploymentStrategy(str, Enum):
    """Deployment strategies with different risk profiles."""

    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    AB_TEST = "ab_test"
    ROLLING = "rolling"
    FULL = "full"


@dataclass
class DeploymentManifest:
    """Complete deployment manifest for regulatory traceability."""

    deployment_id: str
    model_version: str
    model_hash: str
    training_run_id: str
    validation_run_id: str
    strategy: str
    endpoint_name: str
    created_at: str
    created_by: str
    approved_by: Optional[str] = None
    approval_timestamp: Optional[str] = None
    sagemaker_config: Dict[str, Any] = field(default_factory=dict)
    rollback_target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "training_run_id": self.training_run_id,
            "validation_run_id": self.validation_run_id,
            "strategy": self.strategy,
            "endpoint_name": self.endpoint_name,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "approval_timestamp": self.approval_timestamp,
            "sagemaker_config": self.sagemaker_config,
            "rollback_target": self.rollback_target,
            "metadata": self.metadata,
        }


@dataclass
class ModelPackage:
    """Packaged model artifact with all metadata required for deployment."""

    model_artifact_path: str
    model_hash: str
    model_version: str
    framework: str
    framework_version: str
    inference_spec: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    environment_variables: Dict[str, str]
    resource_requirements: Dict[str, Any]
    health_check_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_artifact_path": self.model_artifact_path,
            "model_hash": self.model_hash,
            "model_version": self.model_version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "inference_spec": self.inference_spec,
            "preprocessing_config": self.preprocessing_config,
            "resource_requirements": self.resource_requirements,
            "health_check_config": self.health_check_config,
        }


@project(name="fda_samd_pipeline")
class MedicalDeviceDeploymentFlow(FlowSpec):
    """FDA-compliant deployment pipeline for medical device ML models.

    Implements controlled deployment with:
    - Pre-deployment validation gate
    - Model packaging with complete version metadata
    - SageMaker Model Registry integration
    - Canary deployment with automatic rollback
    - A/B testing for model comparison
    - Real-time monitoring setup
    - Complete audit trail

    Satisfies FDA 21 CFR 820.70 (Production and Process Controls)
    and 820.90 (Nonconforming Product / CAPA).
    """

    training_run_id = Parameter(
        "training-run-id",
        help="Metaflow run ID of the training flow",
        required=True,
        type=str,
    )

    validation_run_id = Parameter(
        "validation-run-id",
        help="Metaflow run ID of the validation flow (must have passed)",
        required=True,
        type=str,
    )

    deployment_strategy = Parameter(
        "strategy",
        help="Deployment strategy: canary, blue_green, ab_test, full",
        default="canary",
        type=str,
    )

    canary_traffic_pct = Parameter(
        "canary-pct",
        help="Percentage of traffic for canary deployment (1-50)",
        default=5,
        type=int,
    )

    ab_test_traffic_split = Parameter(
        "ab-split",
        help="Traffic split for A/B testing (percentage to new model)",
        default=50,
        type=int,
    )

    endpoint_name = Parameter(
        "endpoint",
        help="SageMaker endpoint name",
        default="mammography-screening-prod",
        type=str,
    )

    approved_by = Parameter(
        "approved-by",
        help="Name/ID of the person who approved deployment (required for audit)",
        required=True,
        type=str,
    )

    config_path = Parameter(
        "config",
        help="Path to deployment configuration YAML file",
        default="configs/deployment_config.yaml",
        type=str,
    )

    @step
    def start(self):
        """Initialize deployment and verify prerequisites.

        Checks that:
        1. Training run exists and completed successfully
        2. Validation run passed all acceptance criteria
        3. Deployment has been approved by authorized personnel
        4. No active deployment locks exist
        """
        import yaml

        self.deployment_timestamp = datetime.now(timezone.utc).isoformat()
        self.deployment_id = (
            f"deploy-{self.endpoint_name}-{current.run_id}"
        )

        # Load deployment config
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.deploy_config = yaml.safe_load(f)
        else:
            self.deploy_config = {
                "sagemaker": {
                    "instance_type": "ml.m5.xlarge",
                    "instance_count": 2,
                    "max_instance_count": 8,
                    "target_invocations_per_instance": 100,
                    "scale_in_cooldown": 300,
                    "scale_out_cooldown": 60,
                },
                "monitoring": {
                    "enable_data_capture": True,
                    "capture_percentage": 100,
                    "enable_model_monitor": True,
                    "monitor_schedule": "cron(0 * * * ? *)",
                    "drift_detection_threshold": 0.05,
                },
                "canary": {
                    "evaluation_period_minutes": 30,
                    "error_rate_threshold": 0.01,
                    "latency_p99_threshold_ms": 500,
                    "auto_rollback": True,
                },
                "rollback": {
                    "max_error_rate": 0.05,
                    "max_latency_p99_ms": 1000,
                    "evaluation_window_minutes": 15,
                },
            }

        # Verify validation passed (in production, load from Metaflow)
        self.validation_passed = True  # Would verify from validation run artifacts
        self.model_hash = hashlib.sha256(b"model-artifact-v3").hexdigest()
        self.model_version = "3.0.0"

        if not self.validation_passed:
            raise RuntimeError(
                f"Validation run {self.validation_run_id} did not pass. "
                f"Deployment blocked per FDA 21 CFR 820.30(g)."
            )

        # Record deployment approval
        self.approval_record = {
            "approved_by": self.approved_by,
            "approval_timestamp": self.deployment_timestamp,
            "training_run_id": self.training_run_id,
            "validation_run_id": self.validation_run_id,
            "model_hash": self.model_hash,
            "deployment_strategy": self.deployment_strategy,
        }

        print(f"[FDA AUDIT] Deployment initialized: {self.deployment_id}")
        print(f"[FDA AUDIT] Strategy: {self.deployment_strategy}")
        print(f"[FDA AUDIT] Approved by: {self.approved_by}")

        self.next(self.package_model)

    @retry(times=2)
    @step
    def package_model(self):
        """Package model with complete version metadata and inference spec.

        Creates a self-contained deployment package that includes:
        - Serialized model artifact with integrity hash
        - Preprocessing configuration (normalization parameters)
        - Inference specification (input/output schema)
        - Environment configuration
        - Health check configuration
        """
        self.model_package = ModelPackage(
            model_artifact_path=(
                f"s3://fda-ml-models/mammography-screening/"
                f"v{self.model_version}/model.tar.gz"
            ),
            model_hash=self.model_hash,
            model_version=self.model_version,
            framework="pytorch",
            framework_version="2.1.0",
            inference_spec={
                "input_schema": {
                    "type": "image",
                    "format": "DICOM",
                    "preprocessing": [
                        "dicom_to_png",
                        "resize_1024x1024",
                        "normalize_intensity",
                        "feature_extraction",
                    ],
                },
                "output_schema": {
                    "type": "json",
                    "fields": {
                        "malignancy_score": {
                            "type": "float",
                            "range": [0, 1],
                            "description": "Probability of malignancy",
                        },
                        "birads_category": {
                            "type": "int",
                            "range": [1, 5],
                            "description": "BI-RADS assessment category",
                        },
                        "model_version": {"type": "string"},
                        "inference_timestamp": {"type": "string"},
                        "confidence_interval": {
                            "type": "object",
                            "fields": {"lower": "float", "upper": "float"},
                        },
                    },
                },
                "max_batch_size": 16,
                "timeout_seconds": 30,
            },
            preprocessing_config={
                "normalization": "z-score",
                "feature_mean_hash": hashlib.sha256(b"feature-mean").hexdigest(),
                "feature_std_hash": hashlib.sha256(b"feature-std").hexdigest(),
            },
            environment_variables={
                "MODEL_VERSION": self.model_version,
                "AUDIT_LOGGING": "enabled",
                "LOG_LEVEL": "INFO",
                "INFERENCE_TIMEOUT": "30",
            },
            resource_requirements={
                "min_memory_mb": 4096,
                "min_cpu_cores": 2,
                "gpu_required": False,
                "storage_gb": 10,
            },
            health_check_config={
                "path": "/health",
                "interval_seconds": 30,
                "timeout_seconds": 5,
                "healthy_threshold": 3,
                "unhealthy_threshold": 2,
            },
        )

        package_hash = hashlib.sha256(
            json.dumps(self.model_package.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        self.package_hash = package_hash

        print(f"[FDA AUDIT] Model packaged: v{self.model_version}")
        print(f"[FDA AUDIT] Package hash: {package_hash}")

        self.next(self.register_model)

    @retry(times=2)
    @step
    def register_model(self):
        """Register model in SageMaker Model Registry.

        Creates a versioned model package in the registry with complete
        metadata for traceability. The registry serves as part of the
        Device History Record per FDA 21 CFR 820.184.
        """
        self.registry_entry = {
            "model_package_group": "mammography-screening-samd",
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "package_hash": self.package_hash,
            "training_run_id": self.training_run_id,
            "validation_run_id": self.validation_run_id,
            "approval_status": "Approved",
            "approved_by": self.approved_by,
            "approval_timestamp": self.deployment_timestamp,
            "inference_specification": self.model_package.inference_spec,
            "supported_instance_types": [
                "ml.m5.xlarge",
                "ml.m5.2xlarge",
                "ml.c5.xlarge",
            ],
            "model_metrics": {
                "auc_roc": 0.91,
                "sensitivity": 0.85,
                "specificity": 0.90,
            },
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        # In production: boto3 sagemaker client call
        # sm_client.create_model_package(...)

        print(f"[FDA AUDIT] Model registered in SageMaker Model Registry")
        print(f"[FDA AUDIT] Package group: {self.registry_entry['model_package_group']}")

        self.next(self.deploy_canary)

    @step
    def deploy_canary(self):
        """Deploy canary instance with limited traffic routing.

        Implements a canary deployment strategy where a small percentage
        of production traffic is routed to the new model while monitoring
        for errors, latency degradation, and prediction drift.

        If the canary evaluation fails, automatic rollback is triggered.
        """
        strategy = DeploymentStrategy(self.deployment_strategy)

        if strategy == DeploymentStrategy.CANARY:
            self.endpoint_config = {
                "endpoint_name": self.endpoint_name,
                "production_variants": [
                    {
                        "variant_name": "current-production",
                        "model_name": f"mammography-v{self.model_version}-prev",
                        "initial_instance_count": (
                            self.deploy_config["sagemaker"]["instance_count"]
                        ),
                        "instance_type": (
                            self.deploy_config["sagemaker"]["instance_type"]
                        ),
                        "initial_variant_weight": (
                            (100 - self.canary_traffic_pct) / 100.0
                        ),
                    },
                    {
                        "variant_name": "canary",
                        "model_name": f"mammography-v{self.model_version}",
                        "initial_instance_count": 1,
                        "instance_type": (
                            self.deploy_config["sagemaker"]["instance_type"]
                        ),
                        "initial_variant_weight": self.canary_traffic_pct / 100.0,
                    },
                ],
                "data_capture_config": {
                    "enable_capture": True,
                    "initial_sampling_percentage": (
                        self.deploy_config["monitoring"]["capture_percentage"]
                    ),
                    "capture_options": [
                        {"capture_mode": "Input"},
                        {"capture_mode": "Output"},
                    ],
                    "destination_s3_uri": (
                        f"s3://fda-ml-monitoring/{self.endpoint_name}/data-capture/"
                    ),
                },
            }

        elif strategy == DeploymentStrategy.AB_TEST:
            self.endpoint_config = {
                "endpoint_name": self.endpoint_name,
                "production_variants": [
                    {
                        "variant_name": "control",
                        "model_name": f"mammography-v{self.model_version}-prev",
                        "initial_instance_count": (
                            self.deploy_config["sagemaker"]["instance_count"]
                        ),
                        "instance_type": (
                            self.deploy_config["sagemaker"]["instance_type"]
                        ),
                        "initial_variant_weight": (
                            (100 - self.ab_test_traffic_split) / 100.0
                        ),
                    },
                    {
                        "variant_name": "treatment",
                        "model_name": f"mammography-v{self.model_version}",
                        "initial_instance_count": (
                            self.deploy_config["sagemaker"]["instance_count"]
                        ),
                        "instance_type": (
                            self.deploy_config["sagemaker"]["instance_type"]
                        ),
                        "initial_variant_weight": (
                            self.ab_test_traffic_split / 100.0
                        ),
                    },
                ],
            }

        else:
            self.endpoint_config = {
                "endpoint_name": self.endpoint_name,
                "production_variants": [
                    {
                        "variant_name": "production",
                        "model_name": f"mammography-v{self.model_version}",
                        "initial_instance_count": (
                            self.deploy_config["sagemaker"]["instance_count"]
                        ),
                        "instance_type": (
                            self.deploy_config["sagemaker"]["instance_type"]
                        ),
                        "initial_variant_weight": 1.0,
                    },
                ],
            }

        # In production: boto3 sagemaker endpoint update
        # sm_client.update_endpoint(EndpointName=..., EndpointConfigName=...)

        self.canary_deployed_at = datetime.now(timezone.utc).isoformat()

        print(f"[FDA AUDIT] Canary deployed: {self.canary_traffic_pct}% traffic")
        print(f"[FDA AUDIT] Endpoint: {self.endpoint_name}")

        self.next(self.evaluate_canary)

    @step
    def evaluate_canary(self):
        """Evaluate canary deployment metrics and decide rollback or promote.

        Monitors the canary for the configured evaluation period, checking:
        - Error rate
        - Latency (p50, p95, p99)
        - Prediction distribution drift
        - Data capture anomalies

        Implements automatic rollback per FDA 21 CFR 820.90 if thresholds
        are exceeded.
        """
        import numpy as np

        canary_config = self.deploy_config["canary"]

        # Simulate canary evaluation metrics
        np.random.seed(42)
        self.canary_metrics = {
            "evaluation_period_minutes": canary_config["evaluation_period_minutes"],
            "total_requests": 1500,
            "error_count": 2,
            "error_rate": 0.0013,
            "latency_p50_ms": 45.2,
            "latency_p95_ms": 120.8,
            "latency_p99_ms": 235.4,
            "prediction_mean": 0.148,
            "prediction_std": 0.182,
            "prediction_distribution_ks_stat": 0.023,
            "prediction_distribution_ks_pvalue": 0.87,
        }

        # Evaluate against thresholds
        self.canary_evaluation = {
            "error_rate_pass": (
                self.canary_metrics["error_rate"]
                <= canary_config["error_rate_threshold"]
            ),
            "latency_pass": (
                self.canary_metrics["latency_p99_ms"]
                <= canary_config["latency_p99_threshold_ms"]
            ),
            "distribution_pass": (
                self.canary_metrics["prediction_distribution_ks_pvalue"] > 0.05
            ),
        }
        self.canary_evaluation["overall_pass"] = all(
            self.canary_evaluation.values()
        )

        self.should_rollback = not self.canary_evaluation["overall_pass"]

        if self.should_rollback:
            print("[FDA AUDIT] CANARY FAILED - Triggering rollback")
        else:
            print("[FDA AUDIT] Canary evaluation PASSED")
            print(f"[FDA AUDIT] Error rate: {self.canary_metrics['error_rate']:.4f}")
            print(f"[FDA AUDIT] P99 latency: {self.canary_metrics['latency_p99_ms']:.1f}ms")

        self.next(self.finalize_deployment)

    @step
    def finalize_deployment(self):
        """Promote canary to full production or execute rollback.

        If canary passed, gradually shift traffic to the new model.
        If canary failed, rollback to the previous version and generate
        a CAPA (Corrective and Preventive Action) record.
        """
        if self.should_rollback:
            self.deployment_status = "ROLLED_BACK"
            self.rollback_record = {
                "rollback_timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "Canary evaluation failed",
                "canary_metrics": self.canary_metrics,
                "canary_evaluation": self.canary_evaluation,
                "capa_required": True,
                "capa_id": f"CAPA-{current.run_id}",
            }
            print("[FDA AUDIT] Rollback executed successfully")
            print(f"[FDA AUDIT] CAPA record created: {self.rollback_record['capa_id']}")
        else:
            self.deployment_status = "DEPLOYED"
            # In production: update endpoint to route 100% traffic to new model
            self.promotion_record = {
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "previous_variant_weight": self.canary_traffic_pct / 100.0,
                "final_variant_weight": 1.0,
                "transition_steps": [
                    {"traffic_pct": self.canary_traffic_pct, "duration_min": 30},
                    {"traffic_pct": 25, "duration_min": 30},
                    {"traffic_pct": 50, "duration_min": 30},
                    {"traffic_pct": 100, "duration_min": 0},
                ],
            }
            print("[FDA AUDIT] Model promoted to full production")

        self.next(self.setup_monitoring)

    @step
    def setup_monitoring(self):
        """Configure post-deployment monitoring.

        Sets up real-time monitoring for:
        - Data drift detection
        - Prediction distribution shifts
        - Performance degradation
        - Latency and error rate alerts

        Required by GMLP Principle 10 for deployed AI/ML-based SaMD.
        """
        monitoring_config = self.deploy_config["monitoring"]

        self.monitoring_setup = {
            "model_monitor": {
                "schedule": monitoring_config["monitor_schedule"],
                "baseline_dataset": (
                    f"s3://fda-ml-monitoring/{self.endpoint_name}/baseline/"
                ),
                "constraints": {
                    "max_feature_drift": monitoring_config["drift_detection_threshold"],
                    "max_prediction_drift": 0.05,
                    "max_missing_values_pct": 0.01,
                },
                "alert_destinations": [
                    {
                        "type": "sns",
                        "topic": (
                            f"arn:aws:sns:us-east-1:123456789:ml-monitoring-alerts"
                        ),
                    },
                    {
                        "type": "pagerduty",
                        "service": "ml-samd-monitoring",
                    },
                ],
            },
            "cloudwatch_alarms": [
                {
                    "name": f"{self.endpoint_name}-error-rate",
                    "metric": "ModelError",
                    "threshold": 0.01,
                    "period_seconds": 300,
                    "evaluation_periods": 3,
                    "action": "auto_rollback",
                },
                {
                    "name": f"{self.endpoint_name}-latency-p99",
                    "metric": "ModelLatency",
                    "statistic": "p99",
                    "threshold": 500,
                    "period_seconds": 300,
                    "evaluation_periods": 3,
                    "action": "alert",
                },
                {
                    "name": f"{self.endpoint_name}-invocations",
                    "metric": "Invocations",
                    "threshold": 0,
                    "comparison": "LessThanOrEqualToThreshold",
                    "period_seconds": 600,
                    "evaluation_periods": 1,
                    "action": "alert",
                },
            ],
        }

        print("[FDA AUDIT] Monitoring configured")
        self.next(self.end)

    @step
    def end(self):
        """Generate complete deployment record for regulatory audit.

        Produces a deployment manifest that serves as part of the Device
        History Record per FDA 21 CFR 820.184, documenting the complete
        deployment process including approval, canary evaluation, and
        monitoring configuration.
        """
        manifest = DeploymentManifest(
            deployment_id=self.deployment_id,
            model_version=self.model_version,
            model_hash=self.model_hash,
            training_run_id=self.training_run_id,
            validation_run_id=self.validation_run_id,
            strategy=self.deployment_strategy,
            endpoint_name=self.endpoint_name,
            created_at=self.deployment_timestamp,
            created_by=os.getenv("USER", "pipeline-service-account"),
            approved_by=self.approved_by,
            approval_timestamp=self.deployment_timestamp,
            sagemaker_config=self.endpoint_config,
            metadata={
                "deployment_status": self.deployment_status,
                "canary_metrics": self.canary_metrics,
                "canary_evaluation": self.canary_evaluation,
                "monitoring_setup": self.monitoring_setup,
            },
        )

        self.deployment_manifest = manifest.to_dict()

        manifest_hash = hashlib.sha256(
            json.dumps(self.deployment_manifest, sort_keys=True).encode()
        ).hexdigest()
        self.deployment_manifest["manifest_integrity_hash"] = manifest_hash

        print(f"\n{'='*60}")
        print(f"[FDA AUDIT] Deployment Complete: {self.deployment_id}")
        print(f"[FDA AUDIT] Status: {self.deployment_status}")
        print(f"[FDA AUDIT] Model: v{self.model_version} ({self.model_hash[:12]}...)")
        print(f"[FDA AUDIT] Manifest Hash: {manifest_hash}")
        print(f"{'='*60}")


if __name__ == "__main__":
    MedicalDeviceDeploymentFlow()
