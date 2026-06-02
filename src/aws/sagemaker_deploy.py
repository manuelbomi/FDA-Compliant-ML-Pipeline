"""
SageMaker Deployment Utilities for FDA-Compliant ML Models.

Provides model registry management, endpoint configuration, auto-scaling
policies, A/B testing setup, and model monitoring for medical device ML
models deployed on AWS SageMaker.

Designed for use within the FDA-compliant deployment pipeline, ensuring
all deployment actions are auditable and reversible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelRegistryEntry:
    """Model registry entry for SageMaker Model Registry."""

    model_package_group: str
    model_version: str
    model_hash: str
    training_run_id: str
    validation_run_id: str
    approval_status: str
    inference_spec: Dict[str, Any]
    model_metrics: Dict[str, float]
    supported_instance_types: List[str]
    created_at: str = ""
    approved_by: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_package_group": self.model_package_group,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "training_run_id": self.training_run_id,
            "validation_run_id": self.validation_run_id,
            "approval_status": self.approval_status,
            "inference_spec": self.inference_spec,
            "model_metrics": self.model_metrics,
            "supported_instance_types": self.supported_instance_types,
            "created_at": self.created_at,
            "approved_by": self.approved_by,
        }


@dataclass
class EndpointConfig:
    """SageMaker endpoint configuration."""

    endpoint_name: str
    instance_type: str
    instance_count: int
    model_name: str
    variant_name: str = "primary"
    variant_weight: float = 1.0
    data_capture_enabled: bool = True
    data_capture_percentage: int = 100

    def to_sagemaker_config(self) -> Dict[str, Any]:
        """Convert to SageMaker API format."""
        config = {
            "EndpointConfigName": f"{self.endpoint_name}-config",
            "ProductionVariants": [
                {
                    "VariantName": self.variant_name,
                    "ModelName": self.model_name,
                    "InstanceType": self.instance_type,
                    "InitialInstanceCount": self.instance_count,
                    "InitialVariantWeight": self.variant_weight,
                }
            ],
        }
        if self.data_capture_enabled:
            config["DataCaptureConfig"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": self.data_capture_percentage,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "DestinationS3Uri": (
                    f"s3://fda-ml-monitoring/{self.endpoint_name}/data-capture/"
                ),
            }
        return config


class SageMakerModelRegistry:
    """Manage models in SageMaker Model Registry.

    Provides a version-controlled registry of approved model packages,
    serving as part of the Device History Record for FDA compliance.
    """

    def __init__(self, region: str = "us-east-1"):
        """Initialize the registry manager.

        Args:
            region: AWS region for SageMaker.
        """
        self.region = region
        self.entries: Dict[str, ModelRegistryEntry] = {}

    def register_model(
        self,
        model_package_group: str,
        model_version: str,
        model_hash: str,
        training_run_id: str,
        validation_run_id: str,
        model_metrics: Dict[str, float],
        inference_spec: Dict[str, Any],
        supported_instance_types: Optional[List[str]] = None,
    ) -> ModelRegistryEntry:
        """Register a new model version in the registry.

        Args:
            model_package_group: Model package group name.
            model_version: Semantic version string.
            model_hash: SHA-256 hash of the model artifact.
            training_run_id: Associated training run ID.
            validation_run_id: Associated validation run ID.
            model_metrics: Validated performance metrics.
            inference_spec: Inference input/output specification.
            supported_instance_types: Compatible SageMaker instance types.

        Returns:
            The created registry entry.
        """
        entry = ModelRegistryEntry(
            model_package_group=model_package_group,
            model_version=model_version,
            model_hash=model_hash,
            training_run_id=training_run_id,
            validation_run_id=validation_run_id,
            approval_status="PendingManualApproval",
            inference_spec=inference_spec,
            model_metrics=model_metrics,
            supported_instance_types=supported_instance_types or [
                "ml.m5.xlarge",
                "ml.m5.2xlarge",
                "ml.c5.xlarge",
            ],
        )

        key = f"{model_package_group}/{model_version}"
        self.entries[key] = entry

        # In production:
        # sm_client = boto3.client("sagemaker", region_name=self.region)
        # sm_client.create_model_package(
        #     ModelPackageGroupName=model_package_group,
        #     ModelPackageDescription=f"Model v{model_version}",
        #     InferenceSpecification={...},
        #     ModelApprovalStatus="PendingManualApproval",
        #     ModelMetrics={...},
        # )

        return entry

    def approve_model(
        self,
        model_package_group: str,
        model_version: str,
        approved_by: str,
    ) -> bool:
        """Approve a model for deployment.

        Args:
            model_package_group: Model package group name.
            model_version: Version to approve.
            approved_by: Approver identifier.

        Returns:
            True if approval was recorded.
        """
        key = f"{model_package_group}/{model_version}"
        entry = self.entries.get(key)
        if entry is None:
            return False

        entry.approval_status = "Approved"
        entry.approved_by = approved_by

        # In production:
        # sm_client.update_model_package(
        #     ModelPackageArn=...,
        #     ModelApprovalStatus="Approved",
        # )

        return True

    def get_latest_approved(
        self, model_package_group: str
    ) -> Optional[ModelRegistryEntry]:
        """Get the latest approved model version.

        Args:
            model_package_group: Model package group to query.

        Returns:
            Latest approved model entry, or None.
        """
        approved = [
            entry for key, entry in self.entries.items()
            if key.startswith(model_package_group)
            and entry.approval_status == "Approved"
        ]
        if not approved:
            return None
        return max(approved, key=lambda e: e.created_at)


class SageMakerEndpointManager:
    """Manage SageMaker endpoint lifecycle for medical device models."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize the endpoint manager.

        Args:
            region: AWS region for SageMaker.
        """
        self.region = region

    def create_endpoint_config(
        self,
        endpoint_name: str,
        variants: List[Dict[str, Any]],
        data_capture_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a SageMaker endpoint configuration.

        Args:
            endpoint_name: Name for the endpoint.
            variants: List of production variant configurations.
            data_capture_config: Optional data capture configuration.

        Returns:
            Endpoint configuration dictionary.
        """
        config = {
            "EndpointConfigName": f"{endpoint_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "ProductionVariants": variants,
        }

        if data_capture_config:
            config["DataCaptureConfig"] = data_capture_config

        return config

    def configure_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str,
        min_instances: int = 1,
        max_instances: int = 8,
        target_invocations_per_instance: int = 100,
        scale_in_cooldown: int = 300,
        scale_out_cooldown: int = 60,
    ) -> Dict[str, Any]:
        """Configure auto-scaling for an endpoint variant.

        Args:
            endpoint_name: Endpoint name.
            variant_name: Production variant name.
            min_instances: Minimum instance count.
            max_instances: Maximum instance count.
            target_invocations_per_instance: Target invocations per instance per minute.
            scale_in_cooldown: Cooldown period for scale-in (seconds).
            scale_out_cooldown: Cooldown period for scale-out (seconds).

        Returns:
            Auto-scaling configuration dictionary.
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        scaling_config = {
            "resource_id": resource_id,
            "scalable_dimension": "sagemaker:variant:DesiredInstanceCount",
            "min_capacity": min_instances,
            "max_capacity": max_instances,
            "scaling_policy": {
                "policy_name": f"{endpoint_name}-scaling-policy",
                "policy_type": "TargetTrackingScaling",
                "target_tracking_configuration": {
                    "target_value": target_invocations_per_instance,
                    "predefined_metric": "SageMakerVariantInvocationsPerInstance",
                    "scale_in_cooldown": scale_in_cooldown,
                    "scale_out_cooldown": scale_out_cooldown,
                },
            },
        }

        # In production:
        # aas_client = boto3.client("application-autoscaling")
        # aas_client.register_scalable_target(...)
        # aas_client.put_scaling_policy(...)

        return scaling_config

    def configure_ab_test(
        self,
        endpoint_name: str,
        model_a_name: str,
        model_b_name: str,
        traffic_split: float = 0.5,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 2,
    ) -> Dict[str, Any]:
        """Configure an A/B test between two model versions.

        Args:
            endpoint_name: Endpoint name.
            model_a_name: Control model name.
            model_b_name: Treatment model name.
            traffic_split: Fraction of traffic to treatment (0-1).
            instance_type: SageMaker instance type.
            instance_count: Number of instances per variant.

        Returns:
            A/B test endpoint configuration.
        """
        return self.create_endpoint_config(
            endpoint_name=endpoint_name,
            variants=[
                {
                    "VariantName": "control",
                    "ModelName": model_a_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": instance_count,
                    "InitialVariantWeight": 1.0 - traffic_split,
                },
                {
                    "VariantName": "treatment",
                    "ModelName": model_b_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": instance_count,
                    "InitialVariantWeight": traffic_split,
                },
            ],
            data_capture_config={
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "DestinationS3Uri": (
                    f"s3://fda-ml-monitoring/{endpoint_name}/ab-test/"
                ),
            },
        )


class SageMakerModelMonitor:
    """Configure SageMaker Model Monitor for deployed models.

    Sets up continuous monitoring for data quality, model quality,
    bias drift, and feature attribution drift -- all required for
    GMLP Principle 10 (monitoring of deployed models).
    """

    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        """Initialize the model monitor.

        Args:
            endpoint_name: Name of the endpoint to monitor.
            region: AWS region.
        """
        self.endpoint_name = endpoint_name
        self.region = region

    def create_data_quality_monitor(
        self,
        baseline_dataset_uri: str,
        schedule_expression: str = "cron(0 * * * ? *)",
        output_s3_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a data quality monitoring schedule.

        Detects drift in input data distributions compared to the
        training data baseline.

        Args:
            baseline_dataset_uri: S3 URI of the baseline dataset.
            schedule_expression: CloudWatch Events schedule expression.
            output_s3_uri: S3 URI for monitoring output.

        Returns:
            Data quality monitor configuration.
        """
        output_uri = output_s3_uri or (
            f"s3://fda-ml-monitoring/{self.endpoint_name}/data-quality/"
        )

        return {
            "monitor_type": "data_quality",
            "endpoint_name": self.endpoint_name,
            "baseline_dataset_uri": baseline_dataset_uri,
            "schedule_expression": schedule_expression,
            "output_s3_uri": output_uri,
            "constraints": {
                "max_missing_values_pct": 0.01,
                "max_feature_drift_distance": 0.05,
            },
            "alert_config": {
                "sns_topic_arn": (
                    f"arn:aws:sns:{self.region}:123456789:ml-data-quality-alerts"
                ),
                "threshold": "violation",
            },
        }

    def create_model_quality_monitor(
        self,
        ground_truth_s3_uri: str,
        performance_thresholds: Dict[str, float],
        schedule_expression: str = "cron(0 0 * * ? *)",
    ) -> Dict[str, Any]:
        """Create a model quality monitoring schedule.

        Monitors model performance against ground truth labels when
        they become available (e.g., after pathology results).

        Args:
            ground_truth_s3_uri: S3 URI of ground truth labels.
            performance_thresholds: Metric thresholds for alerting.
            schedule_expression: Schedule for evaluation.

        Returns:
            Model quality monitor configuration.
        """
        return {
            "monitor_type": "model_quality",
            "endpoint_name": self.endpoint_name,
            "ground_truth_s3_uri": ground_truth_s3_uri,
            "schedule_expression": schedule_expression,
            "problem_type": "BinaryClassification",
            "performance_thresholds": performance_thresholds,
            "metrics_to_monitor": [
                "auc", "f1", "precision", "recall", "accuracy",
            ],
            "alert_config": {
                "sns_topic_arn": (
                    f"arn:aws:sns:{self.region}:123456789:ml-model-quality-alerts"
                ),
            },
        }

    def create_bias_drift_monitor(
        self,
        baseline_bias_config: Dict[str, Any],
        sensitive_attributes: List[str],
        schedule_expression: str = "cron(0 0 * * ? *)",
    ) -> Dict[str, Any]:
        """Create a bias drift monitoring schedule.

        Detects changes in model fairness metrics over time,
        required for ongoing compliance with GMLP principles.

        Args:
            baseline_bias_config: Baseline fairness metrics.
            sensitive_attributes: Demographic attributes to monitor.
            schedule_expression: Schedule for evaluation.

        Returns:
            Bias drift monitor configuration.
        """
        return {
            "monitor_type": "bias_drift",
            "endpoint_name": self.endpoint_name,
            "baseline_bias_config": baseline_bias_config,
            "sensitive_attributes": sensitive_attributes,
            "schedule_expression": schedule_expression,
            "fairness_metrics": [
                "demographic_parity_difference",
                "equalized_odds_difference",
                "calibration_difference",
            ],
            "thresholds": {
                "max_demographic_parity_shift": 0.05,
                "max_equalized_odds_shift": 0.05,
            },
        }

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of all active monitoring configurations.

        Returns:
            Summary of monitoring setup for the endpoint.
        """
        return {
            "endpoint_name": self.endpoint_name,
            "monitors": [
                "data_quality",
                "model_quality",
                "bias_drift",
            ],
            "status": "active",
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }
