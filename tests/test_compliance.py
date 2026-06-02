"""
Tests for FDA compliance features.

Verifies that audit trail, data governance, model card generation,
and change control modules satisfy regulatory requirements.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import numpy as np


# ---- Audit Trail Tests ---- #

class TestAuditTrail:
    """Tests for the immutable audit trail system."""

    def setup_method(self):
        """Create a temporary audit trail database for each test."""
        from src.compliance.audit_trail import AuditTrailStore, AuditLogger

        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_audit.db")
        self.store = AuditTrailStore(db_path=self.db_path)
        self.logger = AuditLogger(
            store=self.store,
            user_id="test-user",
            session_id="test-session",
        )

    def test_append_event(self):
        """Events can be appended to the audit trail."""
        event = self.logger.log_pipeline_start(
            run_id="test-run-001",
            flow_name="TestFlow",
            parameters={"lr": 0.001},
        )
        assert event.event_id is not None
        assert event.event_hash != ""
        assert event.user_id == "test-user"

    def test_hash_chain_integrity(self):
        """Hash chain integrity is maintained across multiple events."""
        for i in range(10):
            self.logger.log_step_complete(
                run_id="test-run-001",
                step_name=f"step_{i}",
                artifact_hash=hashlib.sha256(f"artifact-{i}".encode()).hexdigest(),
            )

        is_valid, violations = self.store.verify_chain_integrity()
        assert is_valid, f"Chain integrity violated: {violations}"
        assert len(violations) == 0

    def test_query_by_run(self):
        """Events can be queried by pipeline run ID."""
        run_id = "test-run-002"
        self.logger.log_pipeline_start(
            run_id=run_id, flow_name="TestFlow", parameters={}
        )
        self.logger.log_step_complete(
            run_id=run_id, step_name="train"
        )

        events = self.store.query_by_run(run_id)
        assert len(events) == 2
        assert all(e["pipeline_run_id"] == run_id for e in events)

    def test_query_by_event_type(self):
        """Events can be filtered by type."""
        from src.compliance.audit_trail import AuditEventType

        self.logger.log_data_access(
            dataset_id="test-dataset",
            access_type="read",
            record_count=1000,
            data_hash="abc123",
        )

        events = self.store.query_by_type(AuditEventType.DATA_ACCESS)
        assert len(events) >= 1
        assert events[0]["event_type"] == "data.access"

    def test_artifact_history(self):
        """Artifact history can be traced by hash."""
        artifact_hash = hashlib.sha256(b"test-model").hexdigest()

        self.logger.log_step_complete(
            run_id="run-001",
            step_name="train",
            artifact_hash=artifact_hash,
        )
        self.logger.log_model_registration(
            run_id="run-001",
            model_version="1.0.0",
            model_hash=artifact_hash,
            registry_id="reg-001",
        )

        history = self.store.get_artifact_history(artifact_hash)
        assert len(history) == 2

    def test_export_for_submission(self):
        """Audit trail can be exported for FDA submission."""
        run_id = "export-test-run"
        self.logger.log_pipeline_start(
            run_id=run_id, flow_name="ExportTest", parameters={}
        )

        export_path = os.path.join(self.temp_dir, "export.json")
        export_hash = self.store.export_for_submission(run_id, export_path)

        assert os.path.exists(export_path)
        assert export_hash is not None

        with open(export_path) as f:
            export_data = json.load(f)
        assert export_data["pipeline_run_id"] == run_id
        assert export_data["chain_integrity_verified"] is True

    def test_immutability(self):
        """Audit trail is append-only (no updates or deletes)."""
        self.logger.log_pipeline_start(
            run_id="immutable-test",
            flow_name="Test",
            parameters={},
        )

        # Verify events exist
        events = self.store.query_by_run("immutable-test")
        assert len(events) == 1

        # Verify chain integrity after multiple appends
        for i in range(5):
            self.logger.log_step_complete(
                run_id="immutable-test",
                step_name=f"step_{i}",
            )

        is_valid, _ = self.store.verify_chain_integrity()
        assert is_valid


# ---- Data Governance Tests ---- #

class TestDataGovernance:
    """Tests for HIPAA compliance and data governance."""

    def test_phi_detection_ssn(self):
        """PHI detector identifies Social Security Numbers."""
        from src.compliance.data_governance import PHIDetector

        detector = PHIDetector()
        text = "Patient SSN is 123-45-6789"
        findings = detector.scan_text(text)

        assert len(findings) >= 1
        ssn_findings = [f for f in findings if f["phi_type"] == "ssn"]
        assert len(ssn_findings) == 1

    def test_phi_detection_email(self):
        """PHI detector identifies email addresses."""
        from src.compliance.data_governance import PHIDetector

        detector = PHIDetector()
        text = "Contact: john.doe@hospital.com"
        findings = detector.scan_text(text)

        email_findings = [f for f in findings if f["phi_type"] == "email"]
        assert len(email_findings) == 1

    def test_phi_scrubbing(self):
        """PHI scrubber removes detected PHI elements."""
        from src.compliance.data_governance import PHIDetector

        detector = PHIDetector()
        text = "Patient SSN: 123-45-6789, Email: test@example.com"
        scrubbed = detector.scrub_text(text)

        assert "123-45-6789" not in scrubbed
        assert "test@example.com" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_phi_column_detection(self):
        """PHI detector flags columns with PHI-like names."""
        from src.compliance.data_governance import PHIDetector

        detector = PHIDetector()
        columns = ["feature_1", "patient_name", "score", "date_of_birth", "auc"]
        flagged = detector.scan_dataframe_columns(columns)

        flagged_names = {f["column_name"] for f in flagged}
        assert "patient_name" in flagged_names
        assert "date_of_birth" in flagged_names
        assert "feature_1" not in flagged_names

    def test_data_lineage_tracking(self):
        """Data lineage tracks transformations correctly."""
        from src.compliance.data_governance import (
            DataLineageTracker,
            DataClassification,
        )

        tracker = DataLineageTracker()

        source = tracker.register_source(
            dataset_id="raw-mammography",
            version="1.0",
            data_hash="abc123",
            classification=DataClassification.PHI,
        )

        transformed = tracker.register_transformation(
            dataset_id="processed-mammography",
            version="1.0",
            operation="phi_scrubbing",
            input_node_ids=[source.node_id],
            output_hash="def456",
            parameters={"method": "safe_harbor"},
        )

        lineage = tracker.get_full_lineage(transformed.node_id)
        assert len(lineage) == 2
        assert lineage[0].node_id == source.node_id

    def test_access_control(self):
        """Access control enforces role-based permissions."""
        from src.compliance.data_governance import (
            AccessControlManager,
            AccessLevel,
            DataClassification,
        )

        manager = AccessControlManager()
        manager.assign_role("scientist-1", "data_scientist")
        manager.assign_role("admin-1", "admin")

        # Data scientist can read but not write
        assert manager.check_access(
            "scientist-1", AccessLevel.READ, "dataset-1", DataClassification.INTERNAL
        )
        assert not manager.check_access(
            "scientist-1", AccessLevel.WRITE, "dataset-1", DataClassification.INTERNAL
        )

        # Admin can do everything
        assert manager.check_access(
            "admin-1", AccessLevel.ADMIN, "dataset-1", DataClassification.PHI
        )

    def test_consent_management(self):
        """Consent manager tracks and verifies consent records."""
        from src.compliance.data_governance import ConsentManager

        manager = ConsentManager()

        consent = manager.record_consent(
            patient_id_hash="hash-patient-001",
            consent_type="research",
            scope=["ml_training", "validation"],
            irb_protocol_id="IRB-2024-001",
            duration_days=365,
        )

        valid, record = manager.verify_consent(
            patient_id_hash="hash-patient-001",
            required_scope="ml_training",
        )
        assert valid
        assert record is not None

        # Non-consented scope
        valid, _ = manager.verify_consent(
            patient_id_hash="hash-patient-001",
            required_scope="commercial_use",
        )
        assert not valid

    def test_consent_revocation(self):
        """Consent can be revoked and is no longer valid."""
        from src.compliance.data_governance import ConsentManager

        manager = ConsentManager()
        consent = manager.record_consent(
            patient_id_hash="hash-patient-002",
            consent_type="research",
            scope=["ml_training"],
            irb_protocol_id="IRB-2024-001",
        )

        assert manager.revoke_consent(consent.consent_id)

        valid, _ = manager.verify_consent(
            patient_id_hash="hash-patient-002",
            required_scope="ml_training",
        )
        assert not valid

    def test_retention_policy(self):
        """Retention manager tracks expiration correctly."""
        from src.compliance.data_governance import RetentionManager, RetentionPolicy

        manager = RetentionManager()
        manager.register_data(
            dataset_id="temp-data",
            data_hash="hash123",
            policy=RetentionPolicy.TEMPORARY,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        # Check after 91 days (past 90-day temporary retention)
        expired = manager.check_expired(
            current_time=datetime(2024, 4, 3, tzinfo=timezone.utc)
        )
        assert len(expired) == 1


# ---- Model Card Tests ---- #

class TestModelCard:
    """Tests for FDA-style model card generation."""

    def test_generate_default_card(self):
        """Default model card contains all required sections."""
        from src.compliance.model_card import generate_default_model_card

        card = generate_default_model_card()

        assert "intended_use" in card
        assert "training_data" in card
        assert "performance" in card
        assert "demographic_performance" in card
        assert "limitations_and_failure_modes" in card
        assert "document_integrity_hash" in card

    def test_intended_use_complete(self):
        """Intended use statement has all FDA-required fields."""
        from src.compliance.model_card import generate_default_model_card

        card = generate_default_model_card()
        iu = card["intended_use"]

        required_fields = [
            "device_name", "intended_use", "indications_for_use",
            "target_condition", "target_population", "intended_user",
            "clinical_setting", "device_type", "samd_category",
        ]
        for field in required_fields:
            assert field in iu, f"Missing required field: {field}"

    def test_performance_metrics_have_ci(self):
        """All performance metrics include confidence intervals."""
        from src.compliance.model_card import generate_default_model_card

        card = generate_default_model_card()
        metrics = card["performance"]["overall_metrics"]

        for metric in metrics:
            assert "confidence_interval" in metric
            ci = metric["confidence_interval"]
            assert "lower" in ci
            assert "upper" in ci
            assert ci["lower"] <= metric["value"] <= ci["upper"]

    def test_document_integrity_hash(self):
        """Model card has a valid integrity hash."""
        from src.compliance.model_card import generate_default_model_card

        card = generate_default_model_card()
        stored_hash = card.pop("document_integrity_hash")

        # Recompute hash without the hash field
        recomputed = hashlib.sha256(
            json.dumps(card, sort_keys=True).encode()
        ).hexdigest()

        assert stored_hash == recomputed


# ---- Change Control Tests ---- #

class TestChangeControl:
    """Tests for change control management."""

    def test_create_change_request(self):
        """Change requests can be created with auto-impact assessment."""
        from src.documentation.change_control import (
            ChangeControlManager,
            ChangeCategory,
            ChangeImpact,
        )

        manager = ChangeControlManager()
        cr = manager.create_change_request(
            title="Update training data",
            description="Add 5000 new mammography studies from Site F",
            category=ChangeCategory.TRAINING_DATA,
            justification="Improve representation of Hispanic population",
            affected_artifacts=["training_dataset_v2.1"],
        )

        assert cr.change_id.startswith("CR-")
        assert cr.impact == ChangeImpact.SIGNIFICANT
        assert len(cr.validation_requirements) > 0

    def test_approval_workflow(self):
        """Change requests follow the approval workflow correctly."""
        from src.documentation.change_control import (
            ChangeControlManager,
            ChangeCategory,
            ApprovalStatus,
        )

        manager = ChangeControlManager()
        cr = manager.create_change_request(
            title="Bug fix in preprocessing",
            description="Fix normalization for Siemens scanners",
            category=ChangeCategory.BUG_FIX,
            justification="Incorrect intensity normalization",
            affected_artifacts=["preprocessor"],
        )

        assert cr.status == ApprovalStatus.DRAFT
        assert manager.submit_for_review(cr.change_id)
        assert cr.status == ApprovalStatus.SUBMITTED

        assert manager.approve_change(cr.change_id, "reviewer-1")
        assert cr.status == ApprovalStatus.APPROVED

    def test_document_revision_tracking(self):
        """Document revisions are tracked with content hashes."""
        from src.documentation.change_control import ChangeControlManager

        manager = ChangeControlManager()
        rev = manager.record_document_revision(
            document_id="DOC-001",
            version="2.0",
            previous_version="1.0",
            change_summary="Updated performance thresholds",
            content="new document content here",
        )

        assert rev.content_hash is not None
        assert rev.version == "2.0"

        history = manager.get_document_history("DOC-001")
        assert len(history) == 1

    def test_impact_assessment(self):
        """Impact assessment correctly classifies change severity."""
        from src.documentation.change_control import (
            ChangeControlManager,
            ChangeCategory,
            ChangeImpact,
        )

        manager = ChangeControlManager()

        # Architecture change should be MAJOR
        cr = manager.create_change_request(
            title="New model architecture",
            description="Switch from ResNet-50 to ViT",
            category=ChangeCategory.MODEL_ARCHITECTURE,
            justification="Better performance",
            affected_artifacts=["model"],
        )
        assert cr.impact == ChangeImpact.MAJOR

        # Bug fix should be MINIMAL
        cr2 = manager.create_change_request(
            title="Fix logging bug",
            description="Fix timestamp format in logs",
            category=ChangeCategory.BUG_FIX,
            justification="Incorrect timestamps",
            affected_artifacts=["logger"],
        )
        assert cr2.impact == ChangeImpact.MINIMAL


# ---- Statistical Tests ---- #

class TestStatisticalTests:
    """Tests for statistical validation functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 1000
        self.y_true = np.random.binomial(1, 0.15, n).astype(np.int32)
        self.y_scores = np.clip(
            self.y_true * 0.7 + np.random.randn(n) * 0.2 + 0.1, 0, 1
        )

    def test_bootstrap_confidence_interval(self):
        """Bootstrap CI produces valid intervals."""
        from src.validation.statistical_tests import BootstrapConfidenceInterval

        ci_calc = BootstrapConfidenceInterval(n_bootstrap=500, random_seed=42)
        result = ci_calc.compute(self.y_true, self.y_scores, "auc_roc")

        assert result["lower"] < result["point_estimate"]
        assert result["upper"] > result["point_estimate"]
        assert result["lower"] > 0.5  # Should be well above chance

    def test_non_inferiority(self):
        """Non-inferiority test produces valid results."""
        from src.validation.statistical_tests import NonInferiorityTest

        ni_test = NonInferiorityTest(margin=0.05, alpha=0.05)
        result = ni_test.test_auc(
            y_true=self.y_true,
            y_scores=self.y_scores,
            predicate_auc=0.80,
            n_bootstrap=500,
        )

        assert "non_inferior" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        assert result["observed_auc"] > 0

    def test_subgroup_analysis(self):
        """Subgroup analysis produces per-group metrics."""
        from src.validation.statistical_tests import SubgroupAnalyzer

        analyzer = SubgroupAnalyzer(alpha=0.05, correction_method="bonferroni")
        groups = np.random.choice(["A", "B", "C"], len(self.y_true))

        result = analyzer.analyze(
            y_true=self.y_true,
            y_scores=self.y_scores,
            group_labels=groups,
            group_name="test_group",
        )

        assert result["n_subgroups"] == 3
        assert "subgroups" in result
        assert len(result["subgroups"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
