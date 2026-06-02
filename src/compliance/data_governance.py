"""
Data Governance Module for FDA-Compliant ML Pipelines.

Implements data lineage tracking, HIPAA compliance checks (PHI detection
and scrubbing), data retention policies, access control enforcement,
and consent management -- all required for medical device AI/ML systems
operating under FDA and HIPAA regulations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class DataClassification(str, Enum):
    """Data classification levels per HIPAA and organizational policy."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PHI = "phi"
    RESTRICTED_PHI = "restricted_phi"


class AccessLevel(str, Enum):
    """Access control levels for data operations."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    AUDIT = "audit"


class RetentionPolicy(str, Enum):
    """Data retention policies aligned with FDA and HIPAA requirements."""

    FDA_DEVICE_RECORD = "fda_device_record"  # Life of device + 2 years
    HIPAA_MEDICAL_RECORD = "hipaa_medical_record"  # 6 years from last use
    TRAINING_DATA = "training_data"  # Per design control plan
    AUDIT_TRAIL = "audit_trail"  # Permanent
    TEMPORARY = "temporary"  # 90 days


@dataclass
class DataLineageNode:
    """Single node in the data lineage graph.

    Tracks the provenance of a dataset or artifact through its complete
    transformation history, as required for FDA traceability.
    """

    node_id: str
    dataset_id: str
    version: str
    operation: str
    input_nodes: List[str]
    output_hash: str
    created_at: str
    created_by: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    classification: DataClassification = DataClassification.INTERNAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "operation": self.operation,
            "input_nodes": self.input_nodes,
            "output_hash": self.output_hash,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "parameters": self.parameters,
            "quality_metrics": self.quality_metrics,
            "classification": self.classification.value,
        }


@dataclass
class ConsentRecord:
    """Patient consent record for data usage in ML training.

    Tracks informed consent per HIPAA Privacy Rule and Common Rule
    for research involving human subjects data.
    """

    consent_id: str
    patient_id_hash: str  # De-identified patient reference
    consent_type: str
    granted_at: str
    expires_at: Optional[str]
    scope: List[str]
    irb_protocol_id: str
    revoked: bool = False
    revoked_at: Optional[str] = None

    def is_valid(self, current_time: Optional[datetime] = None) -> bool:
        """Check if consent is currently valid."""
        if self.revoked:
            return False
        if self.expires_at is None:
            return True
        now = current_time or datetime.now(timezone.utc)
        expiry = datetime.fromisoformat(self.expires_at)
        return now < expiry


class PHIDetector:
    """Detect Protected Health Information (PHI) in datasets.

    Implements detection for the 18 HIPAA identifiers defined in
    the Safe Harbor de-identification standard (45 CFR 164.514(b)(2)).
    """

    # Regex patterns for common PHI elements
    PATTERNS = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(
            r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "mrn": re.compile(r"\bMRN[:\s]?\d{6,10}\b", re.IGNORECASE),
        "date_of_birth": re.compile(
            r"\b(?:DOB|Date of Birth)[:\s]?\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "zip_code": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        "ip_address": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "name_prefix": re.compile(
            r"\b(?:Patient|Name)[:\s]+[A-Z][a-z]+ [A-Z][a-z]+\b"
        ),
        "account_number": re.compile(
            r"\b(?:Account|Acct)[:\s#]?\d{8,12}\b", re.IGNORECASE
        ),
        "certificate_number": re.compile(
            r"\b(?:Certificate|License|DEA)[:\s#]?\w{2}\d{7}\b", re.IGNORECASE
        ),
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """Initialize with optional custom PHI patterns.

        Args:
            custom_patterns: Additional regex patterns to check.
        """
        self.patterns = dict(self.PATTERNS)
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.patterns[name] = re.compile(pattern)

    def scan_text(self, text: str) -> List[Dict[str, Any]]:
        """Scan text for PHI elements.

        Args:
            text: Text content to scan.

        Returns:
            List of detected PHI elements with type and location.
        """
        findings: List[Dict[str, Any]] = []
        for phi_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                findings.append({
                    "phi_type": phi_type,
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(),
                    "severity": "high" if phi_type in {"ssn", "mrn"} else "medium",
                })
        return findings

    def scan_dataframe_columns(
        self, column_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Scan column names for likely PHI fields.

        Args:
            column_names: List of dataset column names to check.

        Returns:
            List of columns flagged as potential PHI.
        """
        phi_column_keywords = {
            "name", "first_name", "last_name", "patient_name",
            "ssn", "social_security", "mrn", "medical_record",
            "dob", "date_of_birth", "birth_date", "birthday",
            "address", "street", "city", "zip", "zip_code", "postal",
            "phone", "telephone", "fax", "email",
            "insurance", "policy_number", "account",
            "ip_address", "device_id", "serial_number",
        }

        flagged = []
        for col_name in column_names:
            col_lower = col_name.lower().replace(" ", "_")
            for keyword in phi_column_keywords:
                if keyword in col_lower:
                    flagged.append({
                        "column_name": col_name,
                        "matched_keyword": keyword,
                        "recommendation": "Remove or de-identify before training",
                    })
                    break
        return flagged

    def scrub_text(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Remove PHI from text by replacing matches with redaction marker.

        Args:
            text: Text to scrub.
            replacement: Replacement string for detected PHI.

        Returns:
            Scrubbed text with PHI replaced.
        """
        scrubbed = text
        # Process in reverse order to maintain positions
        all_matches = []
        for phi_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                all_matches.append((match.start(), match.end(), phi_type))

        all_matches.sort(key=lambda x: x[0], reverse=True)
        for start, end, phi_type in all_matches:
            scrubbed = scrubbed[:start] + f"{replacement}" + scrubbed[end:]
        return scrubbed


class DataLineageTracker:
    """Track complete data lineage for FDA traceability.

    Maintains a directed acyclic graph (DAG) of data transformations,
    enabling full provenance tracking from raw data to model artifacts.
    """

    def __init__(self):
        """Initialize the lineage tracker."""
        self.nodes: Dict[str, DataLineageNode] = {}
        self.edges: List[Tuple[str, str]] = []

    def register_source(
        self,
        dataset_id: str,
        version: str,
        data_hash: str,
        classification: DataClassification,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataLineageNode:
        """Register a source dataset (root node in lineage graph).

        Args:
            dataset_id: Unique dataset identifier.
            version: Dataset version string.
            data_hash: SHA-256 hash of the dataset.
            classification: Data classification level.
            metadata: Additional metadata.

        Returns:
            The created lineage node.
        """
        node = DataLineageNode(
            node_id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            version=version,
            operation="source_ingestion",
            input_nodes=[],
            output_hash=data_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=os.getenv("USER", "system"),
            parameters=metadata or {},
            classification=classification,
        )
        self.nodes[node.node_id] = node
        return node

    def register_transformation(
        self,
        dataset_id: str,
        version: str,
        operation: str,
        input_node_ids: List[str],
        output_hash: str,
        parameters: Dict[str, Any],
        quality_metrics: Optional[Dict[str, float]] = None,
    ) -> DataLineageNode:
        """Register a data transformation step.

        Args:
            dataset_id: Output dataset identifier.
            version: Output dataset version.
            operation: Description of the transformation.
            input_node_ids: IDs of input lineage nodes.
            output_hash: SHA-256 hash of the output dataset.
            parameters: Transformation parameters.
            quality_metrics: Quality metrics after transformation.

        Returns:
            The created lineage node.
        """
        # Inherit highest classification from inputs
        input_classifications = [
            self.nodes[nid].classification
            for nid in input_node_ids
            if nid in self.nodes
        ]
        max_classification = max(
            input_classifications,
            key=lambda c: list(DataClassification).index(c),
            default=DataClassification.INTERNAL,
        )

        node = DataLineageNode(
            node_id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            version=version,
            operation=operation,
            input_nodes=input_node_ids,
            output_hash=output_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=os.getenv("USER", "system"),
            parameters=parameters,
            quality_metrics=quality_metrics or {},
            classification=max_classification,
        )
        self.nodes[node.node_id] = node

        for input_id in input_node_ids:
            self.edges.append((input_id, node.node_id))

        return node

    def get_full_lineage(self, node_id: str) -> List[DataLineageNode]:
        """Trace complete lineage of a node back to source datasets.

        Args:
            node_id: The node to trace.

        Returns:
            List of all ancestor nodes in topological order.
        """
        visited: Set[str] = set()
        lineage: List[DataLineageNode] = []

        def _traverse(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            node = self.nodes[nid]
            for parent_id in node.input_nodes:
                _traverse(parent_id)
            lineage.append(node)

        _traverse(node_id)
        return lineage

    def export_lineage_graph(self) -> Dict[str, Any]:
        """Export the lineage graph for visualization or submission.

        Returns:
            Dictionary representation of the lineage DAG.
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {"source": src, "target": tgt} for src, tgt in self.edges
            ],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }


class AccessControlManager:
    """Enforce data access controls for HIPAA compliance.

    Manages role-based access control (RBAC) for data operations,
    logging all access attempts for the audit trail.
    """

    def __init__(self):
        """Initialize access control with default roles."""
        self.role_permissions: Dict[str, Set[AccessLevel]] = {
            "data_scientist": {AccessLevel.READ},
            "ml_engineer": {AccessLevel.READ, AccessLevel.WRITE},
            "data_engineer": {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN},
            "regulatory_affairs": {AccessLevel.READ, AccessLevel.AUDIT},
            "auditor": {AccessLevel.AUDIT},
            "admin": {
                AccessLevel.READ,
                AccessLevel.WRITE,
                AccessLevel.ADMIN,
                AccessLevel.AUDIT,
            },
        }
        self.user_roles: Dict[str, List[str]] = {}
        self.access_log: List[Dict[str, Any]] = []

    def assign_role(self, user_id: str, role: str) -> None:
        """Assign a role to a user.

        Args:
            user_id: User identifier.
            role: Role name.

        Raises:
            ValueError: If the role is not recognized.
        """
        if role not in self.role_permissions:
            raise ValueError(f"Unknown role: {role}")
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)

    def check_access(
        self,
        user_id: str,
        required_level: AccessLevel,
        resource_id: str,
        classification: DataClassification,
    ) -> bool:
        """Check if a user has the required access level.

        Args:
            user_id: User requesting access.
            required_level: Required access level.
            resource_id: Identifier of the resource being accessed.
            classification: Data classification of the resource.

        Returns:
            True if access is granted, False otherwise.
        """
        user_permissions: Set[AccessLevel] = set()
        for role in self.user_roles.get(user_id, []):
            user_permissions.update(self.role_permissions.get(role, set()))

        granted = required_level in user_permissions

        # PHI requires additional checks
        if classification in {DataClassification.PHI, DataClassification.RESTRICTED_PHI}:
            has_phi_training = True  # Would check training records
            granted = granted and has_phi_training

        self.access_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "resource_id": resource_id,
            "required_level": required_level.value,
            "classification": classification.value,
            "granted": granted,
        })

        return granted


class RetentionManager:
    """Manage data retention policies per FDA and HIPAA requirements.

    Enforces minimum retention periods and handles secure disposal
    of expired data according to the applicable policy.
    """

    RETENTION_PERIODS = {
        RetentionPolicy.FDA_DEVICE_RECORD: timedelta(days=365 * 15),
        RetentionPolicy.HIPAA_MEDICAL_RECORD: timedelta(days=365 * 6),
        RetentionPolicy.TRAINING_DATA: timedelta(days=365 * 10),
        RetentionPolicy.AUDIT_TRAIL: None,  # Permanent
        RetentionPolicy.TEMPORARY: timedelta(days=90),
    }

    def __init__(self):
        """Initialize the retention manager."""
        self.retention_records: List[Dict[str, Any]] = []

    def register_data(
        self,
        dataset_id: str,
        data_hash: str,
        policy: RetentionPolicy,
        created_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Register a dataset with a retention policy.

        Args:
            dataset_id: Unique dataset identifier.
            data_hash: SHA-256 hash of the dataset.
            policy: Applicable retention policy.
            created_at: Dataset creation timestamp.

        Returns:
            Retention record with computed expiry date.
        """
        creation = created_at or datetime.now(timezone.utc)
        period = self.RETENTION_PERIODS[policy]
        expiry = None if period is None else (creation + period).isoformat()

        record = {
            "dataset_id": dataset_id,
            "data_hash": data_hash,
            "policy": policy.value,
            "created_at": creation.isoformat(),
            "expires_at": expiry,
            "status": "active",
        }
        self.retention_records.append(record)
        return record

    def check_expired(
        self, current_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Check for datasets that have exceeded their retention period.

        Args:
            current_time: Time to check against (defaults to now).

        Returns:
            List of expired retention records.
        """
        now = current_time or datetime.now(timezone.utc)
        expired = []
        for record in self.retention_records:
            if record["expires_at"] is None:
                continue
            expiry = datetime.fromisoformat(record["expires_at"])
            if now >= expiry and record["status"] == "active":
                expired.append(record)
        return expired


class ConsentManager:
    """Manage patient consent records for data usage.

    Tracks informed consent for use of patient data in ML model
    training, as required by HIPAA Privacy Rule and institutional
    review board (IRB) protocols.
    """

    def __init__(self):
        """Initialize the consent manager."""
        self.consents: Dict[str, ConsentRecord] = {}

    def record_consent(
        self,
        patient_id_hash: str,
        consent_type: str,
        scope: List[str],
        irb_protocol_id: str,
        duration_days: Optional[int] = None,
    ) -> ConsentRecord:
        """Record a new consent grant.

        Args:
            patient_id_hash: De-identified patient reference.
            consent_type: Type of consent (e.g., "research", "training").
            scope: List of permitted data uses.
            irb_protocol_id: IRB protocol identifier.
            duration_days: Consent validity period in days.

        Returns:
            The created consent record.
        """
        now = datetime.now(timezone.utc)
        consent = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            patient_id_hash=patient_id_hash,
            consent_type=consent_type,
            granted_at=now.isoformat(),
            expires_at=(
                (now + timedelta(days=duration_days)).isoformat()
                if duration_days
                else None
            ),
            scope=scope,
            irb_protocol_id=irb_protocol_id,
        )
        self.consents[consent.consent_id] = consent
        return consent

    def revoke_consent(self, consent_id: str) -> bool:
        """Revoke a previously granted consent.

        Args:
            consent_id: ID of the consent to revoke.

        Returns:
            True if revocation was successful, False if consent not found.
        """
        if consent_id not in self.consents:
            return False
        consent = self.consents[consent_id]
        consent.revoked = True
        consent.revoked_at = datetime.now(timezone.utc).isoformat()
        return True

    def verify_consent(
        self,
        patient_id_hash: str,
        required_scope: str,
    ) -> Tuple[bool, Optional[ConsentRecord]]:
        """Verify that valid consent exists for a specific data use.

        Args:
            patient_id_hash: De-identified patient reference.
            required_scope: The data use to verify consent for.

        Returns:
            Tuple of (consent_valid, matching_consent_record).
        """
        for consent in self.consents.values():
            if (
                consent.patient_id_hash == patient_id_hash
                and required_scope in consent.scope
                and consent.is_valid()
            ):
                return True, consent
        return False, None

    def get_consent_summary(self) -> Dict[str, Any]:
        """Generate summary of consent status for regulatory reporting.

        Returns:
            Summary dictionary with consent statistics.
        """
        total = len(self.consents)
        active = sum(1 for c in self.consents.values() if c.is_valid())
        revoked = sum(1 for c in self.consents.values() if c.revoked)
        expired = total - active - revoked

        return {
            "total_consents": total,
            "active_consents": active,
            "revoked_consents": revoked,
            "expired_consents": expired,
            "consent_rate": active / total if total > 0 else 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
