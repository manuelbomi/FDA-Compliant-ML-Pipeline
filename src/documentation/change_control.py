"""
Change Control Management for FDA-Compliant ML Pipelines.

Implements document and model version control, impact assessment,
approval workflow tracking, and revision history as required by
FDA 21 CFR 820.40 (Document Controls) and the Predetermined Change
Control Plan framework for AI/ML-based SaMD.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ChangeCategory(str, Enum):
    """Categories of changes per FDA SaMD Change Control Plan."""

    MODEL_ARCHITECTURE = "model_architecture"
    TRAINING_DATA = "training_data"
    TRAINING_PROCEDURE = "training_procedure"
    PREPROCESSING = "preprocessing"
    OPERATING_POINT = "operating_point"
    INTENDED_USE = "intended_use"
    DEPLOYMENT_CONFIG = "deployment_config"
    BUG_FIX = "bug_fix"
    DOCUMENTATION = "documentation"


class ChangeImpact(str, Enum):
    """Impact level of a change per risk classification."""

    MINIMAL = "minimal"  # No revalidation needed
    MODERATE = "moderate"  # Targeted revalidation
    SIGNIFICANT = "significant"  # Full revalidation
    MAJOR = "major"  # New submission required


class ApprovalStatus(str, Enum):
    """Status of a change request in the approval workflow."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    CLOSED = "closed"


@dataclass
class ChangeRequest:
    """Formal change request record for regulatory traceability."""

    change_id: str
    title: str
    description: str
    category: ChangeCategory
    impact: ChangeImpact
    justification: str
    requested_by: str
    requested_at: str
    status: ApprovalStatus = ApprovalStatus.DRAFT
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    affected_artifacts: List[str] = field(default_factory=list)
    validation_requirements: List[str] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    implementation_notes: str = ""
    verification_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "impact": self.impact.value,
            "justification": self.justification,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "status": self.status.value,
            "risk_assessment": self.risk_assessment,
            "affected_artifacts": self.affected_artifacts,
            "validation_requirements": self.validation_requirements,
            "approvals": self.approvals,
            "implementation_notes": self.implementation_notes,
            "verification_evidence": self.verification_evidence,
        }


@dataclass
class DocumentRevision:
    """Single revision in a document's history."""

    revision_id: str
    document_id: str
    version: str
    previous_version: Optional[str]
    author: str
    timestamp: str
    change_summary: str
    content_hash: str
    change_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "document_id": self.document_id,
            "version": self.version,
            "previous_version": self.previous_version,
            "author": self.author,
            "timestamp": self.timestamp,
            "change_summary": self.change_summary,
            "content_hash": self.content_hash,
            "change_request_id": self.change_request_id,
        }


class ChangeControlManager:
    """Manage change control for FDA-compliant ML pipelines.

    Tracks all changes to the device software, data, and configuration,
    enforcing approval workflows and impact assessment requirements.
    """

    def __init__(self):
        """Initialize the change control manager."""
        self.change_requests: Dict[str, ChangeRequest] = {}
        self.document_revisions: Dict[str, List[DocumentRevision]] = {}

    def create_change_request(
        self,
        title: str,
        description: str,
        category: ChangeCategory,
        justification: str,
        affected_artifacts: List[str],
        requested_by: Optional[str] = None,
    ) -> ChangeRequest:
        """Create a new change request.

        Automatically performs impact assessment based on the change
        category and affected artifacts.

        Args:
            title: Brief title of the change.
            description: Detailed description.
            category: Category of the change.
            justification: Rationale for the change.
            affected_artifacts: List of affected artifact identifiers.
            requested_by: Person requesting the change.

        Returns:
            The created change request.
        """
        change_id = f"CR-{str(uuid.uuid4())[:8].upper()}"
        impact = self._assess_impact(category, affected_artifacts)
        validation_reqs = self._determine_validation_requirements(category, impact)
        risk_assessment = self._perform_risk_assessment(category, impact)

        cr = ChangeRequest(
            change_id=change_id,
            title=title,
            description=description,
            category=category,
            impact=impact,
            justification=justification,
            requested_by=requested_by or os.getenv("USER", "unknown"),
            requested_at=datetime.now(timezone.utc).isoformat(),
            risk_assessment=risk_assessment,
            affected_artifacts=affected_artifacts,
            validation_requirements=validation_reqs,
        )

        self.change_requests[change_id] = cr
        return cr

    def submit_for_review(self, change_id: str) -> bool:
        """Submit a change request for review.

        Args:
            change_id: ID of the change request.

        Returns:
            True if successfully submitted.
        """
        cr = self.change_requests.get(change_id)
        if cr is None or cr.status != ApprovalStatus.DRAFT:
            return False
        cr.status = ApprovalStatus.SUBMITTED
        return True

    def approve_change(
        self,
        change_id: str,
        approver: str,
        comments: str = "",
    ) -> bool:
        """Record an approval for a change request.

        For significant and major changes, multiple approvals may be
        required per the organization's quality system.

        Args:
            change_id: ID of the change request.
            approver: Person approving the change.
            comments: Approval comments.

        Returns:
            True if approval was recorded.
        """
        cr = self.change_requests.get(change_id)
        if cr is None or cr.status not in {
            ApprovalStatus.SUBMITTED,
            ApprovalStatus.UNDER_REVIEW,
        }:
            return False

        cr.approvals.append({
            "approver": approver,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "approved",
            "comments": comments,
        })

        # Determine if sufficient approvals obtained
        required_approvals = self._get_required_approvals(cr.impact)
        if len(cr.approvals) >= required_approvals:
            cr.status = ApprovalStatus.APPROVED
        else:
            cr.status = ApprovalStatus.UNDER_REVIEW

        return True

    def reject_change(
        self,
        change_id: str,
        reviewer: str,
        reason: str,
    ) -> bool:
        """Reject a change request.

        Args:
            change_id: ID of the change request.
            reviewer: Person rejecting the change.
            reason: Reason for rejection.

        Returns:
            True if rejection was recorded.
        """
        cr = self.change_requests.get(change_id)
        if cr is None:
            return False

        cr.status = ApprovalStatus.REJECTED
        cr.approvals.append({
            "approver": reviewer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "rejected",
            "comments": reason,
        })
        return True

    def record_implementation(
        self,
        change_id: str,
        notes: str,
        verification_evidence: Dict[str, Any],
    ) -> bool:
        """Record that a change has been implemented and verified.

        Args:
            change_id: ID of the change request.
            notes: Implementation notes.
            verification_evidence: Evidence of verification.

        Returns:
            True if recorded successfully.
        """
        cr = self.change_requests.get(change_id)
        if cr is None or cr.status != ApprovalStatus.APPROVED:
            return False

        cr.status = ApprovalStatus.IMPLEMENTED
        cr.implementation_notes = notes
        cr.verification_evidence = verification_evidence
        return True

    def verify_and_close(self, change_id: str, verifier: str) -> bool:
        """Verify implementation and close the change request.

        Args:
            change_id: ID of the change request.
            verifier: Person verifying the implementation.

        Returns:
            True if successfully closed.
        """
        cr = self.change_requests.get(change_id)
        if cr is None or cr.status != ApprovalStatus.IMPLEMENTED:
            return False

        cr.status = ApprovalStatus.CLOSED
        cr.approvals.append({
            "approver": verifier,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "verified_and_closed",
            "comments": "Implementation verified per validation requirements",
        })
        return True

    def record_document_revision(
        self,
        document_id: str,
        version: str,
        previous_version: Optional[str],
        change_summary: str,
        content: str,
        change_request_id: Optional[str] = None,
    ) -> DocumentRevision:
        """Record a new revision of a controlled document.

        Args:
            document_id: Unique document identifier.
            version: New version number.
            previous_version: Previous version number.
            change_summary: Summary of changes.
            content: Document content for hashing.
            change_request_id: Associated change request ID.

        Returns:
            The created document revision record.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        revision = DocumentRevision(
            revision_id=str(uuid.uuid4()),
            document_id=document_id,
            version=version,
            previous_version=previous_version,
            author=os.getenv("USER", "system"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            change_summary=change_summary,
            content_hash=content_hash,
            change_request_id=change_request_id,
        )

        if document_id not in self.document_revisions:
            self.document_revisions[document_id] = []
        self.document_revisions[document_id].append(revision)

        return revision

    def get_document_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get the complete revision history of a document.

        Args:
            document_id: Document to query.

        Returns:
            List of revision records in chronological order.
        """
        revisions = self.document_revisions.get(document_id, [])
        return [r.to_dict() for r in revisions]

    def compare_versions(
        self,
        change_id: str,
    ) -> Dict[str, Any]:
        """Compare the before and after state of a change.

        Args:
            change_id: Change request to compare.

        Returns:
            Dictionary with version comparison details.
        """
        cr = self.change_requests.get(change_id)
        if cr is None:
            return {"error": "Change request not found"}

        return {
            "change_id": change_id,
            "title": cr.title,
            "category": cr.category.value,
            "impact": cr.impact.value,
            "status": cr.status.value,
            "affected_artifacts": cr.affected_artifacts,
            "validation_requirements": cr.validation_requirements,
            "approvals": cr.approvals,
            "risk_assessment": cr.risk_assessment,
        }

    def generate_change_report(self) -> Dict[str, Any]:
        """Generate a summary report of all change requests.

        Returns:
            Summary report suitable for regulatory review.
        """
        total = len(self.change_requests)
        by_status = {}
        by_category = {}
        by_impact = {}

        for cr in self.change_requests.values():
            by_status[cr.status.value] = by_status.get(cr.status.value, 0) + 1
            by_category[cr.category.value] = by_category.get(cr.category.value, 0) + 1
            by_impact[cr.impact.value] = by_impact.get(cr.impact.value, 0) + 1

        return {
            "report_type": "Change Control Summary",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_change_requests": total,
            "by_status": by_status,
            "by_category": by_category,
            "by_impact": by_impact,
            "change_requests": [
                cr.to_dict() for cr in self.change_requests.values()
            ],
        }

    def _assess_impact(
        self,
        category: ChangeCategory,
        affected_artifacts: List[str],
    ) -> ChangeImpact:
        """Assess the impact level of a change.

        Impact assessment follows FDA's Predetermined Change Control Plan
        framework, categorizing changes by their potential to affect
        device safety and effectiveness.
        """
        high_impact_categories = {
            ChangeCategory.MODEL_ARCHITECTURE,
            ChangeCategory.INTENDED_USE,
        }
        moderate_categories = {
            ChangeCategory.TRAINING_DATA,
            ChangeCategory.TRAINING_PROCEDURE,
            ChangeCategory.OPERATING_POINT,
        }

        if category in high_impact_categories:
            return ChangeImpact.MAJOR
        elif category in moderate_categories:
            return ChangeImpact.SIGNIFICANT
        elif category == ChangeCategory.PREPROCESSING:
            return ChangeImpact.MODERATE
        else:
            return ChangeImpact.MINIMAL

    def _determine_validation_requirements(
        self,
        category: ChangeCategory,
        impact: ChangeImpact,
    ) -> List[str]:
        """Determine required validation activities for a change."""
        requirements = []

        if impact in {ChangeImpact.SIGNIFICANT, ChangeImpact.MAJOR}:
            requirements.extend([
                "Full validation study on independent test set",
                "Non-inferiority testing against predicate",
                "Subgroup analysis across all demographics",
                "Bias assessment",
                "Regression testing against previous version",
            ])

        if impact == ChangeImpact.MODERATE:
            requirements.extend([
                "Targeted validation on affected subgroups",
                "Regression testing on overall performance",
            ])

        if impact == ChangeImpact.MINIMAL:
            requirements.extend([
                "Unit testing of changed components",
                "Smoke test of end-to-end pipeline",
            ])

        if category == ChangeCategory.TRAINING_DATA:
            requirements.append("Data quality assessment of new data")
            requirements.append("Distribution comparison with previous dataset")

        return requirements

    def _perform_risk_assessment(
        self,
        category: ChangeCategory,
        impact: ChangeImpact,
    ) -> Dict[str, Any]:
        """Perform automated risk assessment for a change."""
        risk_matrix = {
            ChangeImpact.MINIMAL: {"risk_level": "low", "review_required": False},
            ChangeImpact.MODERATE: {"risk_level": "medium", "review_required": True},
            ChangeImpact.SIGNIFICANT: {"risk_level": "high", "review_required": True},
            ChangeImpact.MAJOR: {"risk_level": "critical", "review_required": True},
        }

        assessment = risk_matrix.get(
            impact, {"risk_level": "unknown", "review_required": True}
        )

        return {
            "risk_level": assessment["risk_level"],
            "review_required": assessment["review_required"],
            "regulatory_notification_required": impact == ChangeImpact.MAJOR,
            "potential_hazards": self._identify_hazards(category),
        }

    def _identify_hazards(self, category: ChangeCategory) -> List[str]:
        """Identify potential hazards associated with a change category."""
        hazard_map = {
            ChangeCategory.MODEL_ARCHITECTURE: [
                "Change in diagnostic accuracy",
                "New failure modes not covered by existing validation",
                "Altered latency characteristics",
            ],
            ChangeCategory.TRAINING_DATA: [
                "Distribution shift from previous data",
                "Potential new biases from data source",
                "Change in effective sample size",
            ],
            ChangeCategory.TRAINING_PROCEDURE: [
                "Model convergence changes",
                "Hyperparameter sensitivity",
                "Reproducibility of results",
            ],
            ChangeCategory.OPERATING_POINT: [
                "Sensitivity/specificity trade-off change",
                "Impact on clinical workflow",
            ],
            ChangeCategory.INTENDED_USE: [
                "New patient populations not previously validated",
                "New clinical settings with different requirements",
                "Regulatory classification may change",
            ],
        }
        return hazard_map.get(category, ["General change risk"])

    def _get_required_approvals(self, impact: ChangeImpact) -> int:
        """Get the number of required approvals based on impact level."""
        approval_map = {
            ChangeImpact.MINIMAL: 1,
            ChangeImpact.MODERATE: 2,
            ChangeImpact.SIGNIFICANT: 3,
            ChangeImpact.MAJOR: 4,
        }
        return approval_map.get(impact, 2)
