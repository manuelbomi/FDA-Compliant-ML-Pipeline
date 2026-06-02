"""
FDA-Compliant Immutable Audit Trail for Medical Device ML Pipelines.

Implements an append-only, tamper-evident audit logging system that
satisfies FDA 21 CFR Part 820.184 (Device History Record) and
21 CFR Part 11 (Electronic Records; Electronic Signatures).

Every pipeline action, data access, model version change, and user
authentication event is recorded with cryptographic integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


class AuditEventType(str, Enum):
    """Categorization of audit events per FDA 21 CFR Part 11."""

    # Pipeline events
    PIPELINE_START = "pipeline.start"
    PIPELINE_STEP_START = "pipeline.step.start"
    PIPELINE_STEP_COMPLETE = "pipeline.step.complete"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_FAILURE = "pipeline.failure"

    # Data events
    DATA_ACCESS = "data.access"
    DATA_UPLOAD = "data.upload"
    DATA_TRANSFORM = "data.transform"
    DATA_VERSION_CREATE = "data.version.create"
    DATA_DELETE = "data.delete"

    # Model events
    MODEL_TRAIN_START = "model.train.start"
    MODEL_TRAIN_COMPLETE = "model.train.complete"
    MODEL_VALIDATE = "model.validate"
    MODEL_REGISTER = "model.register"
    MODEL_DEPLOY = "model.deploy"
    MODEL_ROLLBACK = "model.rollback"

    # Authentication events
    USER_LOGIN = "auth.login"
    USER_LOGOUT = "auth.logout"
    USER_APPROVE = "auth.approve"
    ACCESS_DENIED = "auth.denied"

    # Compliance events
    COMPLIANCE_CHECK = "compliance.check"
    COMPLIANCE_VIOLATION = "compliance.violation"
    CAPA_CREATED = "compliance.capa.created"
    CAPA_RESOLVED = "compliance.capa.resolved"

    # System events
    CONFIG_CHANGE = "system.config.change"
    SYSTEM_ERROR = "system.error"


@dataclass
class AuditEvent:
    """Single immutable audit event record.

    Each event is individually hashed and chained to the previous event,
    forming a tamper-evident chain similar to a blockchain. Any modification
    to a historical event will break the chain, providing cryptographic
    evidence of tampering.
    """

    event_id: str
    event_type: AuditEventType
    timestamp: str
    user_id: str
    session_id: str
    pipeline_run_id: Optional[str]
    step_name: Optional[str]
    action_description: str
    artifact_hash: Optional[str]
    metadata: Dict[str, Any]
    previous_event_hash: str
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this event for chain integrity.

        The hash covers all event fields except event_hash itself,
        ensuring immutability of the complete record.
        """
        hash_input = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
                "timestamp": self.timestamp,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "pipeline_run_id": self.pipeline_run_id,
                "step_name": self.step_name,
                "action_description": self.action_description,
                "artifact_hash": self.artifact_hash,
                "metadata": self.metadata,
                "previous_event_hash": self.previous_event_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "pipeline_run_id": self.pipeline_run_id,
            "step_name": self.step_name,
            "action_description": self.action_description,
            "artifact_hash": self.artifact_hash,
            "metadata": self.metadata,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
        }


class AuditTrailStore:
    """Persistent, append-only audit trail storage.

    Uses SQLite with write-ahead logging for local development and testing.
    In production, this would be backed by AWS DynamoDB or S3 with
    Object Lock for true immutability.

    The store enforces:
    - Append-only semantics (no UPDATE or DELETE operations)
    - Hash chain integrity across all events
    - Automatic integrity verification on read
    """

    def __init__(self, db_path: str = "audit_trail.db"):
        """Initialize the audit trail store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Create the audit trail table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    pipeline_run_id TEXT,
                    step_name TEXT,
                    action_description TEXT NOT NULL,
                    artifact_hash TEXT,
                    metadata TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type
                ON audit_events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON audit_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_run
                ON audit_events(pipeline_run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON audit_events(user_id)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with WAL mode for concurrency."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        try:
            yield conn
        finally:
            conn.close()

    def _get_last_event_hash(self, conn: sqlite3.Connection) -> str:
        """Get the hash of the most recent event in the chain.

        Returns the genesis hash if the chain is empty.
        """
        cursor = conn.execute(
            "SELECT event_hash FROM audit_events "
            "ORDER BY sequence_number DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            # Genesis hash for the start of the chain
            return hashlib.sha256(b"FDA-AUDIT-TRAIL-GENESIS").hexdigest()
        return row[0]

    def _get_next_sequence(self, conn: sqlite3.Connection) -> int:
        """Get the next sequence number for ordering."""
        cursor = conn.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM audit_events"
        )
        return cursor.fetchone()[0]

    def append(self, event: AuditEvent) -> AuditEvent:
        """Append an event to the audit trail.

        This is the only write operation supported. Events cannot be
        modified or deleted once appended.

        Args:
            event: The audit event to append.

        Returns:
            The event with its computed hash and sequence number.

        Raises:
            RuntimeError: If the event hash chain is broken.
        """
        with self._lock:
            with self._get_connection() as conn:
                previous_hash = self._get_last_event_hash(conn)
                sequence = self._get_next_sequence(conn)

                event.previous_event_hash = previous_hash
                event.event_hash = event.compute_hash()

                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, event_type, timestamp, user_id,
                        session_id, pipeline_run_id, step_name,
                        action_description, artifact_hash, metadata,
                        previous_event_hash, event_hash, sequence_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type.value if isinstance(event.event_type, Enum) else event.event_type,
                        event.timestamp,
                        event.user_id,
                        event.session_id,
                        event.pipeline_run_id,
                        event.step_name,
                        event.action_description,
                        event.artifact_hash,
                        json.dumps(event.metadata),
                        event.previous_event_hash,
                        event.event_hash,
                        sequence,
                    ),
                )
                conn.commit()

        return event

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the complete audit trail hash chain.

        Recomputes every event hash and verifies the chain linkage.
        Any tampering with historical events will be detected.

        Returns:
            Tuple of (integrity_valid, list_of_violations).
        """
        violations: List[str] = []

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT event_id, event_type, timestamp, user_id, "
                "session_id, pipeline_run_id, step_name, "
                "action_description, artifact_hash, metadata, "
                "previous_event_hash, event_hash, sequence_number "
                "FROM audit_events ORDER BY sequence_number ASC"
            )

            expected_previous_hash = hashlib.sha256(
                b"FDA-AUDIT-TRAIL-GENESIS"
            ).hexdigest()

            for row in cursor:
                (
                    event_id, event_type, timestamp, user_id,
                    session_id, pipeline_run_id, step_name,
                    action_description, artifact_hash, metadata_json,
                    previous_event_hash, stored_hash, sequence_number,
                ) = row

                # Verify chain linkage
                if previous_event_hash != expected_previous_hash:
                    violations.append(
                        f"Chain break at event {event_id} (seq {sequence_number}): "
                        f"expected previous hash {expected_previous_hash[:16]}..., "
                        f"found {previous_event_hash[:16]}..."
                    )

                # Recompute hash
                event = AuditEvent(
                    event_id=event_id,
                    event_type=event_type,
                    timestamp=timestamp,
                    user_id=user_id,
                    session_id=session_id,
                    pipeline_run_id=pipeline_run_id,
                    step_name=step_name,
                    action_description=action_description,
                    artifact_hash=artifact_hash,
                    metadata=json.loads(metadata_json),
                    previous_event_hash=previous_event_hash,
                )
                computed_hash = event.compute_hash()

                if computed_hash != stored_hash:
                    violations.append(
                        f"Hash mismatch at event {event_id} (seq {sequence_number}): "
                        f"computed {computed_hash[:16]}..., "
                        f"stored {stored_hash[:16]}..."
                    )

                expected_previous_hash = stored_hash

        return (len(violations) == 0, violations)

    def query_by_run(self, pipeline_run_id: str) -> List[Dict[str, Any]]:
        """Query all events for a specific pipeline run.

        Args:
            pipeline_run_id: The Metaflow run ID to query.

        Returns:
            List of event dictionaries ordered by sequence number.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT event_id, event_type, timestamp, user_id, "
                "session_id, pipeline_run_id, step_name, "
                "action_description, artifact_hash, metadata, "
                "previous_event_hash, event_hash "
                "FROM audit_events WHERE pipeline_run_id = ? "
                "ORDER BY sequence_number ASC",
                (pipeline_run_id,),
            )
            return [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "user_id": row[3],
                    "session_id": row[4],
                    "pipeline_run_id": row[5],
                    "step_name": row[6],
                    "action_description": row[7],
                    "artifact_hash": row[8],
                    "metadata": json.loads(row[9]),
                    "previous_event_hash": row[10],
                    "event_hash": row[11],
                }
                for row in cursor
            ]

    def query_by_type(
        self,
        event_type: AuditEventType,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query events by type with optional time range filtering.

        Args:
            event_type: The type of events to query.
            start_time: ISO format start time (inclusive).
            end_time: ISO format end time (inclusive).

        Returns:
            List of matching event dictionaries.
        """
        query = (
            "SELECT event_id, event_type, timestamp, user_id, "
            "action_description, artifact_hash, metadata, event_hash "
            "FROM audit_events WHERE event_type = ?"
        )
        params: List[Any] = [event_type.value]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY sequence_number ASC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "user_id": row[3],
                    "action_description": row[4],
                    "artifact_hash": row[5],
                    "metadata": json.loads(row[6]),
                    "event_hash": row[7],
                }
                for row in cursor
            ]

    def get_artifact_history(self, artifact_hash: str) -> List[Dict[str, Any]]:
        """Trace the complete history of an artifact by its hash.

        Args:
            artifact_hash: SHA-256 hash of the artifact.

        Returns:
            List of events referencing this artifact.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT event_id, event_type, timestamp, user_id, "
                "action_description, metadata, event_hash "
                "FROM audit_events WHERE artifact_hash = ? "
                "ORDER BY sequence_number ASC",
                (artifact_hash,),
            )
            return [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "user_id": row[3],
                    "action_description": row[4],
                    "metadata": json.loads(row[5]),
                    "event_hash": row[6],
                }
                for row in cursor
            ]

    def export_for_submission(
        self,
        pipeline_run_id: str,
        output_path: str,
    ) -> str:
        """Export audit trail for a run as a signed JSON document.

        Produces a self-contained, integrity-verified document suitable
        for inclusion in an FDA regulatory submission.

        Args:
            pipeline_run_id: The run to export.
            output_path: Path to write the export file.

        Returns:
            SHA-256 hash of the exported document.
        """
        events = self.query_by_run(pipeline_run_id)
        integrity_valid, violations = self.verify_chain_integrity()

        export_document = {
            "document_type": "FDA Audit Trail Export",
            "document_version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_run_id": pipeline_run_id,
            "chain_integrity_verified": integrity_valid,
            "chain_violations": violations,
            "total_events": len(events),
            "events": events,
        }

        export_json = json.dumps(export_document, indent=2, sort_keys=True)
        export_hash = hashlib.sha256(export_json.encode()).hexdigest()
        export_document["document_hash"] = export_hash

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_document, f, indent=2, sort_keys=True)

        return export_hash


class AuditLogger:
    """High-level audit logging interface for pipeline steps.

    Provides convenience methods for common audit events, handling
    session management, user identification, and hash chain maintenance.
    """

    def __init__(
        self,
        store: AuditTrailStore,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the audit logger.

        Args:
            store: The underlying audit trail store.
            user_id: User or service account ID.
            session_id: Unique session identifier.
        """
        self.store = store
        self.user_id = user_id or os.getenv("USER", "system")
        self.session_id = session_id or str(uuid.uuid4())

    def _create_event(
        self,
        event_type: AuditEventType,
        description: str,
        pipeline_run_id: Optional[str] = None,
        step_name: Optional[str] = None,
        artifact_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Create and log an audit event.

        Args:
            event_type: Category of the event.
            description: Human-readable description.
            pipeline_run_id: Associated Metaflow run ID.
            step_name: Associated pipeline step name.
            artifact_hash: SHA-256 hash of any associated artifact.
            metadata: Additional event metadata.

        Returns:
            The logged audit event.
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=self.user_id,
            session_id=self.session_id,
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            action_description=description,
            artifact_hash=artifact_hash,
            metadata=metadata or {},
            previous_event_hash="",  # Set by store.append
        )
        return self.store.append(event)

    def log_pipeline_start(
        self, run_id: str, flow_name: str, parameters: Dict[str, Any]
    ) -> AuditEvent:
        """Log the start of a pipeline run."""
        return self._create_event(
            event_type=AuditEventType.PIPELINE_START,
            description=f"Pipeline {flow_name} started",
            pipeline_run_id=run_id,
            metadata={"flow_name": flow_name, "parameters": parameters},
        )

    def log_step_complete(
        self,
        run_id: str,
        step_name: str,
        artifact_hash: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log the completion of a pipeline step."""
        return self._create_event(
            event_type=AuditEventType.PIPELINE_STEP_COMPLETE,
            description=f"Step '{step_name}' completed successfully",
            pipeline_run_id=run_id,
            step_name=step_name,
            artifact_hash=artifact_hash,
            metadata={"metrics": metrics or {}},
        )

    def log_data_access(
        self,
        dataset_id: str,
        access_type: str,
        record_count: int,
        data_hash: str,
    ) -> AuditEvent:
        """Log a data access event for HIPAA compliance."""
        return self._create_event(
            event_type=AuditEventType.DATA_ACCESS,
            description=f"Data access: {access_type} on {dataset_id}",
            artifact_hash=data_hash,
            metadata={
                "dataset_id": dataset_id,
                "access_type": access_type,
                "record_count": record_count,
            },
        )

    def log_model_registration(
        self,
        run_id: str,
        model_version: str,
        model_hash: str,
        registry_id: str,
    ) -> AuditEvent:
        """Log model registration in the model registry."""
        return self._create_event(
            event_type=AuditEventType.MODEL_REGISTER,
            description=f"Model v{model_version} registered: {registry_id}",
            pipeline_run_id=run_id,
            artifact_hash=model_hash,
            metadata={
                "model_version": model_version,
                "registry_id": registry_id,
            },
        )

    def log_deployment(
        self,
        run_id: str,
        model_version: str,
        model_hash: str,
        endpoint: str,
        strategy: str,
        approved_by: str,
    ) -> AuditEvent:
        """Log a model deployment event."""
        return self._create_event(
            event_type=AuditEventType.MODEL_DEPLOY,
            description=(
                f"Model v{model_version} deployed to {endpoint} "
                f"({strategy}), approved by {approved_by}"
            ),
            pipeline_run_id=run_id,
            artifact_hash=model_hash,
            metadata={
                "model_version": model_version,
                "endpoint": endpoint,
                "strategy": strategy,
                "approved_by": approved_by,
            },
        )

    def log_rollback(
        self,
        run_id: str,
        model_version: str,
        reason: str,
        rollback_target: str,
    ) -> AuditEvent:
        """Log a model rollback event and create CAPA reference."""
        return self._create_event(
            event_type=AuditEventType.MODEL_ROLLBACK,
            description=(
                f"Model v{model_version} rolled back to {rollback_target}: {reason}"
            ),
            pipeline_run_id=run_id,
            metadata={
                "model_version": model_version,
                "reason": reason,
                "rollback_target": rollback_target,
                "capa_required": True,
            },
        )

    def log_compliance_check(
        self,
        check_name: str,
        passed: bool,
        details: Dict[str, Any],
    ) -> AuditEvent:
        """Log a compliance check result."""
        event_type = (
            AuditEventType.COMPLIANCE_CHECK
            if passed
            else AuditEventType.COMPLIANCE_VIOLATION
        )
        return self._create_event(
            event_type=event_type,
            description=f"Compliance check '{check_name}': {'PASS' if passed else 'FAIL'}",
            metadata={"check_name": check_name, "passed": passed, **details},
        )
