"""
audit.py — Append-only audit logger for compliance and regulatory requirements.

Records all actions taken by the agent and humans, ensuring immutable audit trail
for compliance with regulatory and audit requirements.
"""

from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger("sre_audit")
logger.setLevel(logging.INFO)


class AuditLogger:
    """Append-only JSONL audit trail logger."""

    def __init__(self, log_file: str = "audit.log") -> None:
        self.log_file = log_file
        logger.info(f"AuditLogger | Initialised at {log_file}")

    def log(
        self,
        node: str,
        action: str,
        actor: str,
        decision: str,
        incident_id: str,
        proposal_id: Optional[str] = None,
        confidence: float = 0.0,
        rationale: str = "",
        rollback_plan: str = "",
    ) -> str:
        """
        Append an immutable audit entry to the log.
        
        Args:
            node: Which graph node emitted this event
            action: Name of the action (e.g., 'scale_replicas')
            actor: 'agent' or 'human'
            decision: 'approved', 'rejected', 'escalated', 'skipped'
            incident_id: Associated incident ID
            proposal_id: Associated proposal ID
            confidence: RL confidence score
            rationale: Human-readable explanation
            rollback_plan: Documented rollback procedure
            
        Returns:
            Event ID for traceability
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        entry = {
            "event_id": event_id,
            "timestamp": timestamp,
            "node": node,
            "action": action,
            "actor": actor,
            "incident_id": incident_id,
            "proposal_id": proposal_id,
            "decision": decision,
            "confidence": round(confidence, 3),
            "rationale": rationale,
            "rollback_plan": rollback_plan,
        }

        # Append to JSONL (one JSON record per line, never overwritten)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"AuditLogger | {event_id} | {actor} {decision} {action}")
        return event_id

    def log_incident_detected(
        self,
        incident_id: str,
        severity: str,
        summary: str,
    ) -> str:
        """Log when an anomaly is detected."""
        return self.log(
            node="analyzer",
            action="detect_incident",
            actor="agent",
            decision="detected",
            incident_id=incident_id,
            rationale=f"Detected {severity} severity incident: {summary}",
        )

    def log_proposal_generated(
        self,
        incident_id: str,
        proposal_id: str,
        action: str,
        confidence: float,
        rationale: str,
    ) -> str:
        """Log when a remediation proposal is generated."""
        return self.log(
            node="proposer",
            action=action,
            actor="agent",
            decision="proposed",
            incident_id=incident_id,
            proposal_id=proposal_id,
            confidence=confidence,
            rationale=rationale,
        )

    def log_approval_decision(
        self,
        incident_id: str,
        proposal_id: str,
        action: str,
        approved: bool,
        rationale: str = "",
    ) -> str:
        """Log human approval or rejection decision."""
        decision = "approved" if approved else "rejected"
        return self.log(
            node="human_in_the_loop",
            action=action,
            actor="human",
            decision=decision,
            incident_id=incident_id,
            proposal_id=proposal_id,
            rationale=rationale or f"Human {decision} the proposal",
        )

    def log_action_execution(
        self,
        incident_id: str,
        proposal_id: str,
        action: str,
        params: dict,
        rollback_action: str,
        rollback_params: dict,
    ) -> str:
        """Log action execution with rollback information."""
        return self.log(
            node="human_in_the_loop",
            action=action,
            actor="agent",
            decision="executed",
            incident_id=incident_id,
            proposal_id=proposal_id,
            rationale=f"Executed action with params: {params}",
            rollback_plan=f"{rollback_action}({rollback_params})",
        )

    def log_escalation(
        self,
        incident_id: str,
        proposal_id: Optional[str],
        reason: str,
    ) -> str:
        """Log escalation to human operators or on-call."""
        return self.log(
            node="human_in_the_loop",
            action="escalate",
            actor="agent",
            decision="escalated",
            incident_id=incident_id,
            proposal_id=proposal_id,
            rationale=reason,
        )

    def read_audit_trail(self) -> list[dict]:
        """Read and parse the entire audit trail (for compliance reporting)."""
        entries = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except FileNotFoundError:
            logger.warning(f"AuditLogger | Audit file not found: {self.log_file}")
        return entries

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """Log an error occurrence for debugging and monitoring."""
        return self.log(
            node="error_handler",
            action="error_occurred",
            actor="system",
            decision="logged",
            incident_id="error-" + str(uuid.uuid4())[:8],
            rationale=f"{error_type}: {error_message}",
            rollback_plan=f"Context: {context or {}}"
        )


# Global audit logger instance
_audit_instance: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = AuditLogger()
    return _audit_instance
