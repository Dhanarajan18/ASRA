"""
persistence.py — SQLite-backed persistence layer for the Autonomous SRE Agent.

Manages storage of incidents, proposals, RL experiences, and episode metrics.
Provides query interfaces for dashboard and analytics.
"""

from __future__ import annotations

import sqlite3
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np

from autonomous_sre.core.state import IncidentState, RemediationProposal

logger = logging.getLogger("sre_persistence")
logger.setLevel(logging.INFO)


def _utc_now_iso() -> str:
    """Return an RFC3339 UTC timestamp with explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SREDatabase:
    """SQLite-backed database for SRE agent operational data."""

    def __init__(self, db_path: str = "sre_agent.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()
        logger.info(f"SREDatabase | Initialised at {db_path}")

    def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        cursor = self.conn.cursor()

        # Incidents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                service TEXT NOT NULL,
                severity TEXT NOT NULL,
                anomaly_summary TEXT NOT NULL,
                rag_context TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Proposals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                action TEXT NOT NULL,
                action_params TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                risk_rationale TEXT NOT NULL,
                rollback_action TEXT NOT NULL,
                rollback_params TEXT NOT NULL,
                human_approved INTEGER,
                reward REAL,
                executed_at TEXT,
                FOREIGN KEY (incident_id) REFERENCES incidents(id)
            )
        """)

        # RL experiences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_vec TEXT NOT NULL,
                action TEXT NOT NULL,
                reward REAL NOT NULL,
                next_state_vec TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        # Episode metrics table (for learning curve visualization)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_metrics (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action_chosen TEXT NOT NULL,
                confidence REAL NOT NULL,
                reward REAL NOT NULL,
                was_correct INTEGER
            )
        """)

        # Escalations table (tracks manual handoff work)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                proposal_id TEXT,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                assignee TEXT,
                assigned_by TEXT,
                workstream TEXT,
                resolution_summary TEXT,
                resolution_outcome TEXT,
                created_at TEXT NOT NULL,
                assigned_at TEXT,
                resolved_at TEXT,
                FOREIGN KEY (incident_id) REFERENCES incidents(id),
                FOREIGN KEY (proposal_id) REFERENCES proposals(id)
            )
        """)

        # Escalation activity log table (who did what and when)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS escalation_logs (
                id TEXT PRIMARY KEY,
                escalation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                author TEXT NOT NULL,
                phase TEXT NOT NULL,
                note TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (escalation_id) REFERENCES escalations(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalations_status_created
            ON escalations(status, created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalation_logs_escalation
            ON escalation_logs(escalation_id, timestamp DESC)
        """)

        self.conn.commit()
        logger.info("SREDatabase | Tables created/verified")

    def save_incident(self, incident: IncidentState, service: str) -> str:
        """
        Save an incident to the database.
        
        Args:
            incident: IncidentState object
            service: Service name
            
        Returns:
            Generated incident ID
        """
        incident_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        # Serialize RAG context as JSON
        rag_context_json = json.dumps(incident.rag_context)

        cursor.execute("""
            INSERT INTO incidents (id, timestamp, service, severity, anomaly_summary, rag_context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id,
            _utc_now_iso(),
            service,
            incident.severity.value,
            incident.anomaly_summary,
            rag_context_json,
            _utc_now_iso(),
        ))

        self.conn.commit()
        logger.info(f"SREDatabase | Saved incident {incident_id}")
        return incident_id

    def save_proposal(
        self,
        proposal: RemediationProposal,
        incident_id: str,
        approved: Optional[bool] = None,
        reward: float = 0.0,
    ) -> str:
        """
        Save a remediation proposal to the database.
        
        Args:
            proposal: RemediationProposal object
            incident_id: Associated incident ID
            approved: Whether human approved (None = pending)
            reward: RL reward value
            
        Returns:
            Generated proposal ID
        """
        proposal_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO proposals 
            (id, incident_id, action, action_params, confidence_score, risk_rationale, 
             rollback_action, rollback_params, human_approved, reward, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            proposal_id,
            incident_id,
            proposal.action,
            json.dumps(proposal.action_params),
            proposal.confidence_score,
            proposal.risk_rationale,
            proposal.rollback_action,
            json.dumps(proposal.rollback_params),
            1 if approved else (0 if approved is False else None),
            reward,
            _utc_now_iso() if approved is not None else None,
        ))

        self.conn.commit()
        logger.info(f"SREDatabase | Saved proposal {proposal_id} for incident {incident_id}")
        return proposal_id

    def update_proposal_approval(
        self,
        proposal_id: str,
        approved: bool,
        reward: float = 0.0,
    ) -> None:
        """Update a proposal with approval decision and reward."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE proposals
            SET human_approved = ?, reward = ?, executed_at = ?
            WHERE id = ?
        """, (1 if approved else 0, reward, _utc_now_iso(), proposal_id))
        self.conn.commit()
        logger.info(f"SREDatabase | Updated proposal {proposal_id}: approved={approved}, reward={reward:.3f}")

    def save_experience(
        self,
        state_vec: np.ndarray,
        action: str,
        reward: float,
        next_state_vec: np.ndarray,
    ) -> None:
        """Save an RL experience to replay buffer."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO rl_experiences (state_vec, action, reward, next_state_vec, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            json.dumps(state_vec.tolist()),
            action,
            reward,
            json.dumps(next_state_vec.tolist()),
            _utc_now_iso(),
        ))
        self.conn.commit()

    def save_episode_metric(
        self,
        action_chosen: str,
        confidence: float,
        reward: float,
        was_correct: bool = False,
    ) -> None:
        """Save episode-level metrics for learning curve visualization."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episode_metrics (timestamp, action_chosen, confidence, reward, was_correct)
            VALUES (?, ?, ?, ?, ?)
        """, (
            _utc_now_iso(),
            action_chosen,
            confidence,
            reward,
            1 if was_correct else 0,
        ))
        self.conn.commit()

    def get_recent_incidents(self, n: int = 20) -> list[dict]:
        """Get the n most recent incidents."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, service, severity, anomaly_summary, rag_context, created_at
            FROM incidents
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_incident_by_id(self, incident_id: str) -> dict | None:
        """Get one incident by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, service, severity, anomaly_summary, rag_context, created_at
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_proposals(self, incident_id: str) -> list[dict]:
        """Get all proposals for a specific incident."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, incident_id, action, action_params, confidence_score, risk_rationale,
                   rollback_action, rollback_params, human_approved, reward, executed_at
            FROM proposals
            WHERE incident_id = ?
            ORDER BY executed_at DESC
        """, (incident_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_proposal_by_id(self, proposal_id: str) -> dict | None:
        """Get a proposal by proposal ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, incident_id, action, action_params, confidence_score, risk_rationale,
                   rollback_action, rollback_params, human_approved, reward, executed_at
            FROM proposals
            WHERE id = ?
            """,
            (proposal_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_proposal_for_incident(self, incident_id: str) -> dict | None:
        """Get the latest proposal for a given incident."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, incident_id, action, action_params, confidence_score, risk_rationale,
                   rollback_action, rollback_params, human_approved, reward, executed_at
            FROM proposals
            WHERE incident_id = ?
            ORDER BY COALESCE(executed_at, '') DESC, id DESC
            LIMIT 1
            """,
            (incident_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_metrics_summary(self) -> dict:
        """Return aggregate metrics for dashboard."""
        cursor = self.conn.cursor()

        # Total incidents today
        cursor.execute("""
            SELECT COUNT(*) as total FROM incidents
            WHERE date(created_at) = date('now')
        """)
        total_incidents = cursor.fetchone()["total"]

        # Auto-resolved (approved proposals) today
        cursor.execute("""
            SELECT COUNT(*) as resolved FROM proposals
            WHERE human_approved = 1 AND date(executed_at) = date('now')
        """)
        auto_resolved = cursor.fetchone()["resolved"]

        # Escalated (rejected or low confidence today)
        cursor.execute("""
            SELECT COUNT(*) as escalated FROM proposals
            WHERE (human_approved = 0 OR human_approved IS NULL)
              AND date(executed_at) = date('now')
        """)
        escalated = cursor.fetchone()["escalated"]

        # Average confidence
        cursor.execute("""
            SELECT AVG(confidence_score) as avg_conf FROM proposals
            WHERE date(executed_at) = date('now')
        """)
        avg_confidence = cursor.fetchone()["avg_conf"] or 0.0

        # Top actions
        cursor.execute("""
            SELECT action, COUNT(*) as count FROM proposals
            WHERE date(executed_at) = date('now')
            GROUP BY action
            ORDER BY count DESC
            LIMIT 5
        """)
        top_actions_rows = cursor.fetchall()
        top_actions = [row["action"] for row in top_actions_rows]

        return {
            "total_incidents": total_incidents,
            "auto_resolved": auto_resolved,
            "escalated": escalated,
            "avg_confidence": float(avg_confidence),
            "top_actions": top_actions,
        }

    def get_episode_metrics(self, limit: int = 500) -> list[dict]:
        """Get episode metrics for learning curve visualization."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT episode_id, timestamp, action_chosen, confidence, reward, was_correct
            FROM episode_metrics
            ORDER BY episode_id ASC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def create_escalation(
        self,
        incident_id: str,
        reason: str,
        proposal_id: Optional[str] = None,
    ) -> str:
        """Create an escalation record and return its ID."""
        escalation_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO escalations (
                id, incident_id, proposal_id, status, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                escalation_id,
                incident_id,
                proposal_id,
                "open",
                reason,
                _utc_now_iso(),
            ),
        )
        self.conn.commit()
        return escalation_id

    def assign_escalation(
        self,
        escalation_id: str,
        assignee: str,
        assigned_by: str,
        workstream: Optional[str] = None,
    ) -> bool:
        """Assign an escalation to a team member and mark as in_progress."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE escalations
            SET assignee = ?, assigned_by = ?, workstream = ?, status = ?, assigned_at = ?
            WHERE id = ?
            """,
            (
                assignee,
                assigned_by,
                workstream,
                "in_progress",
                _utc_now_iso(),
                escalation_id,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def resolve_escalation(
        self,
        escalation_id: str,
        resolved_by: str,
        resolution_summary: str,
        outcome: str = "resolved",
    ) -> bool:
        """Resolve an escalation and store the final fix summary."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE escalations
            SET status = ?, resolution_summary = ?, resolution_outcome = ?, resolved_at = ?
            WHERE id = ?
            """,
            (
                "resolved",
                f"{resolved_by}: {resolution_summary}",
                outcome,
                _utc_now_iso(),
                escalation_id,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_escalation_log(
        self,
        escalation_id: str,
        author: str,
        note: str,
        phase: str = "update",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Append an operator worklog entry to an escalation."""
        log_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO escalation_logs (
                id, escalation_id, timestamp, author, phase, note, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                escalation_id,
                _utc_now_iso(),
                author,
                phase,
                note,
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()
        return log_id

    def get_escalations(self, status: Optional[str] = None, limit: int = 100) -> list[dict]:
        """List escalations, optionally filtered by status."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute(
                """
                SELECT id, incident_id, proposal_id, status, reason, assignee, assigned_by,
                       workstream, resolution_summary, resolution_outcome,
                       created_at, assigned_at, resolved_at
                FROM escalations
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, incident_id, proposal_id, status, reason, assignee, assigned_by,
                       workstream, resolution_summary, resolution_outcome,
                       created_at, assigned_at, resolved_at
                FROM escalations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_escalation_by_id(self, escalation_id: str) -> dict | None:
        """Fetch one escalation by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, incident_id, proposal_id, status, reason, assignee, assigned_by,
                   workstream, resolution_summary, resolution_outcome,
                   created_at, assigned_at, resolved_at
            FROM escalations
            WHERE id = ?
            """,
            (escalation_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_escalation_logs(self, escalation_id: str, limit: int = 200) -> list[dict]:
        """Fetch worklog entries for an escalation."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, escalation_id, timestamp, author, phase, note, metadata
            FROM escalation_logs
            WHERE escalation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (escalation_id, limit),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("SREDatabase | Connection closed")


# Global database instance (to be initialized by main.py)
_db_instance: Optional[SREDatabase] = None


def get_db() -> SREDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SREDatabase()
    return _db_instance


def close_db() -> None:
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
