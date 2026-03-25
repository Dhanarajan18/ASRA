"""
state.py — Core data models and shared state for the Autonomous SRE Agent.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Standard incident severity levels."""
    LOW = "low"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentState(BaseModel):
    """Represents the current understanding of an ongoing anomaly."""
    anomaly_summary: str = Field(description="Summary of the detected anomaly")
    severity: Severity = Field(description="Calculated severity level")
    affected_service: str = Field(description="The primary service affected")
    metrics_snapshot: dict[str, float] = Field(description="Snapshot of key metrics (cpu, mem, cmd, etc.)")
    rag_context: list[str] = Field(default_factory=list, description="Expert guides populated by researcher node")


class RemediationProposal(BaseModel):
    """A concrete, explainable proposal to remediate an incident."""
    action: str = Field(description="Primary action to execute")
    action_params: dict = Field(description="Parameters for the primary action")
    confidence_score: float = Field(description="RL confidence score (0.0 to 1.0)")
    risk_rationale: str = Field(description="Human-readable explanation and audit trail")
    rollback_action: str = Field(description="Inverse operation to revert the action")
    rollback_params: dict = Field(description="Parameters for the rollback action")


class AgentState(TypedDict):
    """The global state object passed through the LangGraph pipeline."""
    telemetry_events: list[dict]
    force_human_review: bool
    incident: IncidentState | None
    incident_id: str | None
    rl_prediction: str | None
    proposal: RemediationProposal | None
    proposal_id: str | None
    human_approved: bool | None
    reward_signal: float | None
