"""
graph.py — LangGraph orchestration.

StateGraph with five nodes and conditional routing.
Wiring together RAG, RL, Tools, and Telemetry.
"""

from __future__ import annotations

import logging
import uuid
import os
import math
import numpy as np
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from autonomous_sre.core.state import AgentState, IncidentState, RemediationProposal, Severity
from autonomous_sre.services.rag import KnowledgeBase
from autonomous_sre.services.learning import LearningEngine, encode_state
from autonomous_sre.infrastructure.tools import TOOL_DISPATCHER, ROLLBACK_DISPATCHER
from autonomous_sre.core.config import settings
from autonomous_sre.infrastructure.persistence import get_db
from autonomous_sre.infrastructure.audit import get_audit_logger
from autonomous_sre.infrastructure.approval_bus import register_pending, pop_decision

logger = logging.getLogger("sre_graph")
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Human-in-the-loop confidence threshold (configurable via env var)
HITL_THRESHOLD = float(os.getenv("HITL_THRESHOLD", str(settings.approval_confidence_threshold)))
API_MODE = os.getenv("API_MODE", "false").lower() == "true"
logger.info(f"HITL_THRESHOLD set to {HITL_THRESHOLD:.2f} (set HITL_THRESHOLD env var to override)")


# ──────────────────────────────────────────────
# Global Instances
# ──────────────────────────────────────────────

kb = KnowledgeBase()
engine = LearningEngine()
db = get_db()
audit_logger = get_audit_logger()


# ──────────────────────────────────────────────
# 1. Analyzer Node
# ──────────────────────────────────────────────

def analyzer_node(state: AgentState) -> dict[str, Any]:
    """Scan telemetry_events for anomalies and build IncidentState."""
    logger.info("analyzer_node | Scanning telemetry batch...")
    events = state.get("telemetry_events", [])
    
    anomalous_metric = None
    for evt in events:
        if evt.get("event_type") == "metric":
            cpu = evt.get("cpu_pct", 0)
            mem = evt.get("mem_pct", 0)
            lat = evt.get("latency_ms", 0)
            if (
                cpu > settings.anomaly_cpu_threshold
                or mem > settings.anomaly_memory_threshold
                or lat > 1500
            ):
                anomalous_metric = evt
                break
                
    if anomalous_metric:
        service = anomalous_metric.get("service", "unknown")
        cpu = anomalous_metric.get("cpu_pct", 0)
        lat = anomalous_metric.get("latency_ms", 0)
        
        severity = Severity.CRITICAL if cpu > (settings.anomaly_cpu_threshold + 10) else Severity.HIGH
        summary = f"Detected high CPU ({cpu:.1f}%) or Latency ({lat:.0f}ms) on {service}"
        
        # Calculate derived metrics for snapshot
        err_rates = [e.get("error") for e in events if e.get("event_type") == "trace" and isinstance(e.get("error"), (bool, int, float))]
        error_rate = sum([float(rate) for rate in err_rates]) / len(err_rates) * 100 if err_rates else 0.0
        
        snapshot = {
            "cpu_pct": cpu,
            "mem_pct": anomalous_metric.get("mem_pct", 50.0),
            "latency_ms": lat,
            "error_rate": error_rate,
            "deploy_age_hours": 1.5,  # Stubbed proxy
            "active_alerts": 3.0      # Stubbed proxy
        }
        
    else:
        # No anomalies
        severity = Severity.LOW
        summary = "No significant anomalies detected in the current telemetry window."
        service = "system"
        snapshot = {"cpu_pct": 20.0, "mem_pct": 40.0, "latency_ms": 50.0}

    incident = IncidentState(
        anomaly_summary=summary,
        severity=severity,
        affected_service=service,
        metrics_snapshot=snapshot,
        rag_context=[]
    )
    
    logger.info(f"analyzer_node | Constructed IncidentState: severity={severity.value} on {service}")
    
    # Save incident to database
    incident_id = db.save_incident(incident, service)
    
    # Log to audit trail
    audit_logger.log_incident_detected(
        incident_id=incident_id,
        severity=severity.value,
        summary=summary
    )
    
    return {"incident": incident, "incident_id": incident_id}


# ──────────────────────────────────────────────
# 2. Researcher Node
# ──────────────────────────────────────────────

def researcher_node(state: AgentState) -> dict[str, Any]:
    """Retrieve expert knowledge via FAISS based on the anomaly summary."""
    incident = state["incident"]
    assert incident is not None
    
    logger.info(f"researcher_node | Fetching guides for: '{incident.anomaly_summary}'")
    guides = kb.query(incident.anomaly_summary, k=3)
    
    # Update incident with RAG context
    incident.rag_context = guides
    logger.info(f"researcher_node | Retrieved {len(guides)} guides.")
    
    return {"incident": incident}


# ──────────────────────────────────────────────
# 3. Predictor Node
# ──────────────────────────────────────────────

def predictor_node(state: AgentState) -> dict[str, Any]:
    """RL Engine encodes state and selects an action."""
    incident = state["incident"]
    assert incident is not None
    
    state_vec = encode_state(incident)
    logger.info(f"predictor_node | Encoded state vector: {state_vec}")
    
    action = engine.select_action(state_vec, epsilon=settings.rl_epsilon)
    logger.info(f"predictor_node | RL selected action: {action}")
    
    return {"rl_prediction": action}


# ──────────────────────────────────────────────
# 4. Proposer Node
# ──────────────────────────────────────────────

def proposer_node(state: AgentState) -> dict[str, Any]:
    """Assemble final RemediationProposal with confidence and rollback mappings."""
    incident = state["incident"]
    action = state["rl_prediction"]
    assert incident is not None and action is not None
    
    svc = incident.affected_service
    
    # ── Map Action -> Params ──
    if action == "scale_replicas":
        params = {"service": svc, "replicas": 3}
        rollback_params = {"service": svc, "previous_count": 1}
    elif action == "restart_pod":
        params = {"service": svc, "pod_id": f"{svc}-pod-{uuid.uuid4().hex[:4]}"}
        rollback_params = {"service": svc, "pod_id": params["pod_id"]}
    elif action == "rollback_deployment":
        params = {"service": svc, "revision": 2}
        rollback_params = {"service": svc, "revision": 2}
    elif action == "increase_memory_limit":
        params = {"service": svc, "limit_mb": 1024}
        rollback_params = {"service": svc, "limit_mb": 512}
    elif action == "flush_cache":
        params = {"service": svc}
        rollback_params = {"service": svc}
    elif action == "reroute_traffic":
        params = {"service": svc, "target": "us-east-2", "weight_pct": 50}
        rollback_params = {"service": svc, "target": "us-east-1", "weight_pct": 100}
    else:  # no_action
        params = {"service": svc}
        rollback_params = {"service": svc}

    rollback_action = ROLLBACK_DISPATCHER.get(action, "rollback_no_action")
    
    # ── Confidence & Rationale ──
    state_vec = encode_state(incident)
    # Sigmoid normalization instead of linear scaling
    # Derives Q-value and applies sigmoid to normalize to (0, 1)
    raw_max_q = float(np.max(engine._policy_weights @ state_vec))
    confidence = 1.0 / (1.0 + math.exp(-raw_max_q * 3))
    
    rag_text = incident.rag_context[0][:120] + "..." if incident.rag_context else "No matching guides."
    
    # UPGRADE: replace hardcoded risk_rationale with LangChain ChatPromptTemplate passing
    # RAG context + RL recommendation to GPT-4 for natural-language rationale generation
    rationale = (
        f"Anomaly: {incident.anomaly_summary}. "
        f"Severity: {incident.severity.value}. "
        f"RL selected '{action}' (confidence {confidence:.2f}). "
        f"RAG matched: '{rag_text}'. "
        f"Rollback: {rollback_action}({rollback_params})."
    )
    
    proposal = RemediationProposal(
        action=action,
        action_params=params,
        confidence_score=confidence,
        risk_rationale=rationale,
        rollback_action=rollback_action,
        rollback_params=rollback_params
    )
    
    logger.info(f"proposer_node | Generated proposal. Confidence: {confidence:.2f}")
    
    # Save proposal to database (human_approved=None as it's pending)
    incident_id = state.get("incident_id")
    if incident_id is None:
        raise ValueError("incident_id is required but was None")
    proposal_id = db.save_proposal(proposal, incident_id, approved=None, reward=0.0)
    
    # Log to audit trail
    audit_logger.log_proposal_generated(
        incident_id=str(incident_id) if incident_id is not None else "",
        proposal_id=str(proposal_id) if proposal_id is not None else "",
        action=action,
        confidence=confidence,
        rationale=rationale
    )
    
    return {"proposal": proposal, "proposal_id": proposal_id}


# ──────────────────────────────────────────────
# 5. Human in the Loop Node
# ──────────────────────────────────────────────

def human_in_the_loop_node(state: AgentState) -> dict[str, Any]:
    """
    Execute action if confidence is high and human approves; update RL policy.
    This node now relies on the LangGraph Checkpointer to pause execution.
    """
    proposal = state["proposal"]
    incident = state["incident"]
    incident_id = state.get("incident_id")
    proposal_id = state.get("proposal_id")
    assert proposal is not None and incident is not None
    
    requires_human_review = bool(state.get("force_human_review", False)) or proposal.confidence_score >= HITL_THRESHOLD

    if requires_human_review and state.get("human_approved") is None:
        if API_MODE and proposal_id:
            logger.info(
                "human_in_the_loop_node | API_MODE enabled; awaiting approval for proposal %s (timeout=300s)",
                proposal_id,
            )
            event = register_pending(proposal_id)
            # Use thread-safe event waiting (works in threads without event loop)
            is_set = event.wait(timeout=300)
            if not is_set:
                logger.warning("human_in_the_loop_node | Approval wait timed out; escalating.")
                pop_decision(proposal_id)
                state["human_approved"] = False
            else:
                decision = pop_decision(proposal_id)
                state["human_approved"] = bool(decision)
        else:
            # CLI fallback mode for local runs without API
            decision = input(f"Approve action '{proposal.action}'? [y/N]: ").strip().lower()
            state["human_approved"] = decision in {"y", "yes"}
        
    logger.info("human_in_the_loop_node | Resuming execution and evaluating outcome.")
    reward = 0.0
    
    # Setup for RL experience logging
    s_vec = encode_state(incident)
    action_name = proposal.action
    
    if requires_human_review:
        # We are resuming from an interrupt. Check the injected state.
        human_approved = state.get("human_approved", False)
        
        if human_approved:
            logger.info("human_in_the_loop_node | Human approved via API. Dispatching tool...")
            tool_func = TOOL_DISPATCHER.get(action_name)
            if tool_func:
                tool_result = tool_func(**proposal.action_params)
            else:
                tool_result = {"status": "error", "detail": f"Unknown action '{action_name}'"}

            if tool_result.get("status") != "success":
                logger.error(
                    "human_in_the_loop_node | Tool execution failed for action=%s detail=%s",
                    action_name,
                    tool_result.get("detail", "unknown error"),
                )

                rollback_name = proposal.rollback_action
                rollback_func = globals().get(rollback_name)
                rollback_result = None
                if callable(rollback_func):
                    rollback_result = rollback_func(**proposal.rollback_params)

                audit_logger.log_error(
                    error_type="tool_execution_failed",
                    error_message=str(tool_result.get("detail", "unknown error")),
                    context={
                        "incident_id": incident_id,
                        "proposal_id": proposal_id,
                        "action": action_name,
                        "rollback_action": rollback_name,
                        "rollback_result": rollback_result,
                    },
                )

                reward = engine.calculate_reward("escalated", 0.0)
                human_approved = False
            else:
                reward = engine.calculate_reward("resolved", 1.0)
            
            # Log approval decision
            audit_logger.log_approval_decision(
                incident_id=str(incident_id) if incident_id is not None else "",
                proposal_id=str(proposal_id) if proposal_id is not None else "",
                action=action_name,
                approved=human_approved,
                rationale=(
                    "Human approved the proposal via API"
                    if human_approved
                    else "Human approved but tool failed; escalated with rollback attempt"
                ),
            )
            
            # Log action execution
            if human_approved:
                audit_logger.log_action_execution(
                    incident_id=incident_id,
                    proposal_id=proposal_id,
                    action=action_name,
                    params=proposal.action_params,
                    rollback_action=proposal.rollback_action,
                    rollback_params=proposal.rollback_params,
                )
            else:
                audit_logger.log_escalation(
                    incident_id=incident_id,
                    proposal_id=proposal_id,
                    reason="Action execution failed; escalated after rollback attempt.",
                )
        else:
            logger.info("human_in_the_loop_node | Human REJECTED proposal via API.")
            reward = engine.calculate_reward("escalated", 0.0)
            
            # Log rejection decision
            audit_logger.log_approval_decision(
                incident_id=incident_id,
                proposal_id=proposal_id,
                action=action_name,
                approved=False,
                rationale="Human rejected the proposal via API"
            )
            
            # Log escalation
            audit_logger.log_escalation(
                incident_id=incident_id,
                proposal_id=proposal_id,
                reason="Human rejected the proposal. Escalating to on-call team."
            )

            escalation_id = db.create_escalation(
                incident_id=str(incident_id) if incident_id is not None else "",
                proposal_id=str(proposal_id) if proposal_id is not None else None,
                reason="Human rejected the proposal. Requires manual investigation and remediation.",
            )
            db.add_escalation_log(
                escalation_id=escalation_id,
                author="system",
                phase="handoff",
                note="Escalation created from rejected AI proposal and handed to SRE team.",
                metadata={"action": action_name},
            )
        
        # Update proposal in database with approval decision and reward
        db.update_proposal_approval(proposal_id, human_approved, reward)
    else:
        # Low confidence -> Auto escalate immediately (no interrupt)
        logger.warning(
            "human_in_the_loop_node | Confidence %.2f below HITL threshold %.2f. Auto-escalating.",
            proposal.confidence_score,
            HITL_THRESHOLD,
        )
        human_approved = False
        reward = engine.calculate_reward("escalated", 0.5)
        
        # Log auto-escalation
        audit_logger.log_escalation(
            incident_id=incident_id,
            proposal_id=proposal_id,
            reason=f"Low confidence score ({proposal.confidence_score:.2f}) below threshold {HITL_THRESHOLD:.2f}. Auto-escalating."
        )

        escalation_id = db.create_escalation(
            incident_id=str(incident_id) if incident_id is not None else "",
            proposal_id=str(proposal_id) if proposal_id is not None else None,
            reason=(
                f"Low confidence score ({proposal.confidence_score:.2f}) below threshold "
                f"{HITL_THRESHOLD:.2f}. Requires human fix."
            ),
        )
        db.add_escalation_log(
            escalation_id=escalation_id,
            author="system",
            phase="handoff",
            note="Escalation created automatically due to low-confidence recommendation.",
            metadata={"action": action_name, "confidence": proposal.confidence_score},
        )
        
        # Update proposal in database
        db.update_proposal_approval(proposal_id, False, reward)

    # ── Continuous Learning Update ──
    s_next = s_vec * 0.0
    engine.store_experience(s_vec, action_name, reward, s_next)
    
    # Save experience to database
    db.save_experience(s_vec, action_name, reward, s_next)
    
    # Save episode metric for learning curve
    was_correct = (action_name != "no_action") if incident.severity != Severity.LOW else True
    db.save_episode_metric(action_name, proposal.confidence_score, reward, was_correct)
    
    engine.update_policy()
    
    return {"human_approved": human_approved, "reward_signal": reward}


# ──────────────────────────────────────────────
# Routing & Graph Assembly
# ──────────────────────────────────────────────

def _route_after_analyzer(state: AgentState) -> str:
    incident = state["incident"]
    if incident and incident.severity == Severity.LOW:
        logger.info("_route_after_analyzer | Severity LOW -> skipping RAG directly to predictor")
        return "predictor"
    logger.info("_route_after_analyzer | Severity HIGH/CRITICAL -> routing to RAG researcher")
    return "researcher"


# Build state graph
graph = StateGraph(AgentState)

graph.add_node("analyzer", analyzer_node)
graph.add_node("researcher", researcher_node)
graph.add_node("predictor", predictor_node)
graph.add_node("proposer", proposer_node)
graph.add_node("human_in_the_loop", human_in_the_loop_node)

graph.set_entry_point("analyzer")

graph.add_conditional_edges(
    "analyzer", 
    _route_after_analyzer,
    {"researcher": "researcher", "predictor": "predictor"}
)
graph.add_edge("researcher", "predictor")
graph.add_edge("predictor", "proposer")
graph.add_edge("proposer", "human_in_the_loop")
graph.add_edge("human_in_the_loop", END)

# Instantiate a global memory checkpointer for state persistence
memory = MemorySaver()

# Export compiled graph with checkpointer attached
compiled_graph = graph.compile(checkpointer=memory)
