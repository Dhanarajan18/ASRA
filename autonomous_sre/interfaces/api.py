"""
api.py — FastAPI server for Autonomous SRE Agent + dashboard endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# API server always runs in API_MODE for approval queue handling.
os.environ.setdefault("API_MODE", "true")

from autonomous_sre.orchestration.graph import compiled_graph
from autonomous_sre.core.state import AgentState
from autonomous_sre.services.telemetry import TelemetrySimulator
from autonomous_sre.infrastructure.persistence import get_db
from autonomous_sre.infrastructure.approval_bus import set_decision, list_pending_ids

logger = logging.getLogger("sre_api")
logger.setLevel(logging.INFO)

db = get_db()
app = FastAPI(title="Autonomous SRE Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Keep compatibility with existing health checks.
@app.get("/healthz")
def healthz() -> dict:
    return health()


@app.get("/readyz")
def readyz() -> dict:
    return {"status": "ready", "pending_approvals": len(list_pending_ids())}


@app.get("/incidents")
def list_incidents(n: int = 20) -> list[dict]:
    incidents = db.get_recent_incidents(n)
    enriched: list[dict] = []
    for inc in incidents:
        proposals = db.get_proposals(inc["id"])
        latest = proposals[0] if proposals else None
        status = "Pending"
        if latest:
            if latest.get("human_approved") == 1:
                status = "Approved"
            elif latest.get("human_approved") == 0:
                status = "Escalated"
        enriched.append(
            {
                "incident_id": inc["id"],
                "timestamp": inc["timestamp"],
                "service": inc["service"],
                "severity": inc["severity"],
                "anomaly_summary": inc["anomaly_summary"],
                "proposal_id": latest.get("id") if latest else None,
                "action": latest.get("action") if latest else None,
                "confidence": latest.get("confidence_score") if latest else 0.0,
                "status": status,
            }
        )
    return enriched


@app.get("/incidents/{incident_id}/proposal")
def get_proposal(incident_id: str) -> list[dict]:
    return db.get_proposals(incident_id)


@app.get("/proposals/pending")
def pending_proposals() -> list[dict]:
    pending_ids = list_pending_ids()
    rows: list[dict] = []
    for proposal_id in pending_ids:
        row = db.get_proposal_by_id(proposal_id)
        if row:
            rows.append(row)
    return rows


@app.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str) -> dict:
    ok = set_decision(proposal_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail="Proposal not found or already resolved")
    return {"status": "approved"}


@app.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str) -> dict:
    ok = set_decision(proposal_id, False)
    if not ok:
        raise HTTPException(status_code=404, detail="Proposal not found or already resolved")
    return {"status": "rejected"}


@app.get("/metrics")
def metrics() -> dict:
    summary = db.get_metrics_summary()
    summary["pending_approvals"] = len(list_pending_ids())
    return summary


@app.get("/metrics/rewards")
def rewards() -> dict:
    rows = db.get_episode_metrics()
    return {
        "episodes": [r["episode_id"] for r in rows],
        "rewards": [r["reward"] for r in rows],
    }


@app.get("/metrics/learning-curve")
def learning_curve() -> dict:
    rows = db.get_episode_metrics()
    return {
        "episodes": [r["episode_id"] for r in rows],
        "rewards": [r["reward"] for r in rows],
        "confidence": [r["confidence"] for r in rows],
    }


async def run_agent_async() -> None:
    simulator = TelemetrySimulator()
    batch = await simulator.collect_batch(n=50)
    state: AgentState = {
        "telemetry_events": batch,
        "incident": None,
        "incident_id": None,
        "rl_prediction": None,
        "proposal": None,
        "proposal_id": None,
        "human_approved": None,
        "reward_signal": None,
    }
    compiled_graph.invoke(
        state,
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )


@app.post("/trigger")
async def trigger_run() -> dict:
    """Manually trigger a new SRE agent run."""
    asyncio.create_task(run_agent_async())
    return {"status": "triggered", "timestamp": datetime.utcnow().isoformat()}


app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
