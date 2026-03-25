"""
api.py — FastAPI server for Autonomous SRE Agent + dashboard endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic_settings import BaseSettings

# API server always runs in API_MODE for approval queue handling.
os.environ.setdefault("API_MODE", "true")

from autonomous_sre.orchestration.graph import compiled_graph
from autonomous_sre.core.state import AgentState
from autonomous_sre.services.telemetry import TelemetrySimulator
from autonomous_sre.infrastructure.persistence import get_db
from autonomous_sre.infrastructure.approval_bus import set_decision, list_pending_ids
from autonomous_sre.infrastructure.audit import get_audit_logger
from autonomous_sre.core.config import settings

logger = logging.getLogger("sre_api")
logger.setLevel(logging.INFO)


def utc_now_iso() -> str:
    """Return an RFC3339 UTC timestamp with explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

db = get_db()
audit_logger = get_audit_logger()
app = FastAPI(title="Autonomous SRE Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Verify API key for protected endpoints."""
    # Skip verification if API key auth is disabled
    if not settings.api_key_enabled:
        return True
    
    # Check various ways to provide the API key
    provided_key = None
    
    # Check Authorization header (Bearer token)
    if authorization and authorization.startswith("Bearer "):
        provided_key = authorization[7:]  # Remove "Bearer " prefix
    # Check X-API-Key header
    elif x_api_key:
        provided_key = x_api_key
    # Check credentials from HTTPBearer
    elif credentials:
        provided_key = credentials.credentials
    
    # If no key provided, raise error
    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify the key
    if provided_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Autonomous SRE Agent API",
        version="1.0.0",
        description="""
        ## Autonomous SRE Agent API
        
        This API provides endpoints for interacting with the Autonomous SRE Agent system.
        
        ### Features:
        - Real-time incident detection and remediation
        - Human-in-the-loop approval workflow
        - Reinforcement learning-based action selection
        - Comprehensive audit trail
        - Live dashboard integration
        
        ### Key Endpoints:
        - `/trigger` - Manually trigger an SRE agent run
        - `/incidents` - List recent incidents
        - `/proposals/pending` - Get pending approval requests
        - `/metrics` - Get system metrics
        - `/health` - Health check endpoint
        """,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Swagger UI endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url or "/docs/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/detailed")
def detailed_health() -> dict:
    """Detailed health check including database connectivity."""
    try:
        # Test database connection
        db.get_metrics_summary()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status,
        "timestamp": utc_now_iso()
    }


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
    """Approve a pending proposal. No auth required for dashboard."""
    ok = set_decision(proposal_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail="Proposal not found or already resolved")
    logger.info(f"Proposal {proposal_id} approved via API")
    return {"status": "approved"}


@app.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str) -> dict:
    """Reject a pending proposal. No auth required for dashboard."""
    ok = set_decision(proposal_id, False)
    if not ok:
        raise HTTPException(status_code=404, detail="Proposal not found or already resolved")
    logger.info(f"Proposal {proposal_id} rejected via API")
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


async def run_agent_async(force_human_review: bool = False) -> None:
    """Run a single iteration of the SRE agent with error handling."""
    try:
        simulator = TelemetrySimulator()
        batch = await simulator.collect_batch(n=50)
        state: AgentState = {
            "telemetry_events": batch,
            "force_human_review": force_human_review,
            "incident": None,
            "incident_id": None,
            "rl_prediction": None,
            "proposal": None,
            "proposal_id": None,
            "human_approved": None,
            "reward_signal": None,
        }
        await asyncio.to_thread(
            compiled_graph.invoke,
            state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    except Exception as e:
        logger.error(f"Error in agent run: {e}")
        # Log to audit trail for debugging
        try:
            audit_logger.log_error(
                error_type="agent_run_error",
                error_message=str(e),
                context={"component": "api.trigger_run"}
            )
        except Exception:
            pass  # Avoid infinite logging loops


@app.post("/trigger")
async def trigger_run(background_tasks: BackgroundTasks, force_human_review: bool = False) -> dict:
    """Manually trigger a new SRE agent run."""
    try:
        if force_human_review:
            background_tasks.add_task(run_agent_async, True)
        else:
            background_tasks.add_task(run_agent_async)
        return {
            "status": "triggered",
            "force_human_review": force_human_review,
            "timestamp": utc_now_iso(),
        }
    except Exception as e:
        logger.error(f"Failed to trigger agent run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger agent run: {str(e)}")


app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
