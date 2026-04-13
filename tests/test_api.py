from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient
import pytest

from autonomous_sre.interfaces import api
from autonomous_sre.infrastructure import approval_bus


@pytest.fixture(autouse=True)
def clear_pending_state():
    approval_bus._pending.clear()
    approval_bus._decisions.clear()
    yield
    approval_bus._pending.clear()
    approval_bus._decisions.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api.app)


def test_health_endpoints(client: TestClient):
    live = client.get("/healthz")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_trigger_endpoint_accepted(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    executed = {"value": False}

    async def fake_run_agent_async():
        executed["value"] = True

    monkeypatch.setattr(api, "run_agent_async", fake_run_agent_async)

    tasks = []

    def fake_create_task(coro):
        task = asyncio.get_event_loop().create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(api.asyncio, "create_task", fake_create_task)

    response = client.post("/trigger")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "triggered"
    assert "timestamp" in body

    for task in tasks:
        task.cancel()


def test_pending_proposals_returns_data(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setitem(api.pending_proposals.__globals__, "list_pending_ids", lambda: ["proposal-1"])

    class _FakeDB:
        @staticmethod
        def get_proposal_by_id(proposal_id: str):
            return {
                "id": proposal_id,
                "incident_id": "incident-1",
                "action": "restart_pod",
                "confidence_score": 0.9,
            }

    monkeypatch.setitem(api.pending_proposals.__globals__, "db", _FakeDB())

    response = client.get("/proposals/pending")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "proposal-1"


def test_approve_returns_404_for_unknown_proposal(client: TestClient):
    response = client.post("/proposals/does-not-exist/approve")
    assert response.status_code == 404


def test_reject_returns_404_for_unknown_proposal(client: TestClient):
    response = client.post("/proposals/does-not-exist/reject")
    assert response.status_code == 404


def test_metrics_endpoint_shape(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_incidents" in body
    assert "pending_approvals" in body


def test_list_escalations(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def get_escalations(status=None, limit=100):
            return [{"id": "esc-1", "status": status or "open", "reason": "manual fix"}]

    monkeypatch.setitem(api.list_escalations.__globals__, "db", _FakeDB())
    response = client.get("/escalations?status=open&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["id"] == "esc-1"


def test_assign_escalation_not_found(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def assign_escalation(**kwargs):
            return False

    monkeypatch.setitem(api.assign_escalation.__globals__, "db", _FakeDB())
    response = client.post(
        "/escalations/esc-missing/assign",
        json={"assignee": "alice", "assigned_by": "lead", "workstream": "cloud", "note": "handle this"},
    )
    assert response.status_code == 404


def test_log_escalation_not_found(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def get_escalation_by_id(escalation_id: str):
            return None

    monkeypatch.setitem(api.append_escalation_log.__globals__, "db", _FakeDB())
    response = client.post(
        "/escalations/esc-missing/log",
        json={"author": "alice", "note": "investigating", "phase": "update", "metadata": {}},
    )
    assert response.status_code == 404


def test_resolve_escalation_not_found(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def resolve_escalation(**kwargs):
            return False

    monkeypatch.setitem(api.resolve_escalation.__globals__, "db", _FakeDB())
    response = client.post(
        "/escalations/esc-missing/resolve",
        json={"resolved_by": "alice", "resolution_summary": "patched deployment", "outcome": "resolved"},
    )
    assert response.status_code == 404


def test_escalation_context_not_found(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def get_escalation_by_id(escalation_id: str):
            return None

    monkeypatch.setitem(api.get_escalation_context.__globals__, "db", _FakeDB())
    response = client.get("/escalations/esc-missing/context")
    assert response.status_code == 404


def test_escalation_context_shape(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class _FakeDB:
        @staticmethod
        def get_escalation_by_id(escalation_id: str):
            return {
                "id": escalation_id,
                "incident_id": "inc-1",
                "proposal_id": "prop-1",
                "status": "in_progress",
                "reason": "manual fix",
            }

        @staticmethod
        def get_incident_by_id(incident_id: str):
            return {
                "id": incident_id,
                "service": "orders",
                "severity": "high",
                "anomaly_summary": "high latency detected",
            }

        @staticmethod
        def get_proposal_by_id(proposal_id: str):
            return {
                "id": proposal_id,
                "action": "restart_pod",
                "risk_rationale": "pod instability suspected",
            }

        @staticmethod
        def get_latest_proposal_for_incident(incident_id: str):
            return None

        @staticmethod
        def get_escalation_logs(escalation_id: str, limit: int = 50):
            return [{"author": "dexter", "phase": "investigation", "note": "checking pods"}]

    monkeypatch.setitem(api.get_escalation_context.__globals__, "db", _FakeDB())
    response = client.get("/escalations/esc-1/context")
    assert response.status_code == 200
    body = response.json()
    assert "what_broke" in body
    assert "recommended_investigation" in body
    assert isinstance(body["recommended_investigation"], list)
