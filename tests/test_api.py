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
