from __future__ import annotations

from autonomous_sre.infrastructure.incident_store import InMemoryIncidentStore


def test_store_upsert_contains_and_count():
    store = InMemoryIncidentStore()
    assert store.count() == 0

    store.upsert("thread-1", {"service": "api-gateway", "action": "restart_pod"})
    assert store.contains("thread-1")
    assert store.count() == 1


def test_store_remove_and_clear():
    store = InMemoryIncidentStore()
    store.upsert("thread-1", {"service": "api-gateway"})
    store.upsert("thread-2", {"service": "order-api"})

    removed = store.remove("thread-1")
    assert removed is not None
    assert not store.contains("thread-1")
    assert store.count() == 1

    store.clear()
    assert store.count() == 0


def test_list_all_returns_copy():
    store = InMemoryIncidentStore()
    store.upsert("thread-1", {"service": "api-gateway"})

    copy_view = store.list_all()
    copy_view["thread-2"] = {"service": "mutated"}

    assert store.count() == 1
    assert not store.contains("thread-2")
