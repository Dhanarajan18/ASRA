"""
incident_store.py - Storage abstraction for incidents awaiting human approval.

This module keeps API state handling isolated so alternate backends
(sqlite/redis/postgres) can be added in separate PRs without modifying
endpoint logic.
"""

from __future__ import annotations

import threading
from typing import Any


class InMemoryIncidentStore:
    """Thread-safe in-memory storage for pending incidents."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def upsert(self, thread_id: str, incident: dict[str, Any]) -> None:
        with self._lock:
            self._items[thread_id] = incident

    def remove(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._items.pop(thread_id, None)

    def contains(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._items

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def list_all(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy to avoid external mutation."""
        with self._lock:
            return dict(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


incident_store = InMemoryIncidentStore()
