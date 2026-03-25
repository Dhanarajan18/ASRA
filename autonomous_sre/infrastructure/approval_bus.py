"""
approval_bus.py — Shared in-memory approval coordination for API-driven HITL.

Uses threading.Event for thread-safe cross-thread signaling.
"""

from __future__ import annotations

import threading
from typing import Dict

_pending: Dict[str, threading.Event] = {}
_decisions: Dict[str, bool] = {}
_lock = threading.Lock()


def register_pending(proposal_id: str) -> threading.Event:
    """Register a proposal as pending human approval. Thread-safe."""
    event = threading.Event()
    with _lock:
        _pending[proposal_id] = event
    return event


def set_decision(proposal_id: str, approved: bool) -> bool:
    """Record a human decision and signal the waiting thread. Thread-safe."""
    with _lock:
        event = _pending.get(proposal_id)
        if event is None:
            return False
        _decisions[proposal_id] = approved
    event.set()
    return True


def pop_decision(proposal_id: str) -> bool | None:
    """Retrieve and clear a decision. Thread-safe."""
    with _lock:
        decision = _decisions.pop(proposal_id, None)
        _pending.pop(proposal_id, None)
    return decision


def list_pending_ids() -> list[str]:
    """List all pending proposal IDs. Thread-safe."""
    with _lock:
        return list(_pending.keys())
