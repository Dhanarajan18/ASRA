"""
approval_bus.py — Shared in-memory approval coordination for API-driven HITL.
"""

from __future__ import annotations

import asyncio
from typing import Dict

_pending: Dict[str, asyncio.Event] = {}
_decisions: Dict[str, bool] = {}


def register_pending(proposal_id: str) -> asyncio.Event:
    event = asyncio.Event()
    _pending[proposal_id] = event
    return event


def set_decision(proposal_id: str, approved: bool) -> bool:
    event = _pending.get(proposal_id)
    if event is None:
        return False
    _decisions[proposal_id] = approved
    event.set()
    return True


def pop_decision(proposal_id: str) -> bool | None:
    decision = _decisions.pop(proposal_id, None)
    _pending.pop(proposal_id, None)
    return decision


def list_pending_ids() -> list[str]:
    return list(_pending.keys())
