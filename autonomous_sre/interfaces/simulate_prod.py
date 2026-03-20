"""
simulate_prod.py - Lightweight production-like traffic simulator.

Generates webhook alerts against the local API at configurable request rates,
polls pending incidents, and optionally auto-approves proposals.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import requests


SERVICES = ["api-gateway", "payment-gateway", "auth-service", "order-api"]
ALERTS = ["HighLatency", "HighCPU", "MemoryLeak", "ErrorSpike"]


@dataclass
class SimulationStats:
    sent: int = 0
    accepted: int = 0
    pending_seen: int = 0
    approved: int = 0
    rejected: int = 0
    failures: int = 0


def _build_payload(spike_mode: bool) -> dict:
    service = random.choice(SERVICES)
    alert = random.choice(ALERTS)
    if spike_mode:
        cpu = random.uniform(92.0, 99.5)
        latency = random.uniform(1800.0, 5000.0)
        severity = "critical"
    else:
        cpu = random.uniform(40.0, 88.0)
        latency = random.uniform(120.0, 1600.0)
        severity = "high" if cpu > 80.0 or latency > 1300.0 else "low"

    return {
        "service": service,
        "alert_name": alert,
        "severity": severity,
        "metrics": {
            "cpu_pct": round(cpu, 2),
            "latency_ms": round(latency, 2),
            "mem_pct": round(random.uniform(45.0, 95.0), 2),
        },
    }


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def run_simulation(
    base_url: str,
    duration_seconds: int,
    rps: float,
    approval_probability: float,
    spike_mode: bool,
    api_key: str | None,
) -> SimulationStats:
    stats = SimulationStats()
    interval = 1.0 / max(rps, 0.1)
    started = time.monotonic()
    deadline = started + duration_seconds

    while time.monotonic() < deadline:
        payload = _build_payload(spike_mode=spike_mode)
        stats.sent += 1
        try:
            # Trigger a synthetic run through the new API contract.
            resp = requests.post(
                f"{base_url}/trigger",
                headers=_headers(api_key),
                timeout=3,
            )
            if resp.status_code == 200:
                stats.accepted += 1
            else:
                stats.failures += 1
        except requests.RequestException:
            stats.failures += 1

        # Every second, inspect pending incidents and auto-decide some of them.
        elapsed = time.monotonic() - started
        if int(elapsed) % 1 == 0:
            try:
                pending = requests.get(f"{base_url}/proposals/pending", timeout=3).json()
                incidents = {row["id"]: row for row in pending}
                stats.pending_seen += len(incidents)

                for thread_id in list(incidents.keys()):
                    decide = random.random()
                    if decide <= approval_probability:
                        action = "approve"
                        stats.approved += 1
                    else:
                        action = "reject"
                        stats.rejected += 1

                    try:
                        requests.post(
                            f"{base_url}/proposals/{thread_id}/{action}",
                            headers=_headers(api_key),
                            timeout=3,
                        )
                    except requests.RequestException:
                        stats.failures += 1
            except requests.RequestException:
                stats.failures += 1

        time.sleep(interval)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local prod-like simulation against Autonomous SRE API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--rps", type=float, default=2.0, help="Webhook requests per second")
    parser.add_argument(
        "--approval-prob",
        type=float,
        default=0.7,
        help="Probability [0..1] of auto-approving pending incidents",
    )
    parser.add_argument("--spike-mode", action="store_true", help="Send consistently severe incidents")
    parser.add_argument("--api-key", default=None, help="Optional x-api-key when API auth is enabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_simulation(
        base_url=args.base_url.rstrip("/"),
        duration_seconds=args.duration,
        rps=args.rps,
        approval_probability=max(0.0, min(1.0, args.approval_prob)),
        spike_mode=args.spike_mode,
        api_key=args.api_key,
    )

    print("Simulation complete")
    print(f"sent={stats.sent}")
    print(f"accepted={stats.accepted}")
    print(f"pending_seen={stats.pending_seen}")
    print(f"approved={stats.approved}")
    print(f"rejected={stats.rejected}")
    print(f"failures={stats.failures}")


if __name__ == "__main__":
    main()
