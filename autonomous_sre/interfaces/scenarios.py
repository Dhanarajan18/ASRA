"""
scenarios.py — Scenario injection CLI for deterministic SRE demo runs.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import datetime

from autonomous_sre.services.telemetry import MetricEvent, TelemetrySimulator
from autonomous_sre.orchestration.graph import compiled_graph
from autonomous_sre.core.state import AgentState

SCENARIOS = {
    "cpu_spike": {
        "cpu_pct": 96.0,
        "mem_pct": 72.0,
        "latency_ms": 450.0,
        "service": "payment-service",
    },
    "memory_leak": {
        "cpu_pct": 45.0,
        "mem_pct": 97.0,
        "latency_ms": 890.0,
        "service": "user-service",
    },
    "latency_storm": {
        "cpu_pct": 55.0,
        "mem_pct": 60.0,
        "latency_ms": 4200.0,
        "service": "api-gateway",
    },
    "cascade_failure": {
        "cpu_pct": 98.0,
        "mem_pct": 98.0,
        "latency_ms": 5000.0,
        "service": "order-service",
    },
}


async def run_scenario(name: str, runs: int) -> None:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")

    simulator = TelemetrySimulator()
    spec = SCENARIOS[name]

    for i in range(runs):
        batch = await simulator.collect_batch(n=25)
        injected = MetricEvent(
            service=spec["service"],
            cpu_pct=spec["cpu_pct"],
            mem_pct=spec["mem_pct"],
            latency_ms=spec["latency_ms"],
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump()
        batch.append(injected)

        state: AgentState = {
            "telemetry_events": batch,
            "incident": None,
            "incident_id": None,
            "rl_prediction": None,
            "proposal": None,
            "proposal_id": None,
            "human_approved": True,
            "reward_signal": None,
        }
        final_state = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        proposal = final_state.get("proposal")

        print(f"\nRun {i + 1}/{runs} | Scenario={name}")
        if proposal:
            print(f"Action: {proposal.action}")
            print(f"Confidence: {proposal.confidence_score:.2f}")
            print(f"Rationale: {proposal.risk_rationale}")
            print(f"Rollback: {proposal.rollback_action}({proposal.rollback_params})")
        print(f"Reward: {final_state.get('reward_signal')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject deterministic SRE failure scenarios")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_scenario(args.scenario, args.runs))
