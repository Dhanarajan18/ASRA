"""
main.py — Demo runner for the Autonomous SRE Agent.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import random
import uuid
from datetime import datetime

# Ensure stdout can handle UTF-8/Emoji on Windows cp1252
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from autonomous_sre.services.telemetry import TelemetrySimulator, MetricEvent, SERVICES
from autonomous_sre.orchestration.graph import compiled_graph
from autonomous_sre.core.state import AgentState
from langgraph.errors import NodeInterrupt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("sre_main")


async def run_training_warmup(n_episodes: int = 50) -> None:
    """
    Run n silent episodes with injected anomalies to pre-train
    the RL policy before the live demo. Auto-approves all actions
    during warm-up (no human prompts).
    """
    logger.info(f"── Starting RL Warmup Training ({n_episodes} episodes) ──")
    simulator = TelemetrySimulator()
    
    for episode in range(n_episodes):
        batch = await simulator.collect_batch(n=30)
        
        # Force at least one CPU anomaly per episode
        batch.append(MetricEvent(
            service=random.choice(SERVICES),
            cpu_pct=random.uniform(85, 99),
            mem_pct=random.uniform(70, 95),
            latency_ms=random.uniform(1500, 3000),
            timestamp=datetime.utcnow().isoformat()
        ).model_dump())
        
        initial_state: AgentState = {
            "telemetry_events": batch,
            "incident": None,
            "incident_id": None,
            "rl_prediction": None,
            "proposal": None,
            "proposal_id": None,
            "human_approved": True,  # Auto-approve during warmup
            "reward_signal": None,
        }
        
        try:
            compiled_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )
        except NodeInterrupt:
            # Expected during warmup; just continue
            pass
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Completed {episode + 1}/{n_episodes} warmup episodes")
    
    logger.info(f"✅ Warmup complete. RL policy pre-trained on {n_episodes} episodes.")


async def main():
    print("── Initialising Telemetry Simulator ──")
    simulator = TelemetrySimulator()
    
    # Pre-train policy before live run for demo stability.
    await run_training_warmup(n_episodes=50)
    
    batch = await simulator.collect_batch(n=50)

    initial_state: AgentState = {
        "telemetry_events": batch,
        "incident": None,
        "incident_id": None,
        "rl_prediction": None,
        "proposal": None,
        "proposal_id": None,
        "human_approved": None,
        "reward_signal": None,
    }

    print("\n── Running Autonomous SRE Agent ──")
    try:
        final_state = compiled_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    except NodeInterrupt as e:
        logger.info(f"Graph paused with NodeInterrupt: {e}")
        # In API mode, this would be handled by the FastAPI server
        final_state = {}
    
    print(f"\n── Final State ──")
    print(f"Approved: {final_state.get('human_approved', False)}")
    
    # Safe float formatting
    reward = final_state.get('reward_signal')
    reward_str = f"{reward:.3f}" if isinstance(reward, float) else "None"
    print(f"Reward:   {reward_str}")
    
    proposal = final_state.get("proposal")
    if proposal:
        print(f"Proposal: {proposal.action}")
        print(f"Confidence: {proposal.confidence_score:.2f}")
        print(f"Rationale: {proposal.risk_rationale}")
        print(f"Rollback Map: {proposal.rollback_action}({proposal.rollback_params})")
    else:
        print("Proposal: None")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
