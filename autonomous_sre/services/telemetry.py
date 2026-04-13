"""
telemetry.py — Async telemetry stream simulator.

Simulates a high-throughput stream of Log, Metric, and Trace events
with injected anomalies for testing the Autonomous SRE Agent.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Pydantic Event Models
# ──────────────────────────────────────────────

class BaseEvent(BaseModel):
    service: str = Field(description="The source service")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class LogEvent(BaseEvent):
    event_type: str = "log"
    level: str = Field(description="Log severity (e.g. INFO, ERROR)")
    message: str = Field(description="The log payload")


class MetricEvent(BaseEvent):
    event_type: str = "metric"
    cpu_pct: float = Field(description="CPU utilisation (%)")
    mem_pct: float = Field(description="Memory utilisation (%)")
    latency_ms: float = Field(description="Request latency (ms)")


class TraceEvent(BaseEvent):
    event_type: str = "trace"
    trace_id: str = Field(description="Distributed trace ID")
    duration_ms: float = Field(description="Trace duration (ms)")
    error: bool = Field(description="True if the trace contains an error")


# ──────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────

SERVICES = [
    "auth-service",
    "payment-service",
    "order-service",
    "user-service",
    "api-gateway",
    "notification-service",
]

# Service dependency map for cascade failure simulation
SERVICE_DEPENDENCIES = {
    "api-gateway": ["auth-service", "order-service", "user-service"],
    "order-service": ["payment-service", "user-service"],
    "payment-service": [],
    "auth-service": [],
    "user-service": [],
    "notification-service": ["order-service"],
}

_SERVICES = SERVICES  # Backwards compatibility
_LOG_MESSAGES = ["Request successful", "Cache miss", "Connection timeout", "DB transaction rolled back"]


class TelemetrySimulator:
    """Async generator yielding a noisy telemetry stream."""

    def __init__(self, anomaly_probability: float = 0.15) -> None:
        self.anomaly_prob = anomaly_probability

    async def stream(self, n: int):
        """
        Async generator yielding `n` random telemetry events.
        Events are weighted: 40% Logs, 40% Metrics, 20% Traces.
        
        Supports cascade failure simulation: when a primary service anomaly 
        is detected, dependent services exhibit degraded (but not anomalous) metrics.
        """
        anomalous_primary_service = None
        
        for _ in range(n):
            await asyncio.sleep(0.05)  # Simulate network ingestion latency

            service = random.choice(_SERVICES)
            event_type = random.choices(
                ["log", "metric", "trace"],
                weights=[0.4, 0.4, 0.2]
            )[0]

            if event_type == "log":
                level = random.choices(["INFO", "WARN", "ERROR"], weights=[0.8, 0.1, 0.1])[0]
                yield LogEvent(
                    service=service,
                    level=level,
                    message=random.choice(_LOG_MESSAGES)
                ).model_dump()

            elif event_type == "metric":
                is_anomaly = random.random() < self.anomaly_prob
                
                # Check if this service is a dependent of an anomalous primary
                is_cascade_effect = False
                if anomalous_primary_service and service in SERVICE_DEPENDENCIES.get(anomalous_primary_service, []):
                    is_cascade_effect = True
                
                if is_anomaly:
                    # Primary service anomaly
                    if random.random() < 0.5:
                        cpu = random.uniform(91.0, 100.0)
                        latency = random.uniform(10.0, 500.0)
                    else:
                        cpu = random.uniform(10.0, 60.0)
                        latency = random.uniform(2001.0, 5000.0)
                    mem = random.uniform(60.0, 95.0)
                    anomalous_primary_service = service  # Mark as primary anomaly
                    
                elif is_cascade_effect:
                    # Dependent service showing degradation (not anomaly)
                    cpu = random.uniform(65.0, 82.0)  # Elevated but below anomaly threshold
                    mem = random.uniform(50.0, 70.0)
                    latency = random.uniform(400.0, 950.0)  # Slightly elevated latency
                    
                else:
                    # Normal operating bounds
                    cpu = random.uniform(10.0, 60.0)
                    mem = random.uniform(20.0, 70.0)
                    latency = random.uniform(10.0, 300.0)
                    # Clear anomalous primary after a few iterations
                    if random.random() < 0.1:
                        anomalous_primary_service = None

                yield MetricEvent(
                    service=service,
                    cpu_pct=round(cpu, 2),
                    mem_pct=round(mem, 2),
                    latency_ms=round(latency, 2)
                ).model_dump()

            else:
                # trace
                is_error = random.random() < self.anomaly_prob
                duration = random.uniform(100.0, 3000.0) if is_error else random.uniform(10.0, 200.0)
                yield TraceEvent(
                    service=service,
                    trace_id=f"tr-{random.randint(1000, 9999)}",
                    duration_ms=round(duration, 2),
                    error=is_error
                ).model_dump()

    async def collect_batch(self, n: int = 50) -> list[dict[str, Any]]:
        """Collect a window of telemetry events into a batch."""
        batch = []
        async for event in self.stream(n):
            batch.append(event)
        return batch

# UPGRADE: replace TelemetrySimulator with aiokafka consumer
#   or FastAPI webhook receiving Prometheus/Datadog JSON payloads
