"""
tools.py — Infrastructure actuators.

Idempotent adapters for scaling, restarting, and rolling back services.
Every tool simulates execution, logs its action, and provides an inverse 
for safe rollbacks.
"""

from __future__ import annotations

import logging
import asyncio
import random
import time
from typing import Dict, Any

logger = logging.getLogger("sre_tools")
logger.setLevel(logging.INFO)


def _simulate_delay():
    """Stub to simulate network/API delays (synchronous for simplicity of demo, or can be awaited)."""
    # Simulate realistic network delay for infrastructure calls
    time.sleep(random.uniform(0.1, 0.5))


def _simulate_failure(failure_rate: float = 0.05) -> bool:
    """Simulate occasional infrastructure failures for realism."""
    return random.random() < failure_rate


# ──────────────────────────────────────────────
# Scale Replicas
# ──────────────────────────────────────────────

def scale_replicas(service: str, replicas: int) -> dict:
    logger.info(f"Executing scale_replicas | service={service} replicas={replicas}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.02):  # 2% failure rate
        logger.warning(f"Scale replicas failed for {service}")
        return {"status": "error", "detail": f"Failed to scale {service} to {replicas} replicas"}
    
    print(f"[Actuator] Scaled {service} to {replicas} replicas.")
    return {"status": "success", "detail": f"Scaled {service} to {replicas} replicas."}

def rollback_scale_replicas(service: str, previous_count: int) -> dict:
    logger.info(f"Rolling back scale_replicas | service={service} count={previous_count}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.02):  # 2% failure rate
        logger.warning(f"Rollback scale replicas failed for {service}")
        return {"status": "error", "detail": f"Failed to rollback {service} to {previous_count} replicas"}
    
    return scale_replicas(service, previous_count)


# ──────────────────────────────────────────────
# Restart Pod
# ──────────────────────────────────────────────

def restart_pod(service: str, pod_id: str) -> dict:
    logger.info(f"Executing restart_pod | service={service} pod_id={pod_id}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Pod restart failed for {service}/{pod_id}")
        return {"status": "error", "detail": f"Failed to restart pod {pod_id} in {service}"}
    
    print(f"[Actuator] Restarted pod {pod_id} in {service}.")
    return {"status": "success", "detail": f"Restarted pod {pod_id}."}

def rollback_restart_pod(service: str, pod_id: str) -> dict:
    logger.info("Rollback restart_pod | manual intervention required")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Manual rollback required for pod restart - {service}/{pod_id}")
        return {"status": "error", "detail": f"Manual rollback required for pod {pod_id} in {service}"}
    
    return {"status": "success", "detail": "Manual rollback completed for pod restart."}


# ──────────────────────────────────────────────
# Rollback Deployment
# ──────────────────────────────────────────────

def rollback_deployment(service: str, revision: int) -> dict:
    logger.info(f"Executing rollback_deployment | service={service} revision={revision}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.04):  # 4% failure rate
        logger.warning(f"Rollback deployment failed for {service} revision {revision}")
        return {"status": "error", "detail": f"Failed to rollback {service} to revision {revision}"}
    
    print(f"[Actuator] Rolled back {service} to revision {revision}.")
    return {"status": "success", "detail": f"Rolled back to revision {revision}."}

def rollback_rollback_deployment(service: str, revision: int) -> dict:
    logger.info(f"Rolling back the rollback | restoring {service} to newer revision {revision+1}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.04):  # 4% failure rate
        logger.warning(f"Rollback rollback deployment failed for {service} revision {revision+1}")
        return {"status": "error", "detail": f"Failed to restore {service} to revision {revision+1}"}
    
    return rollback_deployment(service, revision + 1)


# ──────────────────────────────────────────────
# Increase Memory Limit
# ──────────────────────────────────────────────

def increase_memory_limit(service: str, limit_mb: int) -> dict:
    logger.info(f"Executing increase_memory_limit | service={service} limit={limit_mb}MB")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Increase memory limit failed for {service}")
        return {"status": "error", "detail": f"Failed to increase memory limit for {service} to {limit_mb}MB"}
    
    print(f"[Actuator] Increased memory limit for {service} to {limit_mb}MB.")
    return {"status": "success", "detail": f"Memory limit set to {limit_mb}MB."}

def rollback_increase_memory_limit(service: str, limit_mb: int) -> dict:
    logger.info("Rollback increase_memory_limit | manual intervention required")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Manual rollback required for resource limits - {service}")
        return {"status": "error", "detail": f"Manual rollback required for memory limit on {service}"}
    
    return {"status": "success", "detail": "Manual rollback completed for resource limits."}


# ──────────────────────────────────────────────
# Flush Cache
# ──────────────────────────────────────────────

def flush_cache(service: str) -> dict:
    logger.info(f"Executing flush_cache | service={service}")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.02):  # 2% failure rate
        logger.warning(f"Cache flush failed for {service}")
        return {"status": "error", "detail": f"Failed to flush cache for {service}"}
    
    print(f"[Actuator] Cache flushed for {service}.")
    return {"status": "success", "detail": f"Flushed cache for {service}."}

def rollback_flush_cache(service: str) -> dict:
    logger.info("Rollback flush_cache | manual intervention required")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.02):  # 2% failure rate
        logger.warning(f"Manual rollback required for cache flush - {service}")
        return {"status": "error", "detail": f"Manual rollback required for cache flush on {service}"}
    
    return {"status": "success", "detail": "Manual rollback completed for cache flush."}


# ──────────────────────────────────────────────
# Reroute Traffic
# ──────────────────────────────────────────────

def reroute_traffic(service: str, target: str, weight_pct: int) -> dict:
    logger.info(f"Executing reroute_traffic | service={service} target={target} weight={weight_pct}%")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Reroute traffic failed for {service} to {target}")
        return {"status": "error", "detail": f"Failed to reroute {weight_pct}% of {service} traffic to {target}"}
    
    print(f"[Actuator] Rerouted {weight_pct}% of {service} traffic to {target}.")
    return {"status": "success", "detail": f"Rerouted {weight_pct}% to {target}."}

def rollback_reroute_traffic(service: str, target: str, weight_pct: int) -> dict:
    logger.info("Rollback reroute_traffic | manual intervention required")
    _simulate_delay()
    
    # Simulate occasional failures
    if _simulate_failure(0.03):  # 3% failure rate
        logger.warning(f"Manual rollback required for traffic routing - {service} to {target}")
        return {"status": "error", "detail": f"Manual rollback required for traffic routing on {service} to {target}"}
    
    return {"status": "success", "detail": "Manual rollback completed for traffic routing."}


# ──────────────────────────────────────────────
# No Action
# ──────────────────────────────────────────────

def no_action(service: str) -> dict:
    logger.info(f"Executing no_action | service={service}")
    return {"status": "success", "detail": "No infrastructure changes made."}

def rollback_no_action(service: str) -> dict:
    return {"status": "success", "detail": "No rollback needed."}


# ──────────────────────────────────────────────
# Dispatch Mapping
# ──────────────────────────────────────────────

TOOL_DISPATCHER = {
    "scale_replicas": scale_replicas,
    "restart_pod": restart_pod,
    "rollback_deployment": rollback_deployment,
    "increase_memory_limit": increase_memory_limit,
    "flush_cache": flush_cache,
    "reroute_traffic": reroute_traffic,
    "no_action": no_action,
}

ROLLBACK_DISPATCHER = {
    "scale_replicas": "rollback_scale_replicas",
    "restart_pod": "rollback_restart_pod",
    "rollback_deployment": "rollback_rollback_deployment",
    "increase_memory_limit": "rollback_increase_memory_limit",
    "flush_cache": "rollback_flush_cache",
    "reroute_traffic": "rollback_reroute_traffic",
    "no_action": "rollback_no_action",
}
