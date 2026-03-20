"""
learning.py — Reinforcement learning engine for adaptive remediation.

Translates incident state into an 8-dimensional feature vector, selects
actions via ε-greedy policy, and records experiences for Q-learning updates.

UPGRADE: replace linear _policy_weights with torch nn.Sequential
UPGRADE: use torch.optim.Adam + loss.backward() for deep Q-learning
"""

from __future__ import annotations

import collections
import logging
import random
import os
import numpy as np
from typing import Any

from autonomous_sre.core.state import IncidentState, Severity
from autonomous_sre.core.config import settings

logger = logging.getLogger("sre_learning")
logger.setLevel(logging.INFO)

ACTION_SPACE = [
    "scale_replicas", 
    "restart_pod", 
    "rollback_deployment",
    "increase_memory_limit", 
    "flush_cache", 
    "reroute_traffic",
    "no_action"
]


def encode_state(incident: IncidentState) -> np.ndarray:
    """
    Extracts an 8-dimensional feature vector from the IncidentState.
    Features are: [cpu_pct, mem_pct, latency_ms, severity_encoded,
                   rag_match_score, deploy_age_hours, error_rate, active_alerts]
    All values are normalised to the [0.0, 1.0] range.
    """
    metrics = incident.metrics_snapshot

    # 1. CPU (assumed 0-100)
    cpu = min(metrics.get("cpu_pct", 0.0) / 100.0, 1.0)
    
    # 2. Memory (assumed 0-100)
    mem = min(metrics.get("mem_pct", 0.0) / 100.0, 1.0)
    
    # 3. Latency (assumed max sensible 5000ms)
    lat = min(metrics.get("latency_ms", 0.0) / 5000.0, 1.0)
    
    # 4. Severity (LOW=0.0, HIGH=0.5, CRITICAL=1.0)
    sev_map = {Severity.LOW: 0.0, Severity.HIGH: 0.5, Severity.CRITICAL: 1.0}
    sev = sev_map.get(incident.severity, 0.0)
    
    # 5. RAG match score (proxy based on whether guides were found)
    rag = 1.0 if incident.rag_context else 0.0
    
    # 6. Deploy age (proxy max 168 hours = 1 week)
    deploy_age = min(metrics.get("deploy_age_hours", 2.0) / 168.0, 1.0)
    
    # 7. Error rate (proxy max 100%)
    err = min(metrics.get("error_rate", 0.0) / 100.0, 1.0)
    
    # 8. Active alerts (proxy max 20)
    alerts = min(metrics.get("active_alerts", 1.0) / 20.0, 1.0)

    vec = np.array([cpu, mem, lat, sev, rag, deploy_age, err, alerts], dtype=np.float32)
    return vec


class LearningEngine:
    """RL Engine managing policy weights and experience replay."""

    WEIGHTS_PATH = "sre_policy_weights.npy"

    def __init__(self, state_dim: int = 8, n_actions: int | None = None) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions or len(ACTION_SPACE)
        self._policy_weights = np.random.randn(self.n_actions, self.state_dim) * 0.01
        self._replay_buffer = collections.deque(maxlen=settings.rl_replay_buffer_size)
        
        # Load existing weights or apply warm-start
        self.load_weights()

    def _apply_warm_start_bias(self) -> None:
        """
        Apply warm-start heuristic bias to action weights on the CPU feature (dimension 0).
        This prevents argmax from always selecting no_action during cold-start.
        """
        WARM_START_BIAS = np.array([
            0.80,   # scale_replicas      — high CPU default
            0.60,   # restart_pod
            0.50,   # rollback_deployment
            0.40,   # increase_memory_limit
            0.30,   # flush_cache
            0.30,   # reroute_traffic
            0.05,   # no_action — deprioritised for non-LOW severity
        ], dtype=np.float32)
        self._policy_weights[:, 0] = WARM_START_BIAS
        logger.info("LearningEngine | Applied warm-start bias to policy weights (CPU feature dimension)")

    def select_action(self, state_vec: np.ndarray, epsilon: float = 0.1) -> str:
        """ε-greedy action selection."""
        if random.random() < epsilon:
            idx = random.randint(0, self.n_actions - 1)
            logger.info(f"select_action | explored random action index {idx}")
        else:
            q_values = self._policy_weights @ state_vec
            idx = int(np.argmax(q_values))
            logger.info(f"select_action | exploited max Q-value index {idx} (max Q={q_values[idx]:.3f})")
        
        return ACTION_SPACE[idx]

    def get_confidence(self, state_vec: np.ndarray, action: str) -> float:
        """
        Derives a proxy confidence score [0, 1] for the chosen action 
        based on softmax over Q-values.
        """
        q_values = self._policy_weights @ state_vec
        # Stable softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        idx = ACTION_SPACE.index(action)
        return float(probs[idx])

    def store_experience(self, s: np.ndarray, a: str, r: float, s_next: np.ndarray) -> None:
        """Append to replay buffer."""
        action_idx = ACTION_SPACE.index(a)
        self._replay_buffer.append((s, action_idx, r, s_next))
        logger.info(f"store_experience | stored (r={r:.2f}). Buffer size: {len(self._replay_buffer)}")

    def update_policy(
        self,
        batch_size: int = 32,
        gamma: float | None = None,
        lr: float | None = None,
    ) -> None:
        """Sample mini-batch from replay buffer and apply Temporal Difference update."""
        gamma_val = gamma if gamma is not None else settings.rl_discount_factor
        lr_val = lr if lr is not None else settings.rl_learning_rate

        actual_batch = min(batch_size, len(self._replay_buffer))
        if actual_batch < 4:
            logger.warning(f"update_policy skipped | need >= 4 samples, have {actual_batch}")
            return

        batch = random.sample(list(self._replay_buffer), actual_batch)
        
        total_td_err = 0.0
        for s, a_idx, r, s_next in batch:
            # Q(s, a)
            q_sa = np.dot(self._policy_weights[a_idx], s)
            
            # max(Q(s', a'))
            q_s_next = self._policy_weights @ s_next
            max_q_next = np.max(q_s_next)
            
            # TD Error
            target = r + gamma_val * max_q_next
            td_error = target - q_sa
            total_td_err += td_error

            # Gradient step
            self._policy_weights[a_idx] += lr_val * td_error * s

        avg_td = total_td_err / actual_batch
        logger.info(f"update_policy | updated weights on batch_size={actual_batch} | avg_td_error={avg_td:.4f}")
        
        # Persist weights after every update
        self.save_weights()

    def load_weights(self) -> None:
        """
        Load policy weights from disk if they exist, otherwise apply warm-start bias.
        Called on initialization to enable training continuity across restarts.
        """
        if os.path.exists(self.WEIGHTS_PATH):
            try:
                self._policy_weights = np.load(self.WEIGHTS_PATH)
                logger.info(f"LearningEngine | Loaded existing policy weights from {self.WEIGHTS_PATH}")
                return
            except Exception as e:
                logger.warning(f"LearningEngine | Failed to load weights: {e}. Applying warm-start bias.")
        
        # First run: apply warm-start bias
        self._apply_warm_start_bias()

    def save_weights(self) -> None:
        """
        Save policy weights to disk for training continuity across restarts.
        Called after every policy update.
        """
        try:
            np.save(self.WEIGHTS_PATH, self._policy_weights)
            logger.info(f"LearningEngine | Policy weights saved to {self.WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"LearningEngine | Failed to save weights: {e}")

    def calculate_reward(self, outcome: str, human_feedback: float) -> float:
        """
        Weighted sum: 0.7 * outcome_score + 0.3 * human_feedback
        """
        if outcome == "resolved":
            outcome_score = 1.0
        elif outcome == "escalated":
            outcome_score = -0.5
        elif outcome == "worsened":
            outcome_score = -1.0
        else:
            outcome_score = 0.0
            
        reward = (0.7 * outcome_score) + (0.3 * human_feedback)
        logger.info(f"calculate_reward | outcome='{outcome}' feedback={human_feedback:.2f} -> reward={reward:.3f}")
        return reward
