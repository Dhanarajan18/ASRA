"""
config.py — Centralised settings for the Autonomous SRE Agent.

All configuration is loaded from environment variables (or a .env file)
via pydantic-settings for type-safe, validated access.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide configuration."""

    # ── Runtime ──
    environment: str = Field(default="dev", description="Runtime environment: dev/staging/prod")
    log_level: str = Field(default="INFO", description="Application log level")

    # ── API Security (MVP-friendly, optional) ──
    api_key_enabled: bool = Field(default=False, description="Enable API key auth for mutation endpoints")
    api_key: str = Field(default="", description="Shared API key when auth is enabled")
    api_key_header: str = Field(default="x-api-key", description="Header name containing the API key")
    
    # Security settings
    secret_key: str = Field(default="", description="Secret key for cryptographic operations")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time in minutes")

    # ── LLM ──
    openai_api_key: str = Field(default="sk-placeholder", description="OpenAI API key")
    openai_model_name: str = Field(default="gpt-4o", description="Model to use for LLM calls")

    # ── Anomaly Detection ──
    anomaly_cpu_threshold: float = Field(default=85.0, description="CPU % above which an anomaly is flagged")
    anomaly_memory_threshold: float = Field(default=90.0, description="Memory % above which an anomaly is flagged")
    anomaly_error_rate_threshold: float = Field(default=5.0, description="Error-rate % threshold for anomaly detection")

    # ── RL Engine ──
    rl_learning_rate: float = Field(default=0.001, description="Learning rate for the RL policy network")
    rl_discount_factor: float = Field(default=0.99, description="Discount factor (gamma) for future rewards")
    rl_replay_buffer_size: int = Field(default=10_000, description="Max experiences in the replay buffer")
    rl_epsilon: float = Field(default=0.1, description="Exploration probability for epsilon-greedy policy")

    # ── Human-in-the-loop ──
    approval_confidence_threshold: float = Field(default=0.75, description="Confidence threshold requiring explicit approval")

    # ── RAG / Embeddings ──
    rag_provider: str = Field(default="ollama", description="Embedding provider: ollama, openai, or fake")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server base URL")
    ollama_model: str = Field(default="nomic-embed-text", description="Ollama embedding model name")
    embedding_dim: int = Field(default=384, description="Embedding dimension (384 for nomic-embed-text, 1536 for OpenAI)")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance — import this everywhere
settings = Settings()
