# Autonomous SRE Agent

A production-oriented MVP for autonomous incident analysis and remediation using a layered architecture, LangGraph orchestration, RAG context retrieval, RL-based action selection, Human-in-the-Loop (HITL) approval, and full audit persistence.

## 1. Repository Structure

```text
autonomous_sre/
├── autonomous_sre/
│   ├── core/
│   │   ├── config.py            # Runtime and model configuration
│   │   └── state.py             # Pydantic models and AgentState contract
│   ├── services/
│   │   ├── telemetry.py         # Multi-service telemetry simulation
│   │   ├── rag.py               # Knowledge retrieval service (FAISS + embeddings)
│   │   └── learning.py          # RL engine, warm start, policy persistence
│   ├── infrastructure/
│   │   ├── tools.py             # Idempotent remediation/rollback adapters
│   │   ├── persistence.py       # SQLite persistence layer
│   │   ├── audit.py             # Append-only JSONL audit logger
│   │   ├── approval_bus.py      # In-memory proposal approval queue
│   │   └── incident_store.py    # Thread-safe store abstraction (compat utility)
│   ├── orchestration/
│   │   └── graph.py             # LangGraph node pipeline and routing
│   └── interfaces/
│       ├── api.py               # FastAPI + dashboard endpoints
│       ├── main.py              # CLI live run + warmup training
│       ├── scenarios.py         # Deterministic scenario injection CLI
│       └── simulate_prod.py     # Production-like traffic simulator
├── dashboard/
│   └── index.html               # Live dashboard UI
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 2. System Architecture (Layered)

### 2.1 Core Layer
- Defines the canonical domain model and configuration contract.
- No infrastructure side-effects.
- Files:
    - `autonomous_sre/core/state.py`
    - `autonomous_sre/core/config.py`

### 2.2 Service Layer
- Stateless/algorithmic processing services.
- Files:
    - `autonomous_sre/services/telemetry.py`
    - `autonomous_sre/services/rag.py`
    - `autonomous_sre/services/learning.py`

### 2.3 Infrastructure Layer
- Side-effecting adapters: storage, audit, and action execution.
- Files:
    - `autonomous_sre/infrastructure/persistence.py`
    - `autonomous_sre/infrastructure/audit.py`
    - `autonomous_sre/infrastructure/tools.py`
    - `autonomous_sre/infrastructure/approval_bus.py`

### 2.4 Orchestration Layer
- Business workflow and control flow.
- LangGraph nodes, confidence gating, HITL, routing, reward updates.
- File:
    - `autonomous_sre/orchestration/graph.py`

### 2.5 Interface Layer
- Operator-facing and integration-facing entry points.
- Files:
    - `autonomous_sre/interfaces/api.py`
    - `autonomous_sre/interfaces/main.py`
    - `autonomous_sre/interfaces/scenarios.py`
    - `autonomous_sre/interfaces/simulate_prod.py`

## 3. Runtime Flow

1. Telemetry collected/injected.
2. Analyzer detects anomaly and severity.
3. Researcher enriches context using RAG.
4. RL predictor proposes remediation action.
5. Proposer computes confidence and rollback plan.
6. HITL decides approve/reject/escalate.
7. Tool executes or escalates.
8. Experience, metrics, and audit are persisted.
9. Policy updates and weights are saved.

## 4. Quick Start

### 4.1 Install

```bash
pip install -r requirements.txt
```

### 4.2 CLI Run

```bash
python -m autonomous_sre.interfaces.main
```

### 4.3 API + Dashboard

```bash
uvicorn autonomous_sre.interfaces.api:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000`

### 4.4 Scenario Injection

```bash
python -m autonomous_sre.interfaces.scenarios --scenario cpu_spike
python -m autonomous_sre.interfaces.scenarios --scenario memory_leak --runs 10
```

### 4.5 Docker (Single File)

Build image:

```bash
docker build -t autonomous-sre:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 --name autonomous-sre autonomous-sre:latest
```

Open:
- `http://127.0.0.1:8000`

## 5. Production Integration Guide

### 5.1 Replace Simulated Telemetry
- Replace simulator input with real telemetry/webhook ingestion.
- Change in:
    - `autonomous_sre/interfaces/api.py` (triggering and ingestion APIs)
    - `autonomous_sre/services/telemetry.py` (if simulator retained for fallback)

### 5.2 Replace Simulated Actuators
- Replace stub actions in `tools.py` with real infrastructure SDK/CLI calls.
- Change in:
    - `autonomous_sre/infrastructure/tools.py`

### 5.3 Persistence Hardening
- Move SQLite to PostgreSQL for multi-instance deployment.
- Change in:
    - `autonomous_sre/infrastructure/persistence.py`

### 5.4 Security Hardening
- Add authn/authz to mutation endpoints.
- Change in:
    - `autonomous_sre/interfaces/api.py`
- Externalize secrets/env config.
- Change in:
    - `autonomous_sre/core/config.py`

### 5.5 Model/RAG Upgrades
- Swap fake/local embeddings to enterprise embeddings provider.
- Change in:
    - `autonomous_sre/services/rag.py`
- Upgrade linear Q to deep RL (e.g., DQN).
- Change in:
    - `autonomous_sre/services/learning.py`

## 6. Replication and Scaling

### 6.1 Single-Node Demo (Current)
- In-memory approval queue (`approval_bus.py`).
- SQLite persistence (`sre_agent.db`).

### 6.2 Multi-Instance Production
- Replace in-memory approval queue with Redis/Kafka-backed coordination.
- Replace SQLite with managed Postgres.
- Use centralized log sink for `audit.log` shipping.
- Run API behind a load balancer.
- Persist graph checkpoints in shared durable store if required.

## 7. Change Map: Where to Modify What

- Detection threshold logic:
    - `autonomous_sre/orchestration/graph.py` (analyzer node)
    - `autonomous_sre/core/config.py` (threshold values)
- RL action space, rewards, exploration:
    - `autonomous_sre/services/learning.py`
- HITL threshold and behavior:
    - `autonomous_sre/orchestration/graph.py`
    - env var `HITL_THRESHOLD`
- API contracts/endpoints:
    - `autonomous_sre/interfaces/api.py`
- Dashboard behavior/refresh/layout:
    - `dashboard/index.html`
- Database schema and reporting:
    - `autonomous_sre/infrastructure/persistence.py`
- Audit schema and compliance event payloads:
    - `autonomous_sre/infrastructure/audit.py`
- Tool execution and rollback logic:
    - `autonomous_sre/infrastructure/tools.py`

## 8. Operational Artifacts

Runtime-generated files:
- `sre_agent.db`
- `sre_policy_weights.npy`
- `audit.log`

## 9. Environment Parameters

Important runtime controls:
- `HITL_THRESHOLD` (default `0.60` demo; `0.75` production suggestion)
- `API_MODE` (enabled automatically by API interface)

Additional parameters are in:
- `autonomous_sre/core/config.py`
