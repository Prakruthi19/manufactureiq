---
title: ManufactureIQ
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<!-- The YAML block above is Hugging Face Spaces config (keep it at the very top,
     and keep `sdk: docker`). It's harmless on GitHub. -->

# ManufactureIQ

**An agentic quality-control platform for manufacturing.** ManufactureIQ ingests inspection records and sensor telemetry, then runs them through a 7-agent pipeline that classifies defects, finds root causes, detects batch/machine trends, alerts suppliers, and auto-generates ISO 9001 compliance reports — pausing for human sign-off on critical defects.

Built on **LangGraph** for orchestration, **Celery + Redis** for async processing, **XGBoost + Isolation Forest** for ML, **ChromaDB + GPT-4o** for retrieval-augmented root-cause analysis, and a **React/Vite** live dashboard.

---

## Architecture

Each inspection record flows through a LangGraph state machine as an independent Celery task, so many records process in parallel.

```
                        ┌─────────── Tier 1 · Intake ───────────┐
  inspection + sensors → │  Data Ingestor → Sensor Parser        │
                        └───────────────────┬───────────────────┘
                        ┌─────────── Tier 2 · Analysis ──────────┐
                          Defect Classifier (XGBoost)
                                 │  (no defect ──────────────► finalize)
                          Root Cause Analyzer (RAG + GPT-4o)
                          Trend Detector (Isolation Forest)
                        └───────────────────┬───────────────────┘
                          critical/high? ──► Human Review (pause) ──► approve/reject
                        ┌─────────── Tier 3 · Action ────────────┐
                          Supplier Alert  ‖  Compliance Report   (run concurrently)
                        └───────────────────┬───────────────────┘
                                         finalize
```

| # | Agent | Tier | What it does |
|---|-------|------|--------------|
| 1 | Data Ingestor | Intake | Validates & structures the raw inspection payload |
| 2 | Sensor Parser | Intake | Derives metrics, flags failure risks |
| 3 | Defect Classifier | Analysis | XGBoost multiclass defect prediction (99% test acc.) |
| 4 | Root Cause Analyzer | Analysis | ChromaDB retrieval of similar past defects + GPT-4o causal analysis |
| 5 | Trend Detector | Analysis | Isolation Forest anomaly detection across batches/machines |
| 6 | Supplier Alert | Action | Notifies the supplier when a material defect is detected |
| 7 | Compliance Report | Action | Generates an ISO 9001:2015 corrective/preventive report |

### Tech stack

- **Orchestration:** LangGraph (conditional routing, checkpointing, human-in-the-loop interrupts)
- **API:** FastAPI (`api/main.py`) — submit / poll / (approve)
- **Workers:** Celery, with Redis as broker + result backend
- **ML:** scikit-learn (Isolation Forest), XGBoost — trained on the AI4I 2020 dataset (`models/`)
- **RAG:** ChromaDB + `all-MiniLM-L6-v2` embeddings, GPT-4o for reasoning
- **Frontend:** React 18 + Vite live dashboard (`frontend/`)

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for the dashboard)
- **Docker Desktop** (for local Redis) — or an [Upstash](https://upstash.com) Redis URL
- An **OpenAI API key** (used by agents 4–7)

---

## Local setup

> Runs on Windows (PowerShell), macOS, or Linux. Commands below use PowerShell; the `python`/`pip` invocations are identical on other shells.

### 1. Clone & create a virtualenv

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` in the project root:

```ini
# LLM
OPENAI_API_KEY=sk-...

# Redis (local Docker default)
REDIS_URL=redis://localhost:6379/0
# For Upstash cloud instead, use the TLS scheme:
# REDIS_URL=rediss://default:<token>@<endpoint>:6379

# ChromaDB persistence (keep the model cache off the system drive if you like)
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_CACHE_DIR=./.cache_home

# Silence library telemetry (prevents shutdown crashes in Celery workers)
ANONYMIZED_TELEMETRY=False
CREWAI_DISABLE_TELEMETRY=true
OTEL_SDK_DISABLED=true
LITELLM_TELEMETRY=False
```

### 3. Start Redis

```powershell
docker compose -f docker_compose.yaml up -d
```

### 4. (One-time) train models & seed the vector store

The repo already ships trained models in `models/` and a seeded `data/chroma/`, so you can **skip this** for a demo. To regenerate from scratch:

```powershell
python scripts/train_models.py     # → models/*.pkl + metadata.json
python scripts/seed_chromadb.py    # loads defect history into ChromaDB (run once)
```

### 5. Run the stack (4 terminals)

```powershell
# Terminal 1 — Celery worker
celery -A celery_app worker --loglevel=info --concurrency=4
#   On Windows, add:  --pool=solo   (or --pool=threads)

# Terminal 2 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 3 — Frontend dashboard
cd frontend
npm install
npm run dev            # opens http://localhost:5173

# Terminal 4 (optional) — Celery task monitor
celery -A celery_app flower      # http://localhost:5555
```

Open **http://localhost:5173** — the dashboard proxies `/api` to the backend on `:8000` (see `frontend/vite.config.js`).

### Quick sanity check (no frontend needed)

```powershell
python scripts/run_pipeline.py     # runs one record through all 7 agents, prints results
```

---

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/api/health` | Liveness |
| `POST` | `/api/inspections` | Submit an inspection → returns `task_id` |
| `GET`  | `/api/inspections/{task_id}` | Poll status + final result |

Submit body:

```json
{
  "inspection": { "record_id": "REC-001", "batch_id": "BATCH-0146", "machine_id": "MCH-002", "...": "..." },
  "sensors":    [ { "sensor_id": "MCH-002-S1", "temperature_c": 26.85, "machine_speed_rpm": 2800 } ],
  "full": true
}
```

- `full: true` → the full 7-agent LangGraph pipeline.
- `full: false` → the fast **dry pipeline** (ingestor → sensor_parser → classifier), used for bulk CSV batches so they don't fire N× LLM calls.

---

## Deploy

### Option A — Hugging Face Spaces (free, recommended for a demo)

The whole backend runs in **one free Docker Space**: Redis, the Celery worker, and
FastAPI (which also serves the built React dashboard) all live in a single
container — no external services, **$0 hosting**. The `Dockerfile`, `start.sh`, and
the HF config block at the top of this README make it work out of the box.

1. Create a Space → huggingface.co → **New Space** → **Docker** (blank) → note the git URL.
2. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
3. In the Space → **Settings → Variables and secrets**, add a secret
   **`OPENAI_API_KEY`**. (Everything else is baked into the image.)
4. The Space builds (~10 min the first time — it installs the ML stack) and boots.
   Open the Space URL: the dashboard and API are both there.

Notes: the free Space **sleeps after ~48h idle** (wakes on next visit) and storage
is ephemeral — fine, since `models/` and `data/chroma/` ship inside the image and the
embedding model downloads to `/tmp` on first root-cause call. Only OpenAI usage
(agents 4–7, a few cents per inspection) costs anything.

### Option B — Render (managed, ~$14/mo)

A `render.yaml` blueprint is included (Redis + web + worker + static frontend).
Push to GitHub → Render → **New + → Blueprint** → pick the repo → paste
`OPENAI_API_KEY` when prompted. Note: Render has **no free tier for workers**, so
the `miq-api` and `miq-worker` services are ~$7/mo each; Redis and the static site
are free. If the API service gets a URL other than `miq-api.onrender.com`, update
the rewrite `destination` in `render.yaml`.

### Option C — laptop / single VM

Run the four local terminals from [Local setup](#5-run-the-stack-4-terminals)
against local Docker Redis (or Upstash). Lowest friction for a live, in-person pitch.

---

## Project layout

```
manufactureiq/
├── agents/              # the 7 agents + LangGraph orchestrator
│   ├── ingestor/  classifier/  root_cause/  trend/  supplier/  complaince/
│   └── orchestrator/    # graph.py (LangGraph), dry_pipeline.py
├── api/main.py          # FastAPI REST layer
├── tasks/               # Celery task wrapping the pipeline
├── core/state.py        # Pydantic PipelineState — shared schema for all agents
├── celery_app.py        # Celery config (Redis broker/backend)
├── models/              # trained XGBoost + Isolation Forest + encoders
├── data/                # AI4I 2020 dataset, sample payloads, ChromaDB store
├── scripts/             # train_models.py, seed_chromadb.py, run_pipeline.py
├── frontend/            # React + Vite dashboard
└── docker_compose.yaml  # local Redis
```

---

## Notes

- **Human-in-the-loop:** critical/high-severity defects pause the graph before Tier 3. In async mode the Celery task runs non-interactively (`interrupt=False`); interactive review via `resume_graph()` is exposed for the interactive/CLI path.
- **Windows + Celery:** the default `prefork` pool can misbehave; use `--pool=solo` or `--pool=threads`.
- **ChromaDB cache:** `scripts/seed_chromadb.py` redirects the embedding-model download to `CHROMA_CACHE_DIR` to keep it off the system drive and avoid worker-shutdown crashes.
</content>
</invoke>
