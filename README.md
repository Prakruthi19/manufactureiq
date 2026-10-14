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

**A multi-agent quality-control system for semiconductor fabs, trained on real fab data.**
A wafer comes off the line; seven agents validate it, classify its failure pattern,
explain the likely cause, check whether the problem repeats across a lot, a tool or a
material lot, alert the supplier only when the evidence points at their material, and
write an ISO 9001 corrective-action report.

> 📘 **[docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md)** explains every design decision,
> what the numbers mean, and the questions you'll be asked.
>
> 📗 **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** follows one wafer from the raw file to model input, step by step.

## Results (held-out data, evaluated once)

| Model | Data | Headline | Detail |
|---|---|---|---|
| **Defect Classifier** (XGBoost + SMOTE inside training folds) | WM-811K: 172,950 expert-labelled wafer maps | **Macro-F1 0.861** on 34,000 wafers from unseen lots | catches **90.8%** of defective wafers, **0.6%** false alarms; beat a CNN on the same split (0.840); calibrated confidence (ECE 0.010 → 0.002) |
| **Process monitor** (balanced random forest) | SECOM: 1,567 runs × 590 sensors | **ROC-AUC 0.58** on future months (PR-AUC 0.069 vs 0.055 random) | weak, reported as such, never escalates on its own |

Accuracy is not used for any decision: 86% of wafers have no defect, so "always none"
scores 86%. Splits are **lot-aware** (wafers) and **chronological** (process runs), and
the dashboard's **Model card** tab shows `models/*_metrics.json` verbatim.

Why SECOM tutorials report ~99%: SMOTE applied *before* a random split leaks synthetic
copies of test failures into training. Same model family, same data:
**1.000** (leaky) → **0.80** (random split) → **0.58** (chronological). See the guide, §5.

**In production:** confidences are temperature-calibrated (T = 1.45, fitted on validation),
so wafers below **0.81** go to a human (4.6% on test). A **drift monitor** compares live
wafers' features with the training data (PSI, no labels needed) and shows it on the Model
card. See the guide, §4.7–4.8.

---

## Architecture

```
                 ┌──────────── Tier 1 · Intake ────────────┐
 wafer map  ───► │ Data Ingestor → Sensor Parser (SPC ±3σ) │
 590 sensors     └───────────────────┬─────────────────────┘
                 ┌──────────── Tier 2 · Analysis ──────────┐
                   Defect Classifier (XGBoost, 26 wafer features)
                          │  (no defect → trend history → finalize)
                   Root Cause (similar wafers in ChromaDB + pattern knowledge + LLM)
                   Trend Detector (lot / tool / material repeats + SECOM risk)
                 └───────────────────┬─────────────────────┘
                   critical / low confidence ──► Human Review
                 ┌──────────── Tier 3 · Action ────────────┐
                   Supplier Alert (evidence-gated)  ‖  Compliance Report (ISO 9001)
                 └─────────────────────────────────────────┘
```

| # | Agent | Decides with | Output |
|---|---|---|---|
| 1 | Data Ingestor | validation rules | rejects malformed wafer maps / sensor vectors |
| 2 | Sensor Parser | statistical process control | yield; out-of-control sensors vs a normal run |
| 3 | Defect Classifier | XGBoost on `core/wafer_features.py`, calibrated | pattern, calibrated confidence, severity, human-review flag; logs features for drift |
| 4 | Root Cause | 10 nearest historical wafers (ChromaDB) + `core/defect_knowledge.py` + LLM | cause, evidence, neighbour agreement |
| 5 | Trend Detector | rules over inspection history (`core/history.py`) + SECOM model | lot / tool / material pattern, escalate |
| 6 | Supplier Alert | evidence gate, then LLM | alert only with cross-tool or material-linked evidence |
| 7 | Compliance Report | LLM | ISO 9001 CAPA document |

LLMs only explain and write; every decision is made by a model or a tested rule.

**Stack:** LangGraph · FastAPI · Celery (Redis, or a no-Docker local mode) · XGBoost ·
scikit-learn · imbalanced-learn · ChromaDB · OpenAI (`gpt-4o-mini`, override with
`OPENAI_MODEL`) · React + Vite.

---

## Local setup

> Commands are PowerShell; the `python` lines are identical on macOS / Linux.

### 1. Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend; npm install; cd ..
```

Create `.env`:
```ini
OPENAI_API_KEY=sk-...
# Either Redis (docker compose -f docker_compose.yaml up -d) ...
REDIS_URL=redis://localhost:6379/0
# ... or no Docker at all: folder queue + SQLite results
CELERY_LOCAL=1
ANONYMIZED_TELEMETRY=False
OTEL_SDK_DISABLED=true
```
If `OPENAI_API_KEY` is also set as a Windows user environment variable, that one wins
over `.env`; remove it or keep the two in sync.

### 2. Data + models (one time, ~30 min on a laptop)
```powershell
python scripts/download_data.py         # SECOM + WM-811K; verifies class counts vs the paper
python scripts/build_wafer_features.py  # 26 features + 48×48 CNN maps × 172,950 wafers
python scripts/train_models.py          # both models, similar-wafer index, demo samples
python scripts/train_wafer_cnn.py       # optional: CNN comparison on the same split (~25 min CPU)
python -m pytest tests -q               # no network / LLM needed
```
Trained models, the similar-wafer index and the demo samples are committed, so this
step can be skipped to just run the demo.

**GPU experiments (Colab):** [`notebooks/wafer_cnn_colab.ipynb`](notebooks/wafer_cnn_colab.ipynb) re-trains the CNN on a
free T4, pre-trains it self-supervised (SimCLR) on the ~638k unlabelled WM-811K wafers
(`scripts/download_unlabelled.py`, `scripts/pretrain_wafer_cnn.py`), and tests whether that helps,
with all labels and with only 10% of them. Same lot-aware split, same metrics.

### 3. Run (3 terminals)
```powershell
celery -A celery_app worker --loglevel=info --pool=solo   # worker (solo pool on Windows)
uvicorn api.main:app --port 8000                          # API
cd frontend; npm run dev                                  # http://localhost:5173
```

The dashboard has three tabs:
- **Single wafer**: a held-out wafer through all 7 agents; the expert label is revealed after the prediction.
- **Lot scenarios**: real held-out lots through the ML agents; watch lot / tool / material patterns appear.
- **Model card**: every training metric, including the imbalance-strategy comparison, confusion matrix,
  calibration (reliability chart) and live input drift.

### API
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness |
| `POST` | `/api/inspections` | submit `{inspection, sensors, full}` → `task_id` |
| `GET` | `/api/inspections/{task_id}` | status, progress, final state |
| `GET` | `/api/samples` | held-out demo wafers + lot scenarios |
| `GET` | `/api/model-card` | evaluation metrics |
| `GET` | `/api/drift` | PSI of recent live wafers vs the training data |
| `DELETE` | `/api/history` | clear inspection history |

`inspection.wafer_map` is a WM-811K grid (0 = no die, 1 = pass, 2 = fail);
`inspection.process_sensors` is 590 SECOM readings (`null` = missing).
`full: false` runs the ML-only path (no LLM calls).

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
is ephemeral — fine, since `models/`, `data/chroma/` and the demo samples ship inside
the image. Only OpenAI usage (agents 4, 6, 7 with `gpt-4o-mini`) costs anything.

### Option B — Render (managed, ~$14/mo)

A `render.yaml` blueprint is included (Redis + web + worker + static frontend).
Push to GitHub → Render → **New + → Blueprint** → pick the repo → paste
`OPENAI_API_KEY` when prompted. Note: Render has **no free tier for workers**, so
the `miq-api` and `miq-worker` services are ~$7/mo each; Redis and the static site
are free. If the API service gets a URL other than `miq-api.onrender.com`, update
the rewrite `destination` in `render.yaml`.

### Option C — laptop / single VM

Run the three terminals from [Local setup](#3-run-3-terminals) with `CELERY_LOCAL=1`
(no Docker) or against Redis. Lowest friction for a live, in-person pitch.

---

## Project layout

```
manufactureiq/
├── agents/                 # the 7 agents + LangGraph orchestrator (graph.py, dry_pipeline.py)
├── api/main.py             # FastAPI
├── core/
│   ├── state.py            # PipelineState: shared Pydantic schema
│   ├── wafer_features.py   # 26 wafer-map features (training AND serving)
│   ├── secom.py            # SECOM cleaning + SPC limits
│   ├── calibration.py      # temperature scaling + reliability / ECE
│   ├── drift.py            # PSI input-drift monitor (reference + live log)
│   ├── defect_knowledge.py # pattern → likely causes / severity / material link
│   └── history.py          # inspection history (lot / tool / material patterns)
├── scripts/                # download, features, training, index, demo samples
│   └── legacy_ai4i/        # the original AI4I version, kept for reference
├── models/                 # trained models + *_metrics.json (legacy_ai4i/ = old models)
├── data/                   # chroma/ (similar-wafer index), processed/demo_samples.json
├── tests/                  # pytest suite
├── docs/                   # INTERVIEW_GUIDE.md, DATA_PREPARATION.md
├── notebooks/              # wafer_cnn_colab.ipynb — GPU runs (CNN, self-supervised pre-training)
└── frontend/               # React + Vite dashboard
```

## Data sources
- **WM-811K**: Wu, Jang & Chen, "Wafer Map Failure Pattern Recognition and Similarity
  Ranking for Large-Scale Data Sets", *IEEE Trans. Semiconductor Manufacturing* 28(1), 2015.
  Retrieved from a Hugging Face mirror; class counts verified against the published distribution.
- **SECOM**: McCann & Johnston, UCI Machine Learning Repository, 2008.
- Tool, supplier and material-lot IDs in the demo are simulated, and the two datasets come
  from different fabs; see the guide, §3.
