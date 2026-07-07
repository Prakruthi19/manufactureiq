# ManufactureIQ — single-container image for Hugging Face Spaces (free tier).
# One container runs Redis + Celery worker + FastAPI (which also serves the
# built React dashboard). No external services, no paid add-ons.

# ── Stage 1: build the React dashboard ──────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # → /fe/dist

# ── Stage 2: Python runtime + in-container Redis ────────────────────────────
FROM python:3.11-slim

# redis-server runs locally as the Celery broker; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        redis-server curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as UID 1000 — create that user so files are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    REDIS_URL=redis://localhost:6379/0 \
    CHROMA_CACHE_DIR=/tmp/chroma_cache \
    ANONYMIZED_TELEMETRY=False \
    OTEL_SDK_DISABLED=true \
    CREWAI_DISABLE_TELEMETRY=true \
    LITELLM_TELEMETRY=False

WORKDIR /home/user/app

# Python deps first for layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# App code (includes bundled models/ and data/chroma/) + built frontend.
COPY --chown=user . .
COPY --chown=user --from=frontend /fe/dist ./frontend/dist

# HF Spaces expect the app on port 7860.
EXPOSE 7860
CMD ["bash", "start.sh"]
