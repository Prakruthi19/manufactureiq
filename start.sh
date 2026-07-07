#!/usr/bin/env bash
# Boots the whole ManufactureIQ backend in one container:
#   Redis (broker) → Celery worker → FastAPI (serves API + React dashboard).
set -e

echo "→ starting redis (in-container broker)..."
# --save "" / --appendonly no: it's just a job queue, no need to persist to disk.
# --dir /tmp: the only reliably writable dir on HF Spaces.
redis-server --dir /tmp --save "" --appendonly no --port 6379 &

echo "→ waiting for redis..."
until redis-cli ping >/dev/null 2>&1; do sleep 0.3; done
echo "✓ redis up"

echo "→ starting celery worker..."
celery -A celery_app worker --loglevel=info --concurrency=2 &

echo "→ starting api + dashboard on :7860..."
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
