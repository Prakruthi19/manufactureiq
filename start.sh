#!/usr/bin/env bash
# Boots the whole ManufactureIQ backend in one container:
#   Redis (broker) → Celery worker → FastAPI (serves API + React dashboard).
set -e

# Keep ML libs single-threaded on the small (2 vCPU) box — avoids thread thrash
# and any lingering OpenMP fork hazards.
export OMP_NUM_THREADS=1

echo "→ starting redis (in-container broker)..."
# --save "" / --appendonly no: it's just a job queue, no need to persist to disk.
# --dir /tmp: the only reliably writable dir on HF Spaces.
redis-server --dir /tmp --save "" --appendonly no --port 6379 &

echo "→ waiting for redis..."
until redis-cli ping >/dev/null 2>&1; do sleep 0.3; done
echo "✓ redis up"

echo "→ starting celery worker..."
# IMPORTANT: --pool=solo (no fork). The default 'prefork' pool fork()s workers,
# and XGBoost/sklearn's OpenMP threads deadlock across a fork on Linux — the
# pipeline would hang forever at the classifier. solo runs one task at a time in
# a single process (fine for a demo). Switch to '--pool=threads --concurrency=4'
# if you need concurrent inspections.
celery -A celery_app worker --loglevel=info --pool=solo &

echo "→ starting api + dashboard on :7860..."
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
