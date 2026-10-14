"""
ManufactureIQ — Celery Worker
==============================
Connects to Redis and defines the async task queue.
Each inspection record is processed as an independent Celery task,
allowing multiple workers to run the 7-agent pipeline in parallel.
 
Usage:
    # Start worker (from project root)
    celery -A celery_app worker --loglevel=info --concurrency=4
 
    # Monitor tasks (optional)
    celery -A celery_app flower
"""
 
import os
from celery import Celery
from dotenv import load_dotenv
 
load_dotenv()
 
# ── Redis connection ──────────────────────────────────────────────────────────
# Uses local Docker Redis by default.
# In production, swap REDIS_URL in .env to your Upstash URL.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# CELERY_LOCAL=1 → no Redis needed: a folder-based broker + SQLite result store
# under data/celery_local/. Handy for a local demo when Docker isn't available.
CELERY_LOCAL = os.getenv("CELERY_LOCAL", "").lower() in ("1", "true", "yes")
if CELERY_LOCAL:
    _local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "celery_local")
    _queue_dir = os.path.join(_local_dir, "queue")
    os.makedirs(_queue_dir, exist_ok=True)
    BROKER_URL = "filesystem://"
    BACKEND_URL = "db+sqlite:///" + os.path.join(_local_dir, "results.sqlite").replace("\\", "/")
else:
    BROKER_URL = BACKEND_URL = REDIS_URL

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "manufactureiq",
    broker=BROKER_URL,      # Redis (or local folder) as message broker
    backend=BACKEND_URL,    # Redis (or local SQLite) stores task results
    include=[
        "tasks.inspection_task",   # registered tasks (we add these next)
    ]
)
 
# ── Configuration ─────────────────────────────────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
 
    # Timezone
    timezone="UTC",
    enable_utc=True,
 
    # Task behavior
    task_track_started=True,        # marks task as STARTED when worker picks it up
    task_acks_late=True,            # task only acknowledged after completion (safer)
    worker_prefetch_multiplier=1,   # each worker takes one task at a time
 
    # Result expiry — keep results in Redis for 24 hours
    result_expires=86400,
 
    # Retry policy for failed tasks
    task_max_retries=3,
    task_default_retry_delay=5,     # seconds between retries
 
    # Rate limiting — the dry pipeline is local/fast so no throttle is needed.
    # Re-enable (e.g. "4/m") once the OpenAI-backed agents are wired in Step 3.
    # task_annotations={
    #     "tasks.inspection_task.run_pipeline": {"rate_limit": "4/m"}
    # }
)
 
if CELERY_LOCAL:
    celery_app.conf.broker_transport_options = {
        "data_folder_in": _queue_dir,
        "data_folder_out": _queue_dir,
        "control_folder": os.path.join(_local_dir, "control"),   # else kombu writes ./control
    }

if __name__ == "__main__":
    celery_app.start()
 