"""
ManufactureIQ — Inline task runner (TASK_MODE=inline)
======================================================
Runs the pipeline inside the API process via FastAPI BackgroundTasks — no Celery
worker needed. Used for the single free Render instance.

- Jobs run strictly one at a time (global lock) to stay within OpenAI rate limits.
- Status is written to Redis (key miq:task:{id}, 24h TTL) in the same shape the
  Celery path exposes — state PENDING | PROGRESS | SUCCESS | FAILURE plus
  progress/result/error — so GET /api/inspections/{id} and the dashboard work
  unchanged. Falls back to an in-process dict if Redis isn't reachable.
"""

import json
import logging
import os
import threading
import traceback

log = logging.getLogger("manufactureiq.inline")

_KEY = "miq:task:{}"
_TTL = 86400  # match Celery result_expires

_run_lock = threading.Lock()      # serialises pipeline runs
_mem: dict[str, dict] = {}        # fallback store
_mem_lock = threading.Lock()
_redis = None
_redis_checked = False


def _client():
    """Return a connected Redis client, or None to use the in-memory fallback."""
    global _redis, _redis_checked
    if not _redis_checked:
        _redis_checked = True
        try:
            import redis
            c = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                     socket_connect_timeout=2, socket_timeout=5)
            c.ping()
            _redis = c
            log.info("inline runner: storing task status in Redis")
        except Exception as e:  # noqa: BLE001
            log.warning("inline runner: Redis unavailable (%s) — using in-memory status store", e)
    return _redis


def _save(task_id: str, record: dict) -> None:
    c = _client()
    if c is not None:
        try:
            c.set(_KEY.format(task_id), json.dumps(record), ex=_TTL)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("inline runner: Redis write failed (%s) — falling back to memory", e)
    with _mem_lock:
        _mem[task_id] = record


def get_status(task_id: str) -> dict:
    """Status record for a task; unknown ids read as PENDING (same as Celery)."""
    c = _client()
    if c is not None:
        try:
            raw = c.get(_KEY.format(task_id))
            if raw:
                return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("inline runner: Redis read failed (%s)", e)
    with _mem_lock:
        return _mem.get(task_id, {"state": "PENDING"})


def mark_queued(task_id: str) -> None:
    _save(task_id, {"state": "PENDING"})


def run_job(task_id: str, inspection: dict, sensors: list, full: bool) -> None:
    """BackgroundTasks entry point — runs one record end to end, one job at a time."""
    from tasks.inspection_task import execute_pipeline

    def report(state: str, meta: dict) -> None:
        _save(task_id, {"state": state, "progress": meta})

    with _run_lock:
        try:
            result = execute_pipeline(inspection, sensors, full, report=report)
            _save(task_id, {"state": "SUCCESS", "result": result})
        except Exception as e:  # noqa: BLE001
            log.error("inline task %s failed:\n%s", task_id, traceback.format_exc())
            _save(task_id, {"state": "FAILURE", "error": f"{type(e).__name__}: {e}"})
