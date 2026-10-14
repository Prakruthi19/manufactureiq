"""
ManufactureIQ — Inspection history (SQLite)
============================================
Every classified wafer is recorded here. The Trend Detector reads it back to
answer questions a single wafer can't:

    lot      — are other wafers in THIS LOT showing the same pattern?
    tool     — has THIS TOOL produced the same pattern on other lots recently?
    material — is the same MATERIAL LOT failing on SEVERAL tools?
               (several tools, one material = the material is the common factor)

Why our own history instead of the datasets?
    WM-811K and SECOM have no tool IDs or supplier lots. Rather than invent
    that history, the system builds it from what it actually inspects — the
    same way a fab's MES/yield database works.

SQLite: zero setup, one file, fine for one worker. A multi-worker deployment
would point this at Postgres.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.getenv("HISTORY_DB") or
               Path(__file__).resolve().parent.parent / "data" / "history.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    record_id      TEXT PRIMARY KEY,
    inspected_at   TEXT NOT NULL,
    lot            TEXT,
    tool           TEXT,
    supplier       TEXT,
    material_batch TEXT,
    pattern        TEXT NOT NULL,
    defect         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lot  ON inspections(lot);
CREATE INDEX IF NOT EXISTS ix_tool ON inspections(tool, inspected_at);
CREATE INDEX IF NOT EXISTS ix_mat  ON inspections(material_batch);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.executescript(_SCHEMA)
    return c


def record(record_id: str, lot: str, tool: str, supplier: str | None,
           material_batch: str | None, pattern: str) -> None:
    """Insert or update one inspected wafer (re-running a record replaces it)."""
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?,?,?,?)",
            (record_id, datetime.utcnow().isoformat(), lot, tool, supplier,
             material_batch, pattern, int(pattern != "none")),
        )


def lot_stats(lot: str, pattern: str) -> tuple[int, int]:
    """(wafers seen in this lot, of which show `pattern`)."""
    with closing(_conn()) as c:
        return c.execute(
            "SELECT COUNT(*), COALESCE(SUM(pattern = ?), 0) FROM inspections WHERE lot = ?",
            (pattern, lot),
        ).fetchone()


def tool_recent(tool: str, pattern: str, last_n: int = 25) -> tuple[int, int, list[str]]:
    """Among the tool's last `last_n` wafers: (count with `pattern`, distinct lots, those lots)."""
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT lot, pattern FROM inspections WHERE tool = ? ORDER BY inspected_at DESC LIMIT ?",
            (tool, last_n),
        ).fetchall()
    lots = sorted({lot for lot, p in rows if p == pattern})
    return sum(p == pattern for _, p in rows), len(lots), lots


def material_stats(material_batch: str) -> dict:
    """
    Defects on this material lot: how many tools AND how many production lots
    they span, plus the defect rate on every OTHER material lot as a baseline.
    """
    with closing(_conn()) as c:
        n, defects, tools, lots = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(defect), 0), "
            "COUNT(DISTINCT CASE WHEN defect = 1 THEN tool END), "
            "COUNT(DISTINCT CASE WHEN defect = 1 THEN lot END) "
            "FROM inspections WHERE material_batch = ?",
            (material_batch,),
        ).fetchone()
        base_n, base_def = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(defect), 0) FROM inspections "
            "WHERE material_batch IS NOT ? ", (material_batch,),
        ).fetchone()
    return {
        "wafers": n, "defects": defects, "tools_with_defects": tools, "lots_with_defects": lots,
        "defect_rate": defects / n if n else 0.0,
        "baseline_wafers": base_n,
        "baseline_defect_rate": base_def / base_n if base_n else 0.0,
    }


def clear() -> None:
    """Wipe history (used by tests and the dashboard's reset button)."""
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM inspections")
