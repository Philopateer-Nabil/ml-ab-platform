"""SQLite storage for experiments, predictions, feedback, and bandit state.

We use raw sqlite3 for simplicity and to keep the dependency surface small — the
schema is small enough that an ORM would be overkill. A single module-level
connection factory returns per-call connections; SQLite in WAL mode handles
concurrent reads/writes well enough for this workload.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml_ab_platform.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    status          TEXT NOT NULL,
    routing_strategy TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    model_a         TEXT NOT NULL,
    model_b         TEXT NOT NULL,
    target_metric   TEXT NOT NULL,
    minimum_sample_size INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    stopped_at      TEXT,
    concluded_at    TEXT,
    winner          TEXT,
    conclusion_json TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    user_id         TEXT,
    input_hash      TEXT NOT NULL,
    prediction      INTEGER NOT NULL,
    probability     REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    request_id      TEXT UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_exp ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_exp_model ON predictions(experiment_id, model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT UNIQUE NOT NULL,
    experiment_id   TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    ground_truth    INTEGER NOT NULL,
    received_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_exp ON feedback(experiment_id);
CREATE INDEX IF NOT EXISTS idx_feedback_exp_model ON feedback(experiment_id, model_version);

CREATE TABLE IF NOT EXISTS bandit_state (
    experiment_id   TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    alpha           REAL NOT NULL,
    beta            REAL NOT NULL,
    trials          INTEGER NOT NULL,
    successes       INTEGER NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (experiment_id, model_version)
);

CREATE TABLE IF NOT EXISTS bandit_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    chosen_prob     REAL NOT NULL,
    cumulative_regret REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bandit_history_exp ON bandit_history(experiment_id);

CREATE TABLE IF NOT EXISTS canary_state (
    experiment_id   TEXT PRIMARY KEY,
    current_split   REAL NOT NULL,
    promoted        INTEGER NOT NULL DEFAULT 0,
    degraded        INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL,
    notes           TEXT
);
"""

_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _ensure_db(db_path: str) -> None:
    """Create parent dir, initialise schema (idempotent)."""
    with _init_lock:
        if db_path in _initialized_paths:
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.commit()
        _initialized_paths.add(db_path)


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row_factory set. Commits on success."""
    if db_path is None:
        db_path = get_settings().database.path
    _ensure_db(db_path)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a Row to a plain dict, decoding any *_json fields."""
    if row is None:
        return None
    d = dict(row)
    for k, v in list(d.items()):
        if k.endswith("_json") and isinstance(v, str):
            with contextlib.suppress(json.JSONDecodeError):
                d[k[:-5]] = json.loads(v)
    return d


def reset_db(db_path: str | None = None) -> None:
    """Drop and recreate all tables — used in tests."""
    if db_path is None:
        db_path = get_settings().database.path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (name,) in cur.fetchall():
            if name.startswith("sqlite_"):
                continue
            conn.execute(f"DROP TABLE IF EXISTS {name};")
        conn.commit()
    _initialized_paths.discard(db_path)
    _ensure_db(db_path)
