"""SQLite connection + forward-only migrations for POISE (CLAUDE.md §8).

Times are REAL epoch seconds; JSON blobs are TEXT. Migrations are append-only:
never edit an existing migration — add a new one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Forward-only migrations. Each entry is (version, sql). Append new versions; do
# not edit applied ones.
_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS runs (
            id                    TEXT PRIMARY KEY,
            created_at            REAL NOT NULL,
            mode                  TEXT NOT NULL,
            model_id              TEXT,
            dtype                 TEXT,
            static_depth          INTEGER,
            policy_path           TEXT,
            config_json           TEXT,
            prompt                TEXT,
            tokens                INTEGER,
            tok_per_s             REAL,
            ttft_ms               REAL,
            energy_per_token_j    REAL,
            peak_temp_c           REAL,
            time_above_setpoint_s REAL,
            throttle_events       INTEGER,
            mean_budget           REAL,
            notes                 TEXT
        );

        CREATE TABLE IF NOT EXISTS token_events (
            id          INTEGER PRIMARY KEY,
            run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            idx         INTEGER NOT NULL,
            budget      INTEGER NOT NULL,
            latency_ms  REAL,
            temp_c      REAL,
            power_w     REAL,
            kl_vs_full  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_token_events_run ON token_events(run_id, idx);

        CREATE TABLE IF NOT EXISTS telemetry_samples (
            id            INTEGER PRIMARY KEY,
            run_id        TEXT REFERENCES runs(id) ON DELETE SET NULL,
            ts            REAL NOT NULL,
            temp_c        REAL,
            power_w       REAL,
            gpu_clock_mhz REAL,
            gpu_util      REAL,
            throttled     INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_telemetry_run ON telemetry_samples(run_id, ts);

        CREATE TABLE IF NOT EXISTS calibration_runs (
            id          TEXT PRIMARY KEY,
            created_at  REAL NOT NULL,
            power_mode  TEXT,
            depth       INTEGER,
            ambient_c   REAL,
            params_json TEXT
        );

        CREATE TABLE IF NOT EXISTS calibration_samples (
            id        INTEGER PRIMARY KEY,
            calib_id  TEXT NOT NULL REFERENCES calibration_runs(id) ON DELETE CASCADE,
            ts        REAL NOT NULL,
            temp_c    REAL,
            power_w   REAL
        );
        CREATE INDEX IF NOT EXISTS idx_calib_samples ON calibration_samples(calib_id, ts);

        CREATE TABLE IF NOT EXISTS quality_profile (
            id              INTEGER PRIMARY KEY,
            created_at      REAL NOT NULL,
            depth           INTEGER NOT NULL,
            mean_kl_vs_full REAL,
            perplexity      REAL,
            task_accuracy   REAL,
            dataset_tag     TEXT
        );

        CREATE TABLE IF NOT EXISTS ppo_checkpoints (
            id                       TEXT PRIMARY KEY,
            created_at               REAL NOT NULL,
            path                     TEXT,
            reward_config_json       TEXT,
            env_config_json          TEXT,
            sim_params_json          TEXT,
            train_steps              INTEGER,
            final_mean_reward        REAL,
            converged_depth_hist_json TEXT
        );
        """,
    ),
]


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sane pragmas (FK enforcement, dict-like rows)."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return int(row["v"]) if row and row["v"] is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any pending forward-only migrations. Returns the new schema version."""
    import time

    current = _current_version(conn)
    for version, sql in _MIGRATIONS:
        if version <= current:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, time.time()),
        )
        current = version
    conn.commit()
    return current


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Connect + migrate. Single entrypoint used across the codebase."""
    conn = connect(db_path)
    migrate(conn)
    return conn


__all__ = ["connect", "migrate", "init_db"]
