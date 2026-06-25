"""Table accessors for the POISE store (CLAUDE.md §8).

Thin functional wrappers over sqlite3 — insert + query helpers per table. JSON
columns are (de)serialized here so callers pass/receive plain dicts.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Iterable, Sequence


def _now() -> float:
    return time.time()


def new_id() -> str:
    return str(uuid.uuid4())


def _json(v: Any) -> str | None:
    return None if v is None else json.dumps(v, default=str)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k, v in list(d.items()):
        if k.endswith("_json") and isinstance(v, str):
            try:
                d[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    mode: str,
    model_id: str | None = None,
    dtype: str | None = None,
    static_depth: int | None = None,
    policy_path: str | None = None,
    config_json: dict | None = None,
    prompt: str | None = None,
    notes: str | None = None,
) -> str:
    """Create a run header row. Metrics are filled in later via ``update_run_metrics``."""
    rid = run_id or new_id()
    conn.execute(
        """
        INSERT INTO runs (id, created_at, mode, model_id, dtype, static_depth,
                          policy_path, config_json, prompt, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rid,
            _now(),
            mode,
            model_id,
            dtype,
            static_depth,
            policy_path,
            _json(config_json),
            prompt,
            notes,
        ),
    )
    conn.commit()
    return rid


def update_run_metrics(conn: sqlite3.Connection, run_id: str, metrics: dict[str, Any]) -> None:
    allowed = {
        "tokens",
        "tok_per_s",
        "ttft_ms",
        "energy_per_token_j",
        "peak_temp_c",
        "time_above_setpoint_s",
        "throttle_events",
        "mean_budget",
        "notes",
    }
    fields = {k: v for k, v in metrics.items() if k in allowed}
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE runs SET {assignments} WHERE id = ?",
        (*fields.values(), run_id),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_dict(row)


def list_runs(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# token_events  (high-volume; sampling/disable flag lives in config)
# --------------------------------------------------------------------------- #
def insert_token_events(
    conn: sqlite3.Connection, run_id: str, events: Iterable[dict[str, Any]]
) -> int:
    rows = [
        (
            run_id,
            int(e["idx"]),
            int(e["budget"]),
            e.get("latency_ms"),
            e.get("temp_c"),
            e.get("power_w"),
            e.get("kl_vs_full"),
        )
        for e in events
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO token_events (run_id, idx, budget, latency_ms, temp_c, power_w, kl_vs_full)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def get_token_events(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT idx, budget, latency_ms, temp_c, power_w, kl_vs_full "
        "FROM token_events WHERE run_id = ? ORDER BY idx",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_token_kl(conn: sqlite3.Connection, run_id: str, idx: int, kl: float) -> None:
    conn.execute(
        "UPDATE token_events SET kl_vs_full = ? WHERE run_id = ? AND idx = ?",
        (kl, run_id, idx),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# telemetry_samples
# --------------------------------------------------------------------------- #
def insert_telemetry_sample(
    conn: sqlite3.Connection,
    *,
    ts: float,
    temp_c: float,
    power_w: float,
    gpu_clock_mhz: float,
    gpu_util: float,
    throttled: bool,
    run_id: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO telemetry_samples
            (run_id, ts, temp_c, power_w, gpu_clock_mhz, gpu_util, throttled)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, ts, temp_c, power_w, gpu_clock_mhz, gpu_util, int(bool(throttled))),
    )
    conn.commit()


def get_telemetry_samples(
    conn: sqlite3.Connection, run_id: str | None = None, limit: int = 1000
) -> list[dict[str, Any]]:
    if run_id is None:
        rows = conn.execute(
            "SELECT * FROM telemetry_samples ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM telemetry_samples WHERE run_id = ? ORDER BY ts LIMIT ?",
            (run_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def insert_calibration_run(
    conn: sqlite3.Connection,
    *,
    calib_id: str | None = None,
    power_mode: str | None = None,
    depth: int | None = None,
    ambient_c: float | None = None,
    params_json: dict | None = None,
) -> str:
    cid = calib_id or new_id()
    conn.execute(
        """
        INSERT INTO calibration_runs (id, created_at, power_mode, depth, ambient_c, params_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (cid, _now(), power_mode, depth, ambient_c, _json(params_json)),
    )
    conn.commit()
    return cid


def insert_calibration_samples(
    conn: sqlite3.Connection, calib_id: str, samples: Sequence[dict[str, Any]]
) -> int:
    rows = [(calib_id, s["ts"], s.get("temp_c"), s.get("power_w")) for s in samples]
    if not rows:
        return 0
    conn.executemany(
        "INSERT INTO calibration_samples (calib_id, ts, temp_c, power_w) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def get_calibration_samples(conn: sqlite3.Connection, calib_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT ts, temp_c, power_w FROM calibration_samples WHERE calib_id = ? ORDER BY ts",
        (calib_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# quality_profile  (the step-3 GATE output)
# --------------------------------------------------------------------------- #
def insert_quality_profile(
    conn: sqlite3.Connection,
    *,
    depth: int,
    mean_kl_vs_full: float,
    perplexity: float | None = None,
    task_accuracy: float | None = None,
    dataset_tag: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO quality_profile
            (created_at, depth, mean_kl_vs_full, perplexity, task_accuracy, dataset_tag)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_now(), depth, mean_kl_vs_full, perplexity, task_accuracy, dataset_tag),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_quality_profile(
    conn: sqlite3.Connection, dataset_tag: str | None = None
) -> list[dict[str, Any]]:
    """Latest quality-cost row per depth (most recent ``created_at`` wins)."""
    if dataset_tag is None:
        rows = conn.execute(
            """
            SELECT qp.* FROM quality_profile qp
            JOIN (SELECT depth, MAX(created_at) AS mx FROM quality_profile GROUP BY depth) g
              ON qp.depth = g.depth AND qp.created_at = g.mx
            ORDER BY qp.depth
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT qp.* FROM quality_profile qp
            JOIN (SELECT depth, MAX(created_at) AS mx FROM quality_profile
                  WHERE dataset_tag = ? GROUP BY depth) g
              ON qp.depth = g.depth AND qp.created_at = g.mx
            WHERE qp.dataset_tag = ?
            ORDER BY qp.depth
            """,
            (dataset_tag, dataset_tag),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# ppo_checkpoints
# --------------------------------------------------------------------------- #
def insert_ppo_checkpoint(
    conn: sqlite3.Connection,
    *,
    checkpoint_id: str | None = None,
    path: str,
    reward_config_json: dict,
    env_config_json: dict,
    sim_params_json: dict,
    train_steps: int,
    final_mean_reward: float,
    converged_depth_hist_json: dict,
) -> str:
    cid = checkpoint_id or new_id()
    conn.execute(
        """
        INSERT INTO ppo_checkpoints
            (id, created_at, path, reward_config_json, env_config_json, sim_params_json,
             train_steps, final_mean_reward, converged_depth_hist_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cid,
            _now(),
            path,
            _json(reward_config_json),
            _json(env_config_json),
            _json(sim_params_json),
            train_steps,
            final_mean_reward,
            _json(converged_depth_hist_json),
        ),
    )
    conn.commit()
    return cid


def get_ppo_checkpoint(conn: sqlite3.Connection, checkpoint_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM ppo_checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    return _row_to_dict(row)


__all__ = [
    "new_id",
    "insert_run",
    "update_run_metrics",
    "get_run",
    "list_runs",
    "insert_token_events",
    "get_token_events",
    "update_token_kl",
    "insert_telemetry_sample",
    "get_telemetry_samples",
    "insert_calibration_run",
    "insert_calibration_samples",
    "get_calibration_samples",
    "insert_quality_profile",
    "get_quality_profile",
    "insert_ppo_checkpoint",
    "get_ppo_checkpoint",
]
