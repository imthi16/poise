"""Storage schema + accessors round-trip (CLAUDE.md §8 / §10)."""

from __future__ import annotations

from poise.storage import models
from poise.storage.db import init_db, migrate


def test_migration_idempotent():
    conn = init_db(":memory:")
    v1 = migrate(conn)
    v2 = migrate(conn)  # second run is a no-op
    assert v1 == v2 >= 1
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for t in (
        "runs",
        "token_events",
        "telemetry_samples",
        "calibration_runs",
        "calibration_samples",
        "quality_profile",
        "ppo_checkpoints",
    ):
        assert t in tables


def test_run_and_token_roundtrip(db):
    rid = models.insert_run(db, mode="ppo", model_id="m", dtype="fp16", prompt="hi")
    models.insert_token_events(
        db,
        rid,
        [
            {"idx": 0, "budget": 32, "latency_ms": 1.2, "temp_c": 50.0, "power_w": 20.0},
            {"idx": 1, "budget": 24, "latency_ms": 0.9, "temp_c": 60.0, "power_w": 22.0},
        ],
    )
    models.update_run_metrics(db, rid, {"tokens": 2, "tok_per_s": 18.5, "mean_budget": 28.0})
    run = models.get_run(db, rid)
    assert run["tokens"] == 2 and run["tok_per_s"] == 18.5
    events = models.get_token_events(db, rid)
    assert [e["budget"] for e in events] == [32, 24]


def test_quality_profile_latest_per_depth(db):
    models.insert_quality_profile(db, depth=16, mean_kl_vs_full=0.5, dataset_tag="calib-v1")
    models.insert_quality_profile(db, depth=16, mean_kl_vs_full=0.4, dataset_tag="calib-v1")
    models.insert_quality_profile(db, depth=32, mean_kl_vs_full=0.0, dataset_tag="calib-v1")
    prof = models.get_quality_profile(db, dataset_tag="calib-v1")
    by_depth = {p["depth"]: p["mean_kl_vs_full"] for p in prof}
    assert by_depth[16] == 0.4  # latest wins
    assert by_depth[32] == 0.0


def test_foreign_key_cascade(db):
    rid = models.insert_run(db, mode="pid")
    models.insert_token_events(db, rid, [{"idx": 0, "budget": 32}])
    db.execute("DELETE FROM runs WHERE id = ?", (rid,))
    db.commit()
    assert models.get_token_events(db, rid) == []


def test_ppo_checkpoint_roundtrip(db):
    cid = models.insert_ppo_checkpoint(
        db,
        path="/models/ppo_policy.zip",
        reward_config_json={"w_thr": 1.0},
        env_config_json={"dt_s": 0.25},
        sim_params_json={"r_th": 1.5},
        train_steps=1000,
        final_mean_reward=12.3,
        converged_depth_hist_json={"16": 0.2, "32": 0.8},
    )
    ckpt = models.get_ppo_checkpoint(db, cid)
    assert ckpt["train_steps"] == 1000
    assert ckpt["converged_depth_hist_json"] == {"16": 0.2, "32": 0.8}
