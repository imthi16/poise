"""FastAPI endpoints + Prometheus metrics (CLAUDE.md §7, §10).

Runs against the off-device mock engine (real control loop, simulated forward) via
the FastAPI TestClient — no model required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from poise.config import load_config
from poise.serving.api import InferenceService, create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POISE_DB_PATH", str(tmp_path / "api.db"))
    cfg = load_config()
    service = InferenceService(cfg)  # mock engine, in tmp db
    app = create_app(cfg, service=service)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mode"] in ("static", "pid", "ppo")


def test_generate_returns_metrics_and_trace(client):
    r = client.post("/v1/generate", json={"prompt": "hello world", "max_new_tokens": 8,
                                          "mode": "pid", "return_trace": True})
    assert r.status_code == 200
    body = r.json()
    assert body["tokens"] == 8
    assert set(body["metrics"]) >= {"tok_per_s", "ttft_ms", "energy_per_token_j",
                                    "peak_temp_c", "throttle_events", "mean_budget"}
    assert len(body["trace"]) == 8
    assert all("budget" in t for t in body["trace"])
    # stored run is retrievable
    run = client.get(f"/v1/runs/{body['run_id']}").json()
    assert run["tokens"] == 8 and len(run["trace"]) == 8


def test_generate_static_mode_uses_fixed_depth(client):
    r = client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 5,
                                          "mode": "static", "static_depth": 16})
    body = r.json()
    assert all(t["budget"] == 16 for t in body["trace"])


def test_telemetry_endpoint(client):
    r = client.get("/v1/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert {"temp_c", "power_w", "gpu_clock_mhz", "throttled"} <= set(body)


def test_state_endpoint(client):
    r = client.get("/v1/state")
    body = r.json()
    assert body["setpoint_c"] > 0
    assert "policy_loaded" in body and "current_budget" in body


def test_mode_switch_ok(client):
    r = client.post("/v1/config/mode", json={"mode": "pid"})
    assert r.status_code == 200 and r.json() == {"mode": "pid", "ok": True}


def test_mode_switch_rejects_ppo_without_policy(client):
    r = client.post("/v1/config/mode", json={"mode": "ppo"})
    assert r.status_code == 400
    assert "policy" in r.json()["detail"].lower()


def test_run_not_found_404(client):
    assert client.get("/v1/runs/does-not-exist").status_code == 404


def test_metrics_scrapeable(client):
    client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 4, "mode": "pid"})
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    for name in ("poise_temp_c", "poise_power_w", "poise_current_budget",
                 "poise_tok_per_s", "poise_energy_per_token_j"):
        assert name in text


def test_unknown_mode_rejected(client):
    r = client.post("/v1/generate", json={"prompt": "x", "max_new_tokens": 2, "mode": "bogus"})
    assert r.status_code == 400


def test_service_thread_safe_under_concurrency(tmp_path, monkeypatch):
    """Regression: FastAPI runs endpoints in a THREADPOOL, so the shared SQLite
    connection is used across threads. TestClient is single-threaded and missed this;
    here we drive the service from many threads directly (cross-thread DB use, the
    concurrent lazy-init/migration race, and concurrent writers must all be safe)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("POISE_DB_PATH", str(tmp_path / "concurrent.db"))
    cfg = load_config()
    svc = InferenceService(cfg)
    svc.db()  # create the connection in THIS thread; workers run in others

    errors: list[str] = []

    def call(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for _ in range(15):
            futs.append(ex.submit(call, svc.state))
            futs.append(ex.submit(call, svc.telemetry))
            futs.append(ex.submit(call, lambda: svc.generate("hi", 5, mode="pid")))
        for f in futs:
            f.result()

    assert errors == [], f"concurrency errors: {errors[:3]}"
    # all 15 generates persisted and are retrievable across threads
    n = svc.db().execute("SELECT COUNT(*) c FROM runs").fetchone()["c"]
    assert n == 15
