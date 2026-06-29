"""FastAPI app factory + inference service (CLAUDE.md §7).

``create_app(cfg)`` wires telemetry + control + engine + storage + metrics behind the
§7 endpoints. On the Jetson the real ``AdaptiveRunner`` (loaded model) is used; off-
device a ``MockAdaptiveRunner`` reuses the SAME control path (allocator + telemetry +
trace) with a simulated forward, so the API and dashboard run end-to-end without the
model. Mock generation is a dev/demo convenience — its throughput is NOT an eval
result (those come only from ``eval/report.py``).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

from ..config import load_config
from ..control.budget_allocator import make_allocator
from ..engine.adaptive_runner import AdaptiveRunner
from ..eval.benchmark import summarize_trace
from .metrics import MetricsRegistry

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Mock engine (off-device) — real control loop, simulated forward.
# --------------------------------------------------------------------------- #
class MockAdaptiveRunner(AdaptiveRunner):
    """Drives the real generate() loop (allocator + telemetry + trace) with a
    simulated forward so the dashboard/API work without the model."""

    def __init__(self, cfg, telemetry_reader=None, realtime: bool = False, **kw):
        super().__init__(cfg, model=None, tokenizer=None,
                         telemetry_reader=telemetry_reader, **kw)
        self.realtime = realtime
        self._counter = 0
        self._latency = dict(cfg.sim.depth_latency_map)

    def _encode(self, prompt):
        import numpy as np

        self._counter = 0
        return np.zeros((1, max(1, len(prompt.split()))), dtype=int)

    def _prompt_len(self, x):
        return int(x.shape[1])

    def _eos_id(self):
        return None

    def _prefill(self, x):
        return {"logits_last": 0}

    def _decode_step(self, token_id, position, budget, state):
        # let the mock device "feel" the chosen depth so temperature responds
        if self.telemetry_reader is not None and hasattr(self.telemetry_reader, "set_budget"):
            self.telemetry_reader.set_budget(budget)
        if self.realtime:
            lat = self._interp_latency(budget)
            time.sleep(min(0.05, lat))
        return {"logits_last": 0}

    def _select_token(self, logits_last):
        self._counter += 1
        return self._counter

    def _decode_text(self, token_ids):
        return " ".join(f"tok{t}" for t in token_ids)

    def _interp_latency(self, budget):
        import numpy as np

        ds = sorted(self._latency)
        return float(np.interp(budget, ds, [self._latency[d] for d in ds]))


# --------------------------------------------------------------------------- #
# Inference service (app state)
# --------------------------------------------------------------------------- #
class InferenceService:
    def __init__(
        self,
        cfg: "PoiseConfig",
        *,
        runner: Optional[AdaptiveRunner] = None,
        telemetry_reader=None,
        db_conn=None,
        metrics: Optional[MetricsRegistry] = None,
        realtime: bool = False,
    ):
        from ..telemetry import make_reader

        self.cfg = cfg
        self.telemetry_reader = telemetry_reader or make_reader(cfg)
        self.runner = runner or MockAdaptiveRunner(
            cfg, telemetry_reader=self.telemetry_reader, realtime=realtime
        )
        self.is_mock = runner is None
        self.metrics = metrics or MetricsRegistry()
        self.mode = cfg.control.mode
        self._db_conn = db_conn
        # Serializes all DB access. The single SQLite connection is shared across
        # FastAPI's threadpool threads (check_same_thread=False); this lock makes that
        # safe and prevents the concurrent lazy-init / migration race.
        self._db_lock = threading.RLock()
        self._allocators: dict[str, object] = {}
        self._alloc_lock = threading.Lock()
        self.policy_loaded = False
        # try to load a policy for ppo (None => PID fallback)
        self._ensure_allocator(self.mode)

    # --- db ------------------------------------------------------------------ #
    def db(self):
        with self._db_lock:
            if self._db_conn is None:
                from ..storage.db import init_db

                self._db_conn = init_db(self.cfg.storage.db_path)
            return self._db_conn

    # --- allocators ---------------------------------------------------------- #
    def _ensure_allocator(self, mode: str):
        with self._alloc_lock:
            if mode not in self._allocators:
                alloc = make_allocator(self.cfg, mode=mode)
                if mode == "ppo":
                    self.policy_loaded = alloc.policy is not None
                self._allocators[mode] = alloc
            return self._allocators[mode]

    # --- §7 operations ------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: Optional[str] = None,
        static_depth: Optional[int] = None,
        return_trace: bool = True,
    ) -> dict:
        mode = mode or self.mode
        alloc = self._ensure_allocator(mode)
        if hasattr(alloc, "reset"):
            alloc.reset()
        if mode == "static" and static_depth is not None:
            budget_fn = lambda ctx: static_depth  # noqa: E731
        else:
            budget_fn = alloc

        result = self.runner.generate(prompt, max_new_tokens, budget_fn, return_trace=True)
        metrics = summarize_trace(
            result.trace, ttft_ms=result.ttft_ms,
            temp_setpoint_c=self.cfg.thermal.temp_setpoint_c,
        )
        self.metrics.update_from_run(metrics)
        run_id = self._store(prompt, mode, result, metrics)

        resp = {
            "run_id": run_id,
            "text": result.text,
            "tokens": metrics.tokens,
            "metrics": {
                "tok_per_s": metrics.tok_per_s,
                "ttft_ms": metrics.ttft_ms,
                "energy_per_token_j": metrics.energy_per_token_j,
                "peak_temp_c": metrics.peak_temp_c,
                "throttle_events": metrics.throttle_events,
                "mean_budget": metrics.mean_budget,
            },
        }
        if return_trace:
            resp["trace"] = [
                {"i": t.i, "budget": t.budget, "latency_ms": t.latency_ms,
                 "temp_c": t.temp_c, "power_w": t.power_w}
                for t in result.trace
            ]
        return resp

    def _store(self, prompt, mode, result, metrics) -> str:
        from ..storage import models

        # One atomic, serialized transaction across the shared connection.
        with self._db_lock:
            conn = self.db()
            run_id = models.insert_run(
                conn, mode=mode, model_id=self.cfg.model.model_id, dtype=self.cfg.model.dtype,
                config_json=self.cfg.snapshot(), prompt=prompt,
                notes=("mock-engine" if self.is_mock else None),
            )
            models.update_run_metrics(conn, run_id, {
                "tokens": metrics.tokens, "tok_per_s": metrics.tok_per_s,
                "ttft_ms": metrics.ttft_ms, "energy_per_token_j": metrics.energy_per_token_j,
                "peak_temp_c": metrics.peak_temp_c,
                "time_above_setpoint_s": metrics.time_above_setpoint_s,
                "throttle_events": metrics.throttle_events, "mean_budget": metrics.mean_budget,
            })
            if self.cfg.storage.log_token_events:
                models.insert_token_events(conn, run_id, [
                    {"idx": t.i, "budget": t.budget, "latency_ms": t.latency_ms,
                     "temp_c": t.temp_c, "power_w": t.power_w, "kl_vs_full": t.kl_vs_full}
                    for t in result.trace
                ])
            return run_id

    def telemetry(self) -> dict:
        s = self.telemetry_reader.read()
        self.metrics.update_from_telemetry(s)
        return s.to_dict()

    def state(self) -> dict:
        s = self.telemetry_reader.read()
        alloc = self._allocators.get(self.mode)
        budget = getattr(alloc, "_prev_budget", self.cfg.depth.layer_max)
        return {
            "mode": self.mode,
            "current_budget": budget,
            "temp_c": s.temp_c,
            "setpoint_c": self.cfg.thermal.temp_setpoint_c,
            "throttled": s.throttled,
            "policy_loaded": self.policy_loaded,
        }

    def set_mode(self, mode: str) -> tuple[bool, str]:
        if mode not in ("static", "pid", "ppo"):
            return False, f"unknown mode {mode!r}"
        if mode == "ppo":
            alloc = self._ensure_allocator("ppo")
            if alloc.policy is None:
                return False, "no PPO policy loaded; cannot switch to ppo"
        self.mode = mode
        self._ensure_allocator(mode)
        return True, "ok"

    def get_run(self, run_id: str) -> Optional[dict]:
        from ..storage import models

        with self._db_lock:
            conn = self.db()
            run = models.get_run(conn, run_id)
            if run is None:
                return None
            run["trace"] = models.get_token_events(conn, run_id)
            return run

    def rag_query(self, query: str, k: int = 4) -> dict:
        # Demo layer (synthetic/public data only). Lazy import so faiss/langgraph
        # are optional; returns a clear message if the index isn't built.
        try:
            from ..rag.graph import run_rag

            return run_rag(self.cfg, query, k=k, service=self)
        except Exception as e:  # pragma: no cover - optional demo deps
            return {"answer": f"RAG demo unavailable: {e}", "sources": [], "run_id": None}


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(cfg: Optional["PoiseConfig"] = None, service: Optional[InferenceService] = None):
    from fastapi import FastAPI

    from .routes import build_router

    cfg = cfg or load_config()
    app = FastAPI(title="POISE", version="0.1.0",
                  description="Hardware-state-conditioned variable-depth LLM inference")
    app.state.cfg = cfg
    app.state.service = service or InferenceService(cfg)
    app.include_router(build_router())
    return app


def run() -> None:  # pragma: no cover - CLI (poise-serve)
    import uvicorn

    cfg = load_config()
    service = InferenceService(cfg, realtime=True)
    if service.telemetry_reader is not None:
        service.telemetry_reader.start()
    app = create_app(cfg, service=service)
    uvicorn.run(app, host=cfg.serving.host, port=cfg.serving.port)


__all__ = ["create_app", "InferenceService", "MockAdaptiveRunner", "run"]
