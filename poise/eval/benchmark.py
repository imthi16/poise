"""Throughput / energy / thermal stress harness (CLAUDE.md §6, §11).

The stress protocol: sustained generation at a fixed ambient until thermal steady
state, recording throughput, TTFT, inter-token latency, power, energy-per-token
(∫P dt ÷ tokens), peak temp, time-above-setpoint, and hard-throttle count.

🔒 Every headline number is produced HERE (and aggregated with variance in
``report.py``) — never asserted in prose ahead of measurement. The metric-aggregation
helpers are pure and unit-tested off-device (incl. against the mock telemetry + a fake
generator); the model run itself is the on-device part.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..engine.adaptive_runner import AdaptiveRunner, GenerationResult, TokenTrace


# --------------------------------------------------------------------------- #
# Pure metric helpers
# --------------------------------------------------------------------------- #
def integrate_energy_j(timestamps_s: Sequence[float], powers_w: Sequence[float]) -> float:
    """Trapezoidal ∫P dt over a telemetry/trace window (joules)."""
    t = np.asarray(timestamps_s, dtype=np.float64)
    p = np.asarray(powers_w, dtype=np.float64)
    if t.size < 2:
        return 0.0
    # np.trapz was renamed np.trapezoid in NumPy 2.0.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(p, t))


def trace_from_token_latencies(latencies_s, reader, *, budget: int):
    """Build a per-token trace from measured token latencies + a telemetry reader.

    Used by the fixed-depth baselines (llama.cpp / TRT-LLM) so their throughput /
    energy / thermal metrics are computed by the SAME aggregation as the adaptive
    path. ``reader.read()`` supplies temp/power/throttle per token (mock off-device).
    """
    from ..engine.adaptive_runner import TokenTrace

    rows = []
    for i, lat in enumerate(latencies_s):
        s = reader.read()
        rows.append(
            TokenTrace(
                i=i,
                budget=budget,
                latency_ms=float(lat) * 1000.0,
                temp_c=s.temp_c,
                power_w=s.power_w,
                throttled=s.throttled,
            )
        )
    return rows


def count_throttle_events(throttled_flags: Sequence[bool]) -> int:
    """Number of hard-throttle ONSETS (rising edges False→True)."""
    events = 0
    prev = False
    for f in throttled_flags:
        f = bool(f)
        if f and not prev:
            events += 1
        prev = f
    return events


@dataclass
class RunMetrics:
    tokens: int
    tok_per_s: float
    ttft_ms: float
    inter_token_latency_ms: float
    mean_power_w: float
    energy_per_token_j: float
    peak_temp_c: float
    time_above_setpoint_s: float
    throttle_events: int
    mean_budget: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_trace(
    trace: Sequence["TokenTrace"],
    *,
    ttft_ms: float,
    temp_setpoint_c: float,
    elapsed_s: Optional[float] = None,
) -> RunMetrics:
    """Aggregate a per-token trace into RunMetrics (pure)."""
    n = len(trace)
    if n == 0:
        return RunMetrics(0, 0.0, ttft_ms, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    latencies_ms = np.array([t.latency_ms for t in trace], dtype=np.float64)
    dt_s = latencies_ms / 1000.0
    total_s = elapsed_s if elapsed_s is not None else float(np.sum(dt_s))
    powers = np.array([(t.power_w if t.power_w is not None else 0.0) for t in trace])
    temps = np.array([(t.temp_c if t.temp_c is not None else 0.0) for t in trace])
    budgets = np.array([t.budget for t in trace], dtype=np.float64)
    throttled = [bool(t.throttled) for t in trace if t.throttled is not None]

    energy_j = float(np.sum(powers * dt_s))  # per-token power × per-token dt
    energy_per_token = energy_j / n if n else 0.0
    time_above = float(np.sum(dt_s[temps > temp_setpoint_c])) if temps.size else 0.0
    tok_per_s = (n / total_s) if total_s > 0 else 0.0

    return RunMetrics(
        tokens=n,
        tok_per_s=tok_per_s,
        ttft_ms=ttft_ms,
        inter_token_latency_ms=float(np.mean(latencies_ms)),
        mean_power_w=float(np.mean(powers)) if powers.size else 0.0,
        energy_per_token_j=energy_per_token,
        peak_temp_c=float(np.max(temps)) if temps.size else 0.0,
        time_above_setpoint_s=time_above,
        throttle_events=count_throttle_events(throttled),
        mean_budget=float(np.mean(budgets)),
    )


# --------------------------------------------------------------------------- #
# Stress orchestration
# --------------------------------------------------------------------------- #
@dataclass
class StressSpec:
    max_new_tokens: int = 512
    warmup_tokens: int = 0          # tokens to discard before steady-state metrics
    repeats: int = 1


class StressBenchmark:
    """Drives the engine under sustained generation and summarizes the run.

    Works with the real ``AdaptiveRunner`` on-device and with a fake runner + mock
    telemetry off-device (the orchestration + aggregation are validated that way).
    """

    def __init__(self, cfg: "PoiseConfig", runner: "AdaptiveRunner", budget_fn):
        self.cfg = cfg
        self.runner = runner
        self.budget_fn = budget_fn

    def run(self, prompt: str, spec: StressSpec) -> tuple["GenerationResult", RunMetrics]:
        result = self.runner.generate(
            prompt, spec.max_new_tokens, self.budget_fn, return_trace=True
        )
        trace = result.trace[spec.warmup_tokens:] if spec.warmup_tokens else result.trace
        metrics = summarize_trace(
            trace,
            ttft_ms=result.ttft_ms,
            temp_setpoint_c=self.cfg.thermal.temp_setpoint_c,
        )
        return result, metrics

    def run_and_store(self, prompt: str, spec: StressSpec, conn, mode_label: str) -> str:
        """Run, then persist a runs row + token_events (if enabled)."""
        from ..storage import models

        result, metrics = self.run(prompt, spec)
        run_id = models.insert_run(
            conn,
            mode=mode_label,
            model_id=self.cfg.model.model_id,
            dtype=self.cfg.model.dtype,
            config_json=self.cfg.snapshot(),
            prompt=prompt,
        )
        models.update_run_metrics(
            conn,
            run_id,
            {
                "tokens": metrics.tokens,
                "tok_per_s": metrics.tok_per_s,
                "ttft_ms": metrics.ttft_ms,
                "energy_per_token_j": metrics.energy_per_token_j,
                "peak_temp_c": metrics.peak_temp_c,
                "time_above_setpoint_s": metrics.time_above_setpoint_s,
                "throttle_events": metrics.throttle_events,
                "mean_budget": metrics.mean_budget,
            },
        )
        if self.cfg.storage.log_token_events:
            models.insert_token_events(
                conn,
                run_id,
                [
                    {
                        "idx": t.i,
                        "budget": t.budget,
                        "latency_ms": t.latency_ms,
                        "temp_c": t.temp_c,
                        "power_w": t.power_w,
                        "kl_vs_full": t.kl_vs_full,
                    }
                    for t in result.trace
                ],
            )
        return run_id


def main() -> None:  # pragma: no cover - CLI (poise-benchmark)
    import argparse

    from ..config import load_config
    from ..engine.model_loader import load_model
    from ..engine.adaptive_runner import AdaptiveRunner
    from ..control.budget_allocator import make_allocator
    from ..telemetry import make_reader
    from ..storage.db import init_db

    ap = argparse.ArgumentParser(description="POISE stress benchmark")
    ap.add_argument("--prompt", default="Explain thermal throttling in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config()
    model, tok = load_model(cfg)
    reader = make_reader(cfg)
    reader.start()
    runner = AdaptiveRunner(cfg, model, tok, telemetry_reader=reader)
    allocator = make_allocator(cfg)
    bench = StressBenchmark(cfg, runner, allocator)
    conn = init_db(cfg.storage.db_path)
    run_id = bench.run_and_store(args.prompt, StressSpec(max_new_tokens=args.max_new_tokens),
                                 conn, mode_label=cfg.control.mode)
    print(f"stored run {run_id}")
    conn.close()
    reader.stop()


__all__ = [
    "integrate_energy_j",
    "count_throttle_events",
    "RunMetrics",
    "summarize_trace",
    "StressSpec",
    "StressBenchmark",
    "main",
]
