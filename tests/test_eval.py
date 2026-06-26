"""Eval harness: metrics, stress orchestration, baseline metric path, variance report.

(CLAUDE.md §11 / §10). The model runs are on-device; here we validate the aggregation
and the comparison-set enforcement that guard the headline numbers.
"""

from __future__ import annotations

import numpy as np

from poise.engine.adaptive_runner import AdaptiveRunner, TokenTrace
from poise.eval.benchmark import (
    RunMetrics,
    StressBenchmark,
    StressSpec,
    count_throttle_events,
    integrate_energy_j,
    summarize_trace,
    trace_from_token_latencies,
)
from poise.eval.report import aggregate, check_comparison_set, compare, format_markdown
from poise.telemetry.mock import MockTelemetryReader


# --- pure metric helpers --------------------------------------------------- #
def test_integrate_energy_trapezoidal():
    # constant 10 W over 0..4 s -> 40 J
    assert abs(integrate_energy_j([0, 1, 2, 3, 4], [10, 10, 10, 10, 10]) - 40.0) < 1e-9


def test_count_throttle_events_counts_onsets():
    assert count_throttle_events([False, True, True, False, True]) == 2
    assert count_throttle_events([False, False]) == 0
    assert count_throttle_events([True, True, True]) == 1


def test_summarize_trace_metrics():
    trace = [
        TokenTrace(i=0, budget=32, latency_ms=40.0, temp_c=70.0, power_w=30.0, throttled=False),
        TokenTrace(i=1, budget=24, latency_ms=30.0, temp_c=82.0, power_w=26.0, throttled=False),
        TokenTrace(i=2, budget=16, latency_ms=20.0, temp_c=88.0, power_w=20.0, throttled=True),
    ]
    m = summarize_trace(trace, ttft_ms=12.0, temp_setpoint_c=80.0)
    assert m.tokens == 3
    assert m.peak_temp_c == 88.0
    assert m.throttle_events == 1
    assert abs(m.mean_budget - 24.0) < 1e-9
    # time above setpoint = dt of tokens 1 (0.03) + 2 (0.02) = 0.05 s
    assert abs(m.time_above_setpoint_s - 0.05) < 1e-9
    assert m.energy_per_token_j > 0


# --- stress orchestration with a fake runner + mock telemetry -------------- #
class _FakeRunner(AdaptiveRunner):
    def __init__(self, cfg, n, **kw):
        super().__init__(cfg, model=None, tokenizer=None, **kw)
        self._n = n
        self._c = 0

    def _encode(self, prompt):
        return np.zeros((1, 2), dtype=int)

    def _prompt_len(self, x):
        return 2

    def _eos_id(self):
        return None

    def _prefill(self, x):
        return {"logits_last": 0}

    def _decode_step(self, tid, pos, budget, state):
        return {"logits_last": 0}

    def _select_token(self, logits):
        self._c += 1
        return self._c

    def _decode_text(self, ids):
        return "x"


def test_stress_benchmark_runs_and_summarizes(cfg):
    reader = MockTelemetryReader(ambient_c=25.0, init_temp_c=80.0, load_level=1.0, budget=32)
    runner = _FakeRunner(cfg, n=10, telemetry_reader=reader)
    bench = StressBenchmark(cfg, runner, budget_fn=lambda ctx: 24)
    result, metrics = bench.run("hello", StressSpec(max_new_tokens=10))
    assert metrics.tokens == 10
    assert metrics.mean_budget == 24.0
    assert metrics.peak_temp_c > 0


def test_stress_benchmark_persists_run(cfg, db):
    reader = MockTelemetryReader(init_temp_c=60.0, load_level=0.8, budget=32)
    runner = _FakeRunner(cfg, n=5, telemetry_reader=reader)
    bench = StressBenchmark(cfg, runner, budget_fn=lambda ctx: 32)
    run_id = bench.run_and_store("p", StressSpec(max_new_tokens=5), db, mode_label="pid")
    from poise.storage import models

    run = models.get_run(db, run_id)
    assert run["mode"] == "pid" and run["tokens"] == 5
    assert len(models.get_token_events(db, run_id)) == 5


# --- baseline metric path (no llama.cpp needed) ---------------------------- #
def test_baseline_trace_from_latencies(cfg):
    reader = MockTelemetryReader(init_temp_c=70.0, load_level=1.0, budget=32)
    latencies = [0.03, 0.031, 0.029, 0.030]
    trace = trace_from_token_latencies(latencies, reader, budget=cfg.depth.layer_total)
    assert len(trace) == 4
    assert all(t.budget == cfg.depth.layer_total for t in trace)  # full-depth baseline
    m = summarize_trace(trace, ttft_ms=10.0, temp_setpoint_c=80.0)
    assert m.tokens == 4 and m.mean_budget == cfg.depth.layer_total


# --- report aggregation + comparison-set enforcement ----------------------- #
def _mk_metrics(tps, energy, peak, throttle, budget):
    return RunMetrics(
        tokens=100, tok_per_s=tps, ttft_ms=10.0, inter_token_latency_ms=30.0,
        mean_power_w=30.0, energy_per_token_j=energy, peak_temp_c=peak,
        time_above_setpoint_s=1.0, throttle_events=throttle, mean_budget=budget,
    )


def test_aggregate_reports_mean_and_std():
    ms = [_mk_metrics(10, 1.0, 80, 0, 32), _mk_metrics(12, 1.2, 82, 1, 30)]
    agg = aggregate(ms)
    assert abs(agg["tok_per_s"][0] - 11.0) < 1e-9
    assert agg["tok_per_s"][1] > 0  # nonzero std across the two runs


def test_check_comparison_set_enforced():
    # missing pid/ppo and <2 static reduced => problems reported
    problems = check_comparison_set(["static-full-32"])
    assert any("pid" in p for p in problems)
    assert any("ppo" in p for p in problems)
    assert any("static reduced" in p for p in problems)
    # the full mandated set passes
    ok = check_comparison_set(["static-full-32", "static-24", "static-16", "pid", "ppo"])
    assert ok == []


def test_compare_computes_deltas_and_better_flags():
    results = {
        "static-full-32": [_mk_metrics(10, 2.0, 90, 5, 32)],
        "static-24": [_mk_metrics(14, 1.4, 80, 0, 24)],
        "static-16": [_mk_metrics(18, 1.0, 70, 0, 16)],
        "pid": [_mk_metrics(13, 1.5, 82, 0, 26)],
        "ppo": [_mk_metrics(15, 1.3, 81, 0, 27)],
    }
    comp = compare(results, baseline_label="static-full-32")
    assert comp["problems"] == []  # full set present
    ppo = comp["labels"]["ppo"]
    # ppo has higher throughput (better) and lower energy (better) than baseline
    assert ppo["tok_per_s"]["better"] is True
    assert ppo["energy_per_token_j"]["better"] is True
    md = format_markdown(comp)
    assert "ppo" in md and "baseline" in md
