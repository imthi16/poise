"""Adaptive runner control-flow, tested off-device with a fake forward.

The torch math (``_prefill`` / ``_decode_step``) is the on-device correctness
surface; here we verify the loop is pure mechanism: it asks ``budget_fn`` once per
token, clamps to the physical guard, threads prev_budget, records the trace, honors
EOS + max_new_tokens, and can condition the budget on live telemetry (the whole
point of POISE) — with NO control logic baked into the runner.
"""

from __future__ import annotations

import numpy as np

from poise.engine.adaptive_runner import AdaptiveRunner, BudgetContext
from poise.telemetry.mock import MockTelemetryReader


class FakeRunner(AdaptiveRunner):
    """AdaptiveRunner with the torch forward replaced by a deterministic stub."""

    def __init__(self, cfg, scripted_tokens, eos=None, **kw):
        super().__init__(cfg, model=None, tokenizer=None, **kw)
        self._scripted = list(scripted_tokens)
        self._eos = eos
        self._cursor = 0

    def _encode(self, prompt):
        return np.zeros((1, 3), dtype=int)

    def _prompt_len(self, input_ids):
        return 3

    def _eos_id(self):
        return self._eos

    def _prefill(self, input_ids):
        return {"cache": None, "logits_last": 0}

    def _decode_step(self, token_id, position, budget, state):
        return {"cache": None, "logits_last": 0, "kl_vs_full": 0.001 * budget}

    def _select_token(self, logits_last):
        if self._cursor < len(self._scripted):
            tok = self._scripted[self._cursor]
            self._cursor += 1
            return tok
        return 999

    def _decode_text(self, token_ids):
        return " ".join(str(t) for t in token_ids)


def test_generates_max_new_tokens(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3, 4, 5])
    res = r.generate("hi", max_new_tokens=5, budget_fn=lambda ctx: 24)
    assert res.tokens == 5
    assert res.token_ids == [1, 2, 3, 4, 5]
    assert len(res.trace) == 5
    assert all(t.budget == 24 for t in res.trace)


def test_budget_clamped_to_physical_guard(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3])
    # request absurd values; runner clamps to [layer_min, layer_max] = [16, 32]
    res = r.generate("hi", max_new_tokens=3, budget_fn=lambda ctx: 999 if ctx.step == 0 else 0)
    assert res.trace[0].budget == cfg.depth.layer_max   # 999 -> 32
    assert res.trace[1].budget == cfg.depth.layer_min   # 0   -> 16


def test_budget_fn_called_once_per_token_with_context(cfg):
    seen: list[BudgetContext] = []

    def budget_fn(ctx):
        seen.append(ctx)
        return 20

    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3, 4])
    r.generate("hi", max_new_tokens=4, budget_fn=budget_fn)
    assert [c.step for c in seen] == [0, 1, 2, 3]
    assert seen[0].prev_budget is None
    assert seen[1].prev_budget == 20  # threaded from previous clamped budget


def test_eos_stops_generation(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 7, 9], eos=7)
    res = r.generate("hi", max_new_tokens=10, budget_fn=lambda ctx: 32)
    assert res.token_ids == [1, 2, 7]  # stops at EOS (inclusive)
    assert res.tokens == 3


def test_static_shortcut_uses_constant_depth(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3])
    res = r.generate_static("hi", max_new_tokens=3, depth=16)
    assert all(t.budget == 16 for t in res.trace)
    assert abs(res.mean_budget - 16.0) < 1e-9


def test_kl_recorded_when_requested(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2])
    res = r.generate("hi", max_new_tokens=2, budget_fn=lambda ctx: 28, compute_kl_vs_full=True)
    assert all(t.kl_vs_full is not None for t in res.trace)
    assert abs(res.trace[0].kl_vs_full - 0.001 * 28) < 1e-9


def test_budget_conditioned_on_hardware_state(cfg):
    """The defining behavior: depth chosen from live telemetry, not input content."""
    hot = MockTelemetryReader(ambient_c=25.0, temp_max_c=87.0, init_temp_c=85.0,
                              load_level=1.0, budget=32)
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3, 4], telemetry_reader=hot)

    def thermal_budget(ctx):
        assert ctx.telemetry is not None  # runner surfaced the hardware state
        return 16 if ctx.telemetry.temp_c >= 80.0 else 32

    res = r.generate("hi", max_new_tokens=4, budget_fn=thermal_budget)
    assert all(t.budget == 16 for t in res.trace)  # ran hot => shed depth
    assert all(t.temp_c is not None for t in res.trace)


def test_trace_properties(cfg):
    r = FakeRunner(cfg, scripted_tokens=[1, 2, 3])
    res = r.generate("hi", max_new_tokens=3, budget_fn=lambda ctx: 24)
    assert res.mean_budget == 24.0
    assert res.tok_per_s >= 0.0
