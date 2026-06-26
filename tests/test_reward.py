"""Reward shaping + the reward-hacking guard (⚠ CLAUDE.md §6, §9)."""

from __future__ import annotations

import pytest

from poise.config import RewardConfig
from poise.rl.reward import (
    DepthHistogram,
    QualityCostTable,
    RewardFn,
    synthetic_quality_table,
)
from poise.rl.simulator import SimStep


def _weights(**kw):
    base = dict(w_thr=1.0, w_q=1.0, w_therm=0.5, w_energy=0.3, w_throttle=10.0)
    base.update(kw)
    return RewardConfig(**base)


def test_quality_table_interpolates_and_normalizes():
    q = QualityCostTable({16: 0.4, 24: 0.1, 32: 0.0})
    assert q.cost(16) == 0.4 and q.cost(32) == 0.0
    assert 0.0 < q.cost(20) < 0.4
    assert q.normalized_cost(16) == 1.0  # worst KL -> 1.0
    assert q.normalized_cost(32) == 0.0


def test_synthetic_table_is_labeled_and_monotonic():
    q = synthetic_quality_table([16, 24, 32], layer_total=32)
    assert q.synthetic is True            # not a measured result
    assert q.cost(32) == 0.0              # zero cost at full depth
    assert q.cost(16) > q.cost(24) > q.cost(32)  # shallower costs more


def test_from_db_requires_gate_run(db):
    """No quality profile rows => refuse (run the gate first), never fabricate."""
    with pytest.raises(ValueError):
        QualityCostTable.from_db(db)


def test_reward_components_signs():
    q = synthetic_quality_table([16, 24, 32], 32)
    rf = RewardFn(_weights(), q, temp_setpoint_c=80.0,
                 throughput_norm_tok_s=30.0, energy_norm_j=5.0)
    cool = SimStep(temp_c=60.0, power_w=30.0, throttled=False, tok_per_s=28.0, energy_j=7.5)
    r, c = rf(cool, budget=32)
    assert c.throughput > 0
    assert c.thermal == 0.0              # below setpoint => no thermal penalty
    assert c.throttle == 0.0
    assert c.quality == 0.0              # full depth, synthetic table

    throttled = SimStep(temp_c=88.0, power_w=30.0, throttled=True, tok_per_s=10.0, energy_j=7.5)
    r2, c2 = rf(throttled, budget=32)
    assert c2.throttle < 0 and c2.thermal < 0
    assert r2 < r                         # throttling is heavily penalized


def test_reward_hacking_mindepth_dominates_when_wq_too_small():
    """Demonstrates the guard: tiny w_q => min depth out-rewards full depth."""
    q = synthetic_quality_table([16, 32], 32)
    rf = RewardFn(_weights(w_q=0.001), q, temp_setpoint_c=80.0,
                 throughput_norm_tok_s=30.0, energy_norm_j=5.0)
    shallow = SimStep(temp_c=55.0, power_w=18.0, throttled=False, tok_per_s=55.0, energy_j=4.5)
    deep = SimStep(temp_c=75.0, power_w=35.0, throttled=False, tok_per_s=28.0, energy_j=8.75)
    r_shallow, _ = rf(shallow, budget=16)
    r_deep, _ = rf(deep, budget=32)
    assert r_shallow > r_deep   # exactly the collapse the guard warns about


def test_depth_histogram_detects_collapse():
    h = DepthHistogram()
    for _ in range(95):
        h.update(16)
    for _ in range(5):
        h.update(32)
    dist = h.distribution()
    assert abs(dist[16] - 0.95) < 1e-9
    assert h.collapsed_to_min(layer_min=16, threshold=0.9) is True
    assert 16.0 <= h.mean_budget() < 17.0


def test_depth_histogram_healthy_distribution():
    h = DepthHistogram()
    for b in [16, 20, 24, 28, 32] * 10:
        h.update(b)
    assert h.collapsed_to_min(layer_min=16) is False
    assert abs(h.mean_budget() - 24.0) < 1e-9
