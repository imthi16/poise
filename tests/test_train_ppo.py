"""PPO training diagnostics off-device (⚠ reward-hacking audit; CLAUDE.md §9).

The SB3 training loop runs on Kaggle; here we lock down the auditable parts with a
fake policy: the converged-depth-distribution rollout, the collapse detector, the
quality-table resolution (measured > synthetic, with a warning), and the config
snapshot persisted to the ppo_checkpoints row.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from poise.rl.env import PoiseThermalEnv
from poise.rl.train_ppo import (
    _snapshot_configs,
    make_env_fn,
    resolve_quality_table,
    rollout_depth_distribution,
)


def test_rollout_detects_collapse_to_min_depth(cfg):
    env = PoiseThermalEnv(cfg, seed=0)
    # fake policy that always picks action 0 (== min depth) => reward hacking
    hist = rollout_depth_distribution(lambda obs: 0, env, n_steps=500, seed=0)
    assert hist.collapsed_to_min(cfg.depth.layer_min) is True
    assert hist.distribution()[cfg.depth.budget_set[0]] == 1.0


def test_rollout_healthy_distribution_not_collapsed(cfg):
    env = PoiseThermalEnv(cfg, seed=0)
    rng = np.random.default_rng(0)
    n_actions = len(cfg.depth.budget_set)
    hist = rollout_depth_distribution(
        lambda obs: int(rng.integers(0, n_actions)), env, n_steps=2000, seed=0
    )
    assert hist.collapsed_to_min(cfg.depth.layer_min) is False
    # all depths exercised
    assert set(hist.distribution().keys()) == set(cfg.depth.budget_set)


def test_resolve_quality_table_warns_and_uses_synthetic_without_gate(cfg):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q = resolve_quality_table(cfg, conn=None)
    assert q.synthetic is True
    assert any("SYNTHETIC" in str(x.message) for x in w)


def test_resolve_quality_table_prefers_measured(cfg, db):
    from poise.storage import models

    models.insert_quality_profile(db, depth=16, mean_kl_vs_full=0.5, dataset_tag="g")
    models.insert_quality_profile(db, depth=32, mean_kl_vs_full=0.0, dataset_tag="g")
    q = resolve_quality_table(cfg, conn=db)
    assert q.synthetic is False  # measured table used, not synthetic
    assert q.cost(32) == 0.0


def test_env_fn_builds_env(cfg):
    from poise.rl.reward import synthetic_quality_table

    qt = synthetic_quality_table(list(cfg.depth.budget_set), cfg.depth.layer_total)
    env = make_env_fn(cfg, qt, domain_random=True, seed=1)()
    obs, _ = env.reset()
    assert obs.shape[0] == 8


def test_snapshot_configs_captures_reward_env_sim(cfg):
    reward_cfg, env_cfg, sim_params = _snapshot_configs(cfg)
    assert "w_q" in reward_cfg and "w_throttle" in reward_cfg
    assert "budget_set" in env_cfg
    assert sim_params["params_source"] == "placeholder"
    assert "domain_random" in sim_params
