"""Gymnasium env conformance + dynamics + reward-hacking visibility (CLAUDE.md §6, §10)."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from poise.rl.env import OBS_DIM, ObsSpec, PoiseThermalEnv, build_observation
from poise.rl.domain_random import DomainRandomizer
from poise.rl.simulator import RCThermalSimulator


def test_env_passes_gymnasium_checker(cfg):
    from gymnasium.utils.env_checker import check_env

    env = PoiseThermalEnv(cfg, seed=0)
    # raises if the env violates the Gymnasium API contract
    check_env(env, skip_render_check=True)


def test_reset_and_step_shapes(cfg):
    env = PoiseThermalEnv(cfg, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == len(cfg.depth.budget_set)

    obs2, reward, terminated, truncated, info = env.step(0)
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert "reward_components" in info and "depth_distribution" in info


def test_episode_truncates_at_time_limit(cfg):
    env = PoiseThermalEnv(cfg, seed=1)
    env.reset(seed=1)
    steps = 0
    truncated = False
    while not truncated and steps < 100000:
        _, _, terminated, truncated, _ = env.step(env.action_space.n - 1)
        steps += 1
    expected = int(cfg.ppo.env.episode_seconds / cfg.ppo.env.dt_s)
    assert abs(steps - expected) <= 1


def test_determinism_with_seed(cfg):
    e1 = PoiseThermalEnv(cfg, seed=42)
    e2 = PoiseThermalEnv(cfg, seed=42)
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    np.testing.assert_allclose(o1, o2)
    for a in [0, 1, 2, 1, 0]:
        s1, r1, _, _, _ = e1.step(a)
        s2, r2, _, _, _ = e2.step(a)
        np.testing.assert_allclose(s1, s2)
        assert r1 == r2


def test_domain_randomization_varies_params(cfg):
    dr = DomainRandomizer(enabled=True)
    env = PoiseThermalEnv(cfg, domain_random=dr, seed=0)
    r_ths = set()
    for s in range(5):
        env.reset(seed=s)
        r_ths.add(round(env.sim.r_th, 6))
    assert len(r_ths) > 1  # parameters change across episodes


def test_always_min_depth_shows_collapse(cfg):
    """The reward-hacking signal must be observable: always picking min depth collapses
    the depth distribution (train_ppo logs exactly this)."""
    env = PoiseThermalEnv(cfg, seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(50):
        _, _, _, truncated, info = env.step(0)  # action 0 == min depth
        if truncated:
            break
    dist = info["depth_distribution"]
    assert dist[cfg.depth.budget_set[0]] == 1.0  # 100% at min depth => collapsed


def test_shared_obs_builder_matches(cfg):
    """policy.py must reuse this builder; verify it is a pure function of its inputs."""
    sim = RCThermalSimulator.from_config(cfg)
    spec = ObsSpec.from_config(cfg, sim)
    kw = dict(temp_c=70.0, power_w=30.0, throttled=False, prev_budget=24,
              tok_per_s=25.0, ambient_c=25.0, t_elapsed_s=10.0)
    a = build_observation(spec, **kw)
    b = build_observation(spec, **kw)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (OBS_DIM,)
