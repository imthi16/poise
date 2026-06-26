"""RC simulator dynamics + held-out-trace validation (CLAUDE.md §6, §10).

§10: "RC dynamics reproduce held-out real traces within a reported error bound; no
invented params." Off-device we validate the dynamics against a trace generated from
known parameters (a stand-in for a real held-out trace) and REPORT the fit error.
"""

from __future__ import annotations

import math

import numpy as np

from poise.rl.simulator import RCThermalSimulator


def _make_sim(r_th=1.5, c_th=40.0, ambient=25.0, temp_max=87.0, init=30.0):
    return RCThermalSimulator(
        r_th_c_per_w=r_th,
        c_th_j_per_c=c_th,
        ambient_c=ambient,
        temp_max_c=temp_max,
        depth_power_w={16: 18.0, 24: 26.0, 32: 35.0},
        depth_latency_s={16: 0.018, 24: 0.027, 32: 0.036},
        init_temp_c=init,
    )


def test_power_interpolation_monotonic():
    sim = _make_sim()
    assert sim.power_for_depth(16) == 18.0
    assert sim.power_for_depth(32) == 35.0
    assert 18.0 < sim.power_for_depth(20) < 35.0  # interpolated


def test_steady_state_matches_analytic():
    sim = _make_sim(r_th=1.5, ambient=25.0)
    # T_ss = ambient + P(32)*R_th = 25 + 35*1.5 = 77.5
    for _ in range(2000):
        s = sim.step(32, dt=1.0)
    assert abs(s.temp_c - (25.0 + 35.0 * 1.5)) < 0.5


def test_lower_depth_runs_cooler():
    hot = _make_sim()
    cool = _make_sim()
    for _ in range(500):
        th = hot.step(32, 1.0).temp_c
        tc = cool.step(16, 1.0).temp_c
    assert tc < th


def test_throttle_at_ceiling():
    sim = _make_sim(r_th=3.0, ambient=40.0, temp_max=80.0)  # T_eq=40+35*3=145 -> throttles
    throttled = False
    for _ in range(500):
        s = sim.step(32, 1.0)
        if s.throttled:
            throttled = True
            assert s.tok_per_s < 1.0 / 0.036  # slower than nominal due to clock clamp
            break
    assert throttled


def test_reset_restores_initial_temp():
    sim = _make_sim(init=30.0)
    for _ in range(100):
        sim.step(32, 1.0)
    assert sim.reset(init_temp_c=30.0) == 30.0
    assert sim.temp_c == 30.0


def test_dynamics_reproduce_heldout_trace_within_bound():
    """Validate against a held-out trace from known params; REPORT the RMSE."""
    r_th, c_th, ambient, init = 1.6, 30.0, 25.0, 35.0
    tau = r_th * c_th
    power = 35.0
    dt = 1.0
    # held-out analytic trace at full depth. The sim's k-th step lands at t=(k+1)*dt
    # (it advances dt from the initial temp), so evaluate the analytic solution there.
    n = 200
    t = (np.arange(n) + 1) * dt
    t_eq = ambient + power * r_th
    analytic = t_eq + (init - t_eq) * np.exp(-t / tau)

    sim = RCThermalSimulator(
        r_th_c_per_w=r_th, c_th_j_per_c=c_th, ambient_c=ambient, temp_max_c=200.0,
        depth_power_w={32: power}, depth_latency_s={32: 0.036}, init_temp_c=init,
    )
    sim_temps = [sim.step(32, dt).temp_c for _ in range(n)]
    rmse = float(np.sqrt(np.mean((np.array(sim_temps) - analytic) ** 2)))
    # exact exponential integration => essentially zero error vs the analytic solution
    assert rmse < 1e-6, f"held-out RMSE too high: {rmse}"


def test_from_config_uses_placeholder_off_device(cfg):
    sim = RCThermalSimulator.from_config(cfg)
    assert sim.params_source == "placeholder"  # honest: not calibrated off-device
    assert sim.r_th > 0 and sim.c_th > 0
