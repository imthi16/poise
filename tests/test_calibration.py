"""Calibration: RC recovery from a synthetic trace + sweep→fit pipeline (CLAUDE.md §6).

Real RC params come only from on-board sweeps. Here we verify the FIT MATH recovers
known parameters from a synthetic warm-up curve, the CSV round-trips, and the
sweep→fit pipeline runs deterministically off-device via a fake clock. We also assert
the placeholder simulator.yaml is NOT silently marked calibrated by synthetic data.
"""

from __future__ import annotations

import numpy as np

from poise.calibration.fit import (
    aggregate_fits,
    fit_rc_from_trace,
    rc_step_response,
    steady_state_power_by_depth,
    write_simulator_yaml,
)
from poise.calibration.sweep import (
    SweepSpec,
    collect_trace,
    read_csv,
    run_sweep,
    write_csv,
)
from poise.telemetry.mock import MockTelemetryReader


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_rc_fit_recovers_known_params():
    # Ground-truth RC: R_th=1.6 K/W, C_th=30 J/K => tau=48 s, under P=25 W.
    r_th_true, c_th_true, power, amb, t0 = 1.6, 30.0, 25.0, 25.0, 30.0
    tau_true = r_th_true * c_th_true
    t = np.linspace(0, 240, 120)
    clean = rc_step_response(t, r_th_true, tau_true, power, amb, t0)
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0, 0.05, size=clean.shape)

    fit = fit_rc_from_trace(t, noisy, power_w=power, ambient_c=amb, t0_c=t0)
    assert abs(fit.r_th_c_per_w - r_th_true) < 0.1
    assert abs(fit.c_th_j_per_c - c_th_true) < 3.0
    assert fit.rmse_c < 0.5


def test_steady_state_power_by_depth_uses_tail():
    samples = {16: [10, 12, 18, 18, 18], 32: [20, 30, 35, 35, 35]}
    p = steady_state_power_by_depth(samples, tail_frac=0.4)
    assert abs(p[16] - 18.0) < 1e-6
    assert abs(p[32] - 35.0) < 1e-6


def test_csv_roundtrip(tmp_path):
    rows = [
        {"ts": 0.0, "temp_c": 40.0, "power_w": 20.0, "gpu_clock_mhz": 1300.0,
         "gpu_util": 90.0, "throttled": 0},
        {"ts": 0.25, "temp_c": 41.0, "power_w": 21.0, "gpu_clock_mhz": 1300.0,
         "gpu_util": 91.0, "throttled": 0},
    ]
    meta = {"depth": 24, "power_mode": "MAXN", "synthetic": True}
    path = write_csv(tmp_path / "s.csv", rows, meta)
    rmeta, rrows = read_csv(path)
    assert rmeta["depth"] == "24" and rmeta["synthetic"] == "True"
    assert len(rrows) == 2 and rrows[1]["temp_c"] == 41.0


def test_collect_trace_warms_up_with_fake_clock():
    clk = FakeClock()
    reader = MockTelemetryReader(ambient_c=25.0, init_temp_c=30.0, load_level=1.0,
                                 budget=32, tau_s=20.0, clock_fn=clk)
    rows = collect_trace(reader, n_samples=80, period_s=1.0, sleeper=clk.advance)
    assert len(rows) == 80
    assert rows[-1]["temp_c"] > rows[0]["temp_c"]  # a real warm-up curve


def test_sweep_then_fit_pipeline(cfg, tmp_path):
    """End-to-end off-device: synthetic sweep -> RC fit -> depth-power map."""
    clk = FakeClock()
    reader = MockTelemetryReader(ambient_c=cfg.thermal.ambient_c, init_temp_c=30.0,
                                 load_level=1.0, budget=32, tau_s=20.0, clock_fn=clk)
    spec = SweepSpec(depths=[16, 32], duration_s=60.0, hz=2.0, synthetic=True)
    paths = run_sweep(cfg, spec, out_dir=tmp_path, reader=reader,
                      sleeper=clk.advance, persist_db=False)
    assert len(paths) == 2

    fits = []
    power_by_depth = {}
    for p in paths:
        meta, rows = read_csv(p)
        depth = int(meta["depth"])
        t = [r["ts"] - rows[0]["ts"] for r in rows]
        temp = [r["temp_c"] for r in rows]
        power = float(np.mean([r["power_w"] for r in rows][-10:]))
        fits.append(fit_rc_from_trace(t, temp, power_w=power, ambient_c=cfg.thermal.ambient_c))
        power_by_depth[depth] = power

    result = aggregate_fits(fits, power_by_depth)
    assert result.r_th_c_per_w > 0 and result.c_th_j_per_c > 0
    assert set(result.depth_power_w.keys()) == {16, 32}


def test_write_simulator_yaml_marks_calibrated(tmp_path):
    """write_simulator_yaml flips calibrated.valid true — only ever called with real data."""
    import yaml

    from poise.calibration.fit import CalibrationResult

    result = CalibrationResult(
        r_th_c_per_w=1.6, c_th_j_per_c=30.0, tau_s=48.0, fit_rmse_c=0.3,
        ambient_c=25.0, depth_power_w={16: 18.0, 32: 35.0}, depth_latency_s={},
        source_calib_id="calib-xyz",
    )
    out = tmp_path / "simulator.yaml"
    write_simulator_yaml(result, out)
    data = yaml.safe_load(out.read_text())
    assert data["calibrated"]["valid"] is True
    assert data["depth_power_map"]["source"] == "calibrated"
    assert data["calibrated"]["r_th_c_per_w"] == 1.6


def test_repo_simulator_yaml_still_placeholder():
    """The committed simulator.yaml must remain uncalibrated (no fabricated params)."""
    import yaml
    from pathlib import Path

    repo_yaml = Path(__file__).resolve().parent.parent / "configs" / "simulator.yaml"
    data = yaml.safe_load(repo_yaml.read_text())
    assert data["calibrated"]["valid"] is False
    assert data["depth_power_map"]["source"] == "placeholder"
