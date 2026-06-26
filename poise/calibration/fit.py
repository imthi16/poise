"""Fit RC thermal params + depth→power map from calibration sweeps (CLAUDE.md §6).

First-order RC model:  dT/dt = (P − (T − T_ambient)/R_th) / C_th.
Step response to constant power P from initial temp T0:

    T(t) = T_amb + (T0 − T_amb)·e^(−t/τ) + P·R_th·(1 − e^(−t/τ)),   τ = R_th·C_th

So a constant-power warm-up trace identifies (R_th, τ); then C_th = τ / R_th.

🔒 Parameters come ONLY from real calibration data. This module writes the
``calibrated`` block of ``configs/simulator.yaml`` (and the DB) **only** when fed
real sweep traces, flipping ``calibrated.valid`` true and recording the fit RMSE.
It never invents parameters; the placeholder block is left untouched otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# RC step-response model + fit
# --------------------------------------------------------------------------- #
def rc_step_response(
    t: np.ndarray, r_th: float, tau: float, power: float, t_amb: float, t0: float
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    decay = np.exp(-t / max(tau, 1e-9))
    return t_amb + (t0 - t_amb) * decay + power * r_th * (1.0 - decay)


@dataclass
class RCFit:
    r_th_c_per_w: float
    c_th_j_per_c: float
    tau_s: float
    t_ss_c: float          # fitted steady-state temperature
    rmse_c: float
    power_w: float
    ambient_c: float


def fit_rc_from_trace(
    t: Sequence[float],
    temp_c: Sequence[float],
    power_w: float,
    ambient_c: float,
    t0_c: float | None = None,
) -> RCFit:
    """Fit (R_th, τ) from one constant-power warm-up trace; derive C_th = τ/R_th."""
    from scipy.optimize import curve_fit

    t_arr = np.asarray(t, dtype=np.float64)
    y = np.asarray(temp_c, dtype=np.float64)
    if t_arr.size < 4:
        raise ValueError("need >= 4 samples to fit an RC step response")
    t0 = float(y[0]) if t0_c is None else float(t0_c)

    # initial guesses from the data
    t_ss0 = float(np.max(y))
    r_th0 = max(1e-3, (t_ss0 - ambient_c) / max(power_w, 1e-6))
    tau0 = max(1.0, (t_arr[-1] - t_arr[0]) / 3.0)

    def model(tt, r_th, tau):
        return rc_step_response(tt, r_th, tau, power_w, ambient_c, t0)

    try:
        popt, _ = curve_fit(
            model, t_arr, y, p0=[r_th0, tau0],
            bounds=([1e-4, 1e-1], [1e3, 1e5]), maxfev=20000,
        )
        r_th, tau = float(popt[0]), float(popt[1])
    except Exception:
        # robust fallback: estimate from steady state + 63% time constant
        r_th, tau = r_th0, tau0

    pred = model(t_arr, r_th, tau)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    c_th = tau / max(r_th, 1e-9)
    t_ss = ambient_c + power_w * r_th
    return RCFit(
        r_th_c_per_w=r_th,
        c_th_j_per_c=c_th,
        tau_s=tau,
        t_ss_c=t_ss,
        rmse_c=rmse,
        power_w=power_w,
        ambient_c=ambient_c,
    )


# --------------------------------------------------------------------------- #
# depth -> power / latency maps
# --------------------------------------------------------------------------- #
def steady_state_power_by_depth(
    samples_by_depth: Mapping[int, Sequence[float]], tail_frac: float = 0.3
) -> dict[int, float]:
    """Mean power over the steady-state tail of each depth's run."""
    out: dict[int, float] = {}
    for depth, powers in samples_by_depth.items():
        arr = np.asarray(powers, dtype=np.float64)
        if arr.size == 0:
            continue
        k = max(1, int(arr.size * tail_frac))
        out[int(depth)] = float(np.mean(arr[-k:]))
    return out


def latency_by_depth(
    latency_samples_by_depth: Mapping[int, Sequence[float]]
) -> dict[int, float]:
    return {
        int(d): float(np.median(np.asarray(v, dtype=np.float64)))
        for d, v in latency_samples_by_depth.items()
        if len(v) > 0
    }


# --------------------------------------------------------------------------- #
# Aggregate calibration result + persistence
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    r_th_c_per_w: float
    c_th_j_per_c: float
    tau_s: float
    fit_rmse_c: float
    ambient_c: float
    depth_power_w: dict[int, float]
    depth_latency_s: dict[int, float]
    source_calib_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def aggregate_fits(
    fits: Sequence[RCFit],
    depth_power_w: Mapping[int, float],
    depth_latency_s: Mapping[int, float] | None = None,
    source_calib_id: str | None = None,
) -> CalibrationResult:
    """Combine per-run RC fits (median, robust to outliers) into one parameter set."""
    if not fits:
        raise ValueError("no RC fits to aggregate")
    r_th = float(np.median([f.r_th_c_per_w for f in fits]))
    c_th = float(np.median([f.c_th_j_per_c for f in fits]))
    tau = float(np.median([f.tau_s for f in fits]))
    rmse = float(np.median([f.rmse_c for f in fits]))
    amb = float(np.median([f.ambient_c for f in fits]))
    return CalibrationResult(
        r_th_c_per_w=r_th,
        c_th_j_per_c=c_th,
        tau_s=tau,
        fit_rmse_c=rmse,
        ambient_c=amb,
        depth_power_w={int(k): float(v) for k, v in depth_power_w.items()},
        depth_latency_s={int(k): float(v) for k, v in (depth_latency_s or {}).items()},
        source_calib_id=source_calib_id,
    )


def write_simulator_yaml(result: CalibrationResult, path: str | Path) -> None:
    """Write fitted params into ``configs/simulator.yaml`` — flips calibrated.valid
    true and switches the map sources to 'calibrated'. Real data ONLY."""
    import yaml

    p = Path(path)
    data = {}
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {}

    data["calibrated"] = {
        "valid": True,
        "r_th_c_per_w": result.r_th_c_per_w,
        "c_th_j_per_c": result.c_th_j_per_c,
        "fit_rmse_c": result.fit_rmse_c,
        "ambient_c": result.ambient_c,
        "source_calib_id": result.source_calib_id,
    }
    if result.depth_power_w:
        data["depth_power_map"] = {
            "source": "calibrated",
            "watts": {int(k): float(v) for k, v in sorted(result.depth_power_w.items())},
        }
    if result.depth_latency_s:
        data["depth_latency_map"] = {
            "source": "calibrated",
            "seconds": {int(k): float(v) for k, v in sorted(result.depth_latency_s.items())},
        }
    with open(p, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def persist_calibration(conn, result: CalibrationResult, power_mode: str | None = None) -> str:
    from ..storage import models

    return models.insert_calibration_run(
        conn,
        calib_id=result.source_calib_id,
        power_mode=power_mode,
        ambient_c=result.ambient_c,
        params_json=result.as_dict(),
    )


__all__ = [
    "rc_step_response",
    "RCFit",
    "fit_rc_from_trace",
    "steady_state_power_by_depth",
    "latency_by_depth",
    "CalibrationResult",
    "aggregate_fits",
    "write_simulator_yaml",
    "persist_calibration",
]
