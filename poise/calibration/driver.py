"""Calibration LOAD driver (CLAUDE.md §5 step 5).

The sweep in ``sweep.py`` only *records* telemetry — it assumes "the real workload is
driven externally". This module IS that external workload: it drives sustained
fixed-depth inference through the real ``AdaptiveRunner`` while a background thread
samples the telemetry reader, producing one heating trace per depth. Those traces feed
the existing fit pipeline (``fit.fit_rc_from_trace`` → ``aggregate_fits`` →
``write_simulator_yaml``) to flip ``configs/simulator.yaml`` to calibrated.

🔒 Real params come ONLY from real on-board runs. Off-device (mock backend / tiny model)
this exercises the pipeline end-to-end but its traces are SYNTHETIC — never a real fit.
Run on the Jetson with the real model + jtop for a real calibration.

    python3.10 -m poise.calibration.driver --duration 120 --hz 4          # real (on board)
    python3.10 -m poise.calibration.driver --model sshleifer/tiny-gpt2 \\
        --device cpu --duration 6 --out-yaml /tmp/sim.yaml --dry-run       # pipeline smoke
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


@dataclass
class DepthRun:
    """One depth's heating run: time / temperature / power sample arrays + latency."""
    depth: int
    t_s: list[float] = field(default_factory=list)      # seconds since run start
    temp_c: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
    throttled: list[int] = field(default_factory=list)
    per_token_latency_s: float = 0.0
    tokens: int = 0
    synthetic: bool = True


def _sampler(reader: Any, stop: threading.Event, hz: float, run: DepthRun, t0: float,
             sleeper: Callable[[float], None] = time.sleep) -> None:
    """Background thread: append one telemetry sample per 1/hz until ``stop`` is set."""
    period = 1.0 / hz
    while not stop.is_set():
        s = reader.read()
        run.t_s.append(time.monotonic() - t0)
        run.temp_c.append(float(s.temp_c))
        run.power_w.append(float(s.power_w))
        run.throttled.append(int(bool(s.throttled)))
        sleeper(period)


def drive_depth(
    runner: Any,
    reader: Any,
    depth: int,
    duration_s: float,
    hz: float,
    *,
    prompt: str = "Describe the process of heat dissipation in a dense compute workload.",
    tokens_per_call: int = 64,
    sleeper: Callable[[float], None] = time.sleep,
) -> DepthRun:
    """Sustain fixed-``depth`` generation for ``duration_s`` while sampling telemetry."""
    run = DepthRun(depth=depth)
    # If the reader is the mock (has set_budget/set_load) let it model this depth's heat.
    if hasattr(reader, "set_budget"):
        reader.set_budget(int(depth))
    if hasattr(reader, "set_load"):
        reader.set_load(1.0)

    budget_fn = lambda ctx: depth  # noqa: E731 — constant budget pins the depth
    stop = threading.Event()
    t0 = time.monotonic()
    sampler = threading.Thread(target=_sampler, args=(reader, stop, hz, run, t0, sleeper),
                               daemon=True)
    sampler.start()

    gen_time, gen_tokens = 0.0, 0
    try:
        while time.monotonic() - t0 < duration_s:
            g0 = time.monotonic()
            result = runner.generate(prompt, tokens_per_call, budget_fn)
            dt = time.monotonic() - g0
            n = len(getattr(result, "trace", []) or []) or tokens_per_call
            gen_time += dt
            gen_tokens += n
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    run.tokens = gen_tokens
    run.per_token_latency_s = (gen_time / gen_tokens) if gen_tokens else 0.0
    return run


def run_load_sweep(
    cfg: "PoiseConfig",
    *,
    depths: Optional[Sequence[int]] = None,
    duration_s: float = 120.0,
    hz: float = 4.0,
    cooldown_s: float = 0.0,
    runner: Any = None,
    reader: Any = None,
    synthetic: Optional[bool] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[DepthRun]:
    """Drive a heating run at each depth. Builds a real runner+reader unless injected."""
    from ..telemetry import make_reader

    reader = reader if reader is not None else make_reader(cfg)
    is_mock = hasattr(reader, "set_budget")
    synth = is_mock if synthetic is None else synthetic

    if runner is None:  # build the real engine, rescaling depth config to the model
        from ..engine.model_loader import load_model, get_num_layers
        from ..engine.adaptive_runner import AdaptiveRunner
        from ..hardware import adapt_depth_to_model
        model, tokenizer = load_model(cfg)
        cfg = adapt_depth_to_model(cfg, get_num_layers(model))
        runner = AdaptiveRunner(cfg, model, tokenizer, telemetry_reader=reader)

    depths = list(depths if depths is not None else cfg.depth.budget_set)

    runs: list[DepthRun] = []
    for depth in depths:
        run = drive_depth(runner, reader, depth, duration_s, hz, sleeper=sleeper)
        run.synthetic = synth
        runs.append(run)
        if cooldown_s > 0 and depth != depths[-1]:
            sleeper(cooldown_s)
    return runs


def calibrate(
    cfg: "PoiseConfig",
    *,
    duration_s: float = 120.0,
    hz: float = 4.0,
    cooldown_s: float = 0.0,
    out_dir: str | Path = "data/calibration",
    out_yaml: str | Path = "configs/simulator.yaml",
    write_yaml: bool = True,
    persist_db: bool = True,
    runner: Any = None,
    reader: Any = None,
    synthetic: Optional[bool] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Full calibration: drive per-depth heating runs, fit RC + depth→power/latency,
    persist CSV/DB, and (real data only) write the calibrated block of simulator.yaml."""
    import numpy as np

    from . import fit as fitmod
    from .sweep import write_csv

    runs = run_load_sweep(cfg, duration_s=duration_s, hz=hz, cooldown_s=cooldown_s,
                          runner=runner, reader=reader, synthetic=synthetic, sleeper=sleeper)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ambient = cfg.thermal.ambient_c

    rc_fits = []
    power_by_depth: dict[int, list[float]] = {}
    latency_by_depth_s: dict[int, list[float]] = {}
    any_synthetic = False

    for run in runs:
        any_synthetic = any_synthetic or run.synthetic
        power_by_depth[run.depth] = run.power_w
        latency_by_depth_s[run.depth] = [run.per_token_latency_s]

        # steady-state power over the tail drives this run's constant-power RC fit
        tail = max(1, int(len(run.power_w) * 0.3))
        p_ss = float(np.mean(run.power_w[-tail:])) if run.power_w else 0.0
        if len(run.t_s) >= 4 and p_ss > 0:
            rc = fitmod.fit_rc_from_trace(
                np.asarray(run.t_s), np.asarray(run.temp_c), power_w=p_ss, ambient_c=ambient
            )
            rc_fits.append(rc)

        # persist a CSV compatible with sweep.read_csv
        rows = [
            {"ts": run.t_s[i], "temp_c": run.temp_c[i], "power_w": run.power_w[i],
             "gpu_clock_mhz": 0.0, "gpu_util": 0.0, "throttled": run.throttled[i]}
            for i in range(len(run.t_s))
        ]
        write_csv(out / f"loadsweep_d{run.depth}.csv", rows,
                  {"depth": run.depth, "power_mode": "MAXN", "ambient_c": ambient,
                   "synthetic": run.synthetic, "per_token_latency_s": run.per_token_latency_s})

    if not rc_fits:
        raise RuntimeError("no usable RC fits — traces too short or zero power; "
                           "increase --duration or check the telemetry reader")

    depth_power = fitmod.steady_state_power_by_depth(power_by_depth)
    depth_latency = fitmod.latency_by_depth(latency_by_depth_s)
    result = fitmod.aggregate_fits(rc_fits, depth_power, depth_latency)

    calib_id = None
    if persist_db:
        from ..storage.db import init_db
        conn = init_db(cfg.storage.db_path)
        try:
            calib_id = fitmod.persist_calibration(conn, result, power_mode="MAXN")
        finally:
            conn.close()

    wrote_yaml = False
    if write_yaml:
        if any_synthetic:
            # 🔒 never let synthetic/mock traces masquerade as a real calibration
            print("[calibrate] SYNTHETIC traces (mock backend / tiny model) — refusing to "
                  "write the calibrated block of the real simulator.yaml.")
        else:
            fitmod.write_simulator_yaml(result, out_yaml)
            wrote_yaml = True

    return {"result": result, "runs": runs, "calib_id": calib_id,
            "wrote_yaml": wrote_yaml, "synthetic": any_synthetic}


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser(description="POISE calibration load driver (real workload)")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds of load per depth")
    ap.add_argument("--hz", type=float, default=4.0, help="telemetry sample rate")
    ap.add_argument("--cooldown", type=float, default=0.0, help="idle seconds between depths")
    ap.add_argument("--model", default=None, help="override POISE_MODEL_ID (e.g. tiny-gpt2)")
    ap.add_argument("--device", default=None, help="override POISE_DEVICE (cuda|cpu)")
    ap.add_argument("--out", default="data/calibration")
    ap.add_argument("--out-yaml", default="configs/simulator.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write simulator.yaml (pipeline smoke test)")
    args = ap.parse_args()

    import os
    if args.model:
        os.environ["POISE_MODEL_ID"] = args.model
    if args.device:
        os.environ["POISE_DEVICE"] = args.device

    cfg = load_config()
    print(f"[calibrate] depths={list(cfg.depth.budget_set)} duration={args.duration}s "
          f"hz={args.hz} device={cfg.model.device} backend={cfg.telemetry.backend}")
    t0 = time.monotonic()
    res = calibrate(cfg, duration_s=args.duration, hz=args.hz, cooldown_s=args.cooldown,
                    out_dir=args.out, out_yaml=args.out_yaml, write_yaml=not args.dry_run)
    r = res["result"]
    print(f"[calibrate] done in {time.monotonic()-t0:.0f}s"
          f"  (synthetic={res['synthetic']})")
    print(f"  R_th = {r.r_th_c_per_w:.4f} K/W   C_th = {r.c_th_j_per_c:.1f} J/K   "
          f"tau = {r.tau_s:.1f}s   fit RMSE = {r.fit_rmse_c:.3f} C")
    print(f"  depth->power (W): {r.depth_power_w}")
    print(f"  depth->latency (s/tok): { {k: round(v,4) for k,v in r.depth_latency_s.items()} }")
    print(f"  wrote simulator.yaml: {res['wrote_yaml']}  (calib_id={res['calib_id']})")


if __name__ == "__main__":
    main()
