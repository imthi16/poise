"""Power-mode thermal sweeps (CLAUDE.md §6, from `poise_calibrate.py`).

For each Jetson power mode × layer depth, hold a sustained fixed load and record a
telemetry trace (temperature + power vs time). The warm-up curves feed ``fit.py``,
which identifies the RC parameters and the depth→power map.

On the board this drives a real fixed-depth workload and reads jtop/tegrastats. The
collector is clock-injectable so it is deterministic and fast off-device (mock
backend + a fake clock) for pipeline testing — those CSVs are labeled synthetic and
must never be fed to ``fit.py`` as if they were real calibration (they would just
re-derive the placeholder).
"""

from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..telemetry.reader import BaseTelemetryReader


@dataclass
class SweepSpec:
    depths: Sequence[int]
    power_modes: Sequence[str] = field(default_factory=lambda: ["MAXN"])
    duration_s: float = 120.0
    hz: float = 4.0
    load_level: float = 1.0
    synthetic: bool = False  # True when produced off-device via the mock


def collect_trace(
    reader: "BaseTelemetryReader",
    n_samples: int,
    period_s: float,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    run_id: Optional[str] = None,
) -> list[dict]:
    """Read ``n_samples`` telemetry samples ``period_s`` apart.

    ``sleeper`` advances time between samples — on the board it is ``time.sleep``;
    in tests it advances a fake clock shared with the mock reader so warm-up is
    synthesized deterministically without real waiting.
    """
    rows: list[dict] = []
    for _ in range(n_samples):
        s = reader.read()
        rows.append(
            {
                "ts": s.ts,
                "temp_c": s.temp_c,
                "power_w": s.power_w,
                "gpu_clock_mhz": s.gpu_clock_mhz,
                "gpu_util": s.gpu_util,
                "throttled": int(s.throttled),
            }
        )
        sleeper(period_s)
    return rows


def set_power_mode(mode: str) -> bool:
    """Set the Jetson nvpmodel power mode. No-op (returns False) off-device."""
    try:  # pragma: no cover - hardware path
        subprocess.run(["nvpmodel", "-m", _nvpmodel_index(mode)], check=True)
        subprocess.run(["jetson_clocks"], check=False)
        return True
    except Exception:
        return False


def _nvpmodel_index(mode: str) -> str:
    # MAXN is mode 0 on AGX Orin; named modes map to indices in /etc/nvpmodel.conf.
    table = {"MAXN": "0", "50W": "1", "30W": "2", "15W": "3"}
    return table.get(mode.upper(), mode)


def write_csv(path: str | Path, rows: list[dict], meta: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ts", "temp_c", "power_w", "gpu_clock_mhz", "gpu_util", "throttled"]
    with open(p, "w", newline="") as f:
        # Comment header carries sweep metadata (depth, power_mode, ambient, synthetic).
        f.write("# " + ",".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    return p


def read_csv(path: str | Path) -> tuple[dict, list[dict]]:
    p = Path(path)
    meta: dict = {}
    rows: list[dict] = []
    with open(p) as f:
        first = f.readline()
        if first.startswith("#"):
            for kv in first[1:].strip().split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    meta[k.strip()] = v.strip()
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ts": float(row["ts"]),
                    "temp_c": float(row["temp_c"]),
                    "power_w": float(row["power_w"]),
                    "gpu_clock_mhz": float(row["gpu_clock_mhz"]),
                    "gpu_util": float(row["gpu_util"]),
                    "throttled": int(row["throttled"]),
                }
            )
    return meta, rows


def run_sweep(
    cfg: "PoiseConfig",
    spec: SweepSpec,
    out_dir: str | Path = "data/calibration",
    *,
    reader: Optional["BaseTelemetryReader"] = None,
    sleeper: Callable[[float], None] = time.sleep,
    persist_db: bool = True,
) -> list[Path]:
    """Run the full sweep, writing one CSV per (power_mode, depth) and (optionally)
    persisting traces to the DB. Returns the CSV paths."""
    from ..telemetry import make_reader
    from ..storage.db import init_db
    from ..storage import models

    out = Path(out_dir)
    period = 1.0 / spec.hz
    n_samples = max(4, int(spec.duration_s * spec.hz))
    paths: list[Path] = []

    conn = init_db(cfg.storage.db_path) if persist_db else None
    own_reader = reader is None

    for mode in spec.power_modes:
        applied = set_power_mode(mode)
        synthetic = spec.synthetic or not applied
        for depth in spec.depths:
            r = reader if reader is not None else make_reader(cfg)
            # Configure the mock workload (real workload is driven externally).
            if hasattr(r, "set_budget"):
                r.set_budget(int(depth))
            if hasattr(r, "set_load"):
                r.set_load(spec.load_level)
            if own_reader and hasattr(r, "start"):
                pass  # synchronous read() is fine for the collector

            rows = collect_trace(r, n_samples, period, sleeper=sleeper)
            meta = {
                "depth": depth,
                "power_mode": mode,
                "ambient_c": cfg.thermal.ambient_c,
                "synthetic": synthetic,
                "load_level": spec.load_level,
            }
            path = write_csv(out / f"sweep_{mode}_d{depth}.csv", rows, meta)
            paths.append(path)

            if conn is not None:
                calib_id = models.insert_calibration_run(
                    conn, power_mode=mode, depth=int(depth), ambient_c=cfg.thermal.ambient_c,
                    params_json={"synthetic": synthetic, "load_level": spec.load_level},
                )
                models.insert_calibration_samples(
                    conn, calib_id,
                    [{"ts": r["ts"], "temp_c": r["temp_c"], "power_w": r["power_w"]} for r in rows],
                )
    if conn is not None:
        conn.close()
    return paths


def main() -> None:  # pragma: no cover - CLI entry (poise-calibrate)
    """CLI: run a calibration sweep using the active config. ON-DEVICE.

    Off-device this produces clearly-labeled SYNTHETIC traces (mock backend), useful
    only for exercising the sweep→fit pipeline, never as real calibration.
    """
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser(description="POISE thermal-power calibration sweep")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--modes", default="MAXN")
    ap.add_argument("--out", default="data/calibration")
    args = ap.parse_args()

    cfg = load_config()
    spec = SweepSpec(
        depths=list(cfg.depth.budget_set),
        power_modes=args.modes.split(","),
        duration_s=args.duration,
        hz=cfg.telemetry.hz,
    )
    paths = run_sweep(cfg, spec, out_dir=args.out)
    print(f"wrote {len(paths)} sweep CSVs to {args.out}")
    if cfg.telemetry.backend == "mock":
        print("NOTE: mock backend -> these are SYNTHETIC traces, not real calibration.")


__all__ = [
    "SweepSpec",
    "collect_trace",
    "set_power_mode",
    "write_csv",
    "read_csv",
    "run_sweep",
    "main",
]
