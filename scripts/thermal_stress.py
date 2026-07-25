"""Thermal-throttle stress test (CLAUDE.md §1 objective, §11) — REAL Jetson, reboot-free.

Creates the sustained-thermal-load regime POISE targets by reducing the fan (manual pwm,
nvfancontrol stopped) so junction temp climbs to the hardware passive-throttle point under
load. Then compares control modes under that regime:
  * static-32 : fixed full depth  -> should heat, throttle, throughput sags
  * static-16 : fixed low depth    -> cooler, faster, quality cost
  * pid       : adaptive           -> sheds depth as tj rises, aims to hold throughput/temp

Records per-token time series (t, depth, tj, power, latency) and steady-state tail metrics.

🔒 SAFETY: a monitor thread reads tj-thermal every 1s; if it reaches --abort-temp it
restores the fan to full + restarts nvfancontrol and aborts. The fan/service are ALWAYS
restored on exit (finally). The hardware also self-throttles at ~87C independently.
Needs cached sudo (run `sudo -v` first). Energy/thermal are physical -> Jetson only.

    sudo -v && PYTHONPATH=. python3.10 scripts/thermal_stress.py \\
        --modes static32,static16,pid --duration 150 --fan 0 --abort-temp 92 --tag cap
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import threading
import time
from pathlib import Path

FAN_PWM = "/sys/class/hwmon/hwmon0/pwm1"


def find_tj_path() -> str | None:
    for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        try:
            if open(z + "/type").read().strip() == "tj-thermal":
                return z + "/temp"
        except Exception:
            pass
    return None


def read_c(path: str) -> float:
    return int(open(path).read()) / 1000.0


import os as _os


def _load_sudo_pw() -> str:
    f = _os.environ.get("POISE_SUDO_PW_FILE")
    if f and _os.path.exists(f):
        return open(f).read().strip()
    return _os.environ.get("POISE_SUDO_PW", "")


_SUDO_PW = _load_sudo_pw()


def _sudo(cmd: str) -> None:
    # -S reads the password from stdin (survives nohup / no-TTY, unlike the ticket cache).
    subprocess.run(["sudo", "-S", "bash", "-c", cmd], check=False,
                   input=(_SUDO_PW + "\n"), capture_output=True, text=True)


def set_fan(pwm: int) -> None:
    _sudo(f"echo {int(pwm)} > {FAN_PWM}")


def restore_cooling() -> None:
    _sudo(f"echo 255 > {FAN_PWM}; systemctl start nvfancontrol")


class TjMonitor(threading.Thread):
    def __init__(self, tj_path: str, abort_temp: float):
        super().__init__(daemon=True)
        self.tj_path = tj_path
        self.abort_temp = abort_temp
        self.series: list[tuple[float, float]] = []  # (t, tj)
        self.emergency = threading.Event()
        self.stop_flag = threading.Event()
        self.t0 = time.monotonic()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                tj = read_c(self.tj_path)
                self.series.append((time.monotonic() - self.t0, tj))
                if tj >= self.abort_temp and not self.emergency.is_set():
                    self.emergency.set()
                    if not getattr(self, "no_fan", False):
                        restore_cooling()
            except Exception:
                pass
            time.sleep(1.0)

    def tj_now(self) -> float:
        return self.series[-1][1] if self.series else 0.0


def cool_to(tj_path: str, target_c: float, timeout_s: float = 120.0) -> None:
    """Full fan until tj drops below target (for a fair per-mode baseline)."""
    set_fan(255)
    t0 = time.monotonic()
    while read_c(tj_path) > target_c and time.monotonic() - t0 < timeout_s:
        time.sleep(2.0)


def run_mode(runner, budget_fn, duration_s, mon, prompt, chunk=96):
    """Sustained generation for duration_s; returns per-token rows."""
    rows = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s and not mon.emergency.is_set():
        res = runner.generate(prompt, chunk, budget_fn, return_trace=True)
        now = time.monotonic() - t0
        for t in res.trace:
            rows.append({"t": now, "budget": t.budget, "temp_c": t.temp_c,
                         "power_w": t.power_w, "latency_ms": t.latency_ms})
    return rows


def summarize(rows, tail_frac=0.4):
    import numpy as np
    if not rows:
        return {}
    lat = np.array([r["latency_ms"] for r in rows if r["latency_ms"]])
    k = max(1, int(len(rows) * tail_frac))
    tail = rows[-k:]
    tail_lat = np.array([r["latency_ms"] for r in tail if r["latency_ms"]])
    tail_pw = np.array([r["power_w"] for r in tail if r["power_w"]])
    tps_tail = 1000.0 / np.mean(tail_lat) if tail_lat.size else 0.0
    tps_all = 1000.0 / np.mean(lat) if lat.size else 0.0
    return {
        "tokens": len(rows),
        "tok_per_s_tail": float(tps_tail),
        "tok_per_s_all": float(tps_all),
        "mean_budget_tail": float(np.mean([r["budget"] for r in tail])),
        "peak_temp_c": float(np.max([r["temp_c"] for r in rows])),
        "mean_power_tail_w": float(np.mean(tail_pw)) if tail_pw.size else 0.0,
        "energy_per_token_j_tail": float(np.mean(tail_pw) / tps_tail)
        if tail_pw.size and tps_tail else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Jetson thermal-throttle stress test")
    ap.add_argument("--modes", default="static32,static16,pid")
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--fan", type=int, default=0, help="pwm 0-255 during load (low=hot)")
    ap.add_argument("--abort-temp", type=float, default=92.0)
    ap.add_argument("--cool-to", type=float, default=55.0, help="baseline temp between modes")
    ap.add_argument("--prompt", default="Explain how thermal throttling limits sustained "
                    "compute throughput on an edge accelerator, in detail.")
    ap.add_argument("--tag", default="cap")
    ap.add_argument("--no-fan-control", action="store_true",
                    help="do not touch the fan/nvfancontrol (no sudo); run at normal cooling")
    ap.add_argument("--out", default="data/results/thermal_stress")
    args = ap.parse_args()
    NO_FAN = args.no_fan_control

    tj_path = find_tj_path()
    if not tj_path:
        raise SystemExit("tj-thermal zone not found")
    if not NO_FAN:
        chk = subprocess.run(["sudo", "-S", "true"], input=(_SUDO_PW + "\n"),
                             capture_output=True, text=True)
        if chk.returncode != 0:
            raise SystemExit("sudo failed — set POISE_SUDO_PW(_FILE) or use --no-fan-control")

    import os
    from poise.config import load_config
    from poise.engine.model_loader import load_model, get_num_layers
    from poise.engine.adaptive_runner import AdaptiveRunner
    from poise.hardware import adapt_depth_to_model
    from poise.telemetry import make_reader
    from poise.control.budget_allocator import make_allocator

    cfg = load_config()
    model, tok = load_model(cfg)
    cfg = adapt_depth_to_model(cfg, get_num_layers(model))
    L = get_num_layers(model)
    reader = make_reader(cfg)
    reader.start()
    runner = AdaptiveRunner(cfg, model, tok, telemetry_reader=reader)

    def bf_static(d):
        return lambda ctx, _d=d: _d

    mon = TjMonitor(tj_path, args.abort_temp)
    mon.no_fan = NO_FAN
    mon.start()
    results = {}
    print(f"[stress] L={L} fan={args.fan} abort={args.abort_temp}C modes={args.modes} "
          f"dur={args.duration}s tj_start={read_c(tj_path):.0f}C")
    try:
        if not NO_FAN:
            _sudo("systemctl stop nvfancontrol")
        for mode in args.modes.split(","):
            mode = mode.strip()
            if NO_FAN:
                # natural cooling only: wait (bounded) for tj to settle near baseline
                print(f"[stress] settling before {mode} (natural cooling) ...")
                _t = time.monotonic()
                while read_c(tj_path) > args.cool_to and time.monotonic() - _t < 90:
                    time.sleep(2.0)
            else:
                print(f"[stress] cooling to {args.cool_to}C before {mode} ...")
                cool_to(tj_path, args.cool_to)
                set_fan(args.fan)
            if mode == "pid":
                os.environ["POISE_CONTROL_MODE"] = "pid"
                cfg_pid = adapt_depth_to_model(load_config(), L)
                bf = make_allocator(cfg_pid)
            elif mode.startswith("static"):
                bf = bf_static(int(mode.replace("static", "")))
            else:
                print(f"  ? unknown mode {mode}, skipping"); continue
            print(f"[stress] running {mode} for {args.duration}s (tj={read_c(tj_path):.0f}C)")
            rows = run_mode(runner, bf, args.duration, mon, args.prompt)
            s = summarize(rows)
            results[mode] = {"summary": s, "series": rows}
            print(f"  -> {mode}: tok/s_tail={s.get('tok_per_s_tail',0):.2f} "
                  f"peak_tj={s.get('peak_temp_c',0):.1f}C "
                  f"mean_budget_tail={s.get('mean_budget_tail',0):.1f} "
                  f"J/tok={s.get('energy_per_token_j_tail',0):.3f}")
            if mon.emergency.is_set():
                print("  !! ABORTED on over-temp — cooling restored"); break
    finally:
        mon.stop_flag.set()
        if not NO_FAN:
            restore_cooling()
        reader.stop()
        print(f"[stress] done. tj={read_c(tj_path):.0f}C (fan_control={'off' if NO_FAN else 'restored'})")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = {"tag": args.tag, "layer_total": L, "fan": args.fan,
               "abort_temp": args.abort_temp, "duration": args.duration,
               "tj_series": mon.series,
               "modes": {m: results[m]["summary"] for m in results}}
    fp = out / f"thermal_stress_{args.tag}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2)
    # full series separately (larger)
    with open(out / f"thermal_stress_{args.tag}_series.json", "w") as f:
        json.dump({m: results[m]["series"] for m in results}, f)
    print(f"\n[stress] summary table:")
    print(f"{'mode':>10} {'tok/s_tail':>11} {'peak_tj':>8} {'budget':>7} {'J/tok':>8}")
    for m, r in results.items():
        s = r["summary"]
        print(f"{m:>10} {s.get('tok_per_s_tail',0):>11.2f} {s.get('peak_temp_c',0):>7.1f}C "
              f"{s.get('mean_budget_tail',0):>7.1f} {s.get('energy_per_token_j_tail',0):>8.3f}")
    print(f"[stress] wrote {fp}")


if __name__ == "__main__":
    main()
