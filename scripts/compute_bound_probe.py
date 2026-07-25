"""Does depth->power scale in COMPUTE-bound regimes? (follow-up to the flat-decode finding)

Single-stream decode on Orin is memory-bandwidth-bound, so depth->power is ~flat (~47W) and
depth is not a thermal actuator (see thermal_stress). The open question for future scope: in
COMPUTE-bound regimes — long-context PREFILL and BATCHED forwards — arithmetic intensity rises,
so power should scale with FLOPs ∝ depth. If so, depth control could regain thermal authority
there.

Method: temporarily slice the decoder stack to d layers (correct depth-d forward via the real
HF path), run sustained forwards in each regime, and measure steady-state power + throughput
from real telemetry. Compares the depth->power SLOPE against flat decode.

🔒 Physical power -> Jetson only. No hardcoded numbers; all measured.

    PYTHONPATH=. python3.10 scripts/compute_bound_probe.py --depths 16,24,32 \\
        --prefill-seq 2048 --batch 8 --batch-seq 512 --secs 30 --tag orin
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sliced_depth(model, d):
    """Run the model's forward with only the first d decoder layers, then restore."""
    import torch.nn as nn
    from poise.engine.model_loader import _base
    base = _base(model)
    orig = base.layers
    try:
        base.layers = nn.ModuleList(list(orig)[:d])
        yield
    finally:
        base.layers = orig


def measure(model, base_forward, make_inputs, depths, secs, reader, label):
    """For each depth: sustained forward passes for `secs`, mean steady-state power + throughput."""
    import numpy as np
    import torch

    out = {}
    device = next(model.parameters()).device
    for d in depths:
        inp = make_inputs(device)
        n_items = inp["input_ids"].shape[0] * inp["input_ids"].shape[1]  # batch*seq tokens/fwd
        powers, fwds = [], 0
        with sliced_depth(model, d):
            # warmup
            with torch.no_grad():
                base_forward(inp)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.monotonic()
            while time.monotonic() - t0 < secs:
                with torch.no_grad():
                    base_forward(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fwds += 1
                s = reader.read()
                powers.append(float(s.power_w))
        dt = time.monotonic() - t0
        tail = powers[max(0, len(powers) // 2):]  # 2nd half = steady state
        p_ss = float(np.mean(tail)) if tail else 0.0
        tok_s = fwds * n_items / dt if dt else 0.0
        out[d] = {"mean_power_w": p_ss, "tokens_per_s": tok_s, "fwds": fwds,
                  "tokens_per_fwd": n_items}
        print(f"  [{label}] d={d:>2}: {p_ss:5.1f} W | {tok_s:8.0f} tok/s | {fwds} fwds")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute-bound depth->power probe")
    ap.add_argument("--depths", default="16,24,32")
    ap.add_argument("--prefill-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch-seq", type=int, default=512)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--tag", default="orin")
    ap.add_argument("--out", default="data/results/compute_probe")
    args = ap.parse_args()

    import torch

    from poise.config import load_config
    from poise.engine.model_loader import load_model, get_num_layers, _base
    from poise.telemetry import make_reader

    cfg = load_config()
    model, tok = load_model(cfg)
    L = get_num_layers(model)
    depths = [d for d in (int(x) for x in args.depths.split(",")) if 1 <= d <= L]
    reader = make_reader(cfg); reader.start()
    base = _base(model)
    vocab = model.config.vocab_size

    def base_forward(inp):
        base(**inp)  # embed + (sliced) layers + final norm; skips LM head to isolate depth

    def prefill_inputs(device):
        ids = torch.randint(0, vocab, (1, args.prefill_seq), device=device)
        return {"input_ids": ids}

    def batch_inputs(device):
        ids = torch.randint(0, vocab, (args.batch, args.batch_seq), device=device)
        return {"input_ids": ids}

    print(f"[probe] L={L} depths={depths} secs={args.secs} "
          f"prefill_seq={args.prefill_seq} batch={args.batch}x{args.batch_seq}")
    results = {}
    print("PREFILL (batch=1, long seq):")
    results["prefill"] = measure(model, base_forward, prefill_inputs, depths, args.secs,
                                 reader, "prefill")
    print(f"BATCHED (batch={args.batch}, seq={args.batch_seq}):")
    results["batch"] = measure(model, base_forward, batch_inputs, depths, args.secs,
                               reader, "batch")
    reader.stop()

    # slope report: power(d32) - power(d16) vs flat decode (~1-2W)
    print(f"\n{'regime':>10} {'P(d16)':>8} {'P(d32)':>8} {'ΔP(32-16)':>10} {'slope W/layer':>13}")
    print("-" * 54)
    rows = {}
    for reg, r in results.items():
        ds = sorted(r)
        lo, hi = ds[0], ds[-1]
        dP = r[hi]["mean_power_w"] - r[lo]["mean_power_w"]
        slope = dP / (hi - lo) if hi != lo else 0.0
        print(f"{reg:>10} {r[lo]['mean_power_w']:>8.1f} {r[hi]['mean_power_w']:>8.1f} "
              f"{dP:>10.1f} {slope:>13.2f}")
        rows[reg] = {"depths": {str(d): r[d] for d in ds}, "delta_power_w": dP,
                     "slope_w_per_layer": slope}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"compute_probe_{args.tag}.json"
    with open(fp, "w") as f:
        json.dump({"tag": args.tag, "layer_total": L, "prefill_seq": args.prefill_seq,
                   "batch": args.batch, "batch_seq": args.batch_seq, "secs": args.secs,
                   "regimes": rows}, f, indent=2)
    print(f"\n[probe] decode baseline (from calibration) was ~flat: ΔP≈2W over 16→32.")
    print(f"[probe] wrote {fp}")


if __name__ == "__main__":
    main()
