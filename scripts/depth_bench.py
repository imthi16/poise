"""Per-depth throughput / energy / thermal benchmark WITH variance (CLAUDE.md §6, §11).

Runs sustained fixed-depth generation on the REAL engine + REAL telemetry, ``repeats``
times per depth, and aggregates mean±std of tokens/s, energy-per-token (∫P dt ÷ tokens),
and peak temperature. This is the on-device measurement behind the throughput-per-watt
reframe — and the exact protocol the capped throttling stress test reuses.

tok/s-per-watt == 1 / energy_per_token_j (since energy/token = mean_power / tok_per_s).

🔒 Energy/thermal numbers are physical → must run on the Jetson with real telemetry.
No number is asserted; all are measured here and printed with variance.

    python3.10 scripts/depth_bench.py --depths 16,20,24,28,32 --repeats 3 \\
        --max-new-tokens 256 --warmup 32 --tag maxn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-depth throughput/energy/thermal benchmark")
    ap.add_argument("--depths", default="16,20,24,28,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=32, help="tokens discarded before metrics")
    ap.add_argument("--cooldown", type=float, default=10.0, help="idle seconds between depths")
    ap.add_argument("--prompt", default="Explain how thermal throttling limits sustained "
                    "compute throughput on an edge accelerator, in detail.")
    ap.add_argument("--adapter", default=None, help="LayerSkip adapter dir (POISE_ADAPTER_PATH)")
    ap.add_argument("--tag", default="maxn")
    ap.add_argument("--out", default="data/results/depth_bench")
    args = ap.parse_args()

    import os
    if args.adapter:
        os.environ["POISE_ADAPTER_PATH"] = args.adapter

    import numpy as np

    from poise.config import load_config
    from poise.engine.model_loader import load_model, get_num_layers
    from poise.engine.adaptive_runner import AdaptiveRunner
    from poise.hardware import adapt_depth_to_model
    from poise.telemetry import make_reader
    from poise.eval.benchmark import StressBenchmark, StressSpec

    cfg = load_config()
    model, tok = load_model(cfg)
    cfg = adapt_depth_to_model(cfg, get_num_layers(model))
    layer_total = get_num_layers(model)
    depths = [d for d in (int(x) for x in args.depths.split(",")) if 1 <= d <= layer_total]

    reader = make_reader(cfg)
    reader.start()
    runner = AdaptiveRunner(cfg, model, tok, telemetry_reader=reader)
    spec = StressSpec(max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup)

    print(f"[bench] model={cfg.model.model_id} adapter={args.adapter or 'NONE'} "
          f"L={layer_total} depths={depths} repeats={args.repeats} "
          f"tokens={args.max_new_tokens} (warmup {args.warmup})")

    agg = {}
    for d in depths:
        bench = StressBenchmark(cfg, runner, budget_fn=lambda ctx, _d=d: _d)
        tps, ept, peak, powr = [], [], [], []
        for r in range(args.repeats):
            _, m = bench.run(args.prompt, spec)
            tps.append(m.tok_per_s)
            ept.append(m.energy_per_token_j)
            peak.append(m.peak_temp_c)
            powr.append(m.energy_per_token_j * m.tok_per_s)  # mean power W
        agg[d] = {
            "tok_per_s_mean": float(np.mean(tps)), "tok_per_s_std": float(np.std(tps)),
            "energy_per_token_j_mean": float(np.mean(ept)),
            "energy_per_token_j_std": float(np.std(ept)),
            "mean_power_w": float(np.mean(powr)),
            "peak_temp_c_mean": float(np.mean(peak)),
            "tok_per_s_per_watt": float(1.0 / np.mean(ept)) if np.mean(ept) else 0.0,
        }
        a = agg[d]
        print(f"  d={d:>2}: {a['tok_per_s_mean']:5.2f}±{a['tok_per_s_std']:.2f} tok/s | "
              f"{a['energy_per_token_j_mean']:6.3f} J/tok | {a['mean_power_w']:5.1f} W | "
              f"peak {a['peak_temp_c_mean']:4.1f}C | {a['tok_per_s_per_watt']:.3f} tok/s/W")
        if args.cooldown and d != depths[-1]:
            time.sleep(args.cooldown)

    reader.stop()

    # relative to full depth
    full = layer_total if layer_total in agg else max(agg)
    base_tpw = agg[full]["tok_per_s_per_watt"]
    print(f"\n{'depth':>6} {'tok/s':>13} {'J/tok':>9} {'tok/s/W':>9} {'×tok/s/W vs full':>16}")
    print("-" * 60)
    rows = []
    for d in sorted(agg):
        a = agg[d]
        ratio = a["tok_per_s_per_watt"] / base_tpw if base_tpw else 0.0
        print(f"{d:>6} {a['tok_per_s_mean']:>7.2f}±{a['tok_per_s_std']:<4.2f} "
              f"{a['energy_per_token_j_mean']:>9.3f} {a['tok_per_s_per_watt']:>9.3f} "
              f"{ratio:>15.2f}×")
        rows.append({"depth": d, **a, "tok_per_s_per_watt_ratio_vs_full": ratio})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"depth_bench_{args.tag}.json"
    with open(fp, "w") as f:
        json.dump({"tag": args.tag, "model_id": cfg.model.model_id, "adapter": args.adapter,
                   "layer_total": layer_total, "repeats": args.repeats,
                   "max_new_tokens": args.max_new_tokens, "warmup": args.warmup,
                   "rows": rows}, f, indent=2)
    print(f"\n[bench] wrote {fp}")


if __name__ == "__main__":
    main()
