"""Generate the arXiv figures from the measured result JSONs (reproducible).

Reads only committed eval outputs under data/results/ and writes vector PDFs (+ PNG
previews) to docs/arxiv/figs/. No numbers are invented here; every point is measured.

    python3.10 docs/arxiv/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 140, "savefig.bbox": "tight",
})
C = {"blue": "#2E6E9E", "amber": "#E08A1E", "crim": "#C6413B",
     "green": "#3C8C5A", "slate": "#4A4A55"}


def load(p):
    return json.load(open(ROOT / p))


def fig_tradeoff():
    db = load("data/results/depth_bench/depth_bench_maxn.json")["rows"]
    ad = {r["depth"]: r["acc_norm"] for r in
          load("data/results/prong1/prong1_arc_v1adapter.json")["rows"]}
    ba = {r["depth"]: r["acc_norm"] for r in
          load("data/results/prong1/prong1_arc_base.json")["rows"]}
    d = [r["depth"] for r in db]
    tps = [r["tok_per_s_mean"] for r in db]
    tps_e = [r["tok_per_s_std"] for r in db]
    ept = [r["energy_per_token_j_mean"] for r in db]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # panel a: throughput + energy vs depth
    ax1.errorbar(d, tps, yerr=tps_e, marker="o", color=C["blue"], lw=1.8,
                 capsize=3, label="throughput")
    ax1.set_xlabel("layer budget (depth)"); ax1.set_ylabel("tokens / s", color=C["blue"])
    ax1.tick_params(axis="y", labelcolor=C["blue"])
    axb = ax1.twinx(); axb.spines["top"].set_visible(False)
    axb.plot(d, ept, marker="s", color=C["crim"], lw=1.8, label="energy/token")
    axb.set_ylabel("energy / token (J)", color=C["crim"])
    axb.tick_params(axis="y", labelcolor=C["crim"]); axb.grid(False)
    ax1.set_title("(a) halving depth ≈ 2× throughput, ½ energy")

    # panel b: quality vs depth (adapter vs base)
    dd = sorted(ad)
    ax2.plot(dd, [ba[x] for x in dd], marker="^", color=C["slate"], lw=1.6,
             ls="--", label="base model")
    ax2.plot(dd, [ad[x] for x in dd], marker="o", color=C["green"], lw=1.8,
             label="LayerSkip adapter")
    ax2.set_xlabel("layer budget (depth)"); ax2.set_ylabel("ARC accuracy (acc\\_norm)")
    ax2.set_title("(b) quality cost is real"); ax2.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tradeoff.pdf"); fig.savefig(FIGS / "fig_tradeoff.png")
    plt.close(fig)


def fig_flatpower():
    db = {r["depth"]: r["mean_power_w"] for r in
          load("data/results/depth_bench/depth_bench_maxn.json")["rows"]}
    cp = load("data/results/compute_probe/compute_probe_orin_maxn.json")["regimes"]
    regimes = ["decode", "prefill", "batch"]
    p16 = [db[16], cp["prefill"]["depths"]["16"]["mean_power_w"],
           cp["batch"]["depths"]["16"]["mean_power_w"]]
    p32 = [db[32], cp["prefill"]["depths"]["32"]["mean_power_w"],
           cp["batch"]["depths"]["32"]["mean_power_w"]]
    x = np.arange(len(regimes)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    b1 = ax.bar(x - w/2, p16, w, color=C["blue"], label="depth 16")
    b2 = ax.bar(x + w/2, p32, w, color=C["crim"], label="depth 32")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x()+r.get_width()/2, r.get_height()),
                        ha="center", va="bottom", fontsize=8)
    for i in range(len(regimes)):
        ax.annotate(f"ΔP {p32[i]-p16[i]:.1f} W", (x[i], max(p16[i], p32[i])+3.5),
                    ha="center", fontsize=8.5, color=C["slate"])
    ax.set_xticks(x); ax.set_xticklabels(["decode\n(1×stream)", "prefill\n(2048)", "batch\n(8×512)"])
    ax.set_ylabel("steady-state power (W)"); ax.set_ylim(0, max(p32)+10)
    ax.set_title("Power is flat across depth — in every regime")
    ax.legend(frameon=False, ncol=2, loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_flatpower.pdf"); fig.savefig(FIGS / "fig_flatpower.png")
    plt.close(fig)


def _smooth(y, k=15):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    from scipy.ndimage import uniform_filter1d  # edge-aware (no zero-pad dip)
    return uniform_filter1d(y, size=k, mode="nearest")


def fig_thermal():
    s = load("data/results/thermal_stress/thermal_stress_normalfan_series.json")
    styles = {"static32": (C["blue"], "static-32"), "static16": (C["amber"], "static-16"),
              "pid": (C["crim"], "PID (setpoint 50 °C)")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for mode, (col, lab) in styles.items():
        rows = s[mode]
        t = [r["t"] for r in rows]; tj = _smooth([r["temp_c"] for r in rows])
        ax1.plot(t, tj, color=col, lw=1.6, label=lab)
    ax1.axhline(50, color=C["slate"], ls=":", lw=1, label="PID setpoint")
    ax1.set_xlabel("time under load (s)"); ax1.set_ylabel("junction temp (°C)")
    ax1.set_title("(a) shedding depth does not cool")
    ax1.legend(frameon=False, fontsize=8)
    # panel b: pid budget over time
    rows = s["pid"]
    t = [r["t"] for r in rows]; bud = _smooth([r["budget"] for r in rows], 25)
    ax2.plot(t, bud, color=C["crim"], lw=1.6)
    ax2.axhline(32, color=C["slate"], ls=":", lw=1)
    ax2.set_ylim(12, 34); ax2.set_xlabel("time under load (s)")
    ax2.set_ylabel("PID layer budget")
    ax2.set_title("(b) PID sheds depth 32 → ~21 …")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_thermal.pdf"); fig.savefig(FIGS / "fig_thermal.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff(); fig_flatpower(); fig_thermal()
    print("wrote figures to", FIGS)
    for p in sorted(FIGS.glob("*.pdf")):
        print("  ", p.relative_to(ROOT))
