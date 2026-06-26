"""Aggregate runs → comparison tables + plots WITH VARIANCE (CLAUDE.md §6, §9, §11).

🔒 Every headline number is produced here, aggregated across ≥ N seeds/runs and
reported with spread (mean ± std) — never asserted in prose ahead of measurement. The
comparison set MUST include static-full-32, ≥ 2 static reduced depths, PID, and PPO;
``check_comparison_set`` enforces that and refuses to emit a misleading partial report.

Pure aggregation is unit-tested off-device; plotting (matplotlib) is optional/lazy.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .benchmark import RunMetrics

REQUIRED_LABELS = ("static-full-32", "pid", "ppo")
MIN_STATIC_REDUCED = 2

# Higher-is-better vs lower-is-better, for delta direction.
_LOWER_BETTER = {
    "energy_per_token_j",
    "peak_temp_c",
    "time_above_setpoint_s",
    "throttle_events",
    "ttft_ms",
    "inter_token_latency_ms",
}


def aggregate(metrics: Sequence["RunMetrics"]) -> dict[str, tuple[float, float]]:
    """Mean ± std per metric over repeated runs (the variance requirement)."""
    if not metrics:
        return {}
    keys = asdict(metrics[0]).keys()
    out: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in metrics], dtype=np.float64)
        out[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1) if vals.size > 1 else 0.0))
    return out


def check_comparison_set(labels: Sequence[str]) -> list[str]:
    """Return a list of problems (empty == OK). Enforces the mandated comparison set."""
    problems = []
    lset = set(labels)
    for req in REQUIRED_LABELS:
        if req not in lset:
            problems.append(f"missing required comparison: {req!r}")
    n_static_reduced = sum(
        1 for l in labels if l.startswith("static-") and l != "static-full-32"
    )
    if n_static_reduced < MIN_STATIC_REDUCED:
        problems.append(
            f"need >= {MIN_STATIC_REDUCED} static reduced-depth baselines, got {n_static_reduced}"
        )
    return problems


def compare(
    results_by_label: Mapping[str, Sequence["RunMetrics"]],
    baseline_label: str = "static-full-32",
) -> dict:
    """Aggregate every label and compute deltas vs the baseline, with variance."""
    agg = {label: aggregate(list(ms)) for label, ms in results_by_label.items()}
    base = agg.get(baseline_label, {})
    comparison = {"baseline": baseline_label, "labels": {}, "n_runs": {}}
    for label, a in agg.items():
        comparison["n_runs"][label] = len(results_by_label[label])
        row = {}
        for metric, (mean, std) in a.items():
            entry = {"mean": mean, "std": std}
            if base and metric in base and base[metric][0] != 0:
                bmean = base[metric][0]
                pct = 100.0 * (mean - bmean) / abs(bmean)
                # express as improvement sign-aware
                entry["delta_pct_vs_baseline"] = pct
                entry["better"] = (
                    (pct < 0) if metric in _LOWER_BETTER else (pct > 0)
                )
            row[metric] = entry
        comparison["labels"][label] = row
    comparison["problems"] = check_comparison_set(list(results_by_label.keys()))
    return comparison


def format_markdown(comparison: dict, metrics: Sequence[str] | None = None) -> str:
    metrics = list(metrics or ["tok_per_s", "energy_per_token_j", "peak_temp_c",
                              "throttle_events", "mean_budget"])
    labels = list(comparison["labels"].keys())
    lines = [f"# POISE comparison (baseline = {comparison['baseline']})", ""]
    if comparison.get("problems"):
        lines.append("> ⚠ Comparison-set problems: " + "; ".join(comparison["problems"]))
        lines.append("")
    header = "| label | n | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 2)
    lines += [header, sep]
    for label in labels:
        row = comparison["labels"][label]
        n = comparison["n_runs"][label]
        cells = []
        for m in metrics:
            e = row.get(m, {})
            if not e:
                cells.append("-")
                continue
            cell = f"{e['mean']:.3f} ± {e['std']:.3f}"
            if "delta_pct_vs_baseline" in e:
                cell += f" ({e['delta_pct_vs_baseline']:+.1f}%)"
            cells.append(cell)
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "_All numbers are measured eval outputs aggregated across runs (mean ± std). "
        "No figure is asserted ahead of measurement._"
    )
    return "\n".join(lines)


def emit_report(
    comparison: dict,
    out_dir: str | Path = "data/results",
    *,
    emit_plots: bool = True,
    results_by_label: Mapping[str, Sequence["RunMetrics"]] | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    md = format_markdown(comparison)
    (out / "comparison.md").write_text(md)
    paths["markdown"] = str(out / "comparison.md")

    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    paths["json"] = str(out / "comparison.json")

    # flat CSV: one row per (label, metric)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "metric", "mean", "std", "delta_pct_vs_baseline"])
        for label, row in comparison["labels"].items():
            n = comparison["n_runs"][label]
            for metric, e in row.items():
                w.writerow([label, n, metric, e.get("mean"), e.get("std"),
                            e.get("delta_pct_vs_baseline", "")])
    paths["csv"] = str(out / "comparison.csv")

    if emit_plots and results_by_label is not None:
        try:  # pragma: no cover - optional plotting
            paths.update(_emit_plots(comparison, results_by_label, out))
        except Exception:
            pass
    return paths


def _emit_plots(comparison, results_by_label, out: Path) -> dict[str, str]:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    for metric in ("tok_per_s", "energy_per_token_j", "peak_temp_c"):
        labels = list(comparison["labels"].keys())
        means = [comparison["labels"][l][metric]["mean"] for l in labels]
        stds = [comparison["labels"][l][metric]["std"] for l in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (mean ± std)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        p = out / f"plot_{metric}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[f"plot_{metric}"] = str(p)
    return paths


__all__ = [
    "aggregate",
    "check_comparison_set",
    "compare",
    "format_markdown",
    "emit_report",
    "REQUIRED_LABELS",
]
