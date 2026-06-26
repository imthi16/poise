"""Evaluation harness: stress benchmark + quality + variance report (CLAUDE.md §11).

🔒 Every headline number is produced here and reported with spread. No performance
figure is written in prose ahead of these outputs.
"""

from __future__ import annotations

from .benchmark import (
    RunMetrics,
    StressSpec,
    StressBenchmark,
    summarize_trace,
    integrate_energy_j,
    count_throttle_events,
    trace_from_token_latencies,
)
from .quality import (
    mean_kl_from_trace,
    exact_match_accuracy,
    quality_loss_pct,
    QualityEvaluator,
)
from .report import aggregate, compare, check_comparison_set, format_markdown, emit_report

__all__ = [
    "RunMetrics",
    "StressSpec",
    "StressBenchmark",
    "summarize_trace",
    "integrate_energy_j",
    "count_throttle_events",
    "trace_from_token_latencies",
    "mean_kl_from_trace",
    "exact_match_accuracy",
    "quality_loss_pct",
    "QualityEvaluator",
    "aggregate",
    "compare",
    "check_comparison_set",
    "format_markdown",
    "emit_report",
]
