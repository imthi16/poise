"""Calibration: the step-3 quality gate, plus on-board thermal sweeps + RC fit.

- ``quality_profile`` — ⚠ the go/no-go gate: depth→{KL, perplexity, accuracy}.
- ``sweep`` / ``fit`` — on-board thermal-power calibration -> RC params + depth→power.
"""

from __future__ import annotations

from .quality_profile import (
    DepthQuality,
    QualityProfiler,
    kl_divergence_logits,
    perplexity_from_logits,
    recommend_layer_min,
    run_gate,
)
from .fit import (
    RCFit,
    CalibrationResult,
    rc_step_response,
    fit_rc_from_trace,
    steady_state_power_by_depth,
    aggregate_fits,
    write_simulator_yaml,
)
from .sweep import SweepSpec, collect_trace, run_sweep, read_csv, write_csv

__all__ = [
    "DepthQuality",
    "QualityProfiler",
    "kl_divergence_logits",
    "perplexity_from_logits",
    "recommend_layer_min",
    "run_gate",
    "RCFit",
    "CalibrationResult",
    "rc_step_response",
    "fit_rc_from_trace",
    "steady_state_power_by_depth",
    "aggregate_fits",
    "write_simulator_yaml",
    "SweepSpec",
    "collect_trace",
    "run_sweep",
    "read_csv",
    "write_csv",
]
