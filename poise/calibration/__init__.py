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

__all__ = [
    "DepthQuality",
    "QualityProfiler",
    "kl_divergence_logits",
    "perplexity_from_logits",
    "recommend_layer_min",
    "run_gate",
]
