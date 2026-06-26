"""Serving: FastAPI app (§7 endpoints) + Prometheus metrics.

Consumes the engine/control as a black box; off-device a mock engine reuses the real
control loop so the API + dashboard run without the model.
"""

from __future__ import annotations

from .api import create_app, InferenceService, run
from .metrics import MetricsRegistry

__all__ = ["create_app", "InferenceService", "MetricsRegistry", "run"]
