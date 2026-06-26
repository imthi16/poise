"""Prometheus collectors (CLAUDE.md §7).

Exposes the gauges/counters scraped at ``GET /metrics``:
  poise_temp_c, poise_power_w, poise_current_budget, poise_tok_per_s,
  poise_energy_per_token_j, poise_throttle_events_total, poise_tokens_total.

Each ``MetricsRegistry`` owns a private ``CollectorRegistry`` so multiple app
instances (e.g. in tests) don't collide on the global default registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

if TYPE_CHECKING:  # pragma: no cover
    from ..telemetry.schema import TelemetrySample
    from ..eval.benchmark import RunMetrics


class MetricsRegistry:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.temp_c = Gauge("poise_temp_c", "Junction temperature (C)", registry=self.registry)
        self.power_w = Gauge("poise_power_w", "Power draw (W)", registry=self.registry)
        self.current_budget = Gauge(
            "poise_current_budget", "Current layer budget", registry=self.registry
        )
        self.tok_per_s = Gauge(
            "poise_tok_per_s", "Throughput (tokens/s)", registry=self.registry
        )
        self.energy_per_token_j = Gauge(
            "poise_energy_per_token_j", "Energy per token (J)", registry=self.registry
        )
        self.throttle_events = Counter(
            "poise_throttle_events", "Hard-throttle events", registry=self.registry
        )
        self.tokens = Counter(
            "poise_tokens", "Tokens generated", registry=self.registry
        )

    def update_from_telemetry(self, s: "TelemetrySample", budget: float | None = None) -> None:
        self.temp_c.set(s.temp_c)
        self.power_w.set(s.power_w)
        if budget is not None:
            self.current_budget.set(budget)

    def update_from_run(self, m: "RunMetrics") -> None:
        self.tok_per_s.set(m.tok_per_s)
        self.energy_per_token_j.set(m.energy_per_token_j)
        self.current_budget.set(m.mean_budget)
        if m.tokens:
            self.tokens.inc(m.tokens)
        if m.throttle_events:
            self.throttle_events.inc(m.throttle_events)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


__all__ = ["MetricsRegistry"]
