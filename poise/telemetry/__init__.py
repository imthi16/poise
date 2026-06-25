"""Telemetry: one interface, three backends (jtop | tegrastats | mock).

Use ``make_reader(cfg)`` to get the configured backend. Off-device this returns
the mock; on the Jetson it returns the real jtop/tegrastats reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import TelemetrySample
from .reader import BaseTelemetryReader, JtopReader, TegrastatsReader
from .mock import MockTelemetryReader

if TYPE_CHECKING:  # avoid importing config at module import time
    from ..config import PoiseConfig


def make_reader(cfg: "PoiseConfig") -> BaseTelemetryReader:
    """Construct the telemetry reader for the configured backend."""
    backend = cfg.telemetry.backend
    hz = cfg.telemetry.hz
    if backend == "mock":
        return MockTelemetryReader(
            hz=hz,
            ambient_c=cfg.thermal.ambient_c,
            temp_max_c=cfg.thermal.temp_max_c,
            layer_total=cfg.depth.layer_total,
            budget=cfg.depth.layer_max,
        )
    if backend == "jtop":
        return JtopReader(hz=hz)
    if backend == "tegrastats":
        return TegrastatsReader(hz=hz)
    raise ValueError(f"unknown telemetry backend: {backend!r}")


__all__ = [
    "TelemetrySample",
    "BaseTelemetryReader",
    "JtopReader",
    "TegrastatsReader",
    "MockTelemetryReader",
    "make_reader",
]
