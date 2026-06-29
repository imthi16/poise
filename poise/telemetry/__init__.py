"""Telemetry: one interface, three backends (jtop | tegrastats | mock).

Use ``make_reader(cfg)`` to get the configured backend. Off-device this returns
the mock; on the Jetson it returns the real jtop/tegrastats reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import TelemetrySample
from .reader import BaseTelemetryReader, JtopReader, TegrastatsReader, NvmlReader
from .mock import MockTelemetryReader

if TYPE_CHECKING:  # avoid importing config at module import time
    from ..config import PoiseConfig


def make_reader(cfg: "PoiseConfig") -> BaseTelemetryReader:
    """Construct the telemetry reader for the configured backend.

    ``auto`` picks the richest backend available on this machine (jtop on Jetson,
    NVML on a generic NVIDIA box, else mock). Any real backend that fails to
    initialize degrades to the mock so the stack always runs.
    """
    backend = cfg.telemetry.backend
    hz = cfg.telemetry.hz

    if backend == "auto":
        from ..hardware import detect_telemetry_backend

        backend = detect_telemetry_backend(cfg.model.device)

    def _mock() -> BaseTelemetryReader:
        return MockTelemetryReader(
            hz=hz,
            ambient_c=cfg.thermal.ambient_c,
            temp_max_c=cfg.thermal.temp_max_c,
            layer_total=cfg.depth.layer_total,
            budget=cfg.depth.layer_max,
        )

    if backend == "mock":
        return _mock()
    try:
        if backend == "jtop":
            return JtopReader(hz=hz)
        if backend == "nvml":
            return NvmlReader(hz=hz)
        if backend == "tegrastats":
            return TegrastatsReader(hz=hz)
    except Exception as e:  # real backend unavailable -> degrade to mock, never crash
        import warnings

        warnings.warn(f"telemetry backend {backend!r} unavailable ({e}); using mock.",
                      stacklevel=2)
        return _mock()
    raise ValueError(f"unknown telemetry backend: {backend!r}")


__all__ = [
    "TelemetrySample",
    "BaseTelemetryReader",
    "JtopReader",
    "TegrastatsReader",
    "NvmlReader",
    "MockTelemetryReader",
    "make_reader",
]
