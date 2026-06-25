"""Telemetry sample schema (CLAUDE.md §6)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class TelemetrySample:
    """One immutable snapshot of device physical state.

    Fields mirror what jtop / tegrastats expose on a Jetson AGX Orin. ``throttled``
    must reflect a *real* throttle/clamp signal on the hardware path, never a guess.
    """

    ts: float                       # epoch seconds
    temp_c: float                   # junction / SOC temperature
    power_w: float                  # instantaneous power draw
    gpu_clock_mhz: float
    gpu_util: float                 # 0..100
    throttled: bool
    mem_bw_util: float | None = None  # optional; None when unavailable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["TelemetrySample"]
