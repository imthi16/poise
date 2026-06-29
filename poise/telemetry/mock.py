"""Synthetic telemetry stream (CLAUDE.md §6).

``MockTelemetryReader`` drives a plausible thermal trajectory from an externally
set *load level* (and optional layer *budget*), so the whole stack runs and tests
without a Jetson. Same interface as the real reader.

The thermal model here is a deliberately simple first-order RC warm-up/cool-down.
It is a *development stand-in*, NOT the calibrated simulator used for RL — that
lives in ``poise/rl/simulator.py`` and uses real fitted parameters.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from .reader import BaseTelemetryReader
from .schema import TelemetrySample


class MockTelemetryReader(BaseTelemetryReader):
    """A controllable synthetic device.

    Parameters
    ----------
    load_level : 0..1 external "how hard are we pushing it" knob (set live).
    budget     : current layer budget (higher depth => more heat/power).
    layer_total: depth normalization for the budget term.
    tau_s      : thermal time-constant (seconds) of the first-order response.
    """

    def __init__(
        self,
        hz: float = 4.0,
        *,
        ambient_c: float = 25.0,
        temp_max_c: float = 87.0,
        init_temp_c: Optional[float] = None,
        idle_power_w: float = 8.0,
        max_power_w: float = 40.0,
        nominal_clock_mhz: float = 1300.0,
        tau_s: float = 25.0,
        load_level: float = 0.6,
        budget: int = 32,
        layer_total: int = 32,
        noise_c: float = 0.0,
        seed: Optional[int] = None,
        clock_fn=time.monotonic,
    ):
        super().__init__(hz)
        self.ambient_c = ambient_c
        self.temp_max_c = temp_max_c
        self.idle_power_w = idle_power_w
        self.max_power_w = max_power_w
        self.nominal_clock_mhz = nominal_clock_mhz
        self.tau_s = max(1e-3, tau_s)
        self.layer_total = max(1, layer_total)
        self.noise_c = noise_c
        self._load = _clamp01(load_level)
        self._budget = budget
        self._temp = float(init_temp_c if init_temp_c is not None else ambient_c + 5.0)
        self._rng = random.Random(seed)
        self._clock_fn = clock_fn
        self._last_t = self._clock_fn()

    # --- live control knobs ---------------------------------------------- #
    def set_load(self, load_level: float) -> None:
        self._load = _clamp01(load_level)

    def set_budget(self, budget: int) -> None:
        self._budget = int(budget)

    @property
    def temp_c(self) -> float:
        return self._temp

    # --- internal model --------------------------------------------------- #
    def _drive(self) -> float:
        """Combined thermal drive in 0..1 from load level and layer budget."""
        depth_frac = self._budget / self.layer_total
        # Budget contributes once there is any load; pure-idle stays cool.
        return _clamp01(0.15 + 0.85 * self._load) * (0.4 + 0.6 * depth_frac)

    def _equilibrium_temp(self, drive: float) -> float:
        span = (self.temp_max_c + 6.0) - self.ambient_c  # allow over-setpoint pushes
        return self.ambient_c + drive * span

    def _sample_once(self) -> TelemetrySample:
        # Guard the mutable thermal state: the dashboard polls /v1/state and
        # /v1/telemetry from concurrent threadpool threads, both of which read().
        with self._lock:
            now = self._clock_fn()
            dt = max(0.0, now - self._last_t)
            self._last_t = now

            drive = self._drive()
            t_eq = self._equilibrium_temp(drive)
            # Exact first-order step toward equilibrium (stable for any dt).
            alpha = 1.0 - math.exp(-dt / self.tau_s) if dt > 0 else 0.0
            self._temp += (t_eq - self._temp) * alpha

            throttled = self._temp >= self.temp_max_c
            observed_temp = self._temp
            if self.noise_c > 0:
                observed_temp += self._rng.gauss(0.0, self.noise_c)
            cur_load = self._load
            cur_budget = self._budget

        # Power scales with drive; throttling clamps the clock and trims power.
        power = self.idle_power_w + (self.max_power_w - self.idle_power_w) * drive
        clock = self.nominal_clock_mhz
        if throttled:
            clock *= 0.7
            power *= 0.85
        gpu_util = 100.0 * _clamp01(cur_load)

        return TelemetrySample(
            ts=time.time(),
            temp_c=observed_temp,
            power_w=power,
            gpu_clock_mhz=clock,
            gpu_util=gpu_util,
            throttled=bool(throttled),
            mem_bw_util=100.0 * _clamp01(0.5 * cur_load + 0.5 * (cur_budget / self.layer_total)),
        )


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


__all__ = ["MockTelemetryReader"]
