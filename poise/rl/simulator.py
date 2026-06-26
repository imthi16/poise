"""RC thermal-power simulator — the world the PPO policy trains against (CLAUDE.md §6).

First-order RC model (exact exponential integration, stable for any dt):

    T_eq  = T_ambient + P(budget)·R_th
    T_next = T_eq + (T − T_eq)·e^(−dt/τ),   τ = R_th·C_th

🔒 Parameters come ONLY from real calibration (``calibration/fit.py``). Off-device,
the config exposes clearly-labeled NOMINAL placeholders so the env / PPO plumbing is
runnable; ``cfg.sim.params_source`` says which. ``test_simulator.py`` validates the
dynamics against held-out traces and reports the fit error — we do not invent thermal
parameters, and we surface when we are running on placeholders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


@dataclass
class SimStep:
    temp_c: float
    power_w: float
    throttled: bool
    tok_per_s: float
    energy_j: float       # energy consumed during this step (power · dt)


class RCThermalSimulator:
    """Calibrated RC thermal + depth→power/latency model.

    Parameters are explicit so domain randomization (``domain_random.py``) can vary
    them per episode. Build from config with :meth:`from_config`.
    """

    def __init__(
        self,
        *,
        r_th_c_per_w: float,
        c_th_j_per_c: float,
        ambient_c: float,
        temp_max_c: float,
        depth_power_w: Mapping[int, float],
        depth_latency_s: Mapping[int, float],
        init_temp_c: Optional[float] = None,
        throttle_clock_factor: float = 0.7,
        params_source: str = "placeholder",
    ):
        self.r_th = float(r_th_c_per_w)
        self.c_th = float(c_th_j_per_c)
        self.ambient_c = float(ambient_c)
        self.temp_max_c = float(temp_max_c)
        self.tau = max(1e-6, self.r_th * self.c_th)
        self.throttle_clock_factor = throttle_clock_factor
        self.params_source = params_source

        self._depths = np.array(sorted(depth_power_w.keys()), dtype=np.float64)
        self._powers = np.array([depth_power_w[int(d)] for d in self._depths], dtype=np.float64)
        if depth_latency_s:
            self._lat_depths = np.array(sorted(depth_latency_s.keys()), dtype=np.float64)
            self._lats = np.array(
                [depth_latency_s[int(d)] for d in self._lat_depths], dtype=np.float64
            )
        else:
            self._lat_depths = self._depths
            # fall back to latency proportional to depth if not provided
            self._lats = self._depths / float(self._depths.max()) * 0.03

        self._init_temp = init_temp_c if init_temp_c is not None else self.ambient_c + 5.0
        self.temp_c = float(self._init_temp)

    @classmethod
    def from_config(cls, cfg: "PoiseConfig", init_temp_c: Optional[float] = None) -> "RCThermalSimulator":
        return cls(
            r_th_c_per_w=cfg.sim.r_th_c_per_w,
            c_th_j_per_c=cfg.sim.c_th_j_per_c,
            ambient_c=cfg.sim.ambient_c,
            temp_max_c=cfg.thermal.temp_max_c,
            depth_power_w=cfg.sim.depth_power_map,
            depth_latency_s=cfg.sim.depth_latency_map,
            init_temp_c=init_temp_c if init_temp_c is not None else cfg.sim.init_temp_c,
            params_source=cfg.sim.params_source,
        )

    # --- model lookups ------------------------------------------------------ #
    def power_for_depth(self, budget: float) -> float:
        """Steady-state power at ``budget`` (interp over the calibrated map)."""
        return float(np.interp(budget, self._depths, self._powers))

    def latency_for_depth(self, budget: float, throttled: bool = False) -> float:
        lat = float(np.interp(budget, self._lat_depths, self._lats))
        if throttled:
            lat /= max(1e-6, self.throttle_clock_factor)  # slower clock => longer latency
        return lat

    # --- dynamics ----------------------------------------------------------- #
    def reset(self, ambient_c: Optional[float] = None, init_temp_c: Optional[float] = None) -> float:
        if ambient_c is not None:
            self.ambient_c = float(ambient_c)
        self.temp_c = float(init_temp_c if init_temp_c is not None else self._init_temp)
        return self.temp_c

    def step(self, budget: float, dt: float) -> SimStep:
        """Advance the thermal state by ``dt`` seconds at the given layer budget."""
        power = self.power_for_depth(budget)
        t_eq = self.ambient_c + power * self.r_th
        decay = math.exp(-dt / self.tau) if dt > 0 else 1.0
        self.temp_c = t_eq + (self.temp_c - t_eq) * decay

        throttled = self.temp_c >= self.temp_max_c
        latency = self.latency_for_depth(budget, throttled=throttled)
        tok_per_s = 1.0 / latency if latency > 0 else 0.0
        # When throttled the clock drops, so effective power dissipated also dips a bit.
        eff_power = power * (self.throttle_clock_factor if throttled else 1.0)
        energy = eff_power * dt
        return SimStep(
            temp_c=self.temp_c,
            power_w=eff_power,
            throttled=bool(throttled),
            tok_per_s=tok_per_s,
            energy_j=energy,
        )

    def simulate(self, budgets, dt: float):
        """Convenience: run a sequence of budgets, return list[SimStep]."""
        return [self.step(b, dt) for b in budgets]


def simulate_constant(
    sim: RCThermalSimulator, budget: int, dt: float, n_steps: int
) -> list[SimStep]:
    sim_steps = []
    for _ in range(n_steps):
        sim_steps.append(sim.step(budget, dt))
    return sim_steps


__all__ = ["RCThermalSimulator", "SimStep", "simulate_constant"]
