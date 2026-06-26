"""Per-episode domain randomization (CLAUDE.md §6, §9).

Randomize R_th, C_th, the depth→power map, ambient, and sensor noise within plausible
bounds each episode so the policy does not overfit a single RC fit. This is the
PRIMARY mitigation for the sim-to-real gap; bounds come from calibration uncertainty
(``configs/ppo.yaml`` domain_randomization). The remaining gap is quantified on the
real board in step 9 — randomization reduces it, it does not eliminate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .simulator import RCThermalSimulator

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


@dataclass
class DomainParams:
    r_th_c_per_w: float
    c_th_j_per_c: float
    depth_power_w: dict
    ambient_c: float
    init_temp_c: float
    sensor_noise_c: float


class DomainRandomizer:
    """Samples randomized simulator parameters per episode."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def sample(self, cfg: "PoiseConfig", rng: np.random.Generator) -> DomainParams:
        dr = cfg.ppo.domain_random
        if not (self.enabled and dr.enabled):
            return DomainParams(
                r_th_c_per_w=cfg.sim.r_th_c_per_w,
                c_th_j_per_c=cfg.sim.c_th_j_per_c,
                depth_power_w=dict(cfg.sim.depth_power_map),
                ambient_c=cfg.sim.ambient_c,
                init_temp_c=cfg.sim.init_temp_c,
                sensor_noise_c=0.0,
            )
        r_mul = rng.uniform(*dr.r_th_rel_range)
        c_mul = rng.uniform(*dr.c_th_rel_range)
        p_mul = rng.uniform(*dr.power_rel_range)
        ambient = rng.uniform(*dr.ambient_c_range)
        power_map = {int(d): float(w) * p_mul for d, w in cfg.sim.depth_power_map.items()}
        return DomainParams(
            r_th_c_per_w=cfg.sim.r_th_c_per_w * r_mul,
            c_th_j_per_c=cfg.sim.c_th_j_per_c * c_mul,
            depth_power_w=power_map,
            ambient_c=ambient,
            init_temp_c=ambient + 5.0,
            sensor_noise_c=dr.sensor_noise_c,
        )

    def apply(self, cfg: "PoiseConfig", params: DomainParams) -> RCThermalSimulator:
        return RCThermalSimulator(
            r_th_c_per_w=params.r_th_c_per_w,
            c_th_j_per_c=params.c_th_j_per_c,
            ambient_c=params.ambient_c,
            temp_max_c=cfg.thermal.temp_max_c,
            depth_power_w=params.depth_power_w,
            depth_latency_s=cfg.sim.depth_latency_map,
            init_temp_c=params.init_temp_c,
            params_source=cfg.sim.params_source,
        )


__all__ = ["DomainParams", "DomainRandomizer"]
