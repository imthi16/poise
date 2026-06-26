"""Gymnasium env wrapping the RC simulator + depth→quality-cost table (CLAUDE.md §6).

- Observation (normalized): [temp, power, clock_proxy, throttled, prev_budget,
  recent_throughput, ambient, time_in_episode].
- Action: discrete index into ``BUDGET_SET``.
- Step: apply budget → simulator advances dt → reward (rl/reward.py), using the
  PRECOMPUTED depth→quality table for the quality term. The LLM is NEVER run during
  RL training — that is what makes off-device training feasible.

🔒 The obs-builder (``build_observation`` + ``ObsSpec``) is SHARED with
``control/policy.py`` so the inference-time observation is byte-for-byte identical to
training. Any divergence invalidates sim-to-real transfer.

⚠ The static depth→quality mapping is an approximation (real cost is context-
dependent). On-board validation (step 9) uses true per-token KL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False
    gym = None  # type: ignore

from .reward import DepthHistogram, QualityCostTable, RewardFn, synthetic_quality_table
from .simulator import RCThermalSimulator

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Shared observation builder (env + policy MUST use this identically)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObsSpec:
    ambient_c: float
    temp_max_c: float
    max_power_w: float
    throughput_norm_tok_s: float
    layer_min: int
    layer_max: int
    episode_seconds: float
    throttle_clock_factor: float = 0.7

    @classmethod
    def from_config(cls, cfg: "PoiseConfig", sim: RCThermalSimulator) -> "ObsSpec":
        max_power = float(max(cfg.sim.depth_power_map.values())) if cfg.sim.depth_power_map else 40.0
        return cls(
            ambient_c=cfg.sim.ambient_c,
            temp_max_c=cfg.thermal.temp_max_c,
            max_power_w=max_power,
            throughput_norm_tok_s=cfg.ppo.env.throughput_norm_tok_s,
            layer_min=cfg.depth.layer_min,
            layer_max=cfg.depth.layer_max,
            episode_seconds=cfg.ppo.env.episode_seconds,
            throttle_clock_factor=sim.throttle_clock_factor,
        )


OBS_DIM = 8


def build_observation(
    spec: ObsSpec,
    *,
    temp_c: float,
    power_w: float,
    throttled: bool,
    prev_budget: int,
    tok_per_s: float,
    ambient_c: float,
    t_elapsed_s: float,
) -> np.ndarray:
    """Construct the normalized observation vector. SHARED by env + policy."""
    span = max(1e-6, spec.temp_max_c - spec.ambient_c)
    budget_span = max(1, spec.layer_max - spec.layer_min)
    obs = np.array(
        [
            (temp_c - spec.ambient_c) / span,                         # temp
            power_w / max(1e-6, spec.max_power_w),                    # power
            spec.throttle_clock_factor if throttled else 1.0,        # clock proxy
            1.0 if throttled else 0.0,                               # throttled
            (prev_budget - spec.layer_min) / budget_span,            # prev budget
            tok_per_s / max(1e-6, spec.throughput_norm_tok_s),       # recent throughput
            np.clip((ambient_c - 20.0) / 20.0, 0.0, 1.5),            # ambient
            np.clip(t_elapsed_s / max(1e-6, spec.episode_seconds), 0.0, 1.0),  # time
        ],
        dtype=np.float32,
    )
    return obs


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
if _HAS_GYM:

    class PoiseThermalEnv(gym.Env):
        """Thermal depth-allocation env. One step == one control period (dt)."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            cfg: "PoiseConfig",
            *,
            quality_table: Optional[QualityCostTable] = None,
            domain_random=None,
            seed: Optional[int] = None,
        ):
            super().__init__()
            self.cfg = cfg
            self.budget_set = list(cfg.depth.budget_set)
            self.dt = cfg.ppo.env.dt_s
            self.episode_seconds = cfg.ppo.env.episode_seconds
            self.domain_random = domain_random

            self.sim = RCThermalSimulator.from_config(cfg)
            self.obs_spec = ObsSpec.from_config(cfg, self.sim)

            if quality_table is None:
                # off-device plumbing: clearly-labeled synthetic table (NOT a result)
                quality_table = synthetic_quality_table(self.budget_set, cfg.depth.layer_total)
            self.quality_table = quality_table
            self.reward_fn = RewardFn(
                cfg.ppo.reward,
                quality_table,
                temp_setpoint_c=cfg.thermal.temp_setpoint_c,
                throughput_norm_tok_s=cfg.ppo.env.throughput_norm_tok_s,
                energy_norm_j=cfg.ppo.env.energy_norm_j,
            )

            self.action_space = spaces.Discrete(len(self.budget_set))
            # Bounds are generous: normalized temp can exceed 1 in throttling regimes
            # and ambient/throughput terms vary with domain randomization.
            self.observation_space = spaces.Box(
                low=-1.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32
            )

            self._rng = np.random.default_rng(seed)
            self.depth_hist = DepthHistogram()
            self._t = 0.0
            self._prev_budget = cfg.depth.layer_max
            self._last_tok_per_s = 0.0

        # --- gym API -------------------------------------------------------- #
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)  # sets self._np_random per the Gymnasium contract
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            # Per-episode domain randomization (sim-to-real robustness).
            if self.domain_random is not None:
                params = self.domain_random.sample(self.cfg, self._rng)
                self.sim = self.domain_random.apply(self.cfg, params)
                self.obs_spec = ObsSpec.from_config(self.cfg, self.sim)
                self._sensor_noise_c = params.sensor_noise_c
                init_temp = params.init_temp_c
                ambient = params.ambient_c
            else:
                self._sensor_noise_c = 0.0
                init_temp = self.cfg.sim.init_temp_c
                ambient = self.cfg.sim.ambient_c
            self.sim.reset(ambient_c=ambient, init_temp_c=init_temp)

            self._t = 0.0
            self._prev_budget = self.cfg.depth.layer_max
            self._last_tok_per_s = self.sim.latency_for_depth(self._prev_budget)
            self._last_tok_per_s = 1.0 / self._last_tok_per_s if self._last_tok_per_s > 0 else 0.0
            self.depth_hist = DepthHistogram()

            obs = self._observe(self.sim.temp_c, 0.0, False, self._prev_budget,
                                self._last_tok_per_s, ambient)
            return obs, {"params_source": self.sim.params_source}

        def step(self, action: int):
            budget = self.budget_set[int(action)]
            self.depth_hist.update(budget)
            sstep = self.sim.step(budget, self.dt)
            self._t += self.dt
            reward, comps = self.reward_fn(sstep, budget)

            obs_temp = sstep.temp_c + (
                self._rng.normal(0.0, self._sensor_noise_c) if self._sensor_noise_c > 0 else 0.0
            )
            obs = self._observe(obs_temp, sstep.power_w, sstep.throttled, budget,
                                sstep.tok_per_s, self.sim.ambient_c)
            self._prev_budget = budget
            self._last_tok_per_s = sstep.tok_per_s

            truncated = self._t >= self.episode_seconds
            terminated = False
            info = {
                "budget": budget,
                "temp_c": sstep.temp_c,
                "throttled": sstep.throttled,
                "tok_per_s": sstep.tok_per_s,
                "reward_components": comps.as_dict(),
                "depth_distribution": self.depth_hist.distribution(),
            }
            return obs, float(reward), terminated, truncated, info

        # --- internals ------------------------------------------------------ #
        def _observe(self, temp_c, power_w, throttled, prev_budget, tok_per_s, ambient_c):
            return build_observation(
                self.obs_spec,
                temp_c=temp_c,
                power_w=power_w,
                throttled=throttled,
                prev_budget=prev_budget,
                tok_per_s=tok_per_s,
                ambient_c=ambient_c,
                t_elapsed_s=self._t,
            )

else:  # pragma: no cover - gymnasium not installed

    class PoiseThermalEnv:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("gymnasium is required for PoiseThermalEnv (pip install gymnasium)")


__all__ = ["PoiseThermalEnv", "ObsSpec", "build_observation", "OBS_DIM"]
