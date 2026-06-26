"""PPO policy wrapper for inference-time depth allocation (CLAUDE.md §6, §9).

``Policy.load(path, cfg)`` loads a trained PPO checkpoint; ``Policy.act(obs)`` returns
a discrete action (index into ``BUDGET_SET``).

🔒 The observation is built with the SAME ``build_observation`` + ``ObsSpec`` used by
the training env (``rl/env.py``) — identical features and normalization. If they
diverge, sim-to-real transfer is invalid, so the builder is imported and shared, never
re-implemented here.

Falls back safely: if the checkpoint is missing or the package (stable-baselines3) is
unavailable, ``load`` returns ``None`` and the allocator uses PID instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..rl.env import ObsSpec, build_observation
from ..rl.simulator import RCThermalSimulator

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


class Policy:
    def __init__(self, model, obs_spec: ObsSpec, budget_set: list[int]):
        self._model = model
        self.obs_spec = obs_spec
        self.budget_set = budget_set

    @classmethod
    def load(cls, path: Optional[str], cfg: "PoiseConfig") -> Optional["Policy"]:
        """Load a PPO checkpoint. Returns None (=> PID fallback) if unavailable."""
        if not path:
            return None
        try:
            from pathlib import Path

            if not Path(path).exists():
                return None
            from stable_baselines3 import PPO  # lazy: heavy training dep
        except Exception:
            return None
        try:
            model = PPO.load(path, device="cpu")
        except Exception:
            return None
        sim = RCThermalSimulator.from_config(cfg)
        obs_spec = ObsSpec.from_config(cfg, sim)
        return cls(model, obs_spec, list(cfg.depth.budget_set))

    def build_obs(
        self,
        *,
        temp_c: float,
        power_w: float,
        throttled: bool,
        prev_budget: int,
        tok_per_s: float,
        ambient_c: float,
        t_elapsed_s: float,
    ) -> np.ndarray:
        """Identical to the training env's observation (shared builder)."""
        return build_observation(
            self.obs_spec,
            temp_c=temp_c,
            power_w=power_w,
            throttled=throttled,
            prev_budget=prev_budget,
            tok_per_s=tok_per_s,
            ambient_c=ambient_c,
            t_elapsed_s=t_elapsed_s,
        )

    def act(self, obs: np.ndarray) -> int:
        """Deterministic action (index into budget_set). Raises on out-of-range so the
        allocator can fall back to PID."""
        action, _ = self._model.predict(obs, deterministic=True)
        idx = int(np.asarray(action).reshape(-1)[0])
        if not (0 <= idx < len(self.budget_set)):
            raise ValueError(f"policy produced out-of-range action {idx}")
        return idx

    def act_budget(self, obs: np.ndarray) -> int:
        return self.budget_set[self.act(obs)]


__all__ = ["Policy"]
