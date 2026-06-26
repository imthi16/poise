"""Reward function(s) for the depth-allocation policy (⚠ CLAUDE.md §6, §9).

⚠ RESEARCH-CRITICAL — reward shaping is the experiment, not a constant.

    reward_t =  w_thr     · throughput_norm
              − w_q       · quality_cost(budget)        # from the depth→quality table
              − w_therm   · relu(temp − temp_setpoint)
              − w_energy  · energy_per_token_norm
              − w_throttle· I[throttled]                # large; near-terminal

Mandatory guards encoded here:
  * 🔒 Reward-hacking: if ``w_q`` is too small relative to ``w_thr``/``w_energy``, the
    optimal policy is trivially "always minimum depth". ``train_ppo.py`` MUST sweep
    weights and report the converged depth distribution — a policy that collapses to
    min depth is a FAILED reward, not a result. ``DepthHistogram`` is provided for that.
  * 🔒 The quality term is a HARD-TO-GAME signal (KL/perplexity vs full depth) read
    from a PRECOMPUTED table; it is never a self-reported confidence the policy controls.
  * All ``w_*`` are tunable; none are "the answer".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RewardConfig
    from .simulator import SimStep


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Quality-cost table (the hard-to-game term)
# --------------------------------------------------------------------------- #
class QualityCostTable:
    """depth → quality cost, from the step-3 gate's KL-vs-full measurements.

    ``cost(budget)`` interpolates the measured KL; ``normalized_cost`` scales to
    [0, 1] by the worst measured KL so the term is comparable to throughput/energy.
    """

    def __init__(self, depth_to_kl: Mapping[int, float], *, synthetic: bool = False):
        if not depth_to_kl:
            raise ValueError("quality cost table is empty")
        self.synthetic = synthetic
        self._depths = np.array(sorted(depth_to_kl.keys()), dtype=np.float64)
        self._kl = np.array([depth_to_kl[int(d)] for d in self._depths], dtype=np.float64)
        self._kl_max = float(self._kl.max()) if self._kl.max() > 0 else 1.0

    def cost(self, budget: float) -> float:
        return float(np.interp(budget, self._depths, self._kl))

    def normalized_cost(self, budget: float) -> float:
        return self.cost(budget) / self._kl_max

    @classmethod
    def from_db(cls, conn, dataset_tag: str | None = None) -> "QualityCostTable":
        """Load the REAL measured quality profile (after the gate has run)."""
        from ..storage import models

        rows = models.get_quality_profile(conn, dataset_tag=dataset_tag)
        if not rows:
            raise ValueError(
                "no quality_profile rows — run the step-3 gate "
                "(calibration/quality_profile.run_gate) before RL training."
            )
        return cls({int(r["depth"]): float(r["mean_kl_vs_full"]) for r in rows})


def synthetic_quality_table(depths: Sequence[int], layer_total: int) -> QualityCostTable:
    """A CLEARLY-LABELED synthetic depth→KL table for off-device RL plumbing ONLY.

    This is NOT a measured result. It encodes only the qualitative prior that shallower
    depth costs more KL (monotonic, zero at full depth). Real training reads the
    measured table from the DB via ``QualityCostTable.from_db``. Mirrors the
    simulator's placeholder-vs-calibrated split so nothing fabricated leaks into claims.
    """
    table = {}
    for d in depths:
        frac = d / layer_total
        table[int(d)] = float(max(0.0, (1.0 - frac)) ** 2)  # 0 at full depth, grows shallow
    return QualityCostTable(table, synthetic=True)


# --------------------------------------------------------------------------- #
# Reward computation
# --------------------------------------------------------------------------- #
@dataclass
class RewardComponents:
    throughput: float
    quality: float
    thermal: float
    energy: float
    throttle: float
    total: float

    def as_dict(self) -> dict:
        return {
            "throughput": self.throughput,
            "quality": self.quality,
            "thermal": self.thermal,
            "energy": self.energy,
            "throttle": self.throttle,
            "total": self.total,
        }


class RewardFn:
    """Bundles weights + the quality table + normalization references."""

    def __init__(
        self,
        weights: "RewardConfig",
        quality_table: QualityCostTable,
        *,
        temp_setpoint_c: float,
        throughput_norm_tok_s: float,
        energy_norm_j: float,
    ):
        self.w = weights
        self.q = quality_table
        self.temp_setpoint_c = temp_setpoint_c
        self.thr_norm = max(1e-6, throughput_norm_tok_s)
        self.energy_norm = max(1e-6, energy_norm_j)

    def __call__(self, step: "SimStep", budget: int) -> tuple[float, RewardComponents]:
        thr_term = self.w.w_thr * (step.tok_per_s / self.thr_norm)
        quality_term = self.w.w_q * self.q.normalized_cost(budget)
        thermal_term = self.w.w_therm * relu(step.temp_c - self.temp_setpoint_c)
        # energy per token ~ power / throughput; normalize by a reference.
        e_per_tok = step.power_w / step.tok_per_s if step.tok_per_s > 0 else step.energy_j
        energy_term = self.w.w_energy * (e_per_tok / self.energy_norm)
        throttle_term = self.w.w_throttle * (1.0 if step.throttled else 0.0)

        total = thr_term - quality_term - thermal_term - energy_term - throttle_term
        comps = RewardComponents(
            throughput=thr_term,
            quality=-quality_term,
            thermal=-thermal_term,
            energy=-energy_term,
            throttle=-throttle_term,
            total=total,
        )
        return total, comps


# --------------------------------------------------------------------------- #
# Reward-hacking diagnostic: the converged depth distribution
# --------------------------------------------------------------------------- #
class DepthHistogram:
    """Tracks the distribution of chosen budgets — the reward-hacking check.

    🔒 A policy that collapses to (near-)minimum depth is a FAILED reward, not a
    success. ``train_ppo.py`` logs this and persists it to the ppo_checkpoints row.
    """

    def __init__(self):
        self._counts: Counter = Counter()

    def update(self, budget: int) -> None:
        self._counts[int(budget)] += 1

    def distribution(self) -> dict[int, float]:
        total = sum(self._counts.values())
        if total == 0:
            return {}
        return {int(b): c / total for b, c in sorted(self._counts.items())}

    def collapsed_to_min(self, layer_min: int, threshold: float = 0.9) -> bool:
        dist = self.distribution()
        return dist.get(int(layer_min), 0.0) >= threshold

    def mean_budget(self) -> float:
        total = sum(self._counts.values())
        if total == 0:
            return 0.0
        return sum(b * c for b, c in self._counts.items()) / total


__all__ = [
    "relu",
    "QualityCostTable",
    "synthetic_quality_table",
    "RewardComponents",
    "RewardFn",
    "DepthHistogram",
]
