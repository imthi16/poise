"""RL: the off-device training world (calibrated RC simulator + Gymnasium env + reward).

🔒 Trained OFF-DEVICE (Kaggle) against the simulator — on-board RL is far too slow
(thermal time-constants are minutes). ``train_ppo`` is imported lazily (it needs
stable-baselines3) so this package imports without the RL training dep.
"""

from __future__ import annotations

from .simulator import RCThermalSimulator, SimStep
from .reward import (
    QualityCostTable,
    synthetic_quality_table,
    RewardFn,
    RewardComponents,
    DepthHistogram,
)
from .domain_random import DomainRandomizer, DomainParams
from .env import PoiseThermalEnv, ObsSpec, build_observation, OBS_DIM

__all__ = [
    "RCThermalSimulator",
    "SimStep",
    "QualityCostTable",
    "synthetic_quality_table",
    "RewardFn",
    "RewardComponents",
    "DepthHistogram",
    "DomainRandomizer",
    "DomainParams",
    "PoiseThermalEnv",
    "ObsSpec",
    "build_observation",
    "OBS_DIM",
]
