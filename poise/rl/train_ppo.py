"""PPO training entrypoint — runs OFF-DEVICE on Kaggle (CLAUDE.md §6, §9).

🔒 Train off-device only: on-board RL is far too slow (thermal time-constants are
minutes). The policy trains against the calibrated RC simulator with domain
randomization, then is validated separately on the real board (step 9).

🔒 Reward-hacking guard: ``sweep_weights`` runs PPO across reward-weight configs and
reports the CONVERGED DEPTH DISTRIBUTION for each — a policy that collapses to minimum
depth is a FAILED reward, not a success. The histogram is persisted to the
``ppo_checkpoints`` row so the check is auditable.

stable-baselines3 (and torch) are imported lazily so this module imports without the
RL training dep; the diagnostics (rollout distribution, collapse check, config
snapshot) are unit-tested off-device with a fake policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

from .domain_random import DomainRandomizer
from .env import PoiseThermalEnv
from .reward import DepthHistogram, QualityCostTable, synthetic_quality_table

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Env construction + quality table resolution
# --------------------------------------------------------------------------- #
def resolve_quality_table(cfg: "PoiseConfig", conn=None) -> QualityCostTable:
    """Prefer the MEASURED depth→KL table (step-3 gate). Fall back to the clearly-
    labeled synthetic table only for plumbing, with a loud warning."""
    if conn is not None:
        try:
            return QualityCostTable.from_db(conn)
        except Exception:
            pass
    import warnings

    warnings.warn(
        "Using SYNTHETIC quality table — no measured quality_profile in the DB. Run the "
        "step-3 gate (calibration.quality_profile.run_gate) before producing a policy "
        "you intend to trust. This is plumbing-only.",
        stacklevel=2,
    )
    return synthetic_quality_table(list(cfg.depth.budget_set), cfg.depth.layer_total)


def make_env_fn(
    cfg: "PoiseConfig",
    quality_table: QualityCostTable,
    *,
    domain_random: bool = True,
    seed: Optional[int] = None,
) -> Callable[[], PoiseThermalEnv]:
    def _fn() -> PoiseThermalEnv:
        dr = DomainRandomizer(enabled=domain_random)
        return PoiseThermalEnv(cfg, quality_table=quality_table, domain_random=dr, seed=seed)

    return _fn


# --------------------------------------------------------------------------- #
# Diagnostics (testable off-device)
# --------------------------------------------------------------------------- #
def rollout_depth_distribution(
    act_fn: Callable[[np.ndarray], int],
    env: PoiseThermalEnv,
    n_steps: int = 2000,
    seed: Optional[int] = None,
) -> DepthHistogram:
    """Roll a policy (obs->action) through the env and tally chosen budgets.

    ``act_fn`` lets us test the diagnostic with a fake policy off-device; the real
    path passes the trained PPO model's predict.
    """
    hist = DepthHistogram()
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        action = int(act_fn(obs))
        obs, _, terminated, truncated, info = env.step(action)
        hist.update(info["budget"])
        if terminated or truncated:
            obs, _ = env.reset()
    return hist


@dataclass
class TrainResult:
    policy_path: str
    train_steps: int
    final_mean_reward: float
    converged_depth_hist: dict
    collapsed_to_min: bool
    reward_config: dict
    env_config: dict
    sim_params: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _snapshot_configs(cfg: "PoiseConfig") -> tuple[dict, dict, dict]:
    reward_cfg = asdict(cfg.ppo.reward)
    env_cfg = asdict(cfg.ppo.env)
    env_cfg.update({"budget_set": list(cfg.depth.budget_set), "dt_s": cfg.ppo.env.dt_s})
    sim_params = {
        "r_th_c_per_w": cfg.sim.r_th_c_per_w,
        "c_th_j_per_c": cfg.sim.c_th_j_per_c,
        "params_source": cfg.sim.params_source,
        "domain_random": asdict(cfg.ppo.domain_random),
    }
    return reward_cfg, env_cfg, sim_params


# --------------------------------------------------------------------------- #
# Training (lazy SB3)
# --------------------------------------------------------------------------- #
def train(
    cfg: "PoiseConfig",
    out_path: str | Path = "ppo_policy.zip",
    *,
    conn=None,
    quality_table: Optional[QualityCostTable] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    persist_db: bool = True,
) -> TrainResult:
    """Train one PPO policy and persist it + its audit row. Requires stable-baselines3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.evaluation import evaluate_policy
    except Exception as e:  # pragma: no cover - off-device
        raise RuntimeError(
            "PPO training requires stable-baselines3 (pip install -e .[rl]). Train on "
            "Kaggle/GPU box, not the Jetson (on-board RL is far too slow)."
        ) from e

    if quality_table is None:
        quality_table = resolve_quality_table(cfg, conn)
    seed = cfg.ppo.seed if seed is None else seed
    steps = total_timesteps or cfg.ppo.total_timesteps

    env = DummyVecEnv([make_env_fn(cfg, quality_table, domain_random=True, seed=seed)])
    model = PPO(
        cfg.ppo.policy,
        env,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        learning_rate=cfg.ppo.learning_rate,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        policy_kwargs={"net_arch": list(cfg.ppo.net_arch)},
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=steps)
    model.save(str(out_path))

    # converged depth distribution (the reward-hacking check)
    eval_env = make_env_fn(cfg, quality_table, domain_random=True, seed=seed + 1)()
    hist = rollout_depth_distribution(
        lambda o: int(model.predict(o, deterministic=True)[0]), eval_env, n_steps=4000,
        seed=seed + 1,
    )
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    reward_cfg, env_cfg, sim_params = _snapshot_configs(cfg)
    result = TrainResult(
        policy_path=str(out_path),
        train_steps=int(steps),
        final_mean_reward=float(mean_reward),
        converged_depth_hist=hist.distribution(),
        collapsed_to_min=hist.collapsed_to_min(cfg.depth.layer_min),
        reward_config=reward_cfg,
        env_config=env_cfg,
        sim_params=sim_params,
    )
    if result.collapsed_to_min:
        import warnings

        warnings.warn(
            "⚠ Policy COLLAPSED to minimum depth — this is a FAILED reward, not a result. "
            "Increase w_q / re-shape the reward (CLAUDE.md §9).",
            stacklevel=2,
        )
    if persist_db and conn is not None:
        _persist(conn, result)
    return result


def _persist(conn, result: TrainResult) -> str:
    from ..storage import models

    return models.insert_ppo_checkpoint(
        conn,
        path=result.policy_path,
        reward_config_json=result.reward_config,
        env_config_json=result.env_config,
        sim_params_json=result.sim_params,
        train_steps=result.train_steps,
        final_mean_reward=result.final_mean_reward,
        converged_depth_hist_json=result.converged_depth_hist,
    )


def sweep_weights(
    cfg: "PoiseConfig",
    weight_grid: Sequence[dict],
    out_dir: str | Path = "data/results/ppo_sweep",
    *,
    conn=None,
    total_timesteps: Optional[int] = None,
) -> list[TrainResult]:
    """🔒 Reward-weight sweep. Trains a policy per weight config and reports the
    converged depth distribution for each — the required reward-hacking audit."""
    from dataclasses import replace

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[TrainResult] = []
    for i, w in enumerate(weight_grid):
        reward = replace(cfg.ppo.reward, **w)
        ppo = replace(cfg.ppo, reward=reward)
        cfg_i = replace(cfg, ppo=ppo)
        res = train(
            cfg_i, out / f"policy_{i}.zip", conn=conn,
            total_timesteps=total_timesteps, seed=cfg.ppo.seed + i,
        )
        results.append(res)
        print(
            f"[{i}] weights={w} mean_reward={res.final_mean_reward:.3f} "
            f"collapsed_to_min={res.collapsed_to_min} dist={res.converged_depth_hist}"
        )
    return results


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from ..config import load_config
    from ..storage.db import init_db

    ap = argparse.ArgumentParser(description="POISE PPO training (off-device)")
    ap.add_argument("--out", default="ppo_policy.zip")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    conn = init_db(cfg.storage.db_path)
    result = train(cfg, args.out, conn=conn, total_timesteps=args.timesteps, seed=args.seed)
    print(json.dumps(result.as_dict(), indent=2, default=str))
    conn.close()


__all__ = [
    "resolve_quality_table",
    "make_env_fn",
    "rollout_depth_distribution",
    "TrainResult",
    "train",
    "sweep_weights",
    "main",
]
