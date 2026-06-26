"""State → layer budget (CLAUDE.md §6).

``BudgetAllocator(mode).budget(state, ctx)`` maps the live hardware state to a layer
budget. Modes:
  * ``static`` — constant ``static_depth``.
  * ``pid``    — PID only (reactive thermal control).
  * ``ppo``    — the trained policy, with PID as a SAFE FALLBACK if the policy is
                 unavailable or proposes an out-of-range action.

🔒 Hard rules:
  * always clamp to ``[layer_min, layer_max]`` and snap to ``BUDGET_SET``;
  * if a budget would be chosen while ``temp >= TEMP_MAX``, force the MINIMUM budget
    regardless of policy/PID output. Thermal safety overrides the policy, always.

The allocator is the ``budget_fn`` handed to ``engine.adaptive_runner`` — keeping all
control/policy here and the runner pure mechanism (CLAUDE.md §9). It conditions depth
on hardware state; that is the contribution.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

from .pid import PID

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..telemetry.schema import TelemetrySample
    from ..engine.adaptive_runner import BudgetContext
    from .policy import Policy


def snap_to_budget_set(value: float, budget_set: list[int]) -> int:
    """Nearest allowed depth in BUDGET_SET (ties -> lower depth = safer/cheaper)."""
    return min(budget_set, key=lambda b: (abs(b - value), b))


class BudgetAllocator:
    def __init__(
        self,
        cfg: "PoiseConfig",
        mode: Optional[str] = None,
        *,
        pid: Optional[PID] = None,
        policy: Optional["Policy"] = None,
        safety_override: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.mode = mode or cfg.control.mode
        self.budget_set = list(cfg.depth.budget_set)
        self.layer_min = cfg.depth.layer_min
        self.layer_max = cfg.depth.layer_max
        self.min_budget = min(self.budget_set)
        self.temp_max_c = cfg.thermal.temp_max_c
        self.safety_override = safety_override
        self.policy = policy
        self._clock = clock

        if pid is None:
            pid = PID(
                kp=cfg.control.pid.kp,
                ki=cfg.control.pid.ki,
                kd=cfg.control.pid.kd,
                setpoint=cfg.thermal.temp_setpoint_c,
                output_min=cfg.control.pid.output_min,
                output_max=cfg.control.pid.output_max,
                bias=cfg.control.pid.output_max,
                integral_limit=cfg.control.pid.integral_limit,
            )
        self.pid = pid

        # inference-time state for obs reconstruction (ppo) + dt estimation
        self._prev_budget = self.layer_max
        self._t_start: Optional[float] = None
        self._t_last: Optional[float] = None
        self._period_s = cfg.control.period_ms / 1000.0

    def reset(self) -> None:
        self.pid.reset()
        self._prev_budget = self.layer_max
        self._t_start = None
        self._t_last = None

    # --- callable as the runner's budget_fn -------------------------------- #
    def __call__(self, ctx: "BudgetContext") -> int:
        return self.budget(ctx.telemetry, ctx)

    def budget(self, state: Optional["TelemetrySample"], ctx=None) -> int:
        now = self._clock()
        if self._t_start is None:
            self._t_start = now
        dt = self._period_s if self._t_last is None else max(1e-3, now - self._t_last)
        self._t_last = now

        chosen = self._raw_budget(state, dt, ctx)
        chosen = self._clamp(chosen)

        # 🔒 Thermal-safety override — beats policy/PID, always.
        if self.safety_override and state is not None and state.temp_c >= self.temp_max_c:
            chosen = self.min_budget

        self._prev_budget = chosen
        return chosen

    # --- per-mode raw decision (pre safety/clamp) -------------------------- #
    def _raw_budget(self, state, dt, ctx) -> int:
        if self.mode == "static":
            return self.cfg.depth.static_depth

        if state is None:
            # No telemetry yet: default to full depth (safe, max quality).
            return self.layer_max

        if self.mode == "pid":
            return self._pid_budget(state, dt)

        if self.mode == "ppo":
            if self.policy is not None:
                try:
                    return self._ppo_budget(state, dt, ctx)
                except Exception:
                    # policy failure => fall back to PID (safe), never crash generation
                    return self._pid_budget(state, dt)
            return self._pid_budget(state, dt)

        raise ValueError(f"unknown control mode: {self.mode!r}")

    def _pid_budget(self, state, dt) -> int:
        signal = self.pid.update(state.temp_c, dt)
        return snap_to_budget_set(signal, self.budget_set)

    def _ppo_budget(self, state, dt, ctx) -> int:
        assert self.policy is not None
        t_elapsed = (self._t_last - self._t_start) if self._t_start is not None else 0.0
        tok_per_s = 1.0 / dt if dt > 0 else 0.0
        prev_budget = (
            ctx.prev_budget if (ctx is not None and ctx.prev_budget is not None)
            else self._prev_budget
        )
        obs = self.policy.build_obs(
            temp_c=state.temp_c,
            power_w=state.power_w,
            throttled=state.throttled,
            prev_budget=prev_budget,
            tok_per_s=tok_per_s,
            ambient_c=self.cfg.sim.ambient_c,
            t_elapsed_s=t_elapsed,
        )
        return self.policy.act_budget(obs)

    def _clamp(self, b: float) -> int:
        b = int(round(b))
        b = max(self.layer_min, min(self.layer_max, b))
        return snap_to_budget_set(b, self.budget_set)


def make_allocator(
    cfg: "PoiseConfig", *, policy: Optional["Policy"] = None, mode: Optional[str] = None
) -> BudgetAllocator:
    """Build an allocator for the configured mode. In ``ppo`` mode, loads the policy
    from ``cfg.control.policy_path`` if not supplied (None => PID fallback)."""
    resolved_mode = mode or cfg.control.mode
    if resolved_mode == "ppo" and policy is None:
        from .policy import Policy

        policy = Policy.load(cfg.control.policy_path, cfg)
    return BudgetAllocator(cfg, mode=resolved_mode, policy=policy)


__all__ = ["BudgetAllocator", "snap_to_budget_set", "make_allocator"]
