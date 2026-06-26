"""PID controller + budget allocator, incl. the thermal-safety override (CLAUDE.md §6,§10).

DoD §10: "pid mode demonstrably holds temp near setpoint by reducing depth; safety
override at TEMP_MAX verified; clamping tested." We close the loop here against the
RC simulator off-device.
"""

from __future__ import annotations

from poise.control.pid import PID
from poise.control.budget_allocator import BudgetAllocator, snap_to_budget_set
from poise.rl.simulator import RCThermalSimulator
from poise.telemetry.schema import TelemetrySample


def _sample(temp_c, throttled=False):
    return TelemetrySample(ts=0.0, temp_c=temp_c, power_w=30.0, gpu_clock_mhz=1300.0,
                          gpu_util=90.0, throttled=throttled)


# --- PID unit behavior ----------------------------------------------------- #
def test_pid_full_depth_at_setpoint():
    pid = PID(kp=0.8, ki=0.05, kd=0.2, setpoint=80.0, output_min=16, output_max=32, bias=32)
    out = pid.update(80.0, dt=0.25)  # zero error => bias => full depth
    assert abs(out - 32.0) < 1e-9


def test_pid_reduces_budget_when_hot():
    pid = PID(kp=1.0, ki=0.05, kd=0.0, setpoint=80.0, output_min=16, output_max=32, bias=32)
    out = pid.update(90.0, dt=0.25)  # 10C over setpoint => push down
    assert out < 32.0


def test_pid_raises_budget_when_cool_but_clamped():
    pid = PID(kp=1.0, ki=0.05, kd=0.0, setpoint=80.0, output_min=16, output_max=32, bias=32)
    out = pid.update(60.0, dt=0.25)  # cool => wants more, clamped at max
    assert out == 32.0


def test_pid_anti_windup_bounded():
    pid = PID(kp=1.0, ki=1.0, kd=0.0, setpoint=80.0, output_min=16, output_max=32,
              bias=32, integral_limit=8.0)
    for _ in range(1000):
        pid.update(95.0, dt=0.25)  # sustained over-temp
    assert -8.0 - 1e-6 <= pid._integral <= 8.0 + 1e-6  # integral stays clamped


# --- allocator ------------------------------------------------------------- #
def test_snap_to_budget_set():
    bs = [16, 20, 24, 28, 32]
    assert snap_to_budget_set(23.4, bs) == 24
    assert snap_to_budget_set(17.0, bs) == 16
    assert snap_to_budget_set(99, bs) == 32


def test_safety_override_forces_min_budget(cfg):
    alloc = BudgetAllocator(cfg, mode="pid")
    # temp at/above TEMP_MAX must force the minimum budget regardless of PID
    b = alloc.budget(_sample(cfg.thermal.temp_max_c + 1.0))
    assert b == min(cfg.depth.budget_set)


def test_static_mode_constant(cfg):
    alloc = BudgetAllocator(cfg, mode="static")
    b1 = alloc.budget(_sample(50.0))
    b2 = alloc.budget(_sample(70.0))
    assert b1 == b2 == cfg.depth.static_depth


def test_allocator_clamps_into_budget_set(cfg):
    alloc = BudgetAllocator(cfg, mode="pid")
    b = alloc.budget(_sample(60.0))
    assert b in cfg.depth.budget_set


def test_pid_mode_holds_temp_near_setpoint_by_reducing_depth(cfg):
    """Closed loop: allocator(PID) + RC simulator should settle temperature near the
    setpoint by shedding depth, not by throttling.

    Scenario: ambient=40C, R_th=1.5 => full-depth steady state = 40 + 35*1.5 = 92.5C,
    which EXCEEDS TEMP_MAX (87) => a fixed-depth model would throttle. PID must instead
    hold near the 80C setpoint by dropping to an intermediate budget (~24, T_eq=79C).
    Gains are tuned here (config defaults are an explicit 'starting point — tune').
    """
    ambient, r_th = 40.0, 1.5
    power_map = {16: 18.0, 20: 22.0, 24: 26.0, 28: 30.0, 32: 35.0}
    sim = RCThermalSimulator(
        r_th_c_per_w=r_th, c_th_j_per_c=30.0, ambient_c=ambient,
        temp_max_c=cfg.thermal.temp_max_c,
        depth_power_w=power_map,
        depth_latency_s={16: 0.018, 20: 0.022, 24: 0.027, 28: 0.031, 32: 0.036},
        init_temp_c=45.0,
    )
    setpoint = cfg.thermal.temp_setpoint_c
    # full depth would breach the ceiling (motivates control)
    assert ambient + power_map[32] * r_th > cfg.thermal.temp_max_c

    clk = {"t": 0.0}
    tuned_pid = PID(kp=1.0, ki=0.3, kd=0.0, setpoint=setpoint,
                    output_min=cfg.depth.layer_min, output_max=cfg.depth.layer_max,
                    bias=cfg.depth.layer_max, integral_limit=60.0)
    alloc = BudgetAllocator(cfg, mode="pid", pid=tuned_pid, clock=lambda: clk["t"])

    budget = cfg.depth.layer_max
    temps, budgets = [], []
    for _ in range(600):
        step = sim.step(budget, dt=0.25)
        clk["t"] += 0.25
        budget = alloc.budget(_sample(step.temp_c, throttled=step.throttled))
        temps.append(step.temp_c)
        budgets.append(budget)

    settled = temps[-60:]
    assert max(settled) < cfg.thermal.temp_max_c          # no hard ceiling breach
    assert abs(sum(settled) / len(settled) - setpoint) < 5.0  # held near setpoint
    assert min(budgets) < cfg.depth.layer_max             # achieved by reducing depth


def test_ppo_mode_falls_back_to_pid_without_policy(cfg):
    """ppo mode with no policy loaded must behave (fall back to PID), not crash."""
    alloc = BudgetAllocator(cfg, mode="ppo", policy=None)
    b = alloc.budget(_sample(85.0))
    assert b in cfg.depth.budget_set
