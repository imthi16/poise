"""PID controller (CLAUDE.md §6).

Mapping (business rule): ``error = temp_setpoint − temp_measured``.
  * temp ABOVE setpoint (negative error) → output drives the budget DOWN
  * temp BELOW setpoint (positive error) → output drives the budget UP toward layer_max

With a feed-forward ``bias = layer_max``, zero error => full depth; as temperature
climbs above the setpoint the output falls. The continuous output is clamped to the
output limits; ``budget_allocator`` snaps it to the nearest value in ``BUDGET_SET``.

🔒 REACTIVE ONLY. PID lags slow thermal dynamics (the junction warms over seconds–
minutes). That lag is the MOTIVATION for the PPO layer — it is not a bug to tune away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    setpoint: float
    output_min: float = 16.0
    output_max: float = 32.0
    bias: float = 32.0            # feed-forward: output at zero error (full depth)
    integral_limit: float = 16.0  # anti-windup clamp on the integral term

    _integral: float = 0.0
    _prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement: float, dt: float) -> float:
        """Return the control signal (on the budget scale) for one control period."""
        error = self.setpoint - measurement
        if dt <= 0:
            dt = 1e-3

        # Derivative on error (first call: no derivative).
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        # Tentative integral, clamped (anti-windup).
        integral = self._integral + error * dt
        integral = _clamp(integral, -self.integral_limit, self.integral_limit)

        raw = self.bias + self.kp * error + self.ki * integral + self.kd * derivative
        output = _clamp(raw, self.output_min, self.output_max)

        # Conditional integration (anti-windup): only commit the integral if we are
        # not pushing further into saturation in the same direction.
        saturated_high = raw > self.output_max and error > 0
        saturated_low = raw < self.output_min and error < 0
        if not (saturated_high or saturated_low):
            self._integral = integral

        self._prev_error = error
        return output


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


__all__ = ["PID"]
