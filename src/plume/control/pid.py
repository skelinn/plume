"""Small PID controller with output clamping and conditional-integration anti-windup."""

from __future__ import annotations

import numpy as np


class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        limits: tuple[float, float] = (-np.inf, np.inf),
        i_limit: float = np.inf,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limits = limits
        self.i_limit = i_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error: float | None = None

    def __call__(self, error: float, dt: float, derivative: float | None = None) -> float:
        if derivative is None:
            derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        unclamped = self.kp * error + self.ki * self.integral + self.kd * derivative
        out = float(np.clip(unclamped, *self.limits))
        # integrate only when not pushing further into saturation
        if out == unclamped or np.sign(error) != np.sign(unclamped):
            self.integral = float(np.clip(self.integral + error * dt, -self.i_limit, self.i_limit))
        return out
