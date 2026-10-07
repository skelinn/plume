"""Translational guidance laws.

* :class:`WaypointGuidance` -- PD on position/velocity toward a waypoint (hops, hover).
* :func:`zem_zev_accel` -- Zero-Effort-Miss / Zero-Effort-Velocity optimal feedback
  guidance for powered descent (fixed time-to-go).
* :func:`time_to_go` -- time-to-go estimate for ZEM/ZEV given a thrust budget.
* :func:`thrust_command` -- convert a desired acceleration into throttle + axis,
  respecting throttle limits and a tilt cone.
"""

from __future__ import annotations

import math

import numpy as np


def limit_tilt(direction: np.ndarray, up: np.ndarray, max_tilt: float) -> np.ndarray:
    """Clamp ``direction`` to within ``max_tilt`` radians of ``up``."""
    d = direction / np.linalg.norm(direction)
    c = float(d @ up)
    if c >= math.cos(max_tilt):
        return d
    horiz = d - c * up
    hn = float(np.linalg.norm(horiz))
    if hn < 1e-12:
        return up.copy()
    return math.cos(max_tilt) * up + math.sin(max_tilt) * horiz / hn


def thrust_command(
    accel_des: np.ndarray,
    gravity: np.ndarray,
    mass: float,
    max_thrust: float,
    up: np.ndarray,
    max_tilt: float = math.radians(20),
    throttle_min: float = 0.0,
    throttle_max: float = 1.0,
):
    """Desired (non-gravitational) acceleration -> (throttle, thrust axis, thrust N)."""
    f = mass * (accel_des - gravity)
    fn = float(np.linalg.norm(f))
    axis = limit_tilt(f, up, max_tilt) if fn > 1e-9 else up.copy()
    # only the component along the (limited) axis is useful
    f_along = max(float(f @ axis), 0.0)
    throttle = f_along / max(max_thrust, 1e-9)
    throttle = min(max(throttle, throttle_min), throttle_max)
    return throttle, axis, throttle * max_thrust


class WaypointGuidance:
    """PID guidance to a position/velocity target.

    The integral acts on horizontal position error only (rejects steady wind drag)
    and is clamped to ``integral_limit`` m/s^2 of authority.
    """

    def __init__(
        self,
        bandwidth: float = 0.6,
        damping: float = 1.0,
        accel_limit: float = 6.0,
        integral_gain: float = 0.0,
        integral_limit: float = 1.5,
    ):
        self.wn = bandwidth
        self.zeta = damping
        self.accel_limit = accel_limit
        self.ki = integral_gain
        self.i_limit = integral_limit
        self.integral = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)

    def __call__(
        self, pos, vel, target_pos, target_vel=None, dt: float = 0.0, up=None
    ) -> np.ndarray:
        tv = np.zeros(3) if target_vel is None else np.asarray(target_vel, dtype=float)
        err = np.asarray(target_pos) - pos
        a = self.wn**2 * err + 2 * self.zeta * self.wn * (tv - vel)
        if self.ki > 0 and dt > 0:
            u = np.array([0.0, 0.0, 1.0]) if up is None else up
            horiz = err - (err @ u) * u
            self.integral = self.integral + self.ki * horiz * dt
            n = float(np.linalg.norm(self.integral))
            if n > self.i_limit:
                self.integral *= self.i_limit / n
            a = a + self.integral
        n = float(np.linalg.norm(a))
        if n > self.accel_limit:
            a *= self.accel_limit / n
        return a


def zem_zev_accel(
    r: np.ndarray, v: np.ndarray, r_f: np.ndarray, v_f: np.ndarray, g: np.ndarray, t_go: float
) -> np.ndarray:
    """Optimal (energy) feedback acceleration command for a fixed time-to-go.

    a = 6 ZEM / t^2 - 2 ZEV / t, with ZEM = r_f - (r + v t + g t^2/2), ZEV = v_f - (v + g t).
    The returned vector is the commanded *thrust* (non-gravitational) acceleration;
    the total acceleration is this plus ``g``.
    """
    t = max(t_go, 1e-3)
    zem = r_f - (r + v * t + 0.5 * g * t * t)
    zev = v_f - (v + g * t)
    return 6.0 * zem / (t * t) - 2.0 * zev / t


def time_to_go(
    r: np.ndarray, v: np.ndarray, r_f: np.ndarray, v_f: np.ndarray, g: np.ndarray, a_max: float
) -> float:
    """Smallest time-to-go whose initial ZEM/ZEV thrust acceleration stays within ``a_max``.

    The command magnitude falls as time-to-go grows (over the useful range), so a
    bisection on |a(t_go)| = a_max is robust and cheap enough at control rates.
    """
    lo, hi = 0.2, 400.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(zem_zev_accel(r, v, r_f, v_f, g, mid)) > a_max:
            lo = mid
        else:
            hi = mid
    return hi
