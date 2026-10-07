"""Attitude control: point the vehicle axis along a desired direction.

A PD law on the axis-pointing error produces a desired body torque, which is
allocated first to the engine gimbal (when the engine is producing thrust) and
then to the RCS for whatever the gimbal cannot provide (and always for roll).
"""

from __future__ import annotations

import math

import numpy as np

from plume.physics.sim import RocketSim, State


class AttitudeController:
    def __init__(
        self,
        sim: RocketSim,
        bandwidth: float = 2.5,
        damping: float = 0.9,
        roll_damping: float = 2.0,
    ):
        self.sim = sim
        self.wn = bandwidth
        self.zeta = damping
        self.roll_kd = roll_damping
        self.gimbal_max = sim.engine.gimbal_max
        self.gimbal_z = sim.vehicle.engine.gimbal_z

    @staticmethod
    def pointing_error(state: State, desired_axis_world: np.ndarray) -> np.ndarray:
        """Rotation vector (body frame) that would rotate the body axis onto the target."""
        R = state.rot
        z = R[:, 2]
        d = desired_axis_world / np.linalg.norm(desired_axis_world)
        c = float(np.clip(z @ d, -1.0, 1.0))
        axis = np.cross(z, d)
        s = float(np.linalg.norm(axis))
        if s < 1e-12:
            if c > 0:
                return np.zeros(3)
            axis, s = R[:, 0], 1.0  # 180 deg: pick any perpendicular axis
        angle = math.atan2(s, c)
        return R.T @ (axis / s * angle)

    def desired_torque(self, state: State, desired_axis_world: np.ndarray) -> np.ndarray:
        e = self.pointing_error(state, desired_axis_world)
        I = self.sim.mp.inertia
        w = state.omega
        kp = self.wn**2
        kd = 2 * self.zeta * self.wn
        alpha = np.array([kp * e[0] - kd * w[0], kp * e[1] - kd * w[1], -self.roll_kd * w[2]])
        return I * alpha

    def allocate(
        self, state: State, torque: np.ndarray, thrust: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split a body torque into (normalised gimbal command, normalised RCS command)."""
        gimbal = np.zeros(2)
        remaining = torque.copy()
        rz = self.gimbal_z - state.cg_z
        if thrust > 1.0 and self.gimbal_max > 0 and abs(rz) > 1e-6:
            # small-angle: tau_x = rz * a * T,  tau_y = rz * b * T
            a = torque[0] / (rz * thrust)
            b = torque[1] / (rz * thrust)
            ab = np.array([a, b])
            n = float(np.linalg.norm(ab))
            if n > self.gimbal_max:
                ab *= self.gimbal_max / n
            gimbal = ab / self.gimbal_max
            remaining[0] -= rz * ab[0] * thrust
            remaining[1] -= rz * ab[1] * thrust
        cap = self.sim.rcs.torque_cap
        rcs = np.divide(remaining, cap, out=np.zeros(3), where=cap > 0)
        return np.clip(gimbal, -1, 1), np.clip(rcs, -1, 1)

    def __call__(self, state: State, desired_axis_world: np.ndarray, thrust: float):
        tau = self.desired_torque(state, desired_axis_world)
        return self.allocate(state, tau, thrust)
