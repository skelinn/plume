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
        max_rate_deg_s: float = 20.0,
        deadband_deg: float = 0.5,
        aero_feedforward: bool = True,
    ):
        self.sim = sim
        self.max_rate = np.radians(max_rate_deg_s)
        self.deadband = np.radians(deadband_deg)
        self.aero_feedforward = aero_feedforward
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

    def desired_torque(
        self, state: State, desired_axis_world: np.ndarray, max_rate: float | None = None
    ) -> np.ndarray:
        """Cascaded law: pointing error -> (rate-limited) desired body rate -> torque.

        In the linear region this is a PD with natural frequency ``bandwidth`` and
        damping ``damping``; for large errors the slew rate is capped at ``max_rate``
        so big manoeuvres (e.g. a flip) do not saturate the actuators for long.
        """
        e = self.pointing_error(state, desired_axis_world)
        I = self.sim.mp.inertia
        w = state.omega
        k_w = 2 * self.zeta * self.wn
        k_e = self.wn / (2 * self.zeta)
        w_des = k_e * np.array([e[0], e[1], 0.0])
        rate_cap = self.max_rate if max_rate is None else max_rate
        n = float(np.linalg.norm(w_des))
        if n > rate_cap:
            w_des *= rate_cap / n
        alpha = np.array([k_w * (w_des[0] - w[0]), k_w * (w_des[1] - w[1]), -self.roll_kd * w[2]])
        return I * alpha - self.static_aero_torque(state)

    def static_aero_torque(self, state: State) -> np.ndarray:
        """Aerodynamic moment at the current attitude with zero body rate (pitch/yaw only).

        Fed forward so the PD loop does not settle with a steady-state error against the
        aerodynamic restoring moment (rate damping is left in place: it helps)."""
        if not self.aero_feedforward:
            return np.zeros(3)
        sim = self.sim
        atm = sim.atmosphere.at(state.altitude)
        if atm.density <= 0:
            return np.zeros(3)
        v_b = state.rot.T @ (state.vel_com - state.wind)
        _, tau, _, _ = sim.aero.forces(
            v_b, np.zeros(3), state.cg_z, atm.density, atm.speed_of_sound
        )
        fins = sim.grid_fins
        if fins is not None and fins.deployed:
            _, tau_f = fins.forces(v_b, np.zeros(3), state.cg_z, atm.density, np.zeros(fins.n))
            tau = tau + tau_f
        return np.array([tau[0], tau[1], 0.0])

    def allocate(
        self, state: State, torque: np.ndarray, thrust: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split a body torque into (normalised gimbal command, normalised RCS command).

        Order of preference: engine gimbal, then grid fins (if fitted and deployed;
        commanded directly on the simulator), then RCS for whatever is left.
        """
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
        fins = self.sim.grid_fins
        rcs_gain = 1.0
        if fins is not None:
            if fins.deployed and np.any(np.abs(remaining) > 1e-9):
                atm = self.sim.atmosphere.at(state.altitude)
                v_b = state.rot.T @ (state.vel_com - state.wind)
                cmd, achieved = fins.allocate(remaining, v_b, state.cg_z, atm.density)
                fins.command(cmd)
                remaining = remaining - achieved
                # where the fins have real authority the RCS only assists (saves gas)
                fin_cap = fins.capability(v_b, state.cg_z, atm.density)
                if np.all(fin_cap[:2] > 0.5 * self.sim.rcs.torque_cap[:2]):
                    rcs_gain = 0.3
            else:
                fins.command(np.zeros(fins.n))
        cap = self.sim.rcs.torque_cap
        rcs = rcs_gain * np.divide(remaining, cap, out=np.zeros(3), where=cap > 0)
        return np.clip(gimbal, -1, 1), np.clip(rcs, -1, 1)

    def __call__(
        self,
        state: State,
        desired_axis_world: np.ndarray,
        thrust: float,
        max_rate: float | None = None,
    ):
        if thrust <= 1.0 and self.deadband > 0:
            # engine off: coast inside a small deadband instead of chattering the RCS
            e = self.pointing_error(state, desired_axis_world)
            if (
                np.linalg.norm(e[:2]) < self.deadband
                and np.linalg.norm(state.omega) < 0.2 * self.deadband
            ):
                if self.sim.grid_fins is not None:
                    self.sim.grid_fins.command(np.zeros(self.sim.grid_fins.n))
                return np.zeros(2), np.zeros(3)
        tau = self.desired_torque(state, desired_axis_world, max_rate)
        return self.allocate(state, tau, thrust)
