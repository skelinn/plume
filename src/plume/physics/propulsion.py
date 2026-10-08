"""Main engine (liquid throttleable or solid thrust-curve), gimbal and RCS models."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, nnls

from plume.config import EngineSpec, RCSSpec, VehicleSpec
from plume.constants import G0, P0


# ----------------------------------------------------------------------------- engine
def read_eng(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a RASP ``.eng`` motor file. Returns (curve [[t, F]...], header info)."""
    lines = [
        ln.strip()
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith(";")
    ]
    head = lines[0].split()
    info = {
        "name": head[0],
        "diameter_mm": float(head[1]),
        "length_mm": float(head[2]),
        "propellant_kg": float(head[4]),
        "total_mass_kg": float(head[5]),
        "manufacturer": head[6] if len(head) > 6 else "",
    }
    pts = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    curve = np.array(pts, dtype=float)
    if curve[0, 0] > 0:
        curve = np.vstack([[0.0, 0.0], curve])
    return curve, info


class Engine:
    """Main engine with throttle dynamics, ignition limits and a 2-axis gimbal.

    The gimbal angles ``(a, b)`` rotate the thrust direction about body x then y:
    ``d = (cos a sin b, -sin a, cos a cos b)``.
    """

    def __init__(self, spec: EngineSpec, prop_capacity: float, high_fidelity: bool = False):
        self.spec = spec
        self.high_fidelity = high_fidelity
        self._misalign = np.radians(np.asarray(spec.misalignment_deg, dtype=float))
        self.gimbal_max = math.radians(spec.gimbal_max_deg)
        self.gimbal_rate = math.radians(spec.gimbal_rate_deg_s)
        if spec.type == "liquid":
            self.mdot_max = spec.thrust_vac / (spec.isp_vac * G0)
            self.exit_area = self.mdot_max * G0 * (spec.isp_vac - spec.isp_sea_level) / P0
            self.curve = None
        else:
            if spec.thrust_curve:
                curve = np.asarray(spec.thrust_curve, dtype=float)
            else:
                curve, _ = read_eng(spec.motor_file)
            self.set_thrust_curve(curve, prop_capacity)
        self.reset()

    def set_thrust_curve(self, curve: np.ndarray, prop_mass: float) -> None:
        curve = np.asarray(curve, dtype=float)
        if curve[0, 0] > 0:
            curve = np.vstack([[0.0, 0.0], curve])
        self.curve = curve
        self.total_impulse = float(np.trapezoid(curve[:, 1], curve[:, 0]))
        self.burn_time = float(curve[-1, 0])
        self.isp_eff = self.total_impulse / (max(prop_mass, 1e-9) * G0)
        self.mdot_max = float(curve[:, 1].max()) / (self.isp_eff * G0)
        self.exit_area = 0.0

    def reset(self) -> None:
        self.throttle = 0.0  # actual (lagged) throttle
        self.throttle_cmd = 0.0
        self.on = False
        self.ignitions = 0
        self.gimbal = np.zeros(2)
        self.gimbal_cmd = np.zeros(2)
        self.t = 0.0
        self.burn_clock = 0.0
        # high-fidelity actuator state: piston position/rate, delayed commands, backlash
        self._act_pos = np.zeros(2)
        self._act_rate = np.zeros(2)
        self._cmd_hist: deque[tuple[float, np.ndarray]] = deque()
        self._act_cmd = np.zeros(2)
        self._ign_at: float | None = None

    @property
    def is_solid(self) -> bool:
        return self.curve is not None

    def command(self, throttle: float, gimbal: np.ndarray | tuple[float, float] = (0.0, 0.0)):
        """``throttle`` in [0, throttle_max] (values below half of throttle_min mean "off");
        ``gimbal`` normalised to [-1, 1] per axis."""
        self.throttle_cmd = float(throttle)
        g = np.clip(np.asarray(gimbal, dtype=float), -1.0, 1.0)
        self.gimbal_cmd = g * self.gimbal_max

    def can_ignite(self) -> bool:
        mi = self.spec.max_ignitions
        return mi is None or self.ignitions < mi

    def _update_gimbal(self, dt: float) -> None:
        if self.gimbal_max <= 0:
            return
        if self.high_fidelity:
            self._update_gimbal_actuator(dt)
            return
        tau = max(self.spec.gimbal_tau, dt)
        rate = np.clip((self.gimbal_cmd - self.gimbal) / tau, -self.gimbal_rate, self.gimbal_rate)
        g = self.gimbal + rate * dt
        n = math.hypot(g[0], g[1])
        if n > self.gimbal_max:
            g *= self.gimbal_max / n
        self.gimbal = g

    def direction(self) -> np.ndarray:
        a, b = self.gimbal + self._misalign
        ca = math.cos(a)
        return np.array([ca * math.sin(b), -math.sin(a), ca * math.cos(b)])

    def _update_gimbal_actuator(self, dt: float) -> None:
        """Second-order actuator (natural frequency, damping, rate and acceleration
        limits) driven by the command after a transport delay; the nozzle follows the
        piston through a backlash (free-play) band."""
        s = self.spec
        self._cmd_hist.append((self.t, self.gimbal_cmd.copy()))
        while self._cmd_hist and self._cmd_hist[0][0] <= self.t - s.gimbal_delay_s + 1e-12:
            self._act_cmd = self._cmd_hist.popleft()[1]  # the newest command old enough
        cmd = self._act_cmd
        wn = 2.0 * math.pi * s.gimbal_wn_hz
        acc = wn * wn * (cmd - self._act_pos) - 2.0 * s.gimbal_zeta * wn * self._act_rate
        a_max = math.radians(s.gimbal_accel_deg_s2)
        acc = np.clip(acc, -a_max, a_max)
        n_sub = max(1, math.ceil(dt * wn / 0.2))  # keep the explicit update stable
        h = dt / n_sub
        for _ in range(n_sub):
            self._act_rate = np.clip(self._act_rate + acc * h, -self.gimbal_rate, self.gimbal_rate)
            self._act_pos = self._act_pos + self._act_rate * h
            acc = np.clip(
                wn * wn * (cmd - self._act_pos) - 2.0 * s.gimbal_zeta * wn * self._act_rate,
                -a_max,
                a_max,
            )
        n = math.hypot(*self._act_pos)
        if n > self.gimbal_max:
            self._act_pos *= self.gimbal_max / n
            self._act_rate[:] = 0.0
        half = 0.5 * math.radians(s.gimbal_backlash_deg)
        diff = self._act_pos - self.gimbal
        self.gimbal = self.gimbal + np.sign(diff) * np.maximum(np.abs(diff) - half, 0.0)

    def update(self, dt: float, ambient_pressure: float, prop_available: float):
        """Advance engine state by dt. Returns (thrust N, mdot kg/s) for this step."""
        self._update_gimbal(dt)
        self.t += dt
        if self.is_solid:
            return self._update_solid(dt, prop_available)

        s = self.spec
        want_on = self.throttle_cmd >= 0.5 * s.throttle_min and self.throttle_cmd > 1e-6
        if want_on and not self.on and prop_available > 0 and self.can_ignite():
            if self.high_fidelity and s.ignition_delay_s > 0:
                # igniter, valve sequencing and chamber fill before thrust appears
                if self._ign_at is None:
                    self._ign_at = self.t + s.ignition_delay_s
                if self.t >= self._ign_at:
                    self.on = True
                    self.ignitions += 1
                    self._ign_at = None
            else:
                self.on = True
                self.ignitions += 1
        if not want_on:
            self._ign_at = None
        if not want_on or prop_available <= 0:
            self.on = False
        target = min(max(self.throttle_cmd, s.throttle_min), s.throttle_max) if self.on else 0.0
        if s.throttle_tau > 0:
            a = math.exp(-dt / s.throttle_tau)
            self.throttle = target + (self.throttle - target) * a
            if not self.on and self.throttle < 0.02:
                self.throttle = 0.0
        else:
            self.throttle = target
        if prop_available <= 0:
            self.throttle = 0.0
        if self.throttle <= 0:
            return 0.0, 0.0
        mdot = self.throttle * self.mdot_max
        thrust = max(mdot * s.isp_vac * G0 - ambient_pressure * self.exit_area, 0.0)
        if mdot * dt > prop_available:  # last drops
            scale = prop_available / (mdot * dt)
            mdot *= scale
            thrust *= scale
        return thrust, mdot

    def _update_solid(self, dt: float, prop_available: float):
        if not self.on and self.t >= self.spec.ignition_time and self.ignitions == 0:
            self.on = True
            self.ignitions = 1
        if not self.on:
            self.throttle = 0.0
            return 0.0, 0.0
        tb = self.burn_clock + 0.5 * dt  # midpoint sample
        self.burn_clock += dt
        if tb >= self.burn_time or prop_available <= 0:
            self.on = False
            self.throttle = 0.0
            return 0.0, 0.0
        thrust = float(np.interp(tb, self.curve[:, 0], self.curve[:, 1]))
        mdot = thrust / (self.isp_eff * G0)
        if mdot * dt > prop_available:
            scale = prop_available / (mdot * dt)
            thrust *= scale
            mdot *= scale
        self.throttle = thrust / max(float(self.curve[:, 1].max()), 1e-9)
        return thrust, mdot

    def isp_at(self, ambient_pressure: float) -> float:
        """Effective Isp at full throttle and the given ambient pressure."""
        if self.is_solid:
            return self.isp_eff
        s = self.spec
        return max(s.isp_vac - ambient_pressure * self.exit_area / (self.mdot_max * G0), 0.0)

    def max_thrust(self, ambient_pressure: float = 0.0) -> float:
        if self.is_solid:
            return float(self.curve[:, 1].max())
        return max(
            self.spec.throttle_max * self.mdot_max * self.spec.isp_vac * G0
            - ambient_pressure * self.exit_area,
            0.0,
        )


# ----------------------------------------------------------------------------- RCS
_Z3 = np.zeros(3)


def ring_layout(spec: RCSSpec, hull_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Pods on a ring at height ``spec.z``, each with a +/- tangential thruster pair."""
    r = spec.radius if spec.radius is not None else hull_radius
    pos, dirs = [], []
    for k in range(spec.pods):
        th = 2 * math.pi * k / spec.pods
        p = [r * math.cos(th), r * math.sin(th), spec.z]
        t = [-math.sin(th), math.cos(th), 0.0]
        pos += [p, p]
        dirs += [t, [-c for c in t]]
    return np.array(pos), np.array(dirs)


class RCS:
    """Reaction control system with continuous (PWM-averaged) duty-cycle allocation."""

    def __init__(
        self, spec: RCSSpec, vehicle: VehicleSpec, nominal_cg_z: float, high_fidelity: bool = False
    ):
        self.spec = spec
        self.high_fidelity = high_fidelity
        if spec.thrusters:
            self.pos = np.array([t.pos for t in spec.thrusters], dtype=float)
            d = np.array([t.dir for t in spec.thrusters], dtype=float)
            self.dir = d / np.linalg.norm(d, axis=1, keepdims=True)
        else:
            self.pos, self.dir = ring_layout(spec, vehicle.geometry.radius)
        self.n = len(self.pos)
        self.enabled = spec.enabled and spec.thrust > 0 and self.n > 0
        self._f_unit = self.dir * spec.thrust
        self._m_unit = np.cross(self.pos, self._f_unit)
        self.torque_cap = self._torque_capability(nominal_cg_z) if self.enabled else np.zeros(3)
        self.reset()

    def reset(self) -> None:
        self.duty = np.zeros(self.n)
        self.cmd = np.zeros(3)
        self._active = False
        self._t = 0.0
        self._frame_start = 0.0

    def torque_matrix(self, cg_z: float) -> np.ndarray:
        """3 x n matrix: body torque about the CG per unit duty."""
        r = self.pos - np.array([0.0, 0.0, cg_z])
        return (np.cross(r, self.dir) * self.spec.thrust).T

    def _torque_capability(self, cg_z: float) -> np.ndarray:
        B = self.torque_matrix(cg_z)
        cap = np.zeros(3)
        for axis in range(3):
            # maximise torque about `axis` with zero torque about the others
            others = [i for i in range(3) if i != axis]
            res = linprog(
                -B[axis],
                A_eq=B[others],
                b_eq=np.zeros(2),
                bounds=[(0.0, 1.0)] * self.n,
                method="highs",
            )
            cap[axis] = -res.fun if res.success else 0.0
        return cap

    def command(self, torque_norm: np.ndarray | tuple[float, float, float], cg_z: float) -> None:
        """Request a body torque as a fraction of per-axis capability ([-1, 1]^3)."""
        self.cmd = np.clip(np.asarray(torque_norm, dtype=float), -1.0, 1.0)
        if not self.enabled or not np.any(np.abs(self.cmd) > 1e-4):
            self.duty = np.zeros(self.n)
            self._active = False
            return
        target = self.cmd * self.torque_cap
        duty, _ = nnls(self.torque_matrix(cg_z), target)
        peak = duty.max()
        if peak > 1.0:
            duty /= peak
        if self.high_fidelity:
            s = self.spec
            # pulse-width modulation: on-times shorter than the minimum valve opening
            # are dropped (minimum impulse bit); longer ones are kept
            duty = np.where(duty * s.pwm_period_s < s.min_on_time_s, 0.0, duty)
            self._frame_start = self._t
        self.duty = duty
        self._active = bool(duty.any())

    def mdot(self, prop_available: float, dt: float) -> float:
        if not self.enabled or prop_available <= 0 or not self._active:
            return 0.0
        m = float(self.duty.sum()) * self.spec.thrust / (self.spec.isp * G0)
        return min(m, prop_available / dt)

    def update(self, cg_z: float, prop_available: float, dt: float):
        """Returns (force_body, torque_body about CG, mdot)."""
        t0 = self._t
        self._t += dt
        if not self.enabled or prop_available <= 0 or not self._active:
            return _Z3, _Z3, 0.0
        duty = self.duty
        if self.high_fidelity:
            # each valve is open from (frame start + latency) for duty x period: the
            # fraction of this physics step it is open
            s = self.spec
            ph0 = (t0 - self._frame_start) % s.pwm_period_s - s.valve_delay_s
            on_end = duty * s.pwm_period_s
            duty = np.clip(np.minimum(on_end, ph0 + dt) - np.maximum(0.0, ph0), 0.0, dt) / dt
        force = duty @ self._f_unit
        # torque about CG = sum d_i (pos_i x F_i) - (0, 0, cg_z) x force
        m0 = duty @ self._m_unit
        torque = np.array([m0[0] + cg_z * force[1], m0[1] - cg_z * force[0], m0[2]])
        mdot = float(duty.sum()) * self.spec.thrust / (self.spec.isp * G0)
        if mdot * dt > prop_available:
            scale = prop_available / (mdot * dt)
            force, torque, mdot = force * scale, torque * scale, mdot * scale
        return force, torque, mdot
