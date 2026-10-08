"""Fast 3-DOF point-mass trajectory integrator.

Uses the same atmosphere, gravity, aerodynamic (axial) and engine models as the
6-DOF simulator, with the vehicle assumed to fly at zero angle of attack. It is used
where many trajectories are needed quickly: impact-point prediction for hop
guidance and least-squares fitting of drag/thrust to real flight logs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from plume.config import VehicleSpec, WorldSpec
from plume.constants import G0
from plume.physics.aero import Aero
from plume.physics.atmosphere import atmosphere_from_world
from plume.physics.gravity import gravity_from_world
from plume.physics.propulsion import Engine
from plume.physics.recovery import Recovery


@dataclass
class PMTrajectory:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    thrust: np.ndarray
    accel: np.ndarray  # sensed (non-gravitational) acceleration, m/s^2, world frame
    altitude: np.ndarray

    @property
    def apogee(self) -> float:
        return float(self.altitude.max())

    def at(self, t: float) -> dict[str, np.ndarray]:
        return {
            "pos": np.array([np.interp(t, self.t, self.pos[:, k]) for k in range(3)]),
            "vel": np.array([np.interp(t, self.t, self.vel[:, k]) for k in range(3)]),
        }


class PointMassSim:
    """3-DOF integrator.

    Parameters
    ----------
    thrust_curve:
        Optional override ``[[t, F], ...]`` (solid motors / calibration).
    cd_scale:
        Optional override of the aero coefficient multiplier.
    """

    def __init__(
        self,
        vehicle: VehicleSpec,
        world: WorldSpec | None = None,
        thrust_curve: np.ndarray | None = None,
        cd_scale: float | None = None,
        prop_mass: float | None = None,
        cargo_mass: float | None = None,
        wind_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ):
        self.vehicle = vehicle
        self.wind_fn = wind_fn  # (position, altitude) -> air velocity, frame coordinates
        self.world = world or WorldSpec()
        self.gravity = gravity_from_world(self.world)
        self._rotating = bool(getattr(self.gravity, "rotating", False))
        self.atmosphere = atmosphere_from_world(self.world)
        aero_spec = vehicle.aero.model_copy()
        if cd_scale is not None:
            aero_spec.cd_scale = cd_scale
        self.aero = Aero(aero_spec, vehicle.geometry)
        self.prop0 = vehicle.prop_initial if prop_mass is None else prop_mass
        self.engine = Engine(vehicle.engine, vehicle.prop_capacity)
        if thrust_curve is not None:
            self.engine.set_thrust_curve(np.asarray(thrust_curve, dtype=float), self.prop0)
        self.recovery = Recovery(vehicle.recovery)
        self.chute_scale = 1.0  # calibration multiplier on parachute Cd*A
        self.extra_cda = 0.0  # additional drag area, m^2 (e.g. deployed grid fins)
        cargo = vehicle.cargo.mass if cargo_mass is None else cargo_mass
        self.m_dry = vehicle.mass.dry + cargo + vehicle.rcs.gas

    # -------------------------------------------------------------- engine (no lag)
    def _thrust_mdot(self, t: float, throttle: float, p_amb: float, prop: float):
        e = self.engine
        if prop <= 0:
            return 0.0, 0.0
        if e.is_solid:
            tb = t - self.vehicle.engine.ignition_time
            if tb < 0 or tb > e.burn_time:
                return 0.0, 0.0
            f = float(np.interp(tb, e.curve[:, 0], e.curve[:, 1]))
            return f, f / (e.isp_eff * G0)
        if throttle <= 0:
            return 0.0, 0.0
        s = self.vehicle.engine
        thr = min(max(throttle, s.throttle_min), s.throttle_max)
        mdot = thr * e.mdot_max
        return max(mdot * s.isp_vac * G0 - p_amb * e.exit_area, 0.0), mdot

    def _derivs(self, t, r, v, prop, throttle, direction, tail_first, cda=0.0):
        alt = self.gravity.altitude(r)
        atm = self.atmosphere.at(alt)
        m = self.m_dry + max(prop, 0.0)
        thrust, mdot = self._thrust_mdot(t, throttle, atm.pressure, prop)
        va = v - self.wind_fn(r, alt) if self.wind_fn is not None else v
        speed = math.sqrt(va[0] * va[0] + va[1] * va[1] + va[2] * va[2])
        a_ng = np.zeros(3)
        if thrust > 0:
            a_ng += thrust / m * direction
        if self.aero.enabled and atm.density > 0 and speed > 1e-9:
            mach = speed / atm.speed_of_sound
            ca = self.aero.axial_coefficient(mach, not tail_first)
            drag = 0.5 * atm.density * speed * speed * ca * self.aero.ref_area
            a_ng -= drag / m * (va / speed)
        cda = cda + self.extra_cda
        if cda > 0 and atm.density > 0 and speed > 1e-9:
            a_ng -= 0.5 * atm.density * cda * speed / m * va
        a = a_ng + self.gravity.accel(r)
        if self._rotating:  # Earth-fixed world frame: Coriolis + centrifugal
            a = a + self.gravity.fictitious_accel(r, v)
        return v, a, -mdot, thrust, a_ng, alt

    def run(
        self,
        r0,
        v0,
        t_end: float,
        dt: float = 0.01,
        throttle: Callable[[float, np.ndarray, np.ndarray], float] | float = 1.0,
        direction: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
        rail_length: float = 0.0,
        rail_dir=None,
        tail_first: bool = False,
        stop_on_ground: bool = True,
        ground_altitude: float = 0.0,
        t0: float = 0.0,
        record_every: int = 1,
        dt_fn: Callable[[float, np.ndarray, np.ndarray], float] | None = None,
        stop_fn: Callable[[float, np.ndarray, np.ndarray, float], bool] | None = None,
        prop0: float | None = None,
    ) -> PMTrajectory:
        """Integrate with fixed-step RK4 until ``t_end`` or ground impact.

        ``direction`` gives the unit thrust direction; by default thrust follows the
        velocity (gravity turn), or ``rail_dir``/local up while slow or on the rail.
        """
        r = np.asarray(r0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        prop = float(self.prop0 if prop0 is None else prop0)
        up0 = self.gravity.up(r)
        rail = np.asarray(rail_dir, dtype=float) if rail_dir is not None else up0
        rail = rail / np.linalg.norm(rail)
        r_start = r.copy()
        thr_fn = throttle if callable(throttle) else (lambda t, r, v, _c=float(throttle): _c)

        def dir_fn(t, rr, vv):
            if direction is not None:
                return direction(t, rr, vv)
            if float(np.dot(rr - r_start, rail)) < rail_length:
                return rail
            sp = float(np.linalg.norm(vv))
            return vv / sp if sp > 1.0 else rail

        on_rail = rail_length > 0
        rec = self.recovery
        rec.reset()
        alt0 = self.gravity.altitude(r)

        ts, ps, vs, ms, fs, acc, alts = [], [], [], [], [], [], []
        t = t0
        n = 0
        left_ground = False
        while t < t_end - 1e-12:
            h = min(dt_fn(t, r, v) if dt_fn else dt, t_end - t)
            thr = thr_fn(t, r, v)
            d = dir_fn(t, r, v)
            cda = 0.0
            if rec.enabled:
                rec.update(t, self.gravity.altitude(r) - alt0, float(v @ self.gravity.up(r)))
                cda = rec.cd_area(t) * self.chute_scale
            k1 = self._derivs(t, r, v, prop, thr, d, tail_first, cda)
            if n % record_every == 0:
                ts.append(t)
                ps.append(r.copy())
                vs.append(v.copy())
                ms.append(self.m_dry + prop)
                fs.append(k1[3])
                acc.append(k1[4])
                alts.append(k1[5])
            k2 = self._derivs(
                t + h / 2,
                r + h / 2 * k1[0],
                v + h / 2 * k1[1],
                prop + h / 2 * k1[2],
                thr,
                d,
                tail_first,
                cda,
            )
            k3 = self._derivs(
                t + h / 2,
                r + h / 2 * k2[0],
                v + h / 2 * k2[1],
                prop + h / 2 * k2[2],
                thr,
                d,
                tail_first,
                cda,
            )
            k4 = self._derivs(
                t + h, r + h * k3[0], v + h * k3[1], prop + h * k3[2], thr, d, tail_first, cda
            )
            r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v_new = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            prop = max(prop + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), 0.0)
            # launch rail / pad: no motion into the ground before liftoff
            if on_rail and not left_ground and float(np.dot(r - r_start, rail)) < 0:
                r = r_start.copy()
                v_new = np.zeros(3)
            if on_rail and float(np.dot(r - r_start, rail)) > 1e-6:
                left_ground = True
            if not on_rail:
                left_ground = True
            v = v_new
            t += h
            n += 1
            if stop_fn is not None and stop_fn(t, r, v, prop):
                break
            if (
                stop_on_ground
                and left_ground
                and self.gravity.altitude(r) < ground_altitude
                and float(np.dot(v, self.gravity.up(r))) < 0
            ):
                break
        k = self._derivs(t, r, v, prop, thr_fn(t, r, v), dir_fn(t, r, v), tail_first)
        ts.append(t)
        ps.append(r.copy())
        vs.append(v.copy())
        ms.append(self.m_dry + prop)
        fs.append(k[3])
        acc.append(k[4])
        alts.append(k[5])
        return PMTrajectory(
            np.array(ts),
            np.array(ps),
            np.array(vs),
            np.array(ms),
            np.array(fs),
            np.array(acc),
            np.array(alts),
        )
