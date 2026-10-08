"""Impact-point prediction for point-to-point hops.

* :func:`kepler_impact` -- analytic vacuum two-body impact point (cheap; used for
  ascent steering and pre-flight planning).
* :class:`ImpactPredictor` -- 3-DOF numerical prediction with drag and an optional
  planned entry burn (used for MECO and entry-burn cutoff decisions).
"""

from __future__ import annotations

import math

import numpy as np

from plume.config import HopGuidanceSpec, VehicleSpec, WorldSpec
from plume.constants import G0
from plume.physics.gravity import SphericalGravity
from plume.physics.pointmass import PointMassSim


def kepler_impact(r: np.ndarray, v: np.ndarray, gravity: SphericalGravity, r_target: float):
    """Impact point (frame coordinates) of a vacuum ballistic arc on a sphere of radius
    ``r_target``, or ``None`` if the arc never reaches it.

    For a rotating (Earth-fixed) frame the arc is solved in inertial space and the impact
    point is rotated back by the Earth's rotation over the time of flight.
    """
    rotating = bool(getattr(gravity, "rotating", False))
    rr = r - gravity.center
    if rotating:
        v = v + np.cross(gravity.omega_w, rr)
    rn = float(np.linalg.norm(rr))
    h = np.cross(rr, v)
    hn = float(np.linalg.norm(h))
    if hn < 1e-9:
        return None
    mu = gravity.mu
    e_vec = np.cross(v, h) / mu - rr / rn
    e = float(np.linalg.norm(e_vec))
    p = hn * hn / mu
    if e < 1e-9:
        return None
    cos_nu0 = np.clip((p / rn - 1.0) / e, -1.0, 1.0)
    nu0 = math.acos(cos_nu0)
    if float(rr @ v) < 0:
        nu0 = 2 * math.pi - nu0
    c = (p / r_target - 1.0) / e
    if abs(c) > 1.0:
        return None
    nu_i = 2 * math.pi - math.acos(c)  # descending crossing
    dnu = (nu_i - nu0) % (2 * math.pi)
    rhat = rr / rn
    t_hat = np.cross(h / hn, rhat)
    direction = math.cos(dnu) * rhat + math.sin(dnu) * t_hat
    if rotating and e < 1.0:
        # time of flight from the eccentric anomalies, then undo the Earth's rotation
        a_sm = p / (1.0 - e * e)

        def mean_anomaly(nu):
            ecc = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu / 2.0))
            return ecc - e * math.sin(ecc)

        n = math.sqrt(mu / a_sm**3)
        tof = ((mean_anomaly(nu0 + dnu) - mean_anomaly(nu0)) % (2 * math.pi)) / n
        w = gravity.omega_w
        wn = float(np.linalg.norm(w))
        k = w / wn
        ang = -wn * tof
        direction = (
            direction * math.cos(ang)
            + np.cross(k, direction) * math.sin(ang)
            + k * float(k @ direction) * (1.0 - math.cos(ang))
        )
    return gravity.center + r_target * direction


class ImpactPredictor:
    """Drag-aware 3-DOF impact prediction with the remaining burn plan."""

    def __init__(
        self,
        vehicle: VehicleSpec,
        world: WorldSpec,
        guidance: HopGuidanceSpec,
        ground_altitude: float,
        cargo_mass: float,
    ):
        from plume.physics.wind import wind_from_world

        forecast = wind_from_world(world)
        calm = world.model_copy(update={"wind": world.wind.model_copy(update={"speed": 0.0})})
        self.pm = PointMassSim(vehicle, calm, cargo_mass=cargo_mass)
        grav = self.pm.gravity
        if forecast.speed > 0:  # use the forecast mean wind profile (no gusts)
            self.pm.wind_fn = lambda r, alt: grav.local_to_frame(r, forecast.mean_at(alt))
        gf = vehicle.grid_fins
        if gf is not None:  # deployed for the whole descent the predictor models
            area = gf.area if gf.area is not None else gf.span * gf.chord
            self.pm.extra_cda = gf.count * area * gf.cd0
        self.guidance = guidance
        self.ground_altitude = ground_altitude
        self.gravity = self.pm.gravity
        # the entry burn is flown g-limited (HopAutopilot._g_limited_throttle)
        self.pm.accel_limit = guidance.cargo_g_limit * 0.92 * G0

    def model_drag_accel(self, r, v_air, mass: float, tail_first: bool = True) -> float:
        """Drag deceleration (m/s^2) the predictor's model gives, without its drag scale."""
        pm = self.pm
        atm = pm.atmosphere.at(self.gravity.altitude(r))
        speed = float(np.linalg.norm(v_air))
        if atm.density <= 0 or speed < 1e-6:
            return 0.0
        ca = pm.aero.axial_coefficient(speed / atm.speed_of_sound, not tail_first)
        return 0.5 * atm.density * speed * speed * (ca * pm.aero.ref_area + pm.extra_cda) / mass

    def predict(self, r, v, prop: float, entry_burn: bool, t0: float = 0.0):
        """Predicted impact position (frame), or None if it does not come down."""
        g = self.guidance
        gravity = self.gravity
        burning = {"on": False, "done": not entry_burn}

        def throttle(t, rr, vv):
            if burning["done"]:
                return 0.0
            alt = gravity.altitude(rr)
            descending = float(vv @ gravity.up(rr)) < 0
            speed = float(np.linalg.norm(vv))
            if (
                not burning["on"]
                and descending
                and alt < g.entry_altitude
                and speed > g.entry_speed
            ):
                burning["on"] = True
            if burning["on"] and speed <= g.entry_speed:
                burning["on"] = False
                burning["done"] = True
            return 1.0 if burning["on"] else 0.0

        def direction(t, rr, vv):
            sp = float(np.linalg.norm(vv))
            return -vv / sp if sp > 1e-6 else gravity.up(rr)

        def dt_fn(t, rr, vv):
            return 0.5 if gravity.altitude(rr) > 40_000 and not burning["on"] else 0.1

        traj = self.pm.run(
            r,
            v,
            t_end=t0 + 3000.0,
            t0=t0,
            throttle=throttle,
            direction=direction,
            tail_first=True,
            stop_on_ground=True,
            ground_altitude=self.ground_altitude,
            dt_fn=dt_fn,
            prop0=prop,
            record_every=10_000,
        )
        r_end = traj.pos[-1]
        if gravity.altitude(r_end) > self.ground_altitude + 1000:
            return None
        return r_end
