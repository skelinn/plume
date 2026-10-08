"""Point-to-point cargo hops over 3-D terrain.

Flight plan (``HopAutopilot``)
------------------------------
``liftoff``      full (cargo-g-limited) thrust straight up off pad A
``pitch_kick``   rotate the thrust axis toward B by the kick angle
``gravity_turn`` thrust along the air-relative velocity; small yaw corrections null
                 the predicted crossrange; MECO when the drag-aware impact prediction
                 (with the planned entry burn) reaches B
``coast``        RCS flips the vehicle engine-first (retrograde) above the atmosphere
``entry_burn``   retrograde burn that limits re-entry loads; cut off when the
                 predicted (no further burn) impact point reaches B
``descent``      :class:`LandingAutopilot` takes over: aero steering, hoverslam
                 landing burn, touchdown on unprepared ground
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from plume.config import (
    HopGuidanceSpec,
    MissionSpec,
    data_root,
    load_mission,
    load_vehicle,
)
from plume.constants import G0
from plume.control.attitude import AttitudeController
from plume.control.autopilot import LandingAutopilot
from plume.missions.scoring import MissionResult, score_mission
from plume.missions.targeting import ImpactPredictor, kepler_impact
from plume.physics.gravity import SphericalGravity
from plume.physics.mjcf import GroundTile
from plume.physics.pointmass import PointMassSim
from plume.physics.sim import RocketSim, quat_from_z_axis
from plume.recording import Recorder
from plume.terrain import Heightmap

CONTROL_DT = 0.05


def _terrain_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    return data_root() / "terrain" / f"{name}.yaml"


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(
        math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2]
    )
    y = math.copysign(
        math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0]
    )
    z = math.copysign(
        math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1]
    )
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ----------------------------------------------------------------------------- world
class MissionWorld:
    """Terrain, sites and map/frame conversions for a mission."""

    def __init__(self, spec: MissionSpec, gravity: SphericalGravity):
        self.spec = spec
        self.gravity = gravity
        self.base = Heightmap.load(_terrain_path(spec.terrain.base))
        self.tiles: list[Heightmap] = [
            Heightmap.load(_terrain_path(t))
            for t in (spec.terrain.launch_tile, spec.terrain.landing_tile)
            if t
        ]
        a, b = spec.launch, spec.target
        self.site_a = np.array([a.u, a.v])
        self.site_b = np.array([b.u, b.v])
        self.pad_a = gravity.surface_point(a.u, a.v, self.terrain_height(a.u, a.v))
        self.pad_b = gravity.surface_point(b.u, b.v, self.terrain_height(b.u, b.v))
        # geocentric radius of the landing pad: the impact sphere for ballistic prediction
        # (= R + h on a spherical Earth; varies with latitude on the WGS-84 ellipsoid)
        self.r_target = float(np.linalg.norm(self.pad_b - gravity.center))
        d = self.site_b - self.site_a
        self.range = float(np.linalg.norm(d))
        self.track = d / self.range
        self.cross_dir = np.array([-self.track[1], self.track[0]])

    def terrain_height(self, u: float, v: float) -> float:
        for t in self.tiles:
            if t.contains(u, v):
                return float(t.height(u, v))
        return float(self.base.height(u, v))

    def ground_height(self, p: np.ndarray) -> float:
        u, v, _ = self.gravity.map_coords(p)
        return self.terrain_height(u, v)

    def along_cross(self, p: np.ndarray) -> tuple[float, float]:
        """Along-track and cross-track map distance of a frame point from site A."""
        u, v, _ = self.gravity.map_coords(p)
        rel = np.array([u, v]) - self.site_a
        return float(rel @ self.track), float(rel @ self.cross_dir)

    def miss(self, p: np.ndarray) -> float:
        u, v, _ = self.gravity.map_coords(p)
        return float(np.hypot(u - self.site_b[0], v - self.site_b[1]))

    def local_frame(self, u: float, v: float) -> np.ndarray:
        """Columns: local east, north, up at map coordinates (u, v)."""
        n = self.gravity.surface_normal(u, v)
        e = self.gravity.surface_point(u + 1.0, v, 0.0) - self.gravity.surface_point(
            u - 1.0, v, 0.0
        )
        e -= (e @ n) * n
        e /= np.linalg.norm(e)
        return np.column_stack([e, np.cross(n, e), n])

    def ground_tiles(self) -> list[GroundTile]:
        out = []
        for t in self.tiles:
            uc = 0.5 * (t.x_min + t.x_max)
            vc = 0.5 * (t.y_min + t.y_max)
            R = self.local_frame(uc, vc)
            out.append(
                GroundTile(
                    name=t.name,
                    heights=t.heights,
                    origin=self.gravity.surface_point(uc, vc, 0.0),
                    half_x=0.5 * (t.x_max - t.x_min),
                    half_y=0.5 * (t.y_max - t.y_min),
                    quat=_rot_to_quat(R),
                )
            )
        return out

    def map_grid(self, n: int = 33) -> dict:
        """Residual of the true (WGS-84) map->world mapping against the viewer's reference
        sphere (radius ``earth_radius``, centre straight below the origin), sampled on a
        regular grid over the base terrain. The viewer adds the bilinearly interpolated
        residuals to its spherical mapping, which reproduces the ellipsoid to centimetres."""
        R = self.gravity.earth_radius
        ref = SphericalGravity(radius=R)
        b = self.base
        us = np.linspace(b.x_min, b.x_max, n)
        vs = np.linspace(b.y_min, b.y_max, n)
        dp, dn = [], []
        for v in vs:  # row-major: v outer, u inner
            for u in us:
                dp.extend(
                    np.round(self.gravity.surface_point(u, v, 0.0) - ref.surface_point(u, v), 3)
                )
                dn.extend(np.round(self.gravity.surface_normal(u, v) - ref.surface_normal(u, v), 7))
        return {
            "u0": float(us[0]),
            "v0": float(vs[0]),
            "du": float(us[1] - us[0]),
            "dv": float(vs[1] - vs[0]),
            "nu": n,
            "nv": n,
            "dp": [float(x) for x in dp],  # ENU metres, 3 per node
            "dn": [float(x) for x in dn],  # unit-normal residual, 3 per node
        }

    def scene_meta(self) -> dict:
        t = self.spec.terrain
        geo = {}
        if hasattr(self.gravity, "lat0"):
            geo = {
                "frame": "wgs84",
                "origin": {
                    "lat_deg": math.degrees(self.gravity.lat0),
                    "lon_deg": math.degrees(self.gravity.lon0),
                    "height": self.gravity.h0,
                },
                "map_grid": self.map_grid(),
            }
        return {
            "frame": "spherical",
            **geo,
            "earth_radius": self.gravity.earth_radius,
            "ground": {
                "type": "terrain",
                "terrain_id": t.base,
                "detail_terrain_ids": [x for x in (t.launch_tile, t.landing_tile) if x],
            },
            "pads": [{"name": self.spec.launch.name, "pos": self.pad_a.tolist(), "radius": 12.0}],
            "target": {"pos": self.pad_b.tolist(), "radius": self.spec.target_radius},
        }


# ----------------------------------------------------------------------------- autopilot
class HopAutopilot:
    def __init__(
        self,
        sim: RocketSim,
        mw: MissionWorld,
        guidance: HopGuidanceSpec,
        kick_deg: float,
        rise_time: float,
        ascent_profile: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.sim = sim
        self.rise = rise_time
        # planned flight-path angle vs speed (closed-loop ascent); None = follow velocity
        self.profile = ascent_profile
        self.mw = mw
        self.g = guidance
        self.kick = math.radians(kick_deg)
        self.att = AttitudeController(sim)
        self.landing = LandingAutopilot(
            sim,
            mw.pad_b,
            control_dt=CONTROL_DT,
            max_decel=min(2.0, guidance.cargo_g_limit - 1.5) * G0,
            max_tilt_deg=25.0,
        )
        self.predictor = ImpactPredictor(
            sim.nominal,
            getattr(sim, "nominal_world", sim.world),  # the forecast, not the truth
            guidance,
            ground_altitude=mw.terrain_height(*mw.site_b),
            cargo_mass=sim.cargo_mass,
        )
        self.phase = "liftoff"
        self.events: list[tuple[float, str, str]] = []
        self.up0 = sim.gravity.up(mw.pad_a)
        horiz = mw.local_frame(*mw.site_a) @ np.array([*mw.track, 0.0])
        self.downrange0 = horiz / np.linalg.norm(horiz)
        self._last_pred_t = -1e9
        self._drag_prev = None  # (t, velocity) for in-flight drag estimation
        self.landing.max_divert_m = guidance.max_divert_m
        self.landing.divert_gate = guidance.divert_gate
        self._pred_hist: list[tuple[float, float]] = []  # (t, along-track error)
        self._meco_at: float | None = None
        self.predicted_impact: np.ndarray | None = None
        self._entry_along: float | None = None

    def _set_phase(self, phase: str, label: str, kind: str = "phase") -> None:
        self.phase = phase
        self.events.append((self.sim.t, kind, label))
        if phase in ("pitch_kick", "kick_hold", "gravity_turn") and self.sim.legs_deployed:
            self.sim.set_legs(False)  # legs fold flush against the hull for ascent
            self.events.append((self.sim.t, "phase", "Legs stowed"))
        fins = self.sim.grid_fins
        if phase == "coast" and fins is not None and not fins.deployed:
            fins.deploy(True)  # stowed for ascent, deployed after MECO
            self.events.append((self.sim.t, "phase", "Grid fins deployed"))

    def _track_profile(self, st, vhat: np.ndarray) -> np.ndarray:
        """Closed-loop ascent: pitch the thrust axis (in the vertical plane of the
        velocity) so the flight-path angle tracks the pre-flight plan at this speed, with
        the angle of attack limited (tighter at high dynamic pressure). Open-loop velocity
        following let wind and thrust errors reshape the whole trajectory (apogee 80-220
        km in Monte Carlo)."""
        sp, gam = self.profile
        speed = float(np.linalg.norm(st.vel_com))
        if speed < sp[0] or speed > sp[-1]:
            return vhat
        up = st.up
        s_up = float(vhat @ up)
        g_now = math.asin(max(-1.0, min(1.0, s_up)))
        g_ref = float(np.interp(speed, sp, gam))
        a_max = math.radians(2.0 if st.q_dyn > 20_000.0 else 5.0)
        d = float(np.clip(2.0 * (g_ref - g_now), -a_max, a_max))
        h = vhat - s_up * up
        hn = float(np.linalg.norm(h))
        if hn < 1e-6:
            return vhat
        h /= hn
        g_cmd = g_now + d
        return math.cos(g_cmd) * h + math.sin(g_cmd) * up

    def _g_limited_throttle(self, st) -> float:
        """Throttle that keeps the cargo's sensed acceleration under the limit, counting
        the aerodynamic deceleration (estimated from the sensed load minus thrust)."""
        p_amb = self.sim.atmosphere.at(st.altitude).pressure
        t_max = max(self.sim.engine.max_thrust(p_amb), 1e-6)
        a_other = max(st.cargo_g * G0 - self.sim.thrust / st.mass, 0.0)
        a_allow = max(self.g.cargo_g_limit * 0.92 * G0 - a_other, 0.5 * G0)
        return float(np.clip(a_allow * st.mass / t_max, 0.0, 1.0))

    def _along_error(self, p) -> tuple[float, float]:
        along, cross = self.mw.along_cross(p)
        return along - (self.mw.range + self.g.meco_bias), cross

    def act(self, st):
        sim = self.sim
        t = sim.t
        # ------------------------------------------------------------------ ascent
        if self.phase == "liftoff":
            if t >= self.rise:
                self._set_phase("pitch_kick", "Pitch kick")
            throttle = self._g_limited_throttle(st)
            gimbal, rcs = self.att(st, self.up0, sim.thrust)
            return throttle, gimbal, rcs, "ascent"
        if self.phase == "pitch_kick":
            frac = min((t - self.rise) / self.g.kick_time, 1.0)
            ang = frac * self.kick
            axis = math.cos(ang) * self.up0 + math.sin(ang) * self.downrange0
            if frac >= 1.0:
                self._set_phase("kick_hold", "Kick hold")
            gimbal, rcs = self.att(st, axis, sim.thrust)
            return self._g_limited_throttle(st), gimbal, rcs, "ascent"
        if self.phase == "kick_hold":
            axis = math.cos(self.kick) * self.up0 + math.sin(self.kick) * self.downrange0
            vhat = st.vel_com / max(np.linalg.norm(st.vel_com), 1e-6)
            # wait until the flight path has pitched over as far as the kick. Compare the
            # elevation only: a crosswind keeps the velocity slightly out of the kick plane
            # (corrected later in the gravity turn), and a 3-D alignment test would then
            # time out and loft the trajectory (found by Monte Carlo).
            if (
                float(vhat @ self.up0)
                <= math.cos(self.kick) + math.radians(0.5) * math.sin(self.kick)
                or t > self.rise + self.g.kick_time + 40
            ):
                self._set_phase("gravity_turn", "Gravity turn")
            gimbal, rcs = self.att(st, axis, sim.thrust)
            return self._g_limited_throttle(st), gimbal, rcs, "ascent"
        if self.phase == "gravity_turn":
            # follow the *ground* velocity: at low speed a tail/head wind would otherwise
            # steer the turn (the planner assumes calm air); AoA stays a few degrees
            axis = st.vel_com / np.linalg.norm(st.vel_com)
            if self.profile is not None:
                axis = self._track_profile(st, axis)
            # crossrange correction from the vacuum impact point
            imp = kepler_impact(
                st.com,
                st.vel_com,
                sim.gravity,
                self.mw.r_target,
            )
            if imp is not None:
                along, cross = self._along_error(imp)
                frame = self.mw.local_frame(*self.mw.gravity.map_coords(st.com)[:2])
                cross_w = frame @ np.array([*self.mw.cross_dir, 0.0])
                corr = float(np.clip(-cross / 20_000.0, -0.06, 0.06))
                axis = axis + corr * cross_w
                axis /= np.linalg.norm(axis)
                if along > -0.25 * self.mw.range and t - self._last_pred_t >= 0.5:
                    self._precise_meco_check(st)
            if self._meco_at is not None and t >= self._meco_at:
                self._set_phase("coast", "MECO", "cutoff")
                return 0.0, np.zeros(2), np.zeros(3), "coast"
            gimbal, rcs = self.att(st, axis, sim.thrust)
            if sim.prop_mass < self._landing_reserve(st):  # never eat the landing fuel
                self._set_phase("coast", "MECO (fuel reserve)", "cutoff")
                return 0.0, np.zeros(2), rcs, "coast"
            return self._g_limited_throttle(st), gimbal, rcs, "ascent"

        # ------------------------------------------------------------------ coast / entry
        if self.phase == "coast":
            v_air = st.vel_com - st.wind
            retro = -v_air / max(np.linalg.norm(v_air), 1e-6)
            _, rcs = self.att(st, retro, 0.0, max_rate=math.radians(5.0))  # gentle, gas-saving flip
            descending = st.vertical_speed < 0
            if descending and st.altitude < self.g.entry_altitude and st.speed > self.g.entry_speed:
                self._set_phase("entry_burn", "Entry burn", "ignition")
            elif descending and st.altitude < self.g.entry_altitude:
                self._set_phase("descent", "Aero descent")
            return 0.0, np.zeros(2), rcs, "coast"
        if self.phase == "entry_burn":
            v_air = st.vel_com - st.wind
            retro = -v_air / max(np.linalg.norm(v_air), 1e-6)
            # burn to the target speed; keep burning (down to 80% of it) while the predicted
            # impact is still long -- extending a retro burn shortens the range
            long_miss = self._entry_along is not None and self._entry_along > 30.0
            stop = (st.speed <= self.g.entry_speed and not long_miss) or (
                st.speed <= 0.8 * self.g.entry_speed
            )
            stop = stop or sim.prop_mass <= self._landing_reserve(st)
            if stop:
                self._set_phase("descent", "Entry burn cutoff", "cutoff")
                return 0.0, np.zeros(2), np.zeros(3), "descent"
            # steer the predicted impact point onto the target by tilting off retrograde
            if t - self._last_pred_t >= 0.5:
                self._last_pred_t = t
                imp = self.predictor.predict(
                    st.com, st.vel_com, sim.prop_mass, entry_burn=True, t0=t
                )
                if imp is not None:
                    self.predicted_impact = imp
                    self._entry_along = self._along_error(imp)[0] + self.g.meco_bias
                    miss = imp - self.mw.pad_b
                    up_i = self.sim.gravity.up(imp)
                    miss -= (miss @ up_i) * up_i
                    self._entry_tilt = miss
            axis = retro
            miss = getattr(self, "_entry_tilt", None)
            if miss is not None:
                n = float(np.linalg.norm(miss))
                if n > 1.0:
                    ang = min(n / 1_500.0, 1.0) * math.radians(20.0)
                    side = -miss / n
                    side -= (side @ retro) * retro
                    sn = float(np.linalg.norm(side))
                    if sn > 1e-6:
                        axis = math.cos(ang) * retro + math.sin(ang) * side / sn
            gimbal, rcs = self.att(st, axis, sim.thrust)
            return self._g_limited_throttle(st), gimbal, rcs, "entry_burn"

        # ------------------------------------------------------------------ landing
        # the landing profile ignores drag: keep the engine off until the vehicle is in
        # its subsonic, near-terminal descent
        self.landing.allow_ignition = st.speed < 350.0 and st.agl < 8000.0
        if self.landing.phase == "coast":
            self._estimate_drag(st)
        if self.landing.phase == "coast" and t - self._last_pred_t >= 0.5:
            # unpowered descent: steer on the predicted impact point (drag + forecast wind)
            self._last_pred_t = t
            imp = self.predictor.predict(st.com, st.vel_com, sim.prop_mass, entry_burn=False, t0=t)
            if imp is not None:
                self.predicted_impact = imp
                self.landing.predicted_miss = imp - self.mw.pad_b
        throttle, gimbal, rcs = self.landing.act(st)
        phase = {"coast": "descent", "burn": "landing_burn", "landed": "landed"}.get(
            self.landing.phase, "descent"
        )
        if phase == "landing_burn" and not any(e[2] == "Landing burn" for e in self.events):
            self.events.append((t, "ignition", "Landing burn"))
            if self.landing.retargeted is not None:
                self.events.append(
                    (t, "phase", f"Divert limit: safe landing {self.landing.retargeted:,.0f} m off")
                )
        # legs deploy in the last seconds (low speed / low height): deployed struts lead
        # in engine-first flight and add a destabilising drag moment at speed
        if (
            phase in ("landing_burn", "landed")
            and not sim.legs_deployed
            and (st.speed < 50.0 or st.agl < 200.0 or phase == "landed")
        ):
            sim.set_legs(True)
            self.events.append((t, "phase", "Legs deployed"))
        return throttle, gimbal, rcs, phase

    def _estimate_drag(self, st) -> None:
        """In-flight drag estimation (unpowered descent): the measured non-gravitational
        deceleration along the air path over the predictor's modelled drag, low-pass
        filtered, scales the predictor's drag. Absorbs aero-coefficient, density and
        mass errors that would otherwise grow into a landing miss (found by Monte Carlo)."""
        sim = self.sim
        prev = self._drag_prev
        self._drag_prev = (sim.t, st.vel_com.copy())
        if prev is None or sim.thrust > 0:
            return
        dt = sim.t - prev[0]
        if dt <= 0:
            return
        a = (st.vel_com - prev[1]) / dt - sim.gravity.accel(st.com)
        if getattr(sim.gravity, "rotating", False):
            a = a - sim.gravity.fictitious_accel(st.com, st.vel_com)
        v_air = st.vel_com - self.landing.wind_forecast(st)
        speed = float(np.linalg.norm(v_air))
        if speed < 30.0 or st.q_dyn < 2_000.0:
            return
        meas = -float(a @ v_air) / speed
        model = self.predictor.model_drag_accel(st.com, v_air, st.mass)
        if model <= 0.5 or meas <= 0:
            return
        k = dt / (3.0 + dt)
        est = self.predictor.pm.drag_scale + k * (meas / model - self.predictor.pm.drag_scale)
        self.predictor.pm.drag_scale = float(np.clip(est, 0.5, 2.0))

    def _landing_reserve(self, st) -> float:
        """Propellant to keep for the landing burn: 1.5x the terminal velocity at the
        landing site (body + deployed grid-fin drag) plus 150 m/s for divert and hover."""
        sim = self.sim
        dry = st.mass - sim.prop_mass
        rho = sim.atmosphere.density(self.predictor.ground_altitude)
        cda = sim.aero.axial_coefficient(0.3, nose_first=False) * sim.aero.ref_area
        cda += self.predictor.pm.extra_cda
        v_term = math.sqrt(2.0 * dry * G0 / max(rho * cda, 1e-9))
        dv = 1.5 * v_term + 150.0
        isp = sim.nominal.engine.isp_sea_level
        return dry * (math.exp(dv / (isp * G0)) - 1.0)

    def _precise_meco_check(self, st) -> None:
        """Run the drag + entry-burn predictor and schedule MECO at the zero crossing."""
        t = self.sim.t
        self._last_pred_t = t
        imp = self.predictor.predict(st.com, st.vel_com, self.sim.prop_mass, entry_burn=True, t0=t)
        if imp is None:
            return
        self.predicted_impact = imp
        err, _ = self._along_error(imp)
        self._pred_hist.append((t, err))
        # the engine's thrust tail-off after the cut adds ~T*tau of impulse: cut early
        tail = self.sim.nominal.engine.throttle_tau
        if err >= 0:
            self._meco_at = t
            return
        if len(self._pred_hist) >= 2:
            (t1, e1), (t2, e2) = self._pred_hist[-2], self._pred_hist[-1]
            rate = (e2 - e1) / max(t2 - t1, 1e-6)
            if rate > 0:
                t_cross = t2 + (-e2) / rate - tail
                if t_cross - t < 0.5 + CONTROL_DT:
                    self._meco_at = t_cross


# ----------------------------------------------------------------------------- planning
def _plan_run(spec: MissionSpec, mw: MissionWorld, vehicle, cargo: float, world=None):
    """The planner's 3-DOF ascent model: returns ``run(rise, kick[, profile])`` -> score
    (propellant left at the target-reaching cutoff, penalised for lofted arcs)."""
    g = spec.guidance
    gravity = mw.gravity
    world = (world or spec.world).model_copy(
        update={"wind": (world or spec.world).wind.model_copy(update={"speed": 0.0})}
    )
    pm = PointMassSim(vehicle, world, cargo_mass=cargo)
    pm.accel_limit = g.cargo_g_limit * 0.92 * G0  # the ascent is flown g-limited too
    up0 = gravity.up(mw.pad_a)
    horiz = mw.local_frame(*mw.site_a) @ np.array([*mw.track, 0.0])
    dr = horiz / np.linalg.norm(horiz)
    r_target = mw.r_target
    m0 = pm.m_dry + pm.prop0
    t_max = pm.engine.max_thrust(0.0)

    def run(rise: float, kick: float, profile: list | None = None) -> float:
        kick_axis = math.cos(kick) * up0 + math.sin(kick) * dr
        hold = {"aligned": False}

        def direction(t, r, v):
            if t < rise:
                return up0
            if t < rise + g.kick_time:
                a = (t - rise) / g.kick_time * kick
                return math.cos(a) * up0 + math.sin(a) * dr
            vhat = v / np.linalg.norm(v)
            if not hold["aligned"]:  # hold the kick attitude until the velocity catches up
                if (
                    float(vhat @ kick_axis) > math.cos(math.radians(0.5))
                    or t > rise + g.kick_time + 40
                ):
                    hold["aligned"] = True
                else:
                    return kick_axis
            return vhat

        def throttle(t, r, v):
            return min(1.0, g.cargo_g_limit * G0 * m0 * 0.92 / t_max)

        def stop(t, r, v, prop):
            if t < rise + g.kick_time:
                return False
            imp = kepler_impact(r, v, gravity, r_target)
            return imp is not None and mw.along_cross(imp)[0] >= mw.range

        traj = pm.run(
            mw.pad_a + up0 * 5.0,
            np.zeros(3),
            t_end=400.0,
            dt=0.1,
            throttle=throttle,
            direction=direction,
            stop_on_ground=True,
            ground_altitude=mw.terrain_height(*mw.site_a) - 10.0,
            stop_fn=stop,
            record_every=100_000 if profile is None else 5,
        )
        if profile is not None:  # flight-path angle vs speed through the gravity turn
            for t, rr, vv in zip(traj.t, traj.pos, traj.vel, strict=True):
                sp = float(np.linalg.norm(vv))
                if t > rise + g.kick_time and sp > 1.0:
                    profile.append((sp, math.asin(float(vv @ gravity.up(rr)) / sp)))
        r, v = traj.pos[-1], traj.vel[-1]
        imp = kepler_impact(r, v, gravity, r_target)
        reached = imp is not None and mw.along_cross(imp)[0] >= mw.range * 0.999
        if not reached:
            return -1e9
        # steep (lofted) arcs re-enter steeply: penalise flight-path angles above the cap
        gamma = math.degrees(math.asin(float(v @ gravity.up(r)) / float(np.linalg.norm(v))))
        return traj.mass[-1] - 50.0 * max(gamma - g.max_flight_path_deg, 0.0)

    return run


def plan_ascent(
    spec: MissionSpec, mw: MissionWorld, vehicle, cargo: float, world=None
) -> tuple[float, float]:
    """Choose (vertical rise time, pitch-kick angle) that reach the target range with the
    most propellant left (3-DOF gravity turn, vacuum impact prediction)."""
    g = spec.guidance
    run = _plan_run(spec, mw, vehicle, cargo, world)

    best = (-math.inf, g.rise_time, math.radians(2.0))
    rises = [g.rise_time] if g.kick_angle_deg is not None else [4.0, 7.0, 10.0, 14.0]
    for rise in rises:
        kicks = np.radians(np.arange(1.0, 24.01, 1.0))
        vals = [run(rise, k) for k in kicks]
        i = int(np.argmax(vals))
        if vals[i] > best[0]:
            best = (vals[i], rise, kicks[i])
    _, rise, kick = best
    # golden-section refinement of the kick around the best grid point
    lo, hi = max(kick - math.radians(0.5), math.radians(0.2)), kick + math.radians(0.5)
    gr = 0.5 * (math.sqrt(5) - 1)
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = run(rise, a), run(rise, b)
    for _ in range(10):
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = run(rise, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = run(rise, b)
    return rise, math.degrees(0.5 * (lo + hi))


def ascent_profile(
    spec: MissionSpec, mw: MissionWorld, vehicle, cargo: float, world, rise: float, kick_deg: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Planned flight-path angle (rad) vs speed (m/s) through the gravity turn, from the
    same 3-DOF model and direction law as ``plan_ascent`` (calm air, nominal vehicle)."""
    prof: list[tuple[float, float]] = []
    _plan_run(spec, mw, vehicle, cargo, world)(rise, math.radians(kick_deg), prof)
    if len(prof) < 5:
        return None
    a = np.array(sorted(prof))
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 1e-6])
    return a[keep, 0], a[keep, 1]


# ----------------------------------------------------------------------------- runner
def mission_frame(spec: MissionSpec, fidelity: str | None = None):
    """World frame, gravity model and site coordinates for a mission.

    * fast (default): non-rotating spherical Earth, map coordinates from ``u``/``v``.
    * high: WGS-84 ellipsoid with J2-J6 gravity, rotating Earth; the world frame is the
      local East-North-Up frame at the launch site (``launch.lat``/``lon`` required),
      and a target given by ``lat``/``lon`` is projected to map coordinates with the same
      azimuthal-equidistant projection the terrain uses.
    """
    from plume.physics.gravity import gravity_from_world

    fid = fidelity or spec.world.fidelity
    if fid == "high" or spec.world.gravity == "wgs84":
        a, b = spec.launch, spec.target
        if a.lat is None or a.lon is None:
            raise ValueError("high-fidelity missions need launch.lat / launch.lon (WGS-84 degrees)")
        if b.lat is not None and b.lon is not None:
            from plume.terrain.dem import site_projection

            u, v = site_projection(a.lat, a.lon).forward(b.lat, b.lon)
            spec = spec.model_copy(deep=True)
            spec.target.u, spec.target.v = float(u), float(v)
        earth = spec.world.earth.model_copy(
            update={"origin_lat_deg": a.lat, "origin_lon_deg": a.lon, "origin_height": 0.0}
        )
        world = spec.world.model_copy(
            update={"fidelity": "high", "gravity": "wgs84", "earth": earth, "ground": "none"}
        )
        return spec, world, gravity_from_world(world)
    world = spec.world.model_copy(update={"gravity": "spherical", "ground": "none"})
    return spec, world, SphericalGravity()


@dataclass
class HopRun:
    result: MissionResult
    recorder: Recorder
    kick_deg: float
    rise_time: float = 0.0


def run_mission(
    spec: MissionSpec | str,
    seed: int = 0,
    cargo_mass: float | None = None,
    on_frame=None,
    kick_deg: float | None = None,
    fidelity: str | None = None,
    vehicle=None,
    nominal_vehicle=None,
    nominal_world=None,
    rise_time: float | None = None,
) -> HopRun:
    """Fly a mission. ``vehicle`` overrides the simulated (true) vehicle and
    ``nominal_vehicle`` what guidance believes (Monte Carlo); both default to the mission's
    vehicle preset with the mission cargo. ``nominal_world`` is the forecast environment
    guidance plans with (defaults to the simulated one); ``kick_deg`` + ``rise_time`` fix
    the ascent plan instead of optimising it."""
    spec = load_mission(spec) if isinstance(spec, str) else spec
    spec, world, gravity = mission_frame(spec, fidelity)
    mw = MissionWorld(spec, gravity)
    cargo = (
        cargo_mass
        if cargo_mass is not None
        else (spec.cargo_mass if spec.cargo_mass is not None else None)
    )
    truth = vehicle
    vehicle = nominal_vehicle or load_vehicle(spec.vehicle)
    if cargo is not None and nominal_vehicle is None:
        vehicle = vehicle.with_cargo(cargo)
    truth = truth or vehicle
    cargo = vehicle.cargo.mass
    rise = rise_time if rise_time is not None else spec.guidance.rise_time
    if kick_deg is None:
        if spec.guidance.kick_angle_deg is not None:
            kick_deg = spec.guidance.kick_angle_deg
        else:
            rise, kick_deg = plan_ascent(spec, mw, vehicle, cargo, nominal_world or world)

    sim = RocketSim(
        truth, world, seed=seed, tiles=mw.ground_tiles(), ground_height=mw.ground_height
    )
    sim.nominal = vehicle
    sim.nominal_world = nominal_world or world
    up = gravity.up(mw.pad_a)
    # feet exactly on the pad so the vehicle starts in contact (not "airborne")
    sim.reset(
        pos=mw.pad_a + up * (vehicle.legs.height - 0.005), quat=quat_from_z_axis(up), seed=seed
    )
    from plume.control.navigation import Navigator, navigation_mode

    nav = Navigator(sim, seed=seed) if navigation_mode(world) == "ekf" else None
    profile = None
    if spec.guidance.closed_loop_ascent:
        profile = ascent_profile(spec, mw, vehicle, cargo, nominal_world or world, rise, kick_deg)
    ap = HopAutopilot(sim, mw, spec.guidance, kick_deg, rise, ascent_profile=profile)
    rec = Recorder(
        sim.replay_meta(
            f"Cargo hop: {spec.name}",
            controller="hop-autopilot",
            seed=seed,
            scene=mw.scene_meta(),
            mission={
                "name": spec.name,
                "range_m": mw.range,
                "cargo_kg": cargo,
                "kick_deg": kick_deg,
                "rise_time_s": rise,
            },
        )
    )
    steps = round(CONTROL_DT / sim.dt)
    rec_every = max(1, spec.record_every)
    k = 0
    failure = ""
    landed_t = None
    max_alt = 0.0
    while sim.t < spec.max_time:
        # the flight software acts on the navigation estimate (truth in fast mode)
        st = nav.estimate() if nav is not None else sim.state
        throttle, gimbal, rcs, phase = ap.act(st)
        if landed_t is not None:
            throttle = 0.0
        sim.set_controls(throttle, gimbal, rcs)
        sim.step(steps)
        if nav is not None:
            nav.update(steps * sim.dt)
        st = sim.state
        max_alt = max(max_alt, st.altitude)
        k += 1
        if k % rec_every == 0:
            impact = ap.predicted_impact
            frame = sim.frame(phase, {"impact": impact if impact is not None else mw.pad_b})
            rec.record(frame)
            if on_frame:
                on_frame(frame)
        if sim.ever_body_contact:
            failure = "crash_hull"
            break
        if (
            sim.touchdown is not None
            and sim.touchdown.vertical_speed > vehicle.legs.max_touchdown_speed
        ):
            failure = "crash_legs"
            break
        if st.agl < 0.0 and sim.touchdown is None and sim.t > 30:
            # off the collision tiles the terrain is analytic: judge the arrival there
            u, v, _ = mw.gravity.map_coords(st.pos)
            if not any(tile.contains(u, v) for tile in mw.tiles):
                gentle = (
                    -st.vertical_speed <= vehicle.legs.max_touchdown_speed
                    and st.horizontal_speed < 2.0
                    and st.tilt < math.radians(10.0)
                )
                failure = "landed_off_site" if gentle else "terrain_impact"
                break
            if st.agl < -2.0:
                failure = "terrain_impact"
                break
        if landed_t is None and sim.touchdown is not None and sim.t > 30:
            landed_t = sim.t
        if landed_t is not None and sim.t - landed_t > 5.0:
            break
    else:
        failure = "timeout"
    impact = ap.predicted_impact
    rec.record(
        sim.frame(ap.phase, {"impact": impact if impact is not None else mw.pad_b}), force=True
    )
    for t, kind, label in ap.events:
        rec.event(t, kind, label)
    if sim.touchdown is not None and sim.t > 30:
        rec.event(sim.touchdown.t, "touchdown", "Touchdown")
    result = score_mission(sim, mw, spec, failure, max_alt, prop_initial=float(sim.tank_init.sum()))
    extra = {"kick_deg": kick_deg}
    if nav is not None:
        rec.meta["navigation"] = {"mode": "ekf", **nav.summary()}
        extra["nav_pos_err_max_m"] = nav.summary().get("pos_err_max_m")
    rec.set_outcome(result.success, result.reason, asdict(result) | extra)
    rec.meta["outcome"]["metrics"].pop("success", None)
    rec.meta["outcome"]["metrics"].pop("reason", None)
    return HopRun(result, rec, kick_deg, rise)
