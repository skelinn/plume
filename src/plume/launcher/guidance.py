"""Launch guidance: stack ascent, closed-loop upper-stage guidance to orbit, booster return.

``StackAscent``
    vertical rise, pitch kick toward the launch azimuth, gravity turn (thrust along the
    velocity, a small yaw trim toward the target plane above the dense atmosphere),
    acceleration-limited throttle; main-engine cutoff (MECO) when the first stage is down to
    its return reserve.

``UpperStageGuidance``
    closed-loop terminal guidance in the Earth-centred inertial frame. Every cycle the
    radial and out-of-plane channels are given the linear acceleration profile
    ``a(t) = A + B t`` that reaches the target radius with zero radial velocity and zero
    out-of-plane position/velocity at the predicted burnout time ``t_go`` (from the rocket
    equation with the velocity still to be gained); the rest of the thrust goes
    downrange. Re-targeting stops ``freeze_time`` before burnout. The engine is cut when
    the specific orbital energy reaches the target orbit's (less the thrust tail-off).
    This is a simplified, PEG-like linear-acceleration law (the classic "linear tangent"
    family): robust, not propellant-optimal.

``BoosterReturn``
    return to launch site (RTLS): flip with the RCS, boost-back burn steering the
    predicted (vacuum) impact point onto the landing zone and cut off by the drag- and
    entry-burn-aware predictor; then the cargo-hop autopilot
    (:class:`~plume.missions.hop.HopAutopilot`) flies the coast, entry burn, aero descent
    and landing burn unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from plume.constants import G0
from plume.control.attitude import AttitudeController
from plume.launcher.orbit import InertialFrame
from plume.launcher.spec import (
    AscentSpec,
    BoostbackSpec,
    OrbitTargetSpec,
    UpperGuidanceSpec,
)
from plume.missions.targeting import kepler_impact

V_ORBIT_NOMINAL = 7800.0  # m/s, used only to turn an inertial launch heading into a ground one


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def launch_heading(frame: InertialFrame, p_w: np.ndarray, n_target: np.ndarray, t: float = 0.0):
    """Horizontal world-frame direction to fly at the pad so that the inertial velocity
    builds up in the target plane (``n_target``, ECI). On a rotating Earth the heading
    relative to the ground is corrected for the pad's eastward speed."""
    g = frame.gravity
    r_i, v_i = frame.to_inertial(p_w, np.zeros(3), t)  # v_i = Earth-rotation velocity
    d_i = _unit(np.cross(n_target, _unit(r_i)))
    rel_i = V_ORBIT_NOMINAL * d_i - v_i
    d_w = frame.to_world_direction(rel_i, t)
    up = g.up(p_w)
    d_w = d_w - (d_w @ up) * up
    return _unit(d_w)


# ----------------------------------------------------------------------------- stack ascent
def _ascent_run(vehicle, world, p0, up0, heading, spec: AscentSpec, reserve: float):
    """The planner's 3-DOF stack ascent (calm air, same direction law as ``StackAscent``):
    returns ``run(kick_rad, profile=None) -> flight-path angle at MECO (rad)``."""
    from plume.physics.pointmass import PointMassSim

    calm = world.model_copy(update={"wind": world.wind.model_copy(update={"speed": 0.0})})
    pm = PointMassSim(vehicle, calm)
    pm.accel_limit = spec.max_g * G0
    gravity = pm.gravity
    rise, kt = spec.rise_time, spec.kick_time

    def run(kick: float, profile: list | None = None) -> float:
        kick_axis = math.cos(kick) * up0 + math.sin(kick) * heading
        hold = {"done": False}

        def direction(t, r, v):
            if t < rise:
                return up0
            if t < rise + kt:
                a = (t - rise) / kt * kick
                return math.cos(a) * up0 + math.sin(a) * heading
            if not hold["done"]:
                vhat = _unit(v)
                if float(vhat @ up0) <= math.cos(kick) or t > rise + kt + 30:
                    hold["done"] = True
                else:
                    return kick_axis
            return _unit(v)

        traj = pm.run(
            p0,
            np.zeros(3),
            t_end=600.0,
            dt=0.1,
            throttle=1.0,
            direction=direction,
            stop_on_ground=True,
            ground_altitude=gravity.altitude(p0) - 10.0,
            stop_fn=lambda t, r, v, prop: prop <= reserve,
            record_every=5 if profile is not None else 100_000,
        )
        if profile is not None:
            for t, rr, vv in zip(traj.t, traj.pos, traj.vel, strict=True):
                sp = float(np.linalg.norm(vv))
                if t > rise + kt and sp > 1.0:
                    profile.append((sp, math.asin(float(vv @ gravity.up(rr)) / sp)))
        r, v = traj.pos[-1], traj.vel[-1]
        sp = max(float(np.linalg.norm(v)), 1e-9)
        return math.asin(max(-1.0, min(1.0, float(v @ gravity.up(r)) / sp)))

    return run


def plan_ascent(vehicle, world, p0, up0, heading, spec: AscentSpec, reserve: float):
    """Pitch-kick angle that gives the planned flight-path angle at MECO (bisection on the
    3-DOF model; a larger kick flies a flatter trajectory), and the planned flight-path
    angle vs speed profile that the closed-loop gravity turn tracks."""
    run = _ascent_run(vehicle, world, p0, up0, heading, spec, reserve)
    if spec.kick_angle_deg is not None:
        kick = math.radians(spec.kick_angle_deg)
    else:
        target = math.radians(spec.staging_flight_path_deg)
        lo, hi = math.radians(0.3), math.radians(20.0)
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if run(mid) > target:
                lo = mid  # too steep: kick harder
            else:
                hi = mid
        kick = 0.5 * (lo + hi)
    prof: list[tuple[float, float]] = []
    run(kick, prof)
    profile = None
    if len(prof) >= 5:
        a = np.array(sorted(prof))
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 1e-6])
        profile = (a[keep, 0], a[keep, 1])
    return math.degrees(kick), profile


class StackAscent:
    def __init__(
        self,
        sim,
        spec: AscentSpec,
        frame: InertialFrame,
        n_target: np.ndarray,
        reserve: float,
        kick_deg: float | None = None,
        profile: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.sim = sim
        self.spec = spec
        self.frame = frame
        self.n_target = n_target
        self.reserve = reserve
        self.att = AttitudeController(sim)
        st = sim.state
        self.up0 = sim.gravity.up(st.com)
        self.heading = launch_heading(frame, st.com, n_target)
        if kick_deg is None:
            kick_deg, profile = plan_ascent(
                sim.nominal,
                getattr(sim, "nominal_world", sim.world),
                st.com,
                self.up0,
                self.heading,
                spec,
                reserve,
            )
        self.kick_deg = kick_deg
        self.kick = math.radians(kick_deg)
        self.profile = profile if spec.closed_loop else None
        self.phase = "liftoff"
        self.events: list[tuple[float, str, str]] = []
        self.done = False

    def _event(self, kind: str, label: str) -> None:
        self.events.append((self.sim.t, kind, label))

    def _throttle(self, st) -> float:
        """Full thrust, limited to ``max_g`` of sensed acceleration (thrust + drag)."""
        p_amb = self.sim.atmosphere.at(st.altitude).pressure
        t_max = max(self.sim.engine.max_thrust(p_amb), 1e-6)
        return float(np.clip(self.spec.max_g * G0 * st.mass / t_max, 0.0, 1.0))

    def _track_profile(self, st, vhat: np.ndarray) -> np.ndarray:
        """Closed-loop gravity turn (as in the cargo hop): pitch within the vertical plane
        of the velocity so the flight-path angle tracks the plan at this speed, with the
        angle of attack limited (tighter at high dynamic pressure)."""
        sp, gam = self.profile
        speed = float(np.linalg.norm(st.vel_com))
        if speed < sp[0] or speed > sp[-1]:
            return vhat
        up = st.up
        s_up = float(vhat @ up)
        g_now = math.asin(max(-1.0, min(1.0, s_up)))
        g_ref = float(np.interp(speed, sp, gam))
        a_max = math.radians(1.5 if st.q_dyn > 15_000.0 else 4.0)
        d = float(np.clip(2.0 * (g_ref - g_now), -a_max, a_max))
        h = vhat - s_up * up
        hn = float(np.linalg.norm(h))
        if hn < 1e-6:
            return vhat
        h /= hn
        g_cmd = g_now + d
        return math.cos(g_cmd) * h + math.sin(g_cmd) * up

    def _meco_due(self, st) -> bool:
        # cut a little early for the thrust tail-off (mdot * tau of propellant)
        tail = self.sim.engine.mdot_max * self.sim.nominal.engine.throttle_tau
        return self.sim.prop_mass <= self.reserve + tail

    def act(self, st):
        sim = self.sim
        t = sim.t
        sp = self.spec
        if self.done:
            _, rcs = self.att(st, st.axis, 0.0)
            return 0.0, np.zeros(2), rcs, "coast"
        if self._meco_due(st):
            self.done = True
            self.phase = "coast"
            self._event("cutoff", "MECO")
            return 0.0, np.zeros(2), np.zeros(3), "coast"
        throttle = self._throttle(st)
        if self.phase == "liftoff":
            axis = self.up0
            if t >= sp.rise_time:
                self.phase = "pitch_kick"
                self._event("phase", "Pitch kick")
                if sim.legs_deployed:
                    sim.set_legs(False)
        elif self.phase == "pitch_kick":
            frac = min((t - sp.rise_time) / sp.kick_time, 1.0)
            ang = frac * self.kick
            axis = math.cos(ang) * self.up0 + math.sin(ang) * self.heading
            if frac >= 1.0:
                self.phase = "kick_hold"
        elif self.phase == "kick_hold":
            axis = math.cos(self.kick) * self.up0 + math.sin(self.kick) * self.heading
            vhat = _unit(st.vel_com)
            if (
                float(vhat @ self.up0) <= math.cos(self.kick)
                or t > sp.rise_time + sp.kick_time + 30
            ):
                self.phase = "gravity_turn"
                self._event("phase", "Gravity turn")
        else:
            # follow the *ground* velocity: at low speed a wind would otherwise steer the
            # turn (it turned a 3 deg kick around in a 6 m/s wind); the AoA stays small
            axis = _unit(st.vel_com)
            if self.profile is not None:
                axis = self._track_profile(st, axis)
            if st.q_dyn < 5_000.0 and st.altitude > 20_000.0:
                # above the dense atmosphere: trim the yaw toward the target plane
                _r_i, v_i = self.frame.to_inertial(st.com, st.vel_com, t)
                vn = float(v_i @ self.n_target)
                corr = float(np.clip(-vn / max(np.linalg.norm(v_i), 1.0) * 2.0, -0.03, 0.03))
                n_w = self.frame.to_world_direction(self.n_target, t)
                axis = _unit(axis + corr * n_w)
        gimbal, rcs = self.att(st, axis, sim.thrust)
        return throttle, gimbal, rcs, "ascent"


# ----------------------------------------------------------------------------- upper stage
class UpperStageGuidance:
    def __init__(
        self,
        sim,
        frame: InertialFrame,
        target: OrbitTargetSpec,
        n_target: np.ndarray,
        spec: UpperGuidanceSpec,
        t_ignite: float,
    ):
        self.sim = sim
        self.frame = frame
        self.spec = spec
        self.n = _unit(np.asarray(n_target, dtype=float))
        mu, r_eq = frame.mu, frame.r_eq
        self.mu = mu
        self.r_T = r_eq + target.perigee_altitude
        self.a_T = r_eq + 0.5 * (target.perigee_altitude + target.apogee_altitude)
        self.v_T = math.sqrt(mu * (2.0 / self.r_T - 1.0 / self.a_T))
        self.E_T = -mu / (2.0 * self.a_T)
        self.t_ignite = t_ignite
        self.att = AttitudeController(sim)
        self.phase = "coast_sep"
        self.events: list[tuple[float, str, str]] = []
        self.t_go = None
        self._coef = None  # (t_ref, A, B, An, Bn)
        self.cutoff_t: float | None = None
        self.last_dir = None
        self.throttle = 1.0
        self.trim_time = 3.0  # s of full-thrust energy gain flown at minimum throttle
        self.control_dt = 0.05  # flight-software cycle, s

    def bind(self, sim) -> None:
        """Continue with a new simulator for the same stage (e.g. after fairing jettison)."""
        self.sim = sim
        self.att = AttitudeController(sim)

    def _event(self, kind: str, label: str) -> None:
        self.events.append((self.sim.t, kind, label))

    def steering(self, st) -> np.ndarray:
        """Commanded thrust direction (world frame) from the current state."""
        sim = self.sim
        t = sim.t
        r, v = self.frame.to_inertial(st.com, st.vel_com, t)
        rn = float(np.linalg.norm(r))
        rh = r / rn
        n = self.n
        th = _unit(np.cross(n, rh))
        vr, vt, vn = float(v @ rh), float(v @ th), float(v @ n)
        z = float(r @ n)
        mu = self.mu
        g_eff = mu / rn**2 - vt * vt / rn
        g_end = mu / self.r_T**2 - self.v_T**2 / self.r_T
        eng = sim.nominal.engine
        thrust = max(sim.engine.max_thrust(0.0) * self.throttle, 1e-6)
        ve = eng.isp_vac * G0
        a_t = thrust / st.mass
        tau = st.mass * ve / thrust
        T = self.t_go if self.t_go is not None else 100.0
        for _ in range(4):
            dv_r = -vr + 0.5 * (g_eff + g_end) * T
            dv = math.sqrt((self.v_T - vt) ** 2 + dv_r * dv_r + vn * vn)
            T = tau * (1.0 - math.exp(-min(dv / ve, 20.0)))
        T = max(T, 1e-3)
        self.t_go = T
        if self._coef is None or T > self.spec.freeze_time:
            dr = self.r_T - rn - vr * T
            B = (-12.0 * dr - 6.0 * vr * T) / T**3
            A = (-vr - 0.5 * B * T * T) / T
            dz = -z - vn * T
            Bn = (-12.0 * dz - 6.0 * vn * T) / T**3
            An = (-vn - 0.5 * Bn * T * T) / T
            self._coef = (t, A, B, An, Bn)
        t_ref, A, B, An, Bn = self._coef
        el = t - t_ref
        a_r = g_eff + A + B * el
        a_n = An + Bn * el
        s = math.hypot(a_r, a_n)
        s_max = a_t * math.sin(math.radians(self.spec.max_off_tangent_deg))
        if s > s_max:
            a_r *= s_max / s
            a_n *= s_max / s
        a_h = math.sqrt(max(a_t * a_t - a_r * a_r - a_n * a_n, 0.0))
        d_i = (a_r * rh + a_n * n + a_h * th) / a_t
        return _unit(self.frame.to_world_direction(d_i, t))

    def energy(self, st) -> float:
        r, v = self.frame.to_inertial(st.com, st.vel_com, self.sim.t)
        return 0.5 * float(v @ v) - self.mu / float(np.linalg.norm(r))

    def act(self, st):
        sim = self.sim
        t = sim.t
        if self.phase in ("coast", "orbit"):
            _, rcs = self.att(st, self.last_dir if self.last_dir is not None else st.axis, 0.0)
            return 0.0, np.zeros(2), rcs, self.phase
        axis = self.steering(st)
        self.last_dir = axis
        if self.phase == "coast_sep":
            # slew to the burn attitude on the RCS while the stages drift apart
            _, rcs = self.att(st, axis, 0.0)
            if t >= self.t_ignite:
                self.phase = "burn"
                self._event("ignition", "Upper-stage ignition")
            return 0.0, np.zeros(2), rcs, "coast"
        # burning: cut off on energy, allowing for the thrust tail-off (first-order decay
        # with tau adds ~ a tau of speed) and the flight-software cycle (cut when the
        # target would be crossed within the first half of the next cycle)
        _r, v = self.frame.to_inertial(st.com, st.vel_com, t)
        # dE/dt = v . a_thrust: near burnout the thrust is well off the velocity (radial and
        # plane corrections), so |v| |a| would overstate the tail-off and cut early
        d_i = self.frame.world_to_eci_matrix(t) @ st.axis
        e_rate = max(float(v @ d_i), 0.0) * sim.thrust / st.mass
        tail = e_rate * sim.nominal.engine.throttle_tau
        remaining = self.E_T - tail - self.energy(st)
        if remaining <= 0.5 * e_rate * self.control_dt:
            self.phase = "coast"
            self.cutoff_t = t
            self._event("cutoff", "SECO")
            return 0.0, np.zeros(2), np.zeros(3), "coast"
        if sim.prop_mass <= 0.0:
            self.phase = "coast"
            self.cutoff_t = t
            self._event("cutoff", "Upper stage depleted")
            return 0.0, np.zeros(2), np.zeros(3), "coast"
        # last seconds: throttle down so the cutoff lands closer to the target energy
        e_full = float(np.linalg.norm(v)) * sim.engine.max_thrust(0.0) / st.mass
        if remaining < e_full * self.trim_time:
            self.throttle = sim.nominal.engine.throttle_min
        gimbal, rcs = self.att(st, axis, sim.thrust)
        return self.throttle, gimbal, rcs, "upper_burn"


# ----------------------------------------------------------------------------- booster RTLS
class BoosterReturn:
    def __init__(self, sim, mw, hop_autopilot, spec: BoostbackSpec, t_sep: float):
        self.sim = sim
        self.mw = mw
        self.hop = hop_autopilot
        self.spec = spec
        self.t_sep = t_sep
        self.att = AttitudeController(sim)
        self.phase = "coast_sep"
        self.events: list[tuple[float, str, str]] = []
        self._hist: list[tuple[float, float]] = []
        self._last_pred = -1e9
        self._cut_at: float | None = None
        self._dir_h: np.ndarray | None = None
        self.predicted_impact: np.ndarray | None = None
        self.boostback_prop: float | None = None  # propellant at boost-back ignition
        self.boostback_used: float | None = None

    def _event(self, kind: str, label: str) -> None:
        self.events.append((self.sim.t, kind, label))

    @property
    def all_events(self):
        return self.events + list(self.hop.events)

    def _boostback_axis(self, st) -> np.ndarray:
        """Thrust axis: horizontal, opposite to the predicted (vacuum) miss, pitched up."""
        mw = self.mw
        up = st.up
        imp = kepler_impact(st.com, st.vel_com, self.sim.gravity, mw.r_target)
        if imp is not None:
            miss = imp - mw.pad_b
            self.predicted_impact = imp
        else:  # not coming down on the sphere (should not happen): kill horizontal speed
            miss = st.vel_com * 100.0
        miss_h = miss - (miss @ up) * up
        n = float(np.linalg.norm(miss_h))
        if self._dir_h is None or n > 3_000.0:
            if n > 1e-6:
                self._dir_h = -miss_h / n
            elif self._dir_h is None:
                self._dir_h = _unit(-(st.vel_com - (st.vel_com @ up) * up))
        e = math.radians(self.spec.pitch_deg)
        return _unit(math.cos(e) * self._dir_h + math.sin(e) * up)

    def _along_error(self, p) -> float:
        along, _ = self.mw.along_cross(p)
        return along - (self.mw.range + self.hop.g.meco_bias)

    def act(self, st):
        sim = self.sim
        t = sim.t
        if self.phase == "handover":
            return self.hop.act(st)
        if self.phase == "coast_sep":
            _, rcs = self.att(st, st.axis, 0.0)
            if t - self.t_sep >= self.spec.coast_before_flip:
                self.phase = "flip"
                self._event("phase", "Flip")
            return 0.0, np.zeros(2), rcs, "flip"
        axis = self._boostback_axis(st)
        if self.phase == "flip":
            _, rcs = self.att(st, axis, 0.0, max_rate=math.radians(self.spec.flip_rate_deg_s))
            err = math.degrees(math.acos(float(np.clip(st.axis @ axis, -1.0, 1.0))))
            if err < self.spec.align_deg and np.linalg.norm(st.omega) < math.radians(4.0):
                self.phase = "boostback"
                self.boostback_prop = sim.prop_mass
                self._event("ignition", "Boost-back burn")
            return 0.0, np.zeros(2), rcs, "flip"
        # ---- boost-back burn
        done = self._cut_at is not None and t >= self._cut_at
        if self.predicted_impact is not None and t - self._last_pred >= 0.5:
            err_v = self._along_error(self.predicted_impact)
            if err_v > -40_000.0:
                self._last_pred = t
                imp = self.hop.predictor.predict(
                    st.com, st.vel_com, sim.prop_mass, entry_burn=True, t0=t
                )
                if imp is not None:
                    self.predicted_impact = imp
                    err = self._along_error(imp)
                    self._hist.append((t, err))
                    if err >= 0:
                        done = True
                    elif len(self._hist) >= 2:
                        (t1, e1), (t2, e2) = self._hist[-2], self._hist[-1]
                        rate = (e2 - e1) / max(t2 - t1, 1e-6)
                        tail = sim.nominal.engine.throttle_tau
                        if rate > 0:
                            t_cross = t2 - e2 / rate - tail
                            if t_cross - t < 0.55:
                                self._cut_at = t_cross
        # never eat into the entry and landing propellant
        if sim.prop_mass < 1.6 * self.hop._landing_reserve(st):
            done = True
            self._event("phase", "Boost-back cut short (propellant)")
        if done:
            self.phase = "handover"
            if self.boostback_prop is not None:
                self.boostback_used = self.boostback_prop - sim.prop_mass
            self.hop.phase = "coast"
            fins = sim.grid_fins
            if fins is not None and not fins.deployed:
                fins.deploy(True)
                self._event("phase", "Grid fins deployed")
            self._event("cutoff", "Boost-back cutoff")
            return 0.0, np.zeros(2), np.zeros(3), "coast"
        gimbal, rcs = self.att(st, axis, sim.thrust)
        # a nearly empty stage at full thrust would pull > 10 g: throttle to max_g
        t_max = max(sim.engine.max_thrust(0.0), 1e-6)
        throttle = min(1.0, self.spec.max_g * G0 * st.mass / t_max)
        return throttle, gimbal, rcs, "boostback"
