"""Classical landing autopilot (the PID/guidance baseline).

Phases
------
``coast``    engine off; RCS holds the vehicle engine-first into the airflow.
``burn``     hoverslam: ignite when the descent speed reaches a constant-deceleration
             stopping profile ``v_ref(h) = sqrt(sink^2 + 2 a_v h)``, then
             * vertical channel: PI tracking of ``v_ref`` (feasible by construction),
             * horizontal channel: zero-effort-miss guidance toward the pad with a
               floored time-to-go (the attitude loop is slow; stiff gains oscillate).
``landed``   engine off once a leg touches.

The commanded acceleration is turned into throttle + thrust axis and sent through
:class:`AttitudeController` (gimbal + RCS).
"""

from __future__ import annotations

import math

import numpy as np

from plume.control.attitude import AttitudeController
from plume.control.guidance import limit_tilt
from plume.physics.sim import RocketSim, State


class LandingAutopilot:
    def __init__(
        self,
        sim: RocketSim,
        target: np.ndarray,
        control_dt: float = 0.05,
        decel_fraction: float = 0.5,
        max_tilt_deg: float = 20.0,
        sink: float = 1.0,
        vertical_gain: float = 3.0,
        vertical_ki: float = 0.5,
        min_tgo: float = 6.0,
        aero_aware: bool = True,
        observer: bool = True,
        max_decel: float | None = None,
    ):
        self.sim = sim
        self.target = np.asarray(target, dtype=float)  # pad surface point (world)
        self.dt = control_dt
        self.decel_fraction = decel_fraction
        self.max_tilt = math.radians(max_tilt_deg)
        self.sink = sink
        self.kv = vertical_gain
        self.ki = vertical_ki
        self.min_tgo = min_tgo
        self.aero_aware = aero_aware
        self.use_observer = observer
        self.max_decel = max_decel
        self.allow_ignition = True  # a mission planner may inhibit ignition (e.g. during re-entry)
        self.att = AttitudeController(sim)
        self.reset()

    def reset(self) -> None:
        self.phase = "coast"
        self.integral = 0.0
        self.a_v: float | None = None
        self.disturbance = np.zeros(3)  # estimated unmodelled force (wind), world frame
        self.last_axis = np.array([0.0, 0.0, 1.0])  # last commanded thrust axis
        self._prev: tuple[np.ndarray, np.ndarray] | None = None  # (velocity, predicted accel)

    # ------------------------------------------------------------------ helpers
    def _max_accel(self, st: State) -> tuple[float, float]:
        p_amb = self.sim.atmosphere.at(st.altitude).pressure
        t_max = self.sim.engine.max_thrust(p_amb)
        return t_max, t_max / st.mass

    def v_ref(self, h: float) -> float:
        """Reference descent speed (positive down): constant deceleration ``a_v`` down to ``sink``."""
        return math.sqrt(self.sink * self.sink + 2.0 * self.a_v * max(h, 0.0))

    def _plan_decel(self, a_max: float, g: float) -> None:
        """Use a fraction of the thrust margin over gravity for the stopping profile."""
        self.a_v = max(self.decel_fraction * (a_max - g), 0.3)
        if self.max_decel is not None:  # e.g. a cargo g-load limit
            self.a_v = min(self.a_v, self.max_decel)

    # ------------------------------------------------------------------ main
    def act(self, st: State):
        """Returns (throttle, gimbal[2], rcs[3]) commands."""
        sim = self.sim
        g_vec = sim.gravity.accel(st.com)
        g = float(np.linalg.norm(g_vec))
        up = st.up
        t_max, a_max = self._max_accel(st)
        eng = sim.vehicle.engine
        if self.a_v is None:
            self._plan_decel(a_max, g)

        if st.legs_down > 0 or st.body_contact:
            self.phase = "landed"
        if self.phase == "landed":
            self.last_axis = up
            _, rcs = self.att(st, up, 0.0)
            return 0.0, np.zeros(2), rcs

        if self.use_observer:
            self._observe(st, g_vec)
        h = st.agl
        descent = -st.vertical_speed
        # horizontal channel: zero-effort-miss toward the pad (shared by coast and burn)
        rel = st.com - self.target
        rel_h = rel - (rel @ up) * up
        v_h = st.vel_com - (st.vel_com @ up) * up
        t_go = max(h / max(descent, 1.0) * 1.6, self.min_tgo)
        a_h = -6.0 * (rel_h + v_h * t_go) / (t_go * t_go) + 2.0 * v_h / t_go

        if self.phase == "coast":
            # ignite a little early to absorb throttle lag and the attitude transient
            lead = descent * (0.6 + 3 * eng.throttle_tau)
            wants = descent >= self.v_ref(max(h - lead, 0.0)) or h < 5.0
            if wants and self.allow_ignition and sim.prop_mass > 0:
                self.phase = "burn"
            else:
                # engine-first into the (estimated) airflow, tilted to steer with body lift
                v_air = st.vel_com - self.wind_forecast(st)
                speed = float(np.linalg.norm(v_air))
                base = -v_air / speed if speed > 5.0 else up
                if float(base @ up) < 0.2:  # never point the nose at the ground
                    base = up
                axis, _ = self.allocate(
                    st, st.mass * a_h - self.disturbance, base, math.radians(12.0), 0.0, 0.0
                )
                self.last_axis = axis
                _, rcs = self.att(st, axis, 0.0)
                return 0.0, np.zeros(2), rcs

        # ---- vertical channel: track the stopping profile
        # near the ground the descent rate tapers linearly, whatever the decel profile
        vr = min(self.v_ref(h), self.sink + 0.8 * max(h, 0.0))
        v_h_mag = float(np.linalg.norm(v_h))
        if h < 20.0 and v_h_mag > 1.5:
            # still sliding sideways close to the ground: hover until the drift is killed
            vr = min(vr, max(h - 8.0, 0.0) * 0.3)
        err_v = descent - vr  # >0: falling too fast
        self.integral = float(np.clip(self.integral + err_v * self.dt, -5.0, 5.0))
        # following v_ref exactly needs a constant deceleration a_v (feed-forward)
        ff = self.a_v if descent > self.sink else 0.0
        a_up = g + ff + self.kv * err_v + self.ki * self.integral

        if h < 30.0:  # kill residual drift before touchdown; stop chasing the pad
            w = max(h - 5.0, 0.0) / 25.0
            a_h = w * a_h - 0.6 * v_h
        tilt_cap = self.max_tilt if h > 15.0 else math.radians(12.0)

        # desired non-gravitational force, minus the estimated horizontal disturbance
        d_h = self.disturbance - (self.disturbance @ up) * up
        f_des = st.mass * (a_up * up + a_h) - d_h
        t_min = eng.throttle_min * t_max
        axis, thrust = self.allocate(st, f_des, up, tilt_cap, t_min, eng.throttle_max * t_max)
        throttle = thrust / max(t_max, 1e-9)
        self.last_axis = axis
        gimbal, rcs = self.att(st, axis, max(sim.thrust, 0.5 * thrust))
        return throttle, gimbal, rcs

    def wind_forecast(self, st: State) -> np.ndarray:
        """Mean wind profile (as uplinked before flight); gusts and turbulence are unknown."""
        return self.sim.gravity.local_to_frame(st.com, self.sim.wind.mean_at(st.altitude))

    # ------------------------------------------------------------------ disturbance observer
    def _observe(self, st: State, g_vec: np.ndarray, tau: float = 1.5) -> None:
        """Low-pass estimate of the force the model does not explain (mostly wind)."""
        R = st.rot
        atm = self.sim.atmosphere.at(st.altitude)
        # model of the forces acting now: aero (no wind), actual gimbal thrust direction
        f_a, _, _, _ = self.sim.aero.forces(
            R.T @ (st.vel_com - self.wind_forecast(st)),
            st.omega,
            st.cg_z,
            atm.density,
            atm.speed_of_sound,
        )
        f_model = R @ (f_a + self.sim.thrust * self.sim.engine.direction())
        a_pred = f_model / st.mass + g_vec
        if self._prev is not None and st.legs_down == 0:
            v_prev, a_prev = self._prev
            a_meas = (st.vel_com - v_prev) / self.dt
            k = self.dt / (tau + self.dt)
            self.disturbance += k * (st.mass * (a_meas - a_prev) - self.disturbance)
            n = float(np.linalg.norm(self.disturbance))
            cap = 0.5 * st.mass * 9.81
            if n > cap:
                self.disturbance *= cap / n
        self._prev = (st.vel_com.copy(), a_pred)

    # ------------------------------------------------------------------ allocation
    def predicted_force(
        self,
        st: State,
        axis: np.ndarray,
        side: np.ndarray,
        thrust: float,
        atm,
        info: dict | None = None,
    ):
        """Aerodynamic force plus the gimbal side-force needed to trim the aero torque,
        for the vehicle held at ``axis`` (world frame, zero body rate). ``info['trim']``
        receives the fraction of gimbal authority that trim would use."""
        y_b = np.cross(axis, side)
        R = np.column_stack([side, y_b, axis])  # body -> world
        v_b = R.T @ (st.vel_com - self.wind_forecast(st))
        f_b, tau_b, _, _ = self.sim.aero.forces(
            v_b, np.zeros(3), st.cg_z, atm.density, atm.speed_of_sound
        )
        if thrust <= 0 and info is not None:
            # engine off: the RCS must hold the attitude against the aero torque. Report the
            # fraction of RCS authority needed, scaled so that 25% maps to the 0.6 penalty
            # threshold (holding a large trim would drain the gas)
            cap = self.sim.rcs.torque_cap[:2]
            need = np.abs(tau_b[:2]) / np.maximum(cap, 1e-9)
            info["trim"] = float(need.max()) * 0.6 / 0.25 if self.sim.rcs.enabled else math.inf
        if thrust > 0:
            rz = self.sim.vehicle.engine.gimbal_z - st.cg_z
            if abs(rz) > 1e-6:
                # gimbal torque (0,0,rz) x F = (-rz Fy, rz Fx, 0) must cancel tau_aero
                side_f = np.array([-tau_b[1] / rz, tau_b[0] / rz, 0.0])
                cap = thrust * math.sin(self.sim.engine.gimbal_max)
                n = float(np.linalg.norm(side_f))
                if info is not None:
                    info["trim"] = n / cap if cap > 0 else math.inf
                if n > cap > 0:
                    side_f *= cap / n
                f_b = f_b + side_f
        return R @ f_b

    def allocate(self, st: State, f_des, base, tilt_max, t_min, t_max, n: int = 13):
        """Choose a thrust axis (tilted from ``base`` within the plane of the desired
        horizontal force) and thrust magnitude so that thrust + predicted aero (+ gimbal
        trim side-force) best matches ``f_des``."""
        up = st.up
        if not self.aero_aware:
            axis = limit_tilt(f_des, base, tilt_max) if np.linalg.norm(f_des) > 1e-9 else base
            thrust = float(np.clip(f_des @ axis, t_min, t_max)) if t_max > 0 else 0.0
            return axis, thrust
        f_h = f_des - (f_des @ up) * up
        fh = float(np.linalg.norm(f_h))
        side = f_h / fh if fh > 1e-6 else None
        if side is not None:
            side = side - (side @ base) * base
            sn = float(np.linalg.norm(side))
            side = side / sn if sn > 1e-9 else None
        if side is None:
            thrust = float(np.clip(f_des @ base, t_min, t_max)) if t_max > 0 else 0.0
            return base, thrust
        atm = self.sim.atmosphere.at(st.altitude)
        t_guess = float(np.clip(f_des @ base, t_min, t_max)) if t_max > 0 else 0.0
        reg = (0.01 * st.mass * 9.81) ** 2

        def evaluate(th: float):
            axis = math.cos(th) * base + math.sin(th) * side
            ortho = math.cos(th) * side - math.sin(th) * base
            info = {"trim": 0.0}
            resid = f_des - self.predicted_force(st, axis, ortho, t_guess, atm, info)
            thrust = float(np.clip(resid @ axis, t_min, t_max)) if t_max > 0 else 0.0
            err = resid - thrust * axis
            # throttle owns the vertical channel; the tilt is chosen for the horizontal one
            e_v = float(err @ up)
            e_h = err - e_v * up
            cost = float(e_h @ e_h) + 0.1 * e_v * e_v + reg * (th / tilt_max) ** 2
            # attitudes the gimbal cannot hold against the aero torque are not achievable
            over = max(info["trim"] - 0.6, 0.0)
            cost += (over * st.mass * 9.81) ** 2 * 100.0
            return cost, axis, thrust

        # coarse grid, then golden-section refinement around the best cell
        grid = np.linspace(-tilt_max, tilt_max, n)
        costs = [evaluate(th)[0] for th in grid]
        k = int(np.argmin(costs))
        step = grid[1] - grid[0]
        lo, hi = max(grid[k] - step, -tilt_max), min(grid[k] + step, tilt_max)
        gr = 0.5 * (math.sqrt(5.0) - 1.0)
        a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
        fa, fb = evaluate(a)[0], evaluate(b)[0]
        for _ in range(14):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - gr * (hi - lo)
                fa = evaluate(a)[0]
            else:
                lo, a, fa = a, b, fb
                b = lo + gr * (hi - lo)
                fb = evaluate(b)[0]
        _, axis, thrust = evaluate(0.5 * (lo + hi))
        return axis, thrust
