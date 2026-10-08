"""Convex powered-descent guidance for the landing burn, wrapped for
:class:`plume.control.autopilot.LandingAutopilot`.

* **Ignition** is timed from feasibility: every ``check_dt`` the minimum-fuel / minimum-miss
  problem (:mod:`plume.control.pdg`) is solved from the state predicted one ignition lead
  (ignition delay + throttle lag) and one check interval ahead, with only
  ``ignition_margin`` of the acceleration limit. The engine is lit when waiting one more
  interval would make that plan infeasible or move its landing point away from the pad;
  the remaining acceleration is tracking margin.
* **Burn:** the plan is re-solved from the current state at ``replan_dt`` (receding
  horizon) and tracked with feed-forward thrust + aero acceleration and a PD correction
  on the plan's position and velocity, through the autopilot's aero-aware thrust/tilt
  allocation and attitude control.
* **Drag:** the aerodynamic acceleration along the previous iterate is part of the plan
  (successive convexification) - including the retro-propulsion plume shielding the
  forebody, which removes most of the hull drag as soon as the engine burns.
* **Fallbacks:** if no plan is feasible even at the full acceleration limit, or the solver
  fails, the autopilot reverts to the hoverslam profile for the rest of the burn.
* **Terminal gate:** the plan ends ``gate_height`` above the pad, descending at
  ``gate_speed``; from there the hoverslam's terminal logic (drift kill, soft touchdown)
  lands. A fuel-optimal profile ends at maximum thrust, which leaves no margin for the
  last metres.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from plume.constants import G0
from plume.control.pdg import PDGConfig, PDGSolution, PoweredDescentGuidance

# (altitude array, air-relative velocity (N,3) in the pad frame, mass array, thrust array)
#   -> aerodynamic acceleration (N,3) in the pad frame
AeroAccel = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def axial_drag_model(pm) -> Callable:
    """Axial (engine-first) drag model from a :class:`PointMassSim` - the flight
    software's nominal aerodynamics, grid-fin drag area and in-flight drag scale - with the
    supersonic-retropropulsion reduction of the hull drag when the aero model has one."""
    from plume.physics.aerodb import srp_axial_factor

    aero, atmo = pm.aero, pm.atmosphere
    decay = getattr(aero, "srp_ct_decay", None)

    def drag(alt: float, v_air: np.ndarray, m: float, thrust: float) -> np.ndarray:
        sp = float(np.linalg.norm(v_air))
        atm = atmo.at(float(alt))
        if sp < 1e-3 or atm.density <= 0:
            return np.zeros(3)
        q = 0.5 * atm.density * sp * sp
        ca = aero.axial_coefficient(sp / atm.speed_of_sound, False) * aero.ref_area
        if decay is not None and thrust > 0:
            ca *= srp_axial_factor(thrust / (q * aero.ref_area), decay)
        d = q * (ca + pm.extra_cda) * pm.drag_scale
        return -d / max(m, 1.0) * v_air / sp

    return drag


class ConvexLanding:
    def __init__(
        self,
        ap,
        accel_limit: float,
        ignition_margin: float = 0.8,
        glide_slope_deg: float = 15.0,
        max_tilt_deg: float = 25.0,
        hard_accel_limit: float | None = None,
        drag_model: Callable | None = None,
        replan_dt: float = 0.5,
        tilt_rate_deg_s: float = 10.0,
        check_dt: float = 0.2,
        gate_height: float = 10.0,
        gate_speed: float = 4.0,
        reserve_kg: float = 10.0,
        n: int = 20,
        kp: float = 0.5,
        kv: float = 1.4,
    ):
        from plume.physics.propulsion import Engine

        self.ap = ap
        sim = ap.sim
        self.sim = sim
        self.accel_limit = accel_limit
        self.hard_accel_limit = hard_accel_limit or accel_limit
        self.margin = ignition_margin
        self.replan_dt = replan_dt
        self.check_dt = check_dt
        self.gate_height = gate_height
        self.reserve = reserve_kg
        self.kp, self.kv = kp, kv
        nom = sim.nominal
        self.engine = Engine(nom.engine, nom.prop_capacity, high_fidelity=sim.engine.high_fidelity)
        if drag_model is None:
            from plume.physics.pointmass import PointMassSim

            pm = PointMassSim(nom, sim.world)
            gf = nom.grid_fins
            if gf is not None:
                area = gf.area if gf.area is not None else gf.span * gf.chord
                pm.extra_cda = gf.count * area * gf.cd0
            drag_model = axial_drag_model(pm)
        self.drag = drag_model
        self.cfg = PDGConfig(
            thrust_min=nom.engine.throttle_min * self.engine.max_thrust(0.0),
            thrust_max=nom.engine.throttle_max * self.engine.max_thrust(0.0),
            isp=nom.engine.isp_vac,
            m_dry=1.0,
            accel_max=accel_limit,
            tilt_max=math.radians(max_tilt_deg),
            glide_slope=math.radians(glide_slope_deg),
            h_final=gate_height,
            jerk_max=accel_limit * math.radians(tilt_rate_deg_s),
            v_final=np.array([0.0, 0.0, -gate_speed]),
            n=n,
        )
        self._solvers: dict[float, PoweredDescentGuidance] = {}
        self.reset()
        target = ap.target
        up = sim.gravity.up(target)
        e = np.array([1.0, 0.0, 0.0]) - up[0] * up
        if float(np.linalg.norm(e)) < 1e-6:
            e = np.array([0.0, 1.0, 0.0]) - up[1] * up
        ex = e / np.linalg.norm(e)
        self.B = np.column_stack([ex, np.cross(up, ex), up])  # pad frame -> world
        self.g = self.B.T @ sim.gravity.accel(target)

    def reset(self) -> None:
        self.plan: PDGSolution | None = None
        self.plan_t0 = 0.0
        self.active = False  # burning under convex guidance
        self.failed = False  # fell back to the hoverslam
        self._next_check = -math.inf
        self.log: list[dict] = []  # planning events, for diagnostics

    # ------------------------------------------------------------------ helpers
    def _solver(self, accel: float) -> PoweredDescentGuidance:
        key = round(accel, 3)
        if key not in self._solvers:
            cfg = PDGConfig(**{**self.cfg.__dict__, "accel_max": accel})
            self._solvers[key] = PoweredDescentGuidance(cfg)
        return self._solvers[key]

    def local_state(self, st) -> tuple[np.ndarray, np.ndarray]:
        rel = self.B.T @ (st.com - self.ap.target)
        r = np.array([rel[0], rel[1], st.agl])
        return r, self.B.T @ st.vel_com

    def _ground_alt(self, st) -> float:
        return st.altitude - st.agl

    def _aero_fn(self, st):
        sim, B, alt0 = self.sim, self.B, self._ground_alt(st)
        target = self.ap.target

        def fn(r, v, m, thrust):
            out = np.zeros_like(v)
            for k in range(len(v)):
                alt = alt0 + float(r[k, 2])
                wind = B.T @ sim.gravity.local_to_frame(target, sim.wind.mean_at(alt))
                out[k] = self.drag(alt, v[k] - wind, float(m[k]), float(thrust[k]))
            return out

        return fn

    def _coast(self, st, r, v, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Predict an unpowered fall (gravity + drag) ``dt`` ahead."""
        fn = self._aero_fn(st)
        h = 0.05
        n = max(1, round(dt / h))
        h = dt / n
        m = np.array([st.mass])
        for _ in range(n):
            a = fn(r[None, :], v[None, :], m, np.zeros(1))[0] + self.g
            v = v + a * h
            r = r + v * h
        return r, v

    def _bounds(self, st) -> None:
        cfg = self.cfg
        p = self.sim.atmosphere.at(st.altitude).pressure
        cfg.thrust_max = self.sim.nominal.engine.throttle_max * self.engine.max_thrust(p)
        cfg.thrust_min = self.sim.nominal.engine.throttle_min * self.engine.max_thrust(p)
        cfg.mdot_offset = p * self.engine.exit_area / (cfg.isp * G0)
        cfg.m_dry = st.mass - self.sim.prop_mass + self.reserve

    def solve(self, st, r, v, accel: float, tf_range=None, guess=None) -> PDGSolution | None:
        self._bounds(st)
        pdg = self._solver(accel)
        for k in ("thrust_min", "thrust_max", "mdot_offset", "m_dry"):
            setattr(pdg.cfg, k, getattr(self.cfg, k))
        # the engine-first body axis is the thrust axis once lit
        pdg.axis0 = self.B.T @ st.axis
        if tf_range is None:
            sp = float(np.linalg.norm(v))
            a_net = max(accel - float(np.linalg.norm(self.g)), 1.0)
            t_est = sp / a_net
            tf_range = (max(0.6 * t_est, 1.0), 3.0 * t_est + 6.0)
        try:
            return pdg.solve_free_tf(
                r, v, st.mass, self.g, tf_range, self._aero_fn(st), iterations=2, guess=guess
            )
        except Exception:  # numerical trouble in the solver: treat as infeasible
            return None

    # ------------------------------------------------------------------ ignition
    def wants_ignition(self, st) -> bool:
        """Called every control step while coasting with ignition allowed."""
        t = self.sim.t
        if t < self._next_check:
            return False
        r, v = self.local_state(st)
        descent = -float(v[2])
        if descent <= 0:
            return False
        a_ign = self.margin * self.accel_limit
        a_net = max(a_ign - float(np.linalg.norm(self.g)), 1.0)
        h_stop = descent * descent / (2.0 * a_net)
        if r[2] > 2.5 * h_stop + 500.0:  # far above the burn: don't spend solver time
            self._next_check = t + self.check_dt
            return False
        self._next_check = t + self.check_dt
        eng = self.sim.nominal.engine
        lead = (eng.ignition_delay_s if self.sim.engine.high_fidelity else 0.0) + 2.0 * (
            eng.throttle_tau
        )
        r1, v1 = self._coast(st, r, v, lead + self.check_dt)
        if r1[2] <= self.gate_height + 2.0:
            return True
        nxt = self.solve(st, r1, v1, a_ign)
        if nxt is None:
            self.log.append({"t": t, "ev": "ignite", "why": "infeasible"})
            return True
        if nxt.miss > 1.0:
            r0, v0 = self._coast(st, r, v, lead)
            now = self.solve(st, r0, v0, a_ign)
            if now is None or now.miss + 1.0 < nxt.miss:
                self.log.append({"t": t, "ev": "ignite", "why": f"miss {nxt.miss:.1f}"})
                return True
        return False

    def ignite(self, st) -> None:
        self.active = True
        self.plan = None
        self._replan(st)

    # ------------------------------------------------------------------ burn
    def _replan(self, st) -> None:
        t = self.sim.t
        r, v = self.local_state(st)
        guess = self.plan
        rng = None
        if guess is not None:
            left = max(guess.tf - (t - self.plan_t0), 1.0)
            rng = (max(0.6 * left, 0.8), 1.4 * left + 1.5)
        sol = self.solve(st, r, v, self.accel_limit, rng, guess)
        if sol is None:
            sol = self.solve(st, r, v, self.hard_accel_limit)
        if sol is None:
            self.failed = True
            self.active = False
            self.log.append({"t": t, "ev": "fallback"})
            return
        self.plan, self.plan_t0 = sol, t
        self.log.append(
            {"t": t, "ev": "plan", "tf": sol.tf, "fuel": sol.fuel, "miss": sol.miss, "h": r[2]}
        )

    def command(self, st) -> np.ndarray | None:
        """Desired non-gravitational force (world frame) for the allocator, or None when
        the hoverslam should take over (terminal phase or no feasible plan)."""
        if not self.active:
            return None
        t = self.sim.t
        r, v = self.local_state(st)
        if r[2] < self.gate_height + 1.0:
            self.active = False
            self.log.append({"t": t, "ev": "terminal", "h": r[2], "vz": v[2]})
            return None
        if self.plan is None or t - self.plan_t0 >= self.replan_dt:
            self._replan(st)
            if not self.active:
                return None
        sol = self.plan
        tau = t - self.plan_t0
        if tau > sol.tf - 0.2:
            self.active = False
            self.log.append({"t": t, "ev": "terminal", "h": r[2], "vz": v[2]})
            return None
        r_p, v_p, u_p, a_p = sol.at(tau)
        a_cmd = u_p + a_p + self.kp * (r_p - r) + self.kv * (v_p - v)
        return st.mass * (self.B @ a_cmd)
