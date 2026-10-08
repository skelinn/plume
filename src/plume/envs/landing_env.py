"""Gymnasium environment for powered vertical landing on a pad (``Plume/Landing-v0``).

Observation (23 floats)
    symlog(position of CG relative to the landed CG point) [3], symlog(velocity) [3],
    body z axis in world [3], body x axis in world [3], body rates [3],
    propellant fraction, RCS gas fraction, actual throttle, gimbal angles [2]
    (normalised), symlog(height above ground), forecast mean wind at the current
    altitude [2] (gusts and turbulence are not observed).
Action (``action_mode``)
    ``guidance`` (default, 3 floats in [-1, 1]): throttle (mapped to [0, 1]; below
    ~0.2 the engine shuts down) and the desired thrust-axis tilt east/north
    (fraction of ``max_tilt_deg``). The same :class:`AttitudeController` used by the
    PID baseline turns the axis into gimbal + RCS commands, so PID and RL differ only
    in their guidance.
    ``direct`` (6 floats): throttle, gimbal [2], RCS torque [3] (fractions of the
    per-axis capability) -- the agent must also learn attitude control.

The episode ends on success (all criteria held through ``settle_time`` after
touchdown), crash (hull contact, leg over-speed, loss of control), leaving the
operating box, or the time limit. The engine is cut automatically at touchdown.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from plume.config import LandingEnvSpec, StageSpec, load_landing_env, load_vehicle
from plume.physics.sim import RocketSim, quat_from_axis_angle, quat_from_z_axis, quat_mul
from plume.physics.wind import WindModel
from plume.recording import Recorder

OBS_DIM = 23


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class LandingEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        spec: LandingEnvSpec | str | None = None,
        stage: int = 0,
        record: bool = False,
        fixed_stage: bool = False,
        fidelity: str | None = None,
    ):
        super().__init__()
        self.spec_cfg = (
            spec if isinstance(spec, LandingEnvSpec) else load_landing_env(spec or "landing")
        )
        cfg = self.spec_cfg
        self.vehicle = load_vehicle(cfg.vehicle)
        update = {"dt": cfg.physics_dt}
        if fidelity is not None:  # evaluation at high fidelity (training stays fast)
            update["fidelity"] = fidelity
        world = cfg.world.model_copy(update=update)
        self.sim = RocketSim(self.vehicle, world)
        self.n_sub = max(1, round(cfg.control_dt / cfg.physics_dt))
        self.stages = cfg.curriculum.stages
        self.stage = int(stage)
        self.fixed_stage = fixed_stage
        self.record = record
        self.pad = np.zeros(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_mode = cfg.action_mode
        n_act = 3 if self.action_mode == "guidance" else 6
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32)
        self.tan_tilt = math.tan(math.radians(cfg.max_tilt_deg))
        from plume.control.attitude import AttitudeController

        self.attitude = AttitudeController(self.sim)
        self.recorder: Recorder | None = None
        self.last_replay: dict | None = None
        # reference deceleration for the velocity-profile potential
        e = self.sim.engine
        m_land = (
            self.vehicle.mass.dry + self.vehicle.rcs.propellant + 0.3 * self.vehicle.prop_capacity
        )
        self.a_ref = max(0.5 * (e.max_thrust(101_325.0) / m_land - 9.81), 0.5)

    # ------------------------------------------------------------------ curriculum
    def set_stage(self, stage: int) -> None:
        self.stage = int(np.clip(stage, 0, len(self.stages) - 1))

    def get_stage(self) -> int:
        return self.stage

    def _sample_stage(self) -> int:
        frac = self.spec_cfg.curriculum.replay_fraction
        if not self.fixed_stage and self.stage > 0 and self.np_random.random() < frac:
            return int(self.np_random.integers(0, self.stage))
        return self.stage

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        stage_idx = options.get("stage", self._sample_stage())
        st_spec: StageSpec = self.stages[stage_idx]
        rng = self.np_random

        def u(rng_range):
            lo, hi = rng_range
            return float(rng.uniform(lo, hi)) if hi > lo else float(lo)

        # wind for this episode
        w_seed = int(rng.integers(0, 2**31 - 1))
        self.sim.wind = WindModel(
            speed=u(st_spec.wind_speed),
            from_deg=float(rng.uniform(0, 360)),
            turbulence=st_spec.turbulence,
            gust_rate=st_spec.gust_rate if st_spec.gust_max > 0 else 0.0,
            gust_max=st_spec.gust_max,
            seed=w_seed,
        )

        # initial state
        alt = u(st_spec.altitude)
        off = u(st_spec.offset)
        az = rng.uniform(0, 2 * math.pi)
        hs = u(st_spec.horizontal_speed)
        haz = rng.uniform(0, 2 * math.pi)
        # horizontal velocity = air mass motion (forecast wind) + a random deviation
        w0 = self.sim.wind.mean_at(alt)
        vel = np.array(
            [w0[0] + hs * math.cos(haz), w0[1] + hs * math.sin(haz), -u(st_spec.descent_speed)]
        )
        speed = float(np.linalg.norm(vel))
        # engine-first into the airflow at speed, upright when slow
        up = np.array([0.0, 0.0, 1.0])
        blend = min(max((speed - 5.0) / 25.0, 0.0), 1.0) if vel[2] < 0 else 0.0
        nominal_axis = blend * (-vel / max(speed, 1e-9)) + (1 - blend) * up
        q = quat_from_z_axis(nominal_axis)
        tilt = math.radians(u(st_spec.tilt_deg))
        tax = rng.uniform(0, 2 * math.pi)
        q = quat_mul(quat_from_axis_angle([math.cos(tax), math.sin(tax), 0.0], tilt), q)
        rate = math.radians(u(st_spec.rate_deg_s))
        rdir = rng.normal(size=3)
        omega = rate * rdir / (np.linalg.norm(rdir) + 1e-12)
        legs = self.vehicle.legs
        # the descent is assumed to have been targeted with the wind forecast (aim upwind
        # by the expected drift), so ``offset`` is the residual targeting error
        drift = self._forecast_drift(alt, -vel[2])
        start_h = np.array([off * math.cos(az), off * math.sin(az)]) - drift
        pos = np.array([start_h[0], start_h[1], alt + legs.height])
        prop = u(st_spec.prop_fraction) * self.vehicle.prop_capacity
        self.sim.reset(pos=pos, vel=vel, quat=q, omega=omega, prop=prop, seed=w_seed)

        self.episode_stage = stage_idx
        self.t_touchdown: float | None = None
        self.time_limit = min(self.spec_cfg.max_time, 20.0 + alt / 8.0)
        self.initial_offset = off
        self.initial_distance = float(np.hypot(*start_h))
        self.initial_alt = alt
        self.prop_start = self.sim.prop_mass
        self.done_reason = ""
        self._phi = self._potential(self.sim.state)
        if self.record:
            self.recorder = Recorder(
                self.sim.replay_meta(
                    f"Landing ({st_spec.name})",
                    controller=options.get("controller", "agent"),
                    seed=seed,
                    scene={
                        "frame": "flat",
                        "ground": {"type": "plane"},
                        "pads": [
                            {
                                "name": "LZ",
                                "pos": self.pad.tolist(),
                                "radius": self.spec_cfg.pad_radius,
                            }
                        ],
                        "target": {"pos": self.pad.tolist(), "radius": self.spec_cfg.pad_radius},
                    },
                )
            )
            self.recorder.record(self.sim.frame("descent"))
        return self._obs(), self._info()

    def _forecast_drift(self, alt: float, descent: float, n: int = 40) -> np.ndarray:
        """Horizontal drift from the mean wind during a descent from ``alt``: the vehicle
        is assumed to move with the air, falling at ``descent`` m/s (floored at 30 m/s
        to account for the slow final burn)."""
        if self.sim.wind.speed <= 0 or alt <= 0:
            return np.zeros(2)
        hs = np.linspace(0.0, alt, n + 1)
        mids = 0.5 * (hs[1:] + hs[:-1])
        dt = (alt / n) / max(descent, 30.0)
        return sum(self.sim.wind.mean_at(h)[:2] for h in mids) * dt

    # ------------------------------------------------------------------ observation
    def _landed_cg(self, st) -> np.ndarray:
        return self.pad + np.array([0.0, 0.0, self.vehicle.legs.height + st.cg_z])

    def _obs(self) -> np.ndarray:
        st = self.sim.state
        R = st.rot
        e = self.sim.engine
        gmax = max(e.gimbal_max, 1e-9)
        obs = np.concatenate(
            [
                symlog(st.com - self._landed_cg(st)),
                symlog(st.vel_com),
                R[:, 2],
                R[:, 0],
                st.omega,
                [
                    st.prop_mass / max(self.vehicle.prop_capacity, 1e-9),
                    st.rcs_prop / max(self.vehicle.rcs.propellant, 1e-9),
                    st.throttle,
                    st.gimbal[0] / gmax,
                    st.gimbal[1] / gmax,
                    float(symlog(max(st.agl, 0.0))),
                ],
                self.sim.wind.mean_at(st.altitude)[:2] / 10.0,  # wind forecast
            ]
        )
        return obs.astype(np.float32)

    def reference_velocity(self, st) -> np.ndarray:
        """Reference descent: toward the pad horizontally; vertically a constant-
        deceleration stopping profile (like the PID's hoverslam) tapering near the ground."""
        rel = st.com - self._landed_cg(st)
        h = max(st.agl, 0.0)
        v_down = min(0.8 + 0.8 * h, math.sqrt(0.64 + 2.0 * self.a_ref * h))
        v_h = -0.1 * rel[:2]
        n = float(np.linalg.norm(v_h))
        if n > 15.0:
            v_h *= 15.0 / n
        return np.array([v_h[0], v_h[1], -v_down])

    def velocity_error(self, st) -> float:
        return min(float(np.linalg.norm(st.vel_com - self.reference_velocity(st))), 200.0)

    def _potential(self, st) -> float:
        rw = self.spec_cfg.reward
        rel = st.com - self._landed_cg(st)
        d_h = float(np.hypot(rel[0], rel[1]))
        return -(
            rw.w_distance * min(d_h, 1000.0) / 50.0
            + rw.w_velocity * self.velocity_error(st) / 20.0
            + rw.w_tilt * st.tilt
            + rw.w_rate * min(float(np.linalg.norm(st.omega)), 5.0)
        )

    def _info(self) -> dict[str, Any]:
        sim = self.sim
        st = sim.state
        td = sim.touchdown
        return {
            "stage": self.episode_stage,
            "t": sim.t,
            "fuel_used": self.prop_start - sim.prop_mass,
            "landing_error": float(np.hypot(*(st.pos[:2] - self.pad[:2]))),
            "touchdown_vz": td.vertical_speed if td else None,
            "touchdown_vh": td.horizontal_speed if td else None,
            "tilt_deg": math.degrees(st.tilt),
            "success": False,
            "reason": self.done_reason,
        }

    # ------------------------------------------------------------------ step
    def axis_from_action(self, a1: float, a2: float) -> np.ndarray:
        v = np.array([a1 * self.tan_tilt, a2 * self.tan_tilt, 1.0])
        return v / np.linalg.norm(v)

    def action_from_axis(self, throttle: float, axis: np.ndarray) -> np.ndarray:
        """Inverse of the guidance-mode mapping (used to drive the env with the autopilot)."""
        z = max(float(axis[2]), 1e-3)
        a1 = float(np.clip(axis[0] / z / self.tan_tilt, -1, 1))
        a2 = float(np.clip(axis[1] / z / self.tan_tilt, -1, 1))
        a0 = 2.0 * throttle - 1.0 if throttle > 0 else -1.0
        return np.array([a0, a1, a2], dtype=np.float32)

    def apply_action(self, action: np.ndarray) -> None:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        throttle = 0.5 * (a[0] + 1.0)
        if throttle < 0.2 or self.t_touchdown is not None:
            throttle = 0.0  # engine off / auto cut-off at touchdown
        if self.action_mode == "direct":
            self.sim.set_controls(throttle, a[1:3], a[3:6])
            return
        st = self.sim.state
        axis = self.axis_from_action(a[1], a[2]) if self.t_touchdown is None else st.up
        p_amb = self.sim.atmosphere.at(st.altitude).pressure
        thrust_est = max(self.sim.thrust, 0.5 * throttle * self.sim.engine.max_thrust(p_amb))
        gimbal, rcs = self.attitude(st, axis, thrust_est)
        self.sim.set_controls(throttle, gimbal, rcs)

    def step(self, action):
        cfg = self.spec_cfg
        rw = cfg.reward
        sim = self.sim
        prop0 = sim.prop_mass
        self.apply_action(action)
        sim.step(self.n_sub)
        st = sim.state
        if self.t_touchdown is None and sim.touchdown is not None:
            self.t_touchdown = sim.t
            sim.set_controls(0.0, (0, 0), sim.controls.rcs)

        phi = self._potential(st)
        reward = phi - self._phi
        self._phi = phi
        reward -= rw.fuel_weight * (prop0 - sim.prop_mass)
        if self.t_touchdown is None:  # dense tracking of the reference descent + time cost
            reward -= (rw.track_weight * self.velocity_error(st) + rw.time_weight) * cfg.control_dt
        reward -= rw.rcs_weight * float(np.abs(sim.controls.rcs).sum()) / 3.0

        terminated = False
        truncated = False
        success = False
        td = sim.touchdown
        sc = cfg.success
        d_pad = float(np.hypot(*(st.pos[:2] - self.pad[:2])))
        if sim.ever_body_contact:
            terminated, self.done_reason = True, "crash_hull"
        elif td is not None and td.vertical_speed > self.vehicle.legs.max_touchdown_speed:
            terminated, self.done_reason = True, "crash_legs"
        elif st.tilt > math.radians(100):
            terminated, self.done_reason = True, "lost_control"
        if terminated:
            impact = td.vertical_speed if td else st.speed
            reward -= rw.crash_penalty + rw.crash_speed_penalty * min(
                max(impact - sc.max_vertical_speed, 0.0), 50.0
            )
        elif self.t_touchdown is not None and sim.t - self.t_touchdown >= sc.settle_time:
            terminated = True
            checks = {
                "missed": d_pad <= cfg.pad_radius,
                "hard_vertical": td.vertical_speed <= sc.max_vertical_speed,
                "hard_horizontal": td.horizontal_speed <= sc.max_horizontal_speed,
                "tilted": st.tilt <= math.radians(sc.max_tilt_deg),
                "not_at_rest": st.speed < 0.5 and st.agl < 0.3,
            }
            failed = [k for k, passed in checks.items() if not passed]
            ok = not failed
            if ok:
                success = True
                self.done_reason = "landed"
                reward += rw.success_bonus + rw.accuracy_bonus * (1.0 - d_pad / cfg.pad_radius)
            else:
                self.done_reason = failed[0]
                # partial credit for an upright, gentle touchdown, fading with distance
                gentle = max(0.0, 1.0 - max(td.vertical_speed - sc.max_vertical_speed, 0.0) / 3.0)
                upright = 1.0 if st.tilt <= math.radians(sc.max_tilt_deg) else 0.0
                near = max(0.0, 1.0 - d_pad / (5.0 * cfg.pad_radius))
                reward += rw.partial_landing * gentle * upright * (0.5 + 0.5 * near)
        elif (
            np.hypot(*(st.com[:2] - self.pad[:2])) > max(1000.0, 4.0 * self.initial_distance)
            or st.agl > self.initial_alt + 500.0
        ):
            terminated, self.done_reason = True, "out_of_bounds"
            reward -= rw.fail_penalty
        elif sim.t >= self.time_limit:
            # running out of time is a task failure, not an arbitrary truncation
            terminated, self.done_reason = True, "timeout"
            reward -= rw.fail_penalty

        if self.recorder is not None:
            phase = (
                "landed" if self.t_touchdown is not None else ("burn" if st.thrust > 0 else "coast")
            )
            self.recorder.record(self.sim.frame(phase))
            if terminated or truncated:
                info = self._info()
                self.recorder.set_outcome(
                    success,
                    self.done_reason,
                    {
                        "landing_error_m": info["landing_error"],
                        "fuel_used_kg": info["fuel_used"],
                        "touchdown_vz_mps": info["touchdown_vz"] if td else float("nan"),
                        "touchdown_vh_mps": info["touchdown_vh"] if td else float("nan"),
                        "max_g": sim.max_g,
                        "flight_time_s": sim.t,
                    },
                )
                if td is not None:
                    self.recorder.event(td.t, "touchdown", "Touchdown")
                self.last_replay = self.recorder.to_dict()

        info = self._info()
        info["success"] = success
        return self._obs(), float(reward), terminated, truncated, info


class AutopilotPolicy:
    """Wraps :class:`LandingAutopilot` as an action-producing policy for a LandingEnv."""

    def __init__(self, env: LandingEnv):
        from plume.control.autopilot import LandingAutopilot

        self.env = env
        self.ap = LandingAutopilot(env.sim, env.pad, control_dt=env.spec_cfg.control_dt)

    def reset(self) -> None:
        self.ap.reset()

    def __call__(self, obs=None) -> np.ndarray:
        throttle, gimbal, rcs = self.ap.act(self.env.sim.state)
        if self.env.action_mode == "guidance":
            return self.env.action_from_axis(throttle, self.ap.last_axis)
        a0 = 2.0 * throttle - 1.0 if throttle > 0 else -1.0
        return np.array([a0, *gimbal, *rcs], dtype=np.float32)
