"""The 6-DOF rocket simulator.

``RocketSim`` couples MuJoCo (rigid-body integration with RK4, leg/ground contact)
with Plume's force models. Each physics step:

1. advance engine / RCS state and get mass flow,
2. set the body's mass, CG and inertia to their *mid-step* values (second-order
   accurate mass depletion -- this is what makes Tsiolkovsky come out right),
3. compute gravity (at the mid-step position), thrust, RCS and aerodynamic
   forces/torques about the CG and write them to ``xfrc_applied``,
4. ``mj_step``; then deplete propellant and update bookkeeping.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from plume.config import VehicleSpec, WorldSpec
from plume.constants import G0
from plume.physics.aero import Aero
from plume.physics.atmosphere import Atmosphere
from plume.physics.gravity import gravity_from_world
from plume.physics.massprops import MassModel, MassProps
from plume.physics.mjcf import GroundTile, build_mjcf, leg_angles
from plume.physics.propulsion import RCS, Engine
from plume.physics.recovery import Recovery
from plume.physics.wind import WindModel


# --------------------------------------------------------------------------- helpers
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), *(axis * s)])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_z_axis(axis) -> np.ndarray:
    """Shortest-arc quaternion rotating body +z onto ``axis`` (world)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    c = a[2]
    if c < -0.999999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    v = np.array([-a[1], a[0], 0.0])  # e_z x a
    q = np.array([1.0 + c, *v])
    return q / np.linalg.norm(q)


_ZERO3 = np.zeros(3)


def _cross_z(w: np.ndarray, z: float) -> np.ndarray:
    """w x (0, 0, z)."""
    return np.array([w[1] * z, -w[0] * z, 0.0])


def _z_cross(z: float, f: np.ndarray) -> np.ndarray:
    """(0, 0, z) x f."""
    return np.array([-z * f[1], z * f[0], 0.0])


def _norm(v: np.ndarray) -> float:
    return math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


@dataclass
class Controls:
    throttle: float = 0.0
    gimbal: np.ndarray = field(default_factory=lambda: np.zeros(2))  # normalised [-1, 1]
    rcs: np.ndarray = field(default_factory=lambda: np.zeros(3))  # normalised torque [-1, 1]


@dataclass
class State:
    t: float
    pos: np.ndarray  # body origin (hull base centre), world
    quat: np.ndarray  # [w, x, y, z] body -> world
    vel: np.ndarray  # body-origin velocity, world
    omega: np.ndarray  # body rates, body frame
    com: np.ndarray
    vel_com: np.ndarray
    mass: float
    cg_z: float
    prop_mass: float
    rcs_prop: float
    throttle: float
    thrust: float
    gimbal: np.ndarray
    rcs: np.ndarray
    altitude: float  # CG altitude above the datum (sea level / sphere)
    agl: float  # lowest footpad height above ground
    mach: float
    q_dyn: float
    wind: np.ndarray
    g_load: float  # filtered sensed acceleration at the CG, in g
    cargo_g: float  # filtered sensed acceleration at the cargo, in g
    tilt: float  # angle between body axis and local vertical, rad
    up: np.ndarray  # local vertical
    legs_down: int
    body_contact: bool

    @property
    def rot(self) -> np.ndarray:
        return quat_to_mat(self.quat)

    @property
    def axis(self) -> np.ndarray:
        return self.rot[:, 2]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel_com))

    @property
    def vertical_speed(self) -> float:
        return float(self.vel_com @ self.up)

    @property
    def horizontal_speed(self) -> float:
        v = self.vel_com - (self.vel_com @ self.up) * self.up
        return float(np.linalg.norm(v))


@dataclass
class Touchdown:
    t: float
    vertical_speed: float
    horizontal_speed: float
    tilt: float
    position: np.ndarray


class RocketSim:
    """6-DOF rocket simulation for a single vehicle.

    Parameters
    ----------
    vehicle, world:
        Vehicle and world specifications.
    seed:
        Seed for the wind model.
    tiles:
        Optional heightfield ground tiles (replaces the flat plane).
    ground_height:
        Optional callable ``f(p_world) -> terrain height above datum`` used for AGL
        and crash detection away from tiles. Defaults to the plane at 0.
    """

    def __init__(
        self,
        vehicle: VehicleSpec,
        world: WorldSpec | None = None,
        seed: int | None = None,
        tiles: list[GroundTile] | None = None,
        ground_height: Callable[[np.ndarray], float] | None = None,
    ):
        self.vehicle = vehicle
        self.world = world or WorldSpec()
        self.dt = self.world.dt
        self.gravity = gravity_from_world(self.world)
        self.atmosphere = Atmosphere(self.world.atmosphere, self.world.temperature_offset)
        self.wind = WindModel.from_spec(self.world.wind, seed)
        self.mass_model = MassModel(vehicle)
        self.engine = Engine(vehicle.engine, vehicle.prop_capacity)
        self.tank_init = np.array([t.initial_mass for t in vehicle.tanks], dtype=float)
        nominal = self.mass_model.evaluate(self.tank_init, vehicle.rcs.gas)
        self.rcs = RCS(vehicle.rcs, vehicle, nominal.cg_z)
        self.aero = Aero(vehicle.aero, vehicle.geometry)
        self.recovery = Recovery(vehicle.recovery)
        self.rail: dict | None = None
        self.has_ground = self.world.ground != "none" or bool(tiles)
        self._ground_height = ground_height or (lambda p: 0.0)

        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(vehicle, self.world, tiles, self.mass_model.dry_inertia)
        )
        self.data = mujoco.MjData(self.model)
        for k, tile in enumerate(tiles or []):
            _, _, norm = tile.hfield_params()
            adr = self.model.hfield_adr[k]
            self.model.hfield_data[adr : adr + norm.size] = norm.ravel()
        self.bid = self.model.body("rocket").id
        self.wet_bid = self.model.body("wet").id
        self.dry_z = vehicle.mass.dry_cg_z

        names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.ground_geoms = {i for i, n in enumerate(names) if n.startswith("ground")}
        self.foot_geoms = {i for i, n in enumerate(names) if n.startswith("foot")}
        self.leg_geoms = {i for i, n in enumerate(names) if n.startswith("leg")}
        self.body_geoms = {i for i, n in enumerate(names) if n in {"hull", "engine", "nose"}}
        legs = vehicle.legs
        self.foot_body = np.array(
            [
                [legs.span * math.cos(th), legs.span * math.sin(th), -legs.height]
                for th in leg_angles(legs.count)
            ]
        ).reshape(-1, 3)
        self.cargo_body = np.array([0.0, 0.0, vehicle.cargo.cg_z])
        self._rbuf = np.zeros(9)
        self._state_cache: State | None = None
        self._mp_dirty = True
        self._g_filter_k = 1.0 - math.exp(-self.dt / 0.05)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(
        self,
        pos=(0.0, 0.0, 100.0),
        vel=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        omega=(0.0, 0.0, 0.0),
        prop: float | np.ndarray | None = None,
        rcs_prop: float | None = None,
        cargo_mass: float | None = None,
        seed: int | None = None,
    ) -> State:
        """Place the vehicle. ``vel`` is the velocity of the body origin (world frame)."""
        if prop is None:
            self.tanks = self.tank_init.copy()
        elif np.ndim(prop) == 0:
            cap = self.mass_model.tank_capacity
            self.tanks = (
                cap * min(float(prop) / max(cap.sum(), 1e-12), 1.0) if len(cap) else cap.copy()
            )
        else:
            self.tanks = np.asarray(prop, dtype=float).copy()
        self.rcs_prop = self.vehicle.rcs.gas if rcs_prop is None else float(rcs_prop)
        self.cargo_mass = self.vehicle.cargo.mass if cargo_mass is None else float(cargo_mass)
        self.engine.reset()
        self.rcs.reset()
        self.recovery.reset()
        self.rail = None
        self.wind.reset(seed)
        self.controls = Controls()
        self.t = 0.0
        self.thrust = 0.0
        self.mdot = 0.0
        self.q_dyn = 0.0
        self.mach = 0.0
        self.wind_now = np.zeros(3)
        self.g_load = 0.0
        self.cargo_g = 0.0
        self.max_g = 0.0
        self.max_cargo_g = 0.0
        self.max_q = 0.0
        self.work = {"thrust": 0.0, "aero": 0.0, "rcs": 0.0, "mass_loss_ke": 0.0}
        self.impulse_total = 0.0
        self.prop_used = 0.0
        self.rcs_used = 0.0
        self.touchdown: Touchdown | None = None
        self.airborne = False  # touchdown is only recorded after leaving the ground
        self.legs_down = 0
        self.body_contact = False
        self.ever_body_contact = False
        self._apply_mass(self._mass_props(self.tanks, self.rcs_prop))
        # contact regularisation depends on mass/inertia; mj_setConst also resets qpos
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        q = np.asarray(quat, dtype=float)
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = q / np.linalg.norm(q)
        self.data.qvel[:3] = vel
        self.data.qvel[3:6] = omega
        mujoco.mj_forward(self.model, self.data)
        self._state_cache = None
        self.launch_altitude = self.gravity.altitude(self.com())
        self._prev_v_cg = self._point_velocity(self.mp.cg)
        self._prev_v_cargo = self._point_velocity(self.cargo_body)
        self.airborne = False
        self._contacts()
        return self.state

    def _mass_props(self, tanks, rcs_prop) -> MassProps:
        return self.mass_model.evaluate(tanks, rcs_prop, self.cargo_mass)

    @property
    def mp(self) -> MassProps:
        if self._mp_dirty:
            self._apply_mass(self._mass_props(self.tanks, self.rcs_prop))
        return self._mp

    def _apply_mass(self, mp: MassProps) -> None:
        self._mp = mp
        self._mp_dirty = False
        m = self.model
        # wet part = total - dry (parallel-axis bookkeeping about the respective CGs)
        md = self.vehicle.mass.dry
        mw = mp.mass - md
        if mw > 1e-9:
            zw = (mp.mass * mp.cg_z - md * self.dry_z) / mw
            Id = self.mass_model.dry_inertia
            dd = (self.dry_z - mp.cg_z) ** 2
            dw = (zw - mp.cg_z) ** 2
            lat = mp.inertia[:2] - Id[:2] - md * dd - mw * dw
            iw = (max(lat[0], 1e-9), max(lat[1], 1e-9), max(mp.inertia[2] - Id[2], 1e-9))
        else:
            mw, zw, iw = 1e-9, self.dry_z, (1e-9, 1e-9, 1e-9)
        m.body_mass[self.wet_bid] = mw
        m.body_ipos[self.wet_bid] = (0.0, 0.0, zw)
        m.body_inertia[self.wet_bid] = iw

    # ------------------------------------------------------------------ control
    def set_controls(self, throttle: float = 0.0, gimbal=(0.0, 0.0), rcs=(0.0, 0.0, 0.0)) -> None:
        self.controls = Controls(
            float(throttle), np.asarray(gimbal, dtype=float), np.asarray(rcs, dtype=float)
        )
        self.engine.command(throttle, gimbal)
        self.rcs.command(rcs, self.mp.cg_z)
        self._state_cache = None

    # ------------------------------------------------------------------ kinematics
    @property
    def rot(self) -> np.ndarray:
        mujoco.mju_quat2Mat(self._rbuf, self.data.qpos[3:7])
        return self._rbuf.reshape(3, 3).copy()

    def _point_velocity(self, r_body: np.ndarray) -> np.ndarray:
        R = self.rot
        return self.data.qvel[:3] + R @ np.cross(self.data.qvel[3:6], r_body)

    def com(self) -> np.ndarray:
        return self.data.qpos[:3] + self.rot @ self.mp.cg

    @property
    def prop_mass(self) -> float:
        return float(self.tanks.sum())

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> State:
        for _ in range(n):
            self._physics_step()
        return self.state

    def _physics_step(self) -> None:
        self._state_cache = None
        dt = self.dt
        data = self.data
        R = self.rot
        pos = data.qpos[:3]
        omega_b = data.qvel[3:6].copy()

        # --- atmosphere at current CG
        cg_z0 = self.mp.cg_z
        com = pos + R[:, 2] * cg_z0
        alt = self.gravity.altitude(com)
        atm = self.atmosphere.at(alt)

        # --- propulsion (advances engine/RCS state, returns this step's flows)
        prop = self.prop_mass
        thrust, mdot_e = self.engine.update(dt, atm.pressure, prop)
        mdot_r_est = self.rcs.mdot(self.rcs_prop, dt)

        # --- mid-step mass properties
        share = self.tanks / prop if prop > 0 else self.tanks * 0.0
        tanks_mid = self.tanks - (0.5 * dt * mdot_e) * share
        mp = self._mass_props(tanks_mid, self.rcs_prop - 0.5 * dt * mdot_r_est)
        self._apply_mass(mp)
        cgz = mp.cg_z
        com = pos + R[:, 2] * cgz
        v_com = data.qvel[:3] + R @ _cross_z(omega_b, cgz)

        # --- forces in body frame, torques about the CG
        f_t, tau_t = _ZERO3, _ZERO3
        if thrust > 0:
            f_t = thrust * self.engine.direction()
            tau_t = _z_cross(self.vehicle.engine.gimbal_z - cgz, f_t)
        f_r, tau_r, mdot_r = self.rcs.update(cgz, self.rcs_prop, dt)

        wind = self.wind.at(alt)
        if self.gravity.curved and wind.any():
            wind = self.gravity.local_to_frame(com, wind)
        self.wind_now = wind
        v_air_w = v_com - wind
        f_a, tau_a, q, mach = self.aero.forces(
            R.T @ v_air_w, omega_b, cgz, atm.density, atm.speed_of_sound
        )
        self.q_dyn, self.mach = q, mach

        g = self.gravity.accel(com + (0.5 * dt) * v_com)
        f_w = R @ (f_t + f_r + f_a) + mp.mass * g
        f_chute = None
        if self.recovery.enabled:
            up = self.gravity.up(com)
            self.recovery.update(self.t, alt - self.launch_altitude, float(v_com @ up))
            cda = self.recovery.cd_area(self.t)
            if cda > 0:
                va = float(np.linalg.norm(v_air_w))
                f_chute = -0.5 * atm.density * cda * va * v_air_w
                f_w = f_w + f_chute
        # xfrc_applied acts at the dry body's CG: shift the torque from the total CG
        tau_w = R @ (tau_t + tau_r + tau_a + _z_cross(cgz - self.dry_z, R.T @ f_w))
        xfrc = data.xfrc_applied[self.bid]
        xfrc[:3] = f_w
        xfrc[3:] = tau_w

        # --- bookkeeping (power at the start of the step x dt)
        work = self.work
        if thrust > 0:
            work["thrust"] += (float(f_t @ (R.T @ v_com)) + float(tau_t @ omega_b)) * dt
        if self.aero.enabled:
            work["aero"] += (float(f_a @ (R.T @ v_air_w)) + float(tau_a @ omega_b)) * dt
        if f_chute is not None:
            work["aero"] += float(f_chute @ v_air_w) * dt
        if mdot_r > 0:
            work["rcs"] += (float(f_r @ (R.T @ v_com)) + float(tau_r @ omega_b)) * dt

        mujoco.mj_step(self.model, data)
        if self.rail is not None:
            self._apply_rail()
        self.t += dt
        self.wind.step(dt)

        # --- deplete propellant
        dm_e = mdot_e * dt
        dm_r = mdot_r * dt
        if dm_e > 0:
            self.tanks = np.maximum(self.tanks - dm_e * share, 0.0)
        self.rcs_prop = max(self.rcs_prop - dm_r, 0.0)
        self.prop_used += dm_e
        self.rcs_used += dm_r
        self.impulse_total += thrust * dt
        self.thrust = thrust
        self.mdot = mdot_e
        R1 = self.rot
        w1 = data.qvel[3:6]
        v_new = data.qvel[:3] + R1 @ _cross_z(w1, cgz)
        if dm_e + dm_r > 0:
            v_mid = 0.5 * (v_com + v_new)
            work["mass_loss_ke"] -= 0.5 * float(v_mid @ v_mid) * (dm_e + dm_r)

        # --- sensed acceleration (g-load) at CG and cargo, first-order filtered
        v_cargo = data.qvel[:3] + R1 @ _cross_z(w1, self.cargo_body[2])
        inv_dt = 1.0 / dt
        a_cg = (v_new - self._prev_v_cg) * inv_dt - g
        a_cargo = (v_cargo - self._prev_v_cargo) * inv_dt - g
        self._prev_v_cg = v_new
        self._prev_v_cargo = v_cargo
        k = self._g_filter_k
        self.g_load += k * (_norm(a_cg) / G0 - self.g_load)
        self.cargo_g += k * (_norm(a_cargo) / G0 - self.cargo_g)
        if self.g_load > self.max_g:
            self.max_g = self.g_load
        if self.cargo_g > self.max_cargo_g:
            self.max_cargo_g = self.cargo_g
        if q > self.max_q:
            self.max_q = q

        self._mp_dirty = True
        self._state_cache = None
        if data.ncon or self.legs_down or self.body_contact or not self.airborne:
            self._contacts()

    # ------------------------------------------------------------------ launch rail
    def set_rail(self, length: float, direction=None) -> None:
        """Constrain the vehicle to slide along a launch rail from its current pose."""
        d = self.rot[:, 2] if direction is None else np.asarray(direction, dtype=float)
        self.rail = {
            "start": self.data.qpos[:3].copy(),
            "dir": d / np.linalg.norm(d),
            "quat": self.data.qpos[3:7].copy(),
            "length": float(length),
        }

    def _apply_rail(self) -> None:
        r = self.rail
        data = self.data
        disp = float((data.qpos[:3] - r["start"]) @ r["dir"])
        if disp >= r["length"]:
            self.rail = None  # left the rail
            return
        v_along = float(data.qvel[:3] @ r["dir"])
        if disp <= 0.0:
            disp, v_along = 0.0, max(v_along, 0.0)
        data.qpos[:3] = r["start"] + disp * r["dir"]
        data.qpos[3:7] = r["quat"]
        data.qvel[:3] = v_along * r["dir"]
        data.qvel[3:6] = 0.0

    def _contacts(self) -> None:
        feet = set()
        body = False
        data = self.data
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.ground_geoms:
                other = g2
            elif g2 in self.ground_geoms:
                other = g1
            else:
                continue
            if other in self.foot_geoms or other in self.leg_geoms:
                feet.add(other)
            elif other in self.body_geoms:
                body = True
        self.legs_down = len(feet & self.foot_geoms)
        self.body_contact = body
        self.ever_body_contact |= body
        if not (feet or body):
            self.airborne = True
        elif self.airborne and self.touchdown is None:
            st = self.state
            self.touchdown = Touchdown(
                self.t, -st.vertical_speed, st.horizontal_speed, st.tilt, st.pos.copy()
            )

    # ------------------------------------------------------------------ observers
    def agl(self) -> float:
        """Height of the lowest footpad (or hull base) above the ground."""
        R = self.rot
        pos = self.data.qpos[:3]
        pts = self.foot_body if len(self.foot_body) else np.zeros((1, 3))
        best = math.inf
        for pb in pts:
            p = pos + R @ pb
            best = min(best, self.gravity.altitude(p) - self._ground_height(p))
        return best

    @property
    def state(self) -> State:
        """Current state (cached until the next physics step)."""
        if self._state_cache is not None:
            return self._state_cache
        self._state_cache = st = self._build_state()
        return st

    def _build_state(self) -> State:
        data = self.data
        R = self.rot
        com = data.qpos[:3] + R @ self.mp.cg
        v_com = data.qvel[:3] + R @ _cross_z(data.qvel[3:6], self.mp.cg_z)
        up = self.gravity.up(com)
        tilt = math.acos(max(-1.0, min(1.0, float(R[:, 2] @ up))))
        return State(
            t=self.t,
            pos=data.qpos[:3].copy(),
            quat=data.qpos[3:7].copy(),
            vel=data.qvel[:3].copy(),
            omega=data.qvel[3:6].copy(),
            com=com,
            vel_com=v_com,
            mass=self.mp.mass,
            cg_z=self.mp.cg_z,
            prop_mass=self.prop_mass,
            rcs_prop=self.rcs_prop,
            throttle=self.engine.throttle,
            thrust=self.thrust,
            gimbal=self.engine.gimbal.copy(),
            rcs=self.rcs.cmd.copy(),
            altitude=self.gravity.altitude(com),
            agl=self.agl(),
            mach=self.mach,
            q_dyn=self.q_dyn,
            wind=self.wind_now.copy(),
            g_load=self.g_load,
            cargo_g=self.cargo_g,
            tilt=tilt,
            up=up,
            legs_down=self.legs_down,
            body_contact=self.body_contact,
        )

    def energy(self) -> dict[str, float]:
        """Mechanical energy of the vehicle (J)."""
        st = self.state
        I = self.mp.inertia
        w = st.omega
        ke_t = 0.5 * self.mp.mass * float(st.vel_com @ st.vel_com)
        ke_r = 0.5 * float(w @ (I * w))
        pe = self.mp.mass * self.gravity.potential(st.com)
        return {
            "kinetic": ke_t + ke_r,
            "translational": ke_t,
            "rotational": ke_r,
            "potential": pe,
            "total": ke_t + ke_r + pe,
        }

    def angular_momentum(self) -> np.ndarray:
        """Angular momentum about the CG, world frame."""
        st = self.state
        return st.rot @ (self.mp.inertia * st.omega)

    def frame(self, phase: str | None = None) -> dict:
        """A replay frame for :class:`plume.recording.Recorder`."""
        st = self.state
        f = {
            "t": st.t,
            "pos": st.pos,
            "quat": st.quat,
            "vel": st.vel_com,
            "omega": st.omega,
            "alt": st.agl,
            "throttle": st.throttle,
            "thrust": st.thrust,
            "gimbal": st.gimbal,
            "rcs": st.rcs,
            "prop_mass": st.prop_mass,
            "mass": st.mass,
            "g_load": st.g_load,
            "mach": st.mach,
            "q_dyn": st.q_dyn,
            "wind": st.wind,
        }
        if phase is not None:
            f["phase"] = phase
        return f

    def replay_meta(self, title: str, **extra) -> dict:
        v = self.vehicle
        scene = extra.pop("scene", None) or {
            "frame": "spherical" if self.gravity.curved else "flat",
            "earth_radius": self.gravity.earth_radius,
            "ground": {"type": "plane" if self.has_ground else "none"},
            "pads": [],
            "target": None,
        }
        return {
            "title": title,
            "source": "sim",
            "vehicle": {
                "name": v.name,
                "length": v.geometry.length,
                "diameter": v.geometry.diameter,
                "nose_length": v.geometry.nose_length,
                "legs": {
                    "count": v.legs.count,
                    "span": v.legs.span,
                    "height": v.legs.height,
                    "attach_z": v.legs.attach_z,
                },
                "engine": {
                    "nozzle_radius": v.engine.nozzle_radius,
                    "gimbal_z": v.engine.gimbal_z,
                    "thrust_max": self.engine.max_thrust(0.0),
                },
                "rcs_z": v.rcs.z,
                "cargo_mass": self.cargo_mass,
                "dry_mass": v.mass.dry,
                "prop_mass_initial": float(self.tank_init.sum()),
            },
            "scene": scene,
            **extra,
        }
