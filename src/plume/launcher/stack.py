"""Stacked flight and stage separation.

Before separation the whole stack flies as **one rigid body** in a single
:class:`~plume.physics.sim.RocketSim`: the active (lowest) stage provides the engine,
tanks, RCS, legs, grid fins and aerodynamic model, and everything above it (upper stages
with their unburnt propellant, payload, fairing) is folded into the dry mass with the
correct combined CG and inertia. The geometry is the full stack, so the aerodynamics see
the whole vehicle.

At separation each part becomes its own ``RocketSim``. The new simulators start from the
parent's rigid-body state (velocity field ``v + omega x r``), so mass, linear and angular
momentum carry over exactly; then equal and opposite spring impulses push the parts apart.
Each simulator advances in lock-step with the others from then on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from plume.config import VehicleSpec
from plume.launcher.spec import LauncherSpec
from plume.physics.massprops import MassModel
from plume.physics.sim import RocketSim, quat_mul

# --------------------------------------------------------------------------- mass properties
Part = tuple[float, float, np.ndarray]  # (mass, cg_z, [Ixx, Iyy, Izz] about own CG)


def combine(parts: list[Part]) -> Part:
    """Combine axisymmetric parts on the body z axis (parallel-axis theorem)."""
    m = sum(p[0] for p in parts)
    if m <= 0:
        raise ValueError("no mass to combine")
    z = sum(p[0] * p[1] for p in parts) / m
    inertia = np.zeros(3)
    for mi, zi, ii in parts:
        d2 = (zi - z) ** 2
        inertia += np.asarray(ii, dtype=float) + np.array([mi * d2, mi * d2, 0.0])
    return m, z, inertia


def shell_part(mass: float, radius: float, length: float, z_base: float) -> Part:
    """Thin-walled shell of revolution (a fairing): CG at 40 % of its length."""
    i_lat = mass * (0.5 * radius * radius + length * length / 12.0)
    return mass, z_base + 0.4 * length, np.array([i_lat, i_lat, mass * radius * radius])


def _dry_part(v: VehicleSpec) -> Part:
    return v.mass.dry, v.mass.dry_cg_z, MassModel(v).dry_inertia.copy()


def _loaded_part(v: VehicleSpec, z_shift: float) -> Part:
    """A full stage (initial propellant, RCS gas, cargo) as rigid dead mass."""
    mm = MassModel(v)
    tanks = np.array([t.initial_mass for t in v.tanks], dtype=float)
    mp = mm.evaluate(tanks, v.rcs.gas, v.cargo.mass)
    return mp.mass, mp.cg_z + z_shift, mp.inertia.copy()


def _with_dry(v: VehicleSpec, part: Part, **geometry) -> VehicleSpec:
    m, z, inertia = part
    data = v.model_dump()
    data["mass"] = {"dry": m, "dry_cg_z": z, "dry_inertia": [float(x) for x in inertia]}
    data["geometry"].update(geometry)
    return VehicleSpec.model_validate(data)


def stage_vehicle(launcher: LauncherSpec, k: int, fairing: bool = False) -> VehicleSpec:
    """Stage ``k`` flying alone. The last stage carries the payload as cargo and, with
    ``fairing``, the fairing (added to the dry mass, drawn and flown as its nose)."""
    stage = launcher.stages[k]
    v = stage.vehicle
    if k < len(launcher.stages) - 1:
        return v
    g = v.geometry
    pay = launcher.payload
    data = v.model_dump()
    data["cargo"] = {
        "mass": pay.mass,
        "cg_z": g.length + pay.cg_z,
        "max_mass": None,
    }
    v = VehicleSpec.model_validate(data)
    parts = [_dry_part(v)]
    f = launcher.fairing
    if fairing and f is not None:
        d = f.diameter or g.diameter
        parts.append(shell_part(f.mass, 0.5 * d, f.length, g.length))
        return _with_dry(
            v,
            combine(parts),
            length=g.length + f.length,
            nose_length=f.nose_length,
            diameter=max(d, g.diameter),
        )
    return _with_dry(v, combine(parts), length=g.length + pay.length, nose_length=pay.length)


def stack_vehicle(launcher: LauncherSpec, k: int = 0, fairing: bool = True) -> VehicleSpec:
    """The stack from stage ``k`` up, as one rigid vehicle in stage ``k``'s body frame.

    Stage ``k`` is active (engine, tanks, RCS, legs, fins, aero options); the stages above
    it, the payload and (with ``fairing``) the fairing are folded into the dry mass."""
    last = len(launcher.stages) - 1
    if k == last:
        return stage_vehicle(launcher, k, fairing)
    base = launcher.stage_base(k)
    active = launcher.stages[k].vehicle
    parts = [_dry_part(active)]
    for j in range(k + 1, last + 1):
        # the payload rides on the last stage as cargo; the fairing is added separately
        vj = stage_vehicle(launcher, j, fairing=False)
        parts.append(_loaded_part(vj, launcher.stage_base(j) - base))
    top = launcher.top - base
    f = launcher.fairing
    dia = max(s.vehicle.geometry.diameter for s in launcher.stages[k:])
    if fairing and f is not None:
        d = f.diameter or launcher.stages[last].vehicle.geometry.diameter
        parts.append(shell_part(f.mass, 0.5 * d, f.length, top))
        length, nose = top + f.length, f.nose_length
        dia = max(dia, d)
    else:
        length, nose = top + launcher.payload.length, launcher.payload.length
    data = active.model_dump()
    data["name"] = f"{launcher.name}_{launcher.stages[k].name}_stack"
    data["cargo"] = {"mass": 0.0, "cg_z": 0.0, "max_mass": None}
    v = VehicleSpec.model_validate(data)
    return _with_dry(v, combine(parts), length=length, nose_length=nose, diameter=dia)


# --------------------------------------------------------------------------- separation
def _refresh(sim: RocketSim) -> None:
    """After editing qpos/qvel directly: recompute kinematics and reset the derived
    caches (state cache, g-load differentiators) so no artificial g spike appears."""
    mujoco.mj_forward(sim.model, sim.data)
    sim._state_cache = None
    sim._prev_v_cg = sim._point_velocity(sim.mp.cg)
    sim._prev_v_cargo = sim._point_velocity(sim.cargo_body)


def spawn_from(
    parent: RocketSim,
    child: RocketSim,
    z_offset: float,
    tanks=None,
    rcs_prop: float | None = None,
    seed: int | None = None,
) -> None:
    """Place ``child`` (a fresh simulator) where its part of the rigid ``parent`` is.

    ``z_offset`` is the child's hull base in the parent's body frame. The child gets the
    parent's attitude and body rates and the parent's rigid-body velocity at its origin,
    so its momentum is exactly its share of the parent's."""
    st = parent.state
    R = st.rot
    off = np.array([0.0, 0.0, z_offset])
    pos = st.pos + R @ off
    vel = parent.data.qvel[:3] + R @ np.cross(st.omega, off)
    child.reset(
        pos=pos,
        vel=vel,
        quat=st.quat,
        omega=st.omega,
        prop=tanks,
        rcs_prop=rcs_prop,
        seed=seed,
    )
    child.t = parent.t
    child.airborne = True
    child.g_load = parent.g_load
    child.cargo_g = parent.cargo_g
    _refresh(child)


def separation_impulse(lower: RocketSim, upper: RocketSim, delta_v: float) -> float:
    """Spring separation along the stack axis: equal and opposite impulses giving a
    relative speed ``delta_v``. Returns the impulse (N s)."""
    ml, mu = lower.mp.mass, upper.mp.mass
    axis = lower.state.axis
    impulse = delta_v * ml * mu / (ml + mu)
    lower.data.qvel[:3] -= axis * impulse / ml
    upper.data.qvel[:3] += axis * impulse / mu
    _refresh(lower)
    _refresh(upper)
    return impulse


def momentum(sim: RocketSim) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, linear momentum, angular momentum about the world origin), world frame."""
    st = sim.state
    m = sim.mp.mass
    p = m * st.vel_com
    h = np.cross(st.com, p) + sim.angular_momentum()
    return m, p, h


# --------------------------------------------------------------------------- fairing halves
@dataclass
class FairingHalf:
    """A jettisoned fairing half: ballistic CG plus a constant tumble rate.

    Jettison happens above ~100 km, where the aerodynamic load on a half is negligible
    for the tens of seconds it is tracked (it is drawn in replays, not used for analysis).
    """

    pos: np.ndarray  # hull-base point of the half (the fairing base), world
    vel: np.ndarray
    quat: np.ndarray
    omega_b: np.ndarray  # body rates (constant)
    t: float
    log: list = field(default_factory=list)

    def step(self, dt: float, gravity) -> None:
        def acc(p, v):
            a = gravity.accel(p)
            if getattr(gravity, "rotating", False):
                a = a + gravity.fictitious_accel(p, v)
            return a

        a1 = acc(self.pos, self.vel)
        p2 = self.pos + 0.5 * dt * self.vel
        v2 = self.vel + 0.5 * dt * a1
        a2 = acc(p2, v2)
        self.pos = self.pos + dt * v2
        self.vel = self.vel + dt * a2
        w = self.omega_b
        n = float(np.linalg.norm(w))
        if n > 0:
            ang = n * dt
            dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * w / n)])
            self.quat = quat_mul(self.quat, dq)
            self.quat /= np.linalg.norm(self.quat)
        self.t += dt


def jettison_fairing(sim: RocketSim, z_base: float, speed: float, rate_deg_s: float):
    """Two halves leaving the stage ``sim`` sideways (body +x / -x), tumbling outward.

    Half ``b`` is drawn rotated 180 deg about the vehicle axis, so the two halves have
    the same shape in their own frames. Returns the two :class:`FairingHalf` objects."""
    st = sim.state
    R = st.rot
    off = np.array([0.0, 0.0, z_base])
    base = st.pos + R @ off
    v0 = sim.data.qvel[:3] + R @ np.cross(st.omega, off)
    halves = []
    rate = math.radians(rate_deg_s)
    flip = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about body z
    for sign, q in ((1.0, st.quat.copy()), (-1.0, quat_mul(st.quat, flip))):
        out = R[:, 0] * sign
        # tumble outward: the tip leans away from the axis -> rotation about the body y of
        # the half (its own +x points outward)
        halves.append(
            FairingHalf(
                pos=base.copy(),
                vel=v0 + speed * out,
                quat=q,
                omega_b=np.array([0.0, rate, 0.0]) + st.omega,
                t=sim.t,
            )
        )
    return halves
