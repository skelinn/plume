"""Tether (rope / cable) between a fixed anchor and a point on the vehicle.

Used for tethered hover tests of a hop rig: a rope from a ground anchor below the rig
limits how far it can climb or drift, and a rope from a crane above catches it if the
engine quits. The rope is a spring-damper that only pulls:

    s     = |p - a| - L                          stretch (p: attach point, a: anchor)
    s_dot = d(|p - a|)/dt = u . v_p              u = (p - a) / |p - a|
    T     = max(k s + c s_dot, 0)   if s > 0,    else 0 (slack)
    F     = -T u  on the vehicle (toward the anchor), applied at the attach point

``k`` is the rope's axial stiffness (EA / L for a rope of length L, cross-section A and
modulus E) and ``c`` a viscous damping coefficient. The force is installed with
:meth:`Tether.install`, which appends it to ``RocketSim.extra_forces``; the simulator then
calls it with ``(sim, v_air_w, atm, R, omega_b, mp)`` and expects ``(f_world, tau_body)``
with the torque about the current CG in body axes. In high fidelity the force is
re-evaluated at every RK4 stage, so the undamped spring conserves energy to integrator
accuracy (``tests/test_tether.py``).

Not modelled: rope mass and sag, wave propagation along the rope, bending or friction at
the anchor (a pulley), and the rope snagging on the vehicle. ``breaking_load`` only
flags an overload (the rope keeps pulling); the test checks it against the rated load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TetherSpec:
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world frame, m
    attach: tuple[float, float, float] = (0.0, 0.0, 0.0)  # body frame (z from hull base), m
    length: float = 3.0  # unstretched length, m
    stiffness: float = 2.0e4  # k, N/m
    damping: float = 1.5e3  # c, N s/m
    breaking_load: float | None = None  # N (flag only)

    def __post_init__(self) -> None:
        if self.length <= 0 or self.stiffness <= 0 or self.damping < 0:
            raise ValueError("tether needs length > 0, stiffness > 0 and damping >= 0")

    @staticmethod
    def critical_damping(stiffness: float, mass: float, ratio: float = 1.0) -> float:
        """Damping coefficient for a damping ratio on a mass hanging from the rope."""
        return 2.0 * ratio * math.sqrt(stiffness * mass)


class Tether:
    """Tension-only spring-damper rope, usable as a ``RocketSim.extra_forces`` entry."""

    def __init__(self, spec: TetherSpec | None = None, **kw):
        self.spec = spec or TetherSpec(**kw)
        self.anchor = np.asarray(self.spec.anchor, dtype=float)
        self.attach = np.asarray(self.spec.attach, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.tension = 0.0  # N, at the last evaluation
        self.max_tension = 0.0
        self.stretch = 0.0  # m (negative: slack)
        self.overloaded = False

    def install(self, sim) -> Tether:
        sim.extra_forces.append(self)
        return self

    def remove(self, sim) -> None:
        if self in sim.extra_forces:
            sim.extra_forces.remove(self)

    # ------------------------------------------------------------------ kinematics
    def attach_point(self, sim, R: np.ndarray | None = None) -> np.ndarray:
        R = sim.rot if R is None else R
        return sim.data.qpos[:3] + R @ self.attach

    def geometry(self, p: np.ndarray, v_p: np.ndarray) -> tuple[float, float, np.ndarray]:
        """(stretch, stretch rate, unit vector anchor -> attach point)."""
        d = p - self.anchor
        dist = float(np.linalg.norm(d))
        u = d / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0])
        return dist - self.spec.length, float(u @ v_p), u

    def tension_for(self, stretch: float, rate: float) -> float:
        if stretch <= 0.0:
            return 0.0
        return max(self.spec.stiffness * stretch + self.spec.damping * rate, 0.0)

    def elastic_energy(self, sim) -> float:
        """Spring energy stored in the rope at the simulator's current state (J)."""
        s = float(np.linalg.norm(self.attach_point(sim) - self.anchor)) - self.spec.length
        return 0.5 * self.spec.stiffness * s * s if s > 0 else 0.0

    # ------------------------------------------------------------------ force model
    def __call__(self, sim, v_air_w, atm, R, omega_b, mp):
        # sim.data holds the state being evaluated (the RK4 stage state in high fidelity)
        pos = sim.data.qpos[:3]
        vel = sim.data.qvel[:3]  # body-origin velocity, world frame
        p = pos + R @ self.attach
        v_p = vel + R @ np.cross(omega_b, self.attach)
        if not getattr(sim, "_stage_forces", False):
            # fast fidelity holds forces over the step: evaluate the stretch at the
            # predicted mid-step position (as the simulator does for gravity), which
            # keeps the stiff spring second-order accurate instead of pumping energy in
            p = p + 0.5 * sim.dt * v_p
        stretch, rate, u = self.geometry(p, v_p)
        T = self.tension_for(stretch, rate)
        self.stretch = stretch
        self.tension = T
        if T > self.max_tension:
            self.max_tension = T
        if T <= 0.0:
            return np.zeros(3), np.zeros(3)
        bl = self.spec.breaking_load
        if bl is not None and T > bl:
            self.overloaded = True
        f_w = -T * u
        r_b = self.attach - np.array([0.0, 0.0, mp.cg_z])
        tau_b = np.cross(r_b, R.T @ f_w)
        return f_w, tau_b
