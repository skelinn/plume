"""Propellant slosh: first lateral mode of each tank as an equivalent spring-mass.

For an upright cylindrical tank of radius R filled to height h under an axial
(settling) acceleration a, the first antisymmetric slosh mode (Abramson, NASA SP-106,
1966; Ibrahim, *Liquid Sloshing Dynamics*, 2005) has

    xi = 1.8412                                   (first root of J1')
    omega^2 = (a xi / R) tanh(xi h / R)
    m1 / m_liquid = 2 tanh(xi h / R) / (xi (xi^2 - 1) h / R)

with the sloshing mass m1 placed (R / xi) tanh(xi h / 2R) below the free surface. The
rest of the propellant stays in the rigid mass model. Relative to the tank wall the slosh
mass obeys

    x'' = -omega^2 x - 2 zeta omega x' - a_lat

where a_lat is the lateral specific force of the tank at the slosh-mass height (body
axes, including the centripetal term of the body rotation). Because the rigid model
already carries m1 with the tank, the vehicle feels only the reaction of the relative
motion, F = -m1 x'', applied at the slosh-mass height. Damping zeta ~ 0.005-0.02 for a
smooth wall, 0.03-0.1 with ring baffles.

Validity: small amplitude (|x| < R/2 enforced), settled propellant (axial acceleration
above ``min_settling``; below it, e.g. in ballistic coast, the model is frozen - the
propellant is assumed held by a management device), first mode only.
"""

from __future__ import annotations

import math

import numpy as np

XI1 = 1.8412


def slosh_parameters(m_liquid: float, radius: float, fill_height: float, a_axial: float):
    """(slosh mass kg, natural frequency rad/s, depth of the slosh mass below the free
    surface m) of the first lateral mode."""
    if m_liquid <= 0 or fill_height <= 0 or a_axial <= 0:
        return 0.0, 0.0, 0.0
    k = XI1 * fill_height / radius
    th = math.tanh(k)
    m1 = m_liquid * 2.0 * th / (XI1 * (XI1 * XI1 - 1.0) * fill_height / radius)
    m1 = min(m1, m_liquid)
    w = math.sqrt(a_axial * XI1 / radius * th)
    depth = radius / XI1 * math.tanh(0.5 * k)
    return m1, w, depth


class Slosh:
    def __init__(self, vehicle, min_settling: float = 0.5):
        self.idx = [i for i, t in enumerate(vehicle.tanks) if getattr(t, "slosh", False)]
        self.enabled = bool(self.idx)
        self.R = np.array([vehicle.tanks[i].radius for i in self.idx])
        self.zb = np.array([vehicle.tanks[i].z_bottom for i in self.idx])
        self.H = np.array([vehicle.tanks[i].z_top - vehicle.tanks[i].z_bottom for i in self.idx])
        self.cap = np.array([vehicle.tanks[i].capacity for i in self.idx])
        self.zeta = np.array([vehicle.tanks[i].slosh_damping for i in self.idx])
        self.min_settling = min_settling
        self.reset()

    def reset(self) -> None:
        n = len(self.idx)
        self.x = np.zeros((n, 2))
        self.v = np.zeros((n, 2))
        self.m1 = np.zeros(n)

    def step(self, dt: float, tank_masses, f_spec_b: np.ndarray, omega_b: np.ndarray, cg_z: float):
        """Advance the slosh states; returns (force, torque about the CG), body axes."""
        force = np.zeros(3)
        torque = np.zeros(3)
        a_ax = float(f_spec_b[2])  # settling acceleration toward the tank bottoms
        for j, i in enumerate(self.idx):
            m_l = float(tank_masses[i])
            h = min(m_l / self.cap[j], 1.0) * self.H[j]
            if a_ax < self.min_settling or m_l <= 1e-3:
                self.x[j] *= 0.0  # unsettled: frozen (propellant management device)
                self.v[j] *= 0.0
                self.m1[j] = 0.0
                continue
            m1, w, depth = slosh_parameters(m_l, self.R[j], h, a_ax)
            self.m1[j] = m1
            z1 = self.zb[j] + h - depth
            r = np.array([0.0, 0.0, z1 - cg_z])
            a_c = f_spec_b + np.cross(omega_b, np.cross(omega_b, r))
            acc = -w * w * self.x[j] - 2.0 * self.zeta[j] * w * self.v[j] - a_c[:2]
            self.v[j] = self.v[j] + acc * dt
            self.x[j] = self.x[j] + self.v[j] * dt
            lim = 0.5 * self.R[j]
            n = float(np.linalg.norm(self.x[j]))
            if n > lim:
                self.x[j] *= lim / n
            f = np.array([-m1 * acc[0], -m1 * acc[1], 0.0])
            force += f
            torque += np.cross(r, f)
        return force, torque
