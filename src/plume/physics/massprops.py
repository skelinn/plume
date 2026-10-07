"""Mass properties of the vehicle as a function of propellant load.

Everything is axisymmetric about the body z axis except an optional non-axisymmetric
dry inertia. Tanks drain from the top: the remaining liquid is a solid cylinder
sitting on the tank bottom, so the CG drops as propellant is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plume.config import VehicleSpec


@dataclass(frozen=True, slots=True)
class MassProps:
    mass: float
    cg_z: float
    inertia: np.ndarray  # principal inertia about the CG, body axes [Ixx, Iyy, Izz]

    @property
    def cg(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.cg_z])


def estimate_dry_inertia(spec: VehicleSpec) -> np.ndarray:
    """Thin-walled cylinder estimate for the dry structure (about its CG)."""
    m = spec.mass.dry
    r = spec.geometry.radius
    L = spec.geometry.length
    i_lat = m * (0.5 * r * r + L * L / 12.0)
    return np.array([i_lat, i_lat, m * r * r])


class MassModel:
    """Fast mass-property evaluation for a given vehicle."""

    def __init__(self, spec: VehicleSpec):
        self.spec = spec
        self.dry_inertia = (
            np.asarray(spec.mass.dry_inertia, dtype=float)
            if spec.mass.dry_inertia is not None
            else estimate_dry_inertia(spec)
        )
        self.tank_capacity = np.array([t.capacity for t in spec.tanks], dtype=float)
        self.tank_zb = np.array([t.z_bottom for t in spec.tanks], dtype=float)
        self.tank_h = np.array([t.z_top - t.z_bottom for t in spec.tanks], dtype=float)
        self.tank_r2 = np.array([t.radius**2 for t in spec.tanks], dtype=float)
        self.rcs_z = spec.rcs.z

    def evaluate(self, tank_masses, rcs_mass: float = 0.0, cargo_mass: float | None = None):
        spec = self.spec
        m_cargo = spec.cargo.mass if cargo_mass is None else cargo_mass
        # accumulate sum(m), sum(m z), sum(m z^2) and own inertias, then shift to the CG
        m_d = spec.mass.dry
        z_d = spec.mass.dry_cg_z
        ix, iy, iz = self.dry_inertia
        m_sum, mz, mz2 = m_d, m_d * z_d, m_d * z_d * z_d
        if m_cargo > 0:
            rc = 0.4 * spec.geometry.diameter
            zc = spec.cargo.cg_z
            il = m_cargo * (rc * rc / 4.0 + rc * rc / 3.0)
            ix += il
            iy += il
            iz += 0.5 * m_cargo * rc * rc
            m_sum += m_cargo
            mz += m_cargo * zc
            mz2 += m_cargo * zc * zc
        for k in range(len(self.tank_capacity)):
            tm = max(float(tank_masses[k]), 0.0)
            if tm <= 0.0:
                continue
            h = self.tank_h[k] * min(tm / self.tank_capacity[k], 1.0)
            zc = self.tank_zb[k] + 0.5 * h
            r2 = self.tank_r2[k]
            il = tm * (r2 / 4.0 + h * h / 12.0)
            ix += il
            iy += il
            iz += 0.5 * tm * r2
            m_sum += tm
            mz += tm * zc
            mz2 += tm * zc * zc
        if rcs_mass > 0:
            m_sum += rcs_mass
            mz += rcs_mass * self.rcs_z
            mz2 += rcs_mass * self.rcs_z * self.rcs_z
        cg = mz / m_sum
        shift = mz2 - m_sum * cg * cg  # sum m (z - cg)^2
        return MassProps(m_sum, cg, np.array([ix + shift, iy + shift, iz]))
