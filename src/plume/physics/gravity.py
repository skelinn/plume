"""Gravity models.

Both models work in a local ENU frame whose origin is on the surface at the launch
site. ``SphericalGravity`` puts the Earth's centre at (0, 0, -R) so long hops see
inverse-square gravity, a curved horizon and a rotating local vertical.
"""

from __future__ import annotations

import math

import numpy as np

from plume.constants import G0, MU_EARTH, R_EARTH


class FlatGravity:
    """Uniform gravity, -g0 along z. Altitude is simply z."""

    curved = False
    earth_radius = None

    def __init__(self, g: float = G0):
        self.g = g
        self._vec = np.array([0.0, 0.0, -g])

    def accel(self, p: np.ndarray) -> np.ndarray:
        return self._vec

    def potential(self, p: np.ndarray) -> float:
        """Specific potential energy (J/kg)."""
        return self.g * float(p[2])

    def altitude(self, p: np.ndarray) -> float:
        return float(p[2])

    def up(self, p: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


class SphericalGravity:
    """Point-mass Earth (non-rotating) centred at (0, 0, -R)."""

    curved = True

    def __init__(self, mu: float = MU_EARTH, radius: float = R_EARTH):
        self.mu = mu
        self.earth_radius = radius
        self.center = np.array([0.0, 0.0, -radius])

    def accel(self, p: np.ndarray) -> np.ndarray:
        r = p - self.center
        rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        return -self.mu / rn**3 * r

    def potential(self, p: np.ndarray) -> float:
        return -self.mu / float(np.linalg.norm(p - self.center))

    def altitude(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p - self.center)) - self.earth_radius

    def up(self, p: np.ndarray) -> np.ndarray:
        r = p - self.center
        return r / np.linalg.norm(r)

    # --- map coordinates (azimuthal-equidistant about the launch site) ---------
    def surface_point(self, u: float, v: float, h: float = 0.0) -> np.ndarray:
        """Map coords (east/north arc length, m) + height above the sphere -> frame position."""
        n = self.surface_normal(u, v)
        return self.center + (self.earth_radius + h) * n

    def surface_normal(self, u: float, v: float) -> np.ndarray:
        s = math.hypot(u, v)
        if s < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        th = s / self.earth_radius
        return np.array([math.sin(th) * u / s, math.sin(th) * v / s, math.cos(th)])

    def map_coords(self, p: np.ndarray) -> tuple[float, float, float]:
        """Inverse of ``surface_point``: frame position -> (u, v, height)."""
        r = p - self.center
        rn = float(np.linalg.norm(r))
        n = r / rn
        th = math.acos(max(-1.0, min(1.0, n[2])))
        sh = math.hypot(n[0], n[1])
        if sh < 1e-15:
            return 0.0, 0.0, rn - self.earth_radius
        s = th * self.earth_radius
        return s * n[0] / sh, s * n[1] / sh, rn - self.earth_radius


def gravity_from_world(world) -> FlatGravity | SphericalGravity:
    if world.gravity == "flat":
        return FlatGravity(world.g)
    return SphericalGravity()


def make_gravity(model: str = "flat", **kwargs):
    if model == "flat":
        return FlatGravity(**kwargs)
    if model == "spherical":
        return SphericalGravity(**kwargs)
    raise ValueError(f"unknown gravity model {model!r}")
