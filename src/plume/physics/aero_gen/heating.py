"""Stagnation-point convective heating (Sutton & Graves) and integrated heat load.

    q_dot = k sqrt(rho / R_n) V^3          [W/m^2]

with k = 1.7415e-4 kg^0.5/m for Earth air (Sutton & Graves, "A general
stagnation-point convective-heating equation for arbitrary gas mixtures", NASA TR
R-376, 1971; the value 1.7415e-4 SI is the commonly quoted conversion of their
Earth constant).  Assumptions: continuum, laminar, equilibrium boundary layer, cold
(non-catalytic effects ignored) wall, hypersonic speed; accuracy ~+-10-15 % in the
validity range (V >~ 2-3 km/s); below ~1 km/s it is only an order-of-magnitude
estimate (use it for trends / Monte Carlo margins, not TPS sizing).

Nose radius: pointed noses need an effective bluntness radius; by default
``nose_radius_from_geometry`` uses 5 % of the body diameter (configurable).  For
engine-first flight the stagnation region is the base / engine bay: pass an
effective radius of that blunt face (a flat face behaves like a sphere of roughly
2-3 base radii -- an assumption to be refined with data).
"""

from __future__ import annotations

import numpy as np

SUTTON_GRAVES_K_EARTH = 1.7415e-4


def stagnation_heat_flux(rho, V, nose_radius, k: float = SUTTON_GRAVES_K_EARTH):
    """Convective stagnation-point heat flux, W/m^2 (vectorised)."""
    rho = np.maximum(np.asarray(rho, dtype=float), 0.0)
    rn = np.asarray(nose_radius, dtype=float)
    if np.any(rn <= 0):
        raise ValueError("nose radius must be > 0")
    out = k * np.sqrt(rho / rn) * np.abs(np.asarray(V, dtype=float)) ** 3
    return float(out) if np.ndim(out) == 0 else out


def heat_load(t, rho, V, nose_radius, k: float = SUTTON_GRAVES_K_EARTH) -> float:
    """Integrated stagnation-point heat load (J/m^2) over a trajectory (trapezoid)."""
    t = np.asarray(t, dtype=float)
    q = stagnation_heat_flux(rho, V, nose_radius, k)
    q = np.asarray(q, dtype=float)
    return float(np.sum(0.5 * (q[1:] + q[:-1]) * np.diff(t)))


class HeatingMonitor:
    """Running peak heat flux and heat load (call ``update`` every step)."""

    def __init__(self, nose_radius: float, k: float = SUTTON_GRAVES_K_EARTH):
        self.nose_radius = nose_radius
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.q_peak = 0.0
        self.load = 0.0
        self._last = None

    def update(self, dt: float, rho: float, V: float) -> float:
        q = stagnation_heat_flux(rho, V, self.nose_radius, self.k)
        if self._last is not None:
            self.load += 0.5 * (q + self._last) * dt
        self._last = q
        self.q_peak = max(self.q_peak, q)
        return q


def nose_radius_from_geometry(geometry, bluntness_ratio: float = 0.05) -> float:
    """Effective stagnation radius: ``nose_tip_radius`` if the geometry has one,
    otherwise ``bluntness_ratio`` x diameter (flat face: the body radius)."""
    tip = getattr(geometry, "nose_tip_radius", None)
    if tip:
        return float(tip)
    if getattr(geometry, "nose_length", 0.0) <= 0:
        return 0.5 * float(geometry.diameter)
    return bluntness_ratio * float(geometry.diameter)
