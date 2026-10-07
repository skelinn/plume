"""U.S. Standard Atmosphere 1976 (0–86 km), with an exponential tail above."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from plume.constants import G0, GAMMA_AIR, P0, R_AIR, R_EARTH

# (base geopotential altitude m, base temperature K, lapse rate K/m)
_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.001),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.002),
)
_TOP_GEOPOT = 84_852.0  # geopotential altitude of 86 km geometric


def _layer_base_pressures() -> list[float]:
    p = [P0]
    for (h0, t0, lapse), (h1, _, _) in itertools.pairwise(_LAYERS):
        if lapse == 0.0:
            p.append(p[-1] * math.exp(-G0 * (h1 - h0) / (R_AIR * t0)))
        else:
            t1 = t0 + lapse * (h1 - h0)
            p.append(p[-1] * (t1 / t0) ** (-G0 / (R_AIR * lapse)))
    return p


_P_BASE = _layer_base_pressures()


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m^3
    speed_of_sound: float  # m/s


def _standard(h_geopot: float) -> tuple[float, float]:
    idx = 0
    for k, layer in enumerate(_LAYERS):
        if h_geopot >= layer[0]:
            idx = k
    h0, t0, lapse = _LAYERS[idx]
    p0 = _P_BASE[idx]
    dh = h_geopot - h0
    if lapse == 0.0:
        return t0, p0 * math.exp(-G0 * dh / (R_AIR * t0))
    t = t0 + lapse * dh
    return t, p0 * (t / t0) ** (-G0 / (R_AIR * lapse))


_T_TOP, _P_TOP = _standard(_TOP_GEOPOT)
_SCALE_HEIGHT_TOP = R_AIR * _T_TOP / G0


class Atmosphere:
    """Standard atmosphere, optionally with temperature offset and an "off" switch for tests."""

    def __init__(self, enabled: bool = True, temperature_offset: float = 0.0):
        self.enabled = enabled
        self.temperature_offset = temperature_offset

    def at(self, altitude: float) -> AtmosphereState:
        """Atmospheric state at a geometric altitude above sea level (m)."""
        if not self.enabled:
            return AtmosphereState(_T_TOP, 0.0, 0.0, math.sqrt(GAMMA_AIR * R_AIR * _T_TOP))
        z = max(altitude, -1000.0)
        h = R_EARTH * z / (R_EARTH + z)  # geopotential altitude
        if h <= _TOP_GEOPOT:
            t, p = _standard(h)
        else:
            t = _T_TOP
            p = _P_TOP * math.exp(-(h - _TOP_GEOPOT) / _SCALE_HEIGHT_TOP)
        t_eff = t + self.temperature_offset
        rho = p / (R_AIR * t_eff)
        return AtmosphereState(t_eff, p, rho, math.sqrt(GAMMA_AIR * R_AIR * t_eff))

    def density(self, altitude: float) -> float:
        return self.at(altitude).density

    def pressure(self, altitude: float) -> float:
        return self.at(altitude).pressure
