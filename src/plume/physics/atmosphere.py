"""Atmosphere models.

* :class:`Atmosphere` - U.S. Standard Atmosphere 1976: the 7-layer hydrostatic model to
  86 km and the published US76 tables from 86 to 1000 km (monotone log-cubic
  interpolation), exponential extrapolation above.
* :class:`MsisAtmosphere` - NRLMSISE-00 (date, location, solar and geomagnetic activity);
  needs the ``hifi`` extra (``nrlmsise00``).
* :class:`SoundingAtmosphere` - a measured or forecast temperature/pressure profile
  (radiosonde), blended into US76 above the top of the sounding.

See docs/models/atmosphere.md for equations, sources and validity ranges.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume.constants import G0, GAMMA_AIR, P0, R_AIR

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
_R0 = 6_356_766.0  # US76 effective Earth radius for the geopotential conversion


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

# U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562), Table I: geometric altitude km,
# kinetic temperature K, pressure Pa, density kg/m^3. Above 86 km the air is no longer
# well mixed and US76 is defined by species diffusion; the published values are used.
_UPPER = np.array(
    [
        (86.0, 186.87, 3.7338e-1, 6.958e-6),
        (90.0, 186.87, 1.8359e-1, 3.416e-6),
        (95.0, 188.42, 7.5966e-2, 1.393e-6),
        (100.0, 195.08, 3.2011e-2, 5.604e-7),
        (110.0, 240.00, 7.1042e-3, 9.708e-8),
        (120.0, 360.00, 2.5381e-3, 2.222e-8),
        (130.0, 469.27, 1.2505e-3, 8.152e-9),
        (140.0, 559.63, 7.2028e-4, 3.831e-9),
        (150.0, 634.39, 4.5422e-4, 2.076e-9),
        (160.0, 696.29, 3.0395e-4, 1.233e-9),
        (180.0, 790.07, 1.5271e-4, 5.194e-10),
        (200.0, 854.56, 8.4736e-5, 2.541e-10),
        (300.0, 976.01, 8.7704e-6, 1.916e-11),
        (400.0, 995.83, 1.4518e-6, 2.803e-12),
        (500.0, 999.24, 3.0236e-7, 5.215e-13),
        (600.0, 999.85, 8.2130e-8, 1.137e-13),
        (700.0, 999.97, 3.1908e-8, 3.070e-14),
        (800.0, 999.99, 1.7036e-8, 1.136e-14),
        (900.0, 1000.0, 1.0873e-8, 5.759e-15),
        (1000.0, 1000.0, 7.5138e-9, 3.561e-15),
    ]
)
_Z_UP = _UPPER[:, 0] * 1000.0
_Z_86 = 86_000.0


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic slopes (shape-preserving, no overshoot)."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]
    return m


_LOGP_UP, _LOGR_UP = np.log(_UPPER[:, 2]), np.log(_UPPER[:, 3])
_MP_UP, _MR_UP = _pchip_slopes(_Z_UP, _LOGP_UP), _pchip_slopes(_Z_UP, _LOGR_UP)


def _hermite(z: float, y: np.ndarray, m: np.ndarray) -> float:
    k = int(np.clip(np.searchsorted(_Z_UP, z) - 1, 0, len(_Z_UP) - 2))
    h = _Z_UP[k + 1] - _Z_UP[k]
    t = (z - _Z_UP[k]) / h
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * y[k] + h10 * h * m[k] + h01 * y[k + 1] + h11 * h * m[k + 1]


def us76_upper(z: float) -> tuple[float, float, float]:
    """(temperature K, pressure Pa, density kg/m^3) at geometric altitude z >= 86 km;
    exponential extrapolation above 1000 km."""
    if z >= _Z_UP[-1]:
        hp = (_Z_UP[-1] - _Z_UP[-2]) / (_LOGP_UP[-2] - _LOGP_UP[-1])
        hr = (_Z_UP[-1] - _Z_UP[-2]) / (_LOGR_UP[-2] - _LOGR_UP[-1])
        dz = z - _Z_UP[-1]
        return 1000.0, _UPPER[-1, 2] * math.exp(-dz / hp), _UPPER[-1, 3] * math.exp(-dz / hr)
    t = float(np.interp(z, _Z_UP, _UPPER[:, 1]))
    return t, math.exp(_hermite(z, _LOGP_UP, _MP_UP)), math.exp(_hermite(z, _LOGR_UP, _MR_UP))


def _state(t: float, p: float, rho: float) -> AtmosphereState:
    """State from T, p, rho. The speed of sound uses the local specific gas constant
    p / (rho T), which stays consistent where the mean molar mass drops (above ~90 km)."""
    r_spec = p / (rho * t) if rho > 0 else R_AIR
    return AtmosphereState(t, p, rho, math.sqrt(GAMMA_AIR * r_spec * t))


class Atmosphere:
    """US76 standard atmosphere (0-1000 km), optionally with a temperature offset
    (off-standard day; applied below 86 km at standard pressure) and an "off" switch."""

    name = "us76"

    def __init__(self, enabled: bool = True, temperature_offset: float = 0.0):
        self.enabled = enabled
        self.temperature_offset = temperature_offset

    def locate(self, lat_deg: float, lon_deg: float) -> None:
        """Position hook for location-dependent models (no-op for US76)."""

    def at(self, altitude: float) -> AtmosphereState:
        """Atmospheric state at a geometric altitude above sea level (m)."""
        if not self.enabled:
            return AtmosphereState(_T_TOP, 0.0, 0.0, math.sqrt(GAMMA_AIR * R_AIR * _T_TOP))
        z = max(altitude, -1000.0)
        if z > _Z_86:
            return _state(*us76_upper(z))
        h = _R0 * z / (_R0 + z)  # geopotential altitude (US76 uses r0 = 6356.766 km)
        t, p = _standard(h)
        t_eff = t + self.temperature_offset
        rho = p / (R_AIR * t_eff)
        return AtmosphereState(t_eff, p, rho, math.sqrt(GAMMA_AIR * R_AIR * t_eff))

    def density(self, altitude: float) -> float:
        return self.at(altitude).density

    def pressure(self, altitude: float) -> float:
        return self.at(altitude).pressure


class MsisAtmosphere(Atmosphere):
    """NRLMSISE-00 empirical model (Picone et al. 2002) for an epoch, location and
    solar/geomagnetic activity. For speed, a vertical profile is computed at the current
    location and log-interpolated (1 km steps to 120 km, 5 km above); it is refreshed when
    the vehicle moves more than ``refresh_km``. Valid 0-1000 km; US76 below 0 km."""

    name = "nrlmsise00"

    def __init__(
        self,
        epoch,
        f107: float = 150.0,
        f107a: float = 150.0,
        ap: float = 4.0,
        lat_deg: float = 0.0,
        lon_deg: float = 0.0,
        enabled: bool = True,
        refresh_km: float = 50.0,
    ):
        super().__init__(enabled)
        try:
            from nrlmsise00 import msise_model
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("NRLMSISE-00 needs the 'hifi' extra: uv sync --extra hifi") from e
        self._msis = msise_model
        self.epoch = epoch
        self.f107, self.f107a, self.ap = f107, f107a, ap
        self.refresh_km = refresh_km
        self._z = np.concatenate([np.arange(0.0, 120e3, 1e3), np.arange(120e3, 1000.1e3, 5e3)])
        self._loc = None
        self.locate(lat_deg, lon_deg)

    def point(self, altitude: float, lat_deg: float, lon_deg: float) -> AtmosphereState:
        """Direct (unprofiled) model evaluation."""
        k_b = 1.380649e-23
        d, t = self._msis(
            self.epoch, altitude / 1000.0, lat_deg, lon_deg, self.f107a, self.f107, self.ap
        )
        n = (d[0] + d[1] + d[2] + d[3] + d[4] + d[6] + d[7]) * 1e6  # cm^-3 -> m^-3
        rho = d[5] * 1000.0  # g/cm^3 -> kg/m^3
        return _state(t[1], n * k_b * t[1], rho)

    def locate(self, lat_deg: float, lon_deg: float) -> None:
        if self._loc is not None:
            la, lo = self._loc
            dlon = (lon_deg - lo + 180.0) % 360.0 - 180.0
            dist = 111.2 * math.hypot(lat_deg - la, dlon * math.cos(math.radians(la)))
            if dist < self.refresh_km:
                return
        self._loc = (lat_deg, lon_deg)
        rows = [self.point(z, lat_deg, lon_deg) for z in self._z]
        self._t = np.array([r.temperature for r in rows])
        self._lp = np.log([r.pressure for r in rows])
        self._lr = np.log([r.density for r in rows])

    def at(self, altitude: float) -> AtmosphereState:
        if not self.enabled or altitude < 0.0:
            return Atmosphere.at(self, altitude)
        z = min(altitude, self._z[-1])
        t = float(np.interp(z, self._z, self._t))
        p = math.exp(float(np.interp(z, self._z, self._lp)))
        rho = math.exp(float(np.interp(z, self._z, self._lr)))
        return _state(t, p, rho)


_COLS = {
    "alt": ("altitude_m", "altitude", "alt", "hght", "height", "height_m", "z"),
    "speed": ("speed_mps", "wind_speed_mps", "speed", "wspd", "ws"),
    "speed_kt": ("sknt", "speed_kt", "wind_speed_kt"),
    "dir": ("from_deg", "direction_deg", "direction", "drct", "wdir", "wind_direction"),
    "east": ("east_mps", "u_mps", "u", "east"),
    "north": ("north_mps", "v_mps", "v", "north"),
    "temp_c": ("temp_c", "temperature_c", "temp"),
    "temp_k": ("temp_k", "temperature_k", "temperature"),
    "pres_hpa": ("pressure_hpa", "pres", "pressure_mb", "pres_hpa"),
    "pres_pa": ("pressure_pa",),
}


def read_sounding(path: str | Path) -> dict[str, np.ndarray]:
    """Read a sounding CSV (radiosonde or forecast profile). Column names are matched
    case-insensitively against common conventions (University of Wyoming exports, plain SI
    CSVs). Returns arrays keyed by ``alt`` (m above sea level) plus whichever of ``east`` /
    ``north`` (m/s, wind blowing towards), ``temperature`` (K) and ``pressure`` (Pa) the file
    provides, sorted by altitude. Lines starting with ``#`` are comments."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    header = [h.strip().lower() for h in rows[0]]
    data = {}
    for key, names in _COLS.items():
        for nm in names:
            if nm in header:
                i = header.index(nm)
                vals = []
                for r in rows[1:]:
                    try:
                        vals.append(float(r[i]))
                    except (ValueError, IndexError):
                        vals.append(math.nan)
                data[key] = np.array(vals)
                break
    if "alt" not in data:
        raise ValueError(f"{path}: no altitude column (expected one of {_COLS['alt']})")
    out = {"alt": data["alt"]}
    if "east" in data and "north" in data:
        out["east"], out["north"] = data["east"], data["north"]
    elif "dir" in data and ("speed" in data or "speed_kt" in data):
        sp = data["speed"] if "speed" in data else data["speed_kt"] * 0.514444
        th = np.radians(data["dir"])
        out["east"], out["north"] = -sp * np.sin(th), -sp * np.cos(th)
    if "temp_k" in data:
        out["temperature"] = data["temp_k"]
    elif "temp_c" in data:
        out["temperature"] = data["temp_c"] + 273.15
    if "pres_pa" in data:
        out["pressure"] = data["pres_pa"]
    elif "pres_hpa" in data:
        out["pressure"] = data["pres_hpa"] * 100.0
    keep = np.isfinite(out["alt"])
    out = {k: v[keep] for k, v in out.items()}
    order = np.argsort(out["alt"], kind="stable")
    return {k: v[order] for k, v in out.items()}


class SoundingAtmosphere(Atmosphere):
    """Temperature (and pressure, where measured) from a sounding. Pressure is integrated
    hydrostatically from the lowest level where it is not given. Above the sounding top
    the temperature difference to US76 fades out over ``blend`` metres and the result joins
    US76 (pressure-scaled for continuity)."""

    name = "sounding"

    def __init__(
        self, sounding: dict[str, np.ndarray], enabled: bool = True, blend: float = 10_000.0
    ):
        super().__init__(enabled)
        if "temperature" not in sounding:
            raise ValueError("sounding has no temperature column")
        ok = np.isfinite(sounding["temperature"])
        z = sounding["alt"][ok]
        t = sounding["temperature"][ok]
        std = Atmosphere()
        top = float(z[-1])
        dz_ext = np.arange(500.0, blend + 500.0, 500.0)
        dt_top = float(t[-1]) - std.at(top).temperature
        z_ext = top + dz_ext
        t_ext = np.array([std.at(zz).temperature for zz in z_ext]) + dt_top * np.clip(
            1.0 - dz_ext / blend, 0.0, 1.0
        )
        self._z = np.concatenate([z, z_ext])
        self._t = np.concatenate([t, t_ext])
        p_meas = sounding.get("pressure")
        p_meas = p_meas[ok] if p_meas is not None else np.full(len(z), math.nan)
        p = np.empty(len(self._z))
        p[0] = p_meas[0] if np.isfinite(p_meas[0]) else std.at(float(self._z[0])).pressure
        for k in range(1, len(self._z)):  # hydrostatic with the layer-mean temperature
            g = G0 * (_R0 / (_R0 + self._z[k])) ** 2
            tm = 0.5 * (self._t[k] + self._t[k - 1])
            p[k] = p[k - 1] * math.exp(-g * (self._z[k] - self._z[k - 1]) / (R_AIR * tm))
            if k < len(p_meas) and np.isfinite(p_meas[k]):
                p[k] = p_meas[k]  # measured pressure wins
        self._lp = np.log(p)
        self._top = float(self._z[-1])
        self._scale = p[-1] / std.at(self._top).pressure
        self._std = std

    def at(self, altitude: float) -> AtmosphereState:
        if not self.enabled:
            return Atmosphere.at(self, altitude)
        if altitude > self._top:
            s = self._std.at(altitude)
            return _state(s.temperature, s.pressure * self._scale, s.density * self._scale)
        z = max(altitude, float(self._z[0]) - 2000.0)
        t = float(np.interp(z, self._z, self._t))
        if z < self._z[0]:  # below the lowest level: isothermal hydrostatic extension
            p = math.exp(self._lp[0]) * math.exp(-G0 * (z - self._z[0]) / (R_AIR * t))
        else:
            p = math.exp(float(np.interp(z, self._z, self._lp)))
        return _state(t, p, p / (R_AIR * t))


def resolve_profile(path: str) -> Path:
    """A sounding / wind-profile CSV: a path, or a name under configs/winds/."""
    p = Path(path)
    if p.exists():
        return p
    from plume.config import CONFIG_DIR

    for cand in (CONFIG_DIR / "winds" / path, CONFIG_DIR / "winds" / f"{path}.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"sounding / wind profile not found: {path}")


def atmosphere_from_world(world) -> Atmosphere:
    """Atmosphere model selected by ``WorldSpec.atmosphere_model``."""
    spec = getattr(world, "atmosphere_model", None)
    model = spec.model if spec is not None else "us76"
    if model == "nrlmsise00":
        from datetime import datetime

        epoch = datetime.fromisoformat(spec.epoch.replace("Z", "+00:00")).replace(tzinfo=None)
        es = world.earth
        return MsisAtmosphere(
            epoch,
            spec.f107,
            spec.f107a,
            spec.ap,
            es.origin_lat_deg,
            es.origin_lon_deg,
            world.atmosphere,
        )
    if model == "sounding":
        if not spec.sounding:
            raise ValueError("atmosphere_model.sounding: CSV path required")
        return SoundingAtmosphere(read_sounding(resolve_profile(spec.sounding)), world.atmosphere)
    return Atmosphere(world.atmosphere, world.temperature_offset)
