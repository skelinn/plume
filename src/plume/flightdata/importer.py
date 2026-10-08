"""Import hobby flight-computer CSV logs.

A YAML column mapping (``configs/flightlogs/*.yaml``) says which columns hold time,
barometric altitude, axial acceleration, gyro rates and GPS, and in which units. The
importer converts to SI, zeroes the baro on the pad, detects liftoff (t = 0),
resamples to a uniform rate and fuses baro + accelerometer with a small Kalman
filter to estimate vertical velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from plume.config import LogMappingSpec, load_log_mapping
from plume.constants import G0

UNITS = {
    # time
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    # length
    "m": 1.0,
    "ft": 0.3048,
    "km": 1000.0,
    # acceleration
    "m/s2": 1.0,
    "m/s^2": 1.0,
    "g": G0,
    "ft/s2": 0.3048,
    "ft/s^2": 0.3048,
    # angular rate
    "rad/s": 1.0,
    "deg/s": math.pi / 180.0,
    "dps": math.pi / 180.0,
    # generic
    "si": 1.0,
}


def to_si(values: np.ndarray, unit: str, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    try:
        factor = UNITS[unit.lower()]
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; known: {sorted(UNITS)}") from None
    return (np.asarray(values, dtype=float) * scale + offset) * factor


@dataclass
class FlightLog:
    """A flight in SI units with t = 0 at liftoff, uniformly resampled."""

    t: np.ndarray
    altitude: np.ndarray  # m above the pad (baro)
    accel: np.ndarray | None = None  # axial specific force, m/s^2 (reads +g at rest)
    gyro: np.ndarray | None = None  # body rates, rad/s (n x 3)
    east: np.ndarray | None = None  # GPS-derived local position, m
    north: np.ndarray | None = None
    gps_alt: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))  # vertical, KF estimate
    meta: dict = field(default_factory=dict)

    @property
    def apogee(self) -> float:
        return float(self.altitude.max())

    @property
    def t_apogee(self) -> float:
        return float(self.t[int(np.argmax(self.altitude))])

    @property
    def burnout_time(self) -> float | None:
        """First time after liftoff where the axial specific force turns negative (coast)."""
        if self.accel is None:
            return None
        idx = np.nonzero((self.t > 0.05) & (self.accel < 0.0))[0]
        return float(self.t[idx[0]]) if len(idx) else None

    def window(self, t0: float, t1: float) -> np.ndarray:
        return (self.t >= t0) & (self.t <= t1)


def _hampel(x: np.ndarray, k: int = 7, n_sigma: float = 4.0) -> np.ndarray:
    """Replace outliers (relative to a rolling median) by the median."""
    s = pd.Series(x)
    med = s.rolling(2 * k + 1, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(2 * k + 1, center=True, min_periods=1).median()
    bad = (s - med).abs() > n_sigma * 1.4826 * mad + 1e-9
    out = s.where(~bad, med)
    return out.to_numpy()


def kalman_vertical(
    t: np.ndarray,
    alt: np.ndarray,
    accel: np.ndarray | None,
    sigma_alt: float = 1.0,
    sigma_acc: float = 2.0,
    t_accel_until: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D Kalman filter + RTS smoother on [altitude, vertical velocity].

    The axial accelerometer drives the prediction while the vehicle points up
    (until ``t_accel_until``, e.g. apogee); afterwards a constant-velocity model is used.
    """
    n = len(t)
    xs = np.zeros((n, 2))
    Ps = np.zeros((n, 2, 2))
    xp_all = np.zeros((n, 2))
    Pp_all = np.zeros((n, 2, 2))
    Fs = np.zeros((n, 2, 2))
    x = np.array([alt[0], 0.0])
    P = np.diag([sigma_alt**2, 1.0])
    H = np.array([[1.0, 0.0]])
    R = sigma_alt**2
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        use_acc = accel is not None and (t_accel_until is None or t[k] <= t_accel_until)
        u = (accel[k] - G0) if use_acc else 0.0
        q = sigma_acc**2 if use_acc else 25.0
        B = np.array([0.5 * dt * dt, dt])
        Q = q * np.outer(B, B) + np.diag([1e-6, 1e-6])
        x = F @ x + B * u
        P = F @ P @ F.T + Q
        xp_all[k], Pp_all[k], Fs[k] = x, P, F
        y = alt[k] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T)[:, 0] / S
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H[0])) @ P
        xs[k], Ps[k] = x, P
    # Rauch-Tung-Striebel smoother
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pp_all[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp_all[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp_all[k + 1]) @ C.T
    return xs[:, 0], xs[:, 1]


def _latlon_to_en(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    r = 6_371_000.0
    north = np.radians(lat - lat0) * r
    east = np.radians(lon - lon0) * r * math.cos(math.radians(lat0))
    return east, north


def load_log(csv_path: str | Path, mapping: LogMappingSpec | str | Path) -> FlightLog:
    m = mapping if isinstance(mapping, LogMappingSpec) else load_log_mapping(mapping)
    df = pd.read_csv(
        csv_path,
        sep=m.delimiter,
        comment=m.comment or None,
        skiprows=m.skip_rows,
        skipinitialspace=True,
    )
    df.columns = [c.strip() for c in df.columns]

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            raise KeyError(
                f"column {name!r} not found in {Path(csv_path).name}; have {list(df.columns)}"
            )
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    t = to_si(col(m.time.column), m.time.unit, m.time.scale, m.time.offset)
    alt = to_si(col(m.altitude.column), m.altitude.unit, m.altitude.scale, m.altitude.offset)
    acc = None
    if m.acceleration:
        a = m.acceleration
        acc = to_si(col(a.column), a.unit, a.scale, a.offset)
        if not a.includes_gravity:
            acc = acc + G0
    gyro = None
    if m.gyro:
        gyro = np.column_stack([to_si(col(c), m.gyro.unit) for c in m.gyro.columns])
    lat = lon = galt = None
    if m.gps:
        lat, lon = col(m.gps.lat), col(m.gps.lon)
        galt = to_si(col(m.gps.alt), m.gps.alt_unit) if m.gps.alt else None

    # clean: sort, drop duplicate/invalid timestamps, hold-last GPS gaps
    order = np.argsort(t, kind="stable")
    keep = np.isfinite(t[order]) & np.isfinite(alt[order])
    order = order[keep]
    _, uniq = np.unique(t[order], return_index=True)
    order = order[uniq]

    def pick(x):
        return None if x is None else x[order]

    t, alt, acc, gyro = t[order], alt[order], pick(acc), pick(gyro)
    lat, lon, galt = pick(lat), pick(lon), pick(galt)
    alt = _hampel(alt)

    # liftoff detection
    ld = m.launch_detect
    t_launch = None
    if acc is not None:
        hot = acc > ld.accel_threshold_g * G0
        idx = np.nonzero(hot)[0]
        for i in idx:
            j = np.searchsorted(t, t[i] + 0.1)
            if j < len(t) and hot[i : j + 1].mean() > 0.8:
                # walk back to where the load first rose above ~1.3 g
                k = i
                while k > 0 and acc[k - 1] > 1.3 * G0:
                    k -= 1
                t_launch = t[k]
                break
    if t_launch is None:
        base = np.median(alt[: max(5, len(alt) // 50)])
        idx = np.nonzero(alt - base > ld.altitude_threshold)[0]
        if not len(idx):
            raise ValueError("could not detect a launch in this log")
        t_launch = t[idx[0]] - 0.5
    pad = (t >= t_launch - ld.pad_window) & (t < t_launch)
    alt0 = float(np.median(alt[pad])) if pad.any() else float(alt[0])

    # resample on a uniform grid from 0.5 s before liftoff
    dt = 1.0 / m.resample_hz
    tu = np.arange(max(t[0], t_launch - 0.5), t[-1], dt)

    def interp(x):
        if x is None:
            return None
        ok = np.isfinite(x)
        return np.interp(tu, t[ok], x[ok]) if ok.any() else None

    alt_u = interp(alt) - alt0
    acc_u = interp(acc)
    gyro_u = None if gyro is None else np.column_stack([interp(gyro[:, k]) for k in range(3)])
    east = north = gps_alt = None
    if lat is not None and np.isfinite(lat).any():
        lat_u, lon_u = interp(lat), interp(lon)
        ok = np.isfinite(lat) & np.isfinite(lon)
        east, north = _latlon_to_en(lat_u, lon_u, float(lat[ok][0]), float(lon[ok][0]))
        gps_alt = (
            None
            if galt is None
            else interp(galt) - (np.nanmedian(galt[pad]) if pad.any() else galt[0])
        )
    t_rel = tu - t_launch
    t_apo = float(t_rel[int(np.argmax(alt_u))])
    _, vel = kalman_vertical(t_rel, alt_u, acc_u, t_accel_until=t_apo)
    return FlightLog(
        t=t_rel,
        altitude=alt_u,
        accel=acc_u,
        gyro=gyro_u,
        east=east,
        north=north,
        gps_alt=gps_alt,
        velocity=vel,
        meta={
            "source": str(csv_path),
            "mapping": m.name,
            "launch_time_raw": float(t_launch),
            "pad_altitude_raw": alt0,
            "rate_hz": m.resample_hz,
        },
    )
