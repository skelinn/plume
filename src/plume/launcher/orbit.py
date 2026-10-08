"""Orbital state and classical orbital elements from the simulator's world frame.

The simulator's world frame is an East-North-Up frame at the launch site. In high
fidelity it is fixed to the *rotating* WGS-84 Earth (:class:`~plume.physics.earth.EarthGravity`),
in fast fidelity it belongs to a non-rotating spherical Earth
(:class:`~plume.physics.gravity.SphericalGravity`). :func:`inertial_state` maps a world
position/velocity to an Earth-centred inertial (ECI) frame:

* rotating Earth: ``r = r_ecef``, ``v = v_ecef + omega_E x r_ecef``, both rotated by the
  Earth rotation angle ``omega_E t`` (the ECI frame coincides with ECEF at t = 0, so the
  right ascension of the ascending node is measured from the launch-epoch Greenwich
  meridian);
* non-rotating sphere: the frame is already inertial. Its pole is not defined by the
  gravity model, so it is taken from the launch latitude: ``k = cos(lat) north + sin(lat)
  up`` at the launch site (the same convention the targeting uses), with the x axis on
  the launch meridian.

:func:`elements` converts an ECI state to the classical elements (two-body, osculating).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class OrbitalElements:
    a: float  # semi-major axis, m (negative for hyperbolic)
    e: float
    i: float  # inclination, rad
    raan: float  # rad
    argp: float  # argument of periapsis, rad
    nu: float  # true anomaly, rad
    rp: float  # periapsis radius, m
    ra: float  # apoapsis radius, m (inf if unbound)
    energy: float  # specific orbital energy, J/kg
    h: float  # specific angular momentum, m^2/s

    def perigee_altitude(self, r_eq: float) -> float:
        return self.rp - r_eq

    def apogee_altitude(self, r_eq: float) -> float:
        return self.ra - r_eq

    def period(self, mu: float) -> float:
        return 2 * math.pi * math.sqrt(self.a**3 / mu) if self.a > 0 else math.inf

    def as_dict(self, r_eq: float | None = None) -> dict:
        d = asdict(self)
        d["i_deg"] = math.degrees(self.i)
        if r_eq is not None:
            d["perigee_alt_m"] = self.perigee_altitude(r_eq)
            d["apogee_alt_m"] = self.apogee_altitude(r_eq)
        return d


def elements(r: np.ndarray, v: np.ndarray, mu: float) -> OrbitalElements:
    """Classical orbital elements of an inertial state (Vallado, algorithm 9)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = float(np.linalg.norm(r))
    vn2 = float(v @ v)
    hv = np.cross(r, v)
    h = float(np.linalg.norm(hv))
    energy = 0.5 * vn2 - mu / rn
    ev = ((vn2 - mu / rn) * r - float(r @ v) * v) / mu
    e = float(np.linalg.norm(ev))
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else math.inf
    i = math.acos(max(-1.0, min(1.0, hv[2] / h))) if h > 0 else 0.0
    nv = np.array([-hv[1], hv[0], 0.0])  # k x h
    nn = float(np.linalg.norm(nv))
    raan = math.atan2(nv[1], nv[0]) % (2 * math.pi) if nn > 1e-9 else 0.0
    if e > 1e-10:
        if nn > 1e-9:
            argp = math.acos(max(-1.0, min(1.0, float(nv @ ev) / (nn * e))))
            if ev[2] < 0:
                argp = 2 * math.pi - argp
        else:  # equatorial: longitude of periapsis
            argp = math.atan2(ev[1], ev[0]) % (2 * math.pi)
        nu = math.acos(max(-1.0, min(1.0, float(ev @ r) / (e * rn))))
        if float(r @ v) < 0:
            nu = 2 * math.pi - nu
    else:  # circular: argument of latitude in nu
        argp = 0.0
        ref = nv / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
        nu = math.acos(max(-1.0, min(1.0, float(ref @ r) / rn)))
        if r[2] < 0:
            nu = 2 * math.pi - nu
    p = h * h / mu
    rp = p / (1.0 + e)
    ra = p / (1.0 - e) if e < 1.0 else math.inf
    return OrbitalElements(a, e, i, raan, argp, nu, rp, ra, energy, h)


def state_from_elements(
    a: float, e: float, i: float, raan: float, argp: float, nu: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`elements` (elliptic orbits): ECI position and velocity."""
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * math.cos(nu)) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    def rz(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    Q = rz(raan) @ rx(i) @ rz(argp)
    return Q @ r_pf, Q @ v_pf


class InertialFrame:
    """World (simulator) frame <-> Earth-centred inertial frame for one gravity model."""

    def __init__(self, gravity, launch_lat_deg: float | None = None):
        self.gravity = gravity
        self.mu = float(gravity.mu)
        self.r_eq = float(gravity.earth_radius)
        self.ecef = hasattr(gravity, "to_ecef")
        if self.ecef:
            self.omega = float(gravity.params.omega)
            self.R_we = gravity.R_we  # world -> ECEF (rotation)
            self.R_wi0 = self.R_we
            self.C = None
        else:
            self.omega = 0.0
            lat = math.radians(launch_lat_deg if launch_lat_deg is not None else 0.0)
            k = np.array([0.0, math.cos(lat), math.sin(lat)])  # pole in launch ENU
            up = np.array([0.0, 0.0, 1.0])
            x = up - (up @ k) * k
            n = float(np.linalg.norm(x))
            x = x / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
            y = np.cross(k, x)
            self.C = np.vstack([x, y, k])  # world -> ECI (rows: ECI axes in world coords)

    def _rot(self, t: float) -> np.ndarray:
        """ECEF -> ECI at time t (Earth rotation angle omega t about z)."""
        th = self.omega * t
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def world_to_eci_matrix(self, t: float) -> np.ndarray:
        """Rotation taking world-frame *directions* to ECI at time t."""
        if self.ecef:
            return self._rot(t) @ self.R_we
        return self.C

    def to_inertial(self, p_w, v_w, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        p_w = np.asarray(p_w, dtype=float)
        v_w = np.asarray(v_w, dtype=float)
        g = self.gravity
        if self.ecef:
            r = g.to_ecef(p_w)
            v = self.R_we @ v_w + np.cross([0.0, 0.0, self.omega], r)
            Q = self._rot(t)
            return Q @ r, Q @ v
        return self.C @ (p_w - g.center), self.C @ v_w

    def to_world_direction(self, d_i: np.ndarray, t: float) -> np.ndarray:
        return self.world_to_eci_matrix(t).T @ d_i

    def elements(self, p_w, v_w, t: float = 0.0) -> OrbitalElements:
        r, v = self.to_inertial(p_w, v_w, t)
        return elements(r, v, self.mu)

    def pole(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


def target_plane_normal(r_i: np.ndarray, v_i: np.ndarray, inclination: float | None):
    """Unit normal of a target orbit plane through the position ``r_i`` (ECI) with the given
    inclination, choosing the branch (ascending/descending pass) closest to the current
    velocity's plane. ``None`` keeps the current plane (r x v); an inclination below the
    current latitude is replaced by the latitude (the lowest reachable without a dog-leg)."""
    rh = r_i / np.linalg.norm(r_i)
    h = np.cross(r_i, v_i)
    hn = float(np.linalg.norm(h))
    if inclination is None:
        if hn > 0:
            return h / hn
        inclination = 0.0
    k = np.array([0.0, 0.0, 1.0])
    north = k - (k @ rh) * rh
    nn = float(np.linalg.norm(north))
    if nn < 1e-9:
        return h / hn if hn > 0 else np.array([1.0, 0.0, 0.0])
    north /= nn
    east = np.cross(north, rh)
    cos_lat = nn  # |k - (k.r)r| = cos(latitude)
    cb = max(-1.0, min(1.0, math.cos(inclination) / cos_lat))
    sb = math.sqrt(max(0.0, 1.0 - cb * cb))
    # the velocity n x r_hat of the first candidate heads north-east: the ascending pass
    ascending, descending = cb * north - sb * east, cb * north + sb * east
    if hn > 0:
        hh = h / hn
        if float(descending @ hh) > float(ascending @ hh) + 1e-6:
            return descending
    return ascending
