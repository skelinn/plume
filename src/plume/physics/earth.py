"""High-fidelity Earth model: WGS-84 geodesy, zonal gravity harmonics, rotation.

The simulator's world frame in high-fidelity mode is a **local East-North-Up frame
fixed to the rotating Earth**, with its origin on the WGS-84 ellipsoid (plus an
optional height) at the launch site. Because that frame rotates with the Earth, the
equations of motion need the Coriolis and centrifugal accelerations and an extra
attitude term; :class:`EarthGravity` provides them (see ``physics/sim.py``).

References
----------
* NIMA TR8350.2, "Department of Defense World Geodetic System 1984", 3rd ed. (2000):
  ellipsoid constants, gravitational parameter, rotation rate.
* Lemoine et al., NASA/TP-1998-206861, "The Development of the Joint NASA GSFC and
  NIMA Geopotential Model EGM96": zonal coefficients J2..J6 (unnormalised C_n0 = -J_n).
* Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.: zonal gravity
  gradient via Legendre recursions; geodetic <-> ECEF conversion.
* NASA/TM-2015-218675 (NESC 6-DOF check-cases): verification of all of the above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# WGS-84 defining constants (TR8350.2)
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257_223_563
WGS84_GM = 3.986_004_418e14
WGS84_OMEGA = 7.292_115e-5
# EGM96 zonal harmonics (J_n = -C_n0, unnormalised)
EGM96_J = (
    1.082_629_821_313_3e-3,  # J2
    -2.532_410_518_567_7e-6,  # J3
    -1.619_897_599_917e-6,  # J4
    -2.277_535_907_308e-7,  # J5
    5.406_665_762_838e-7,  # J6
)


@dataclass(frozen=True)
class EarthParams:
    """Shape, gravity and rotation of the Earth model."""

    a: float = WGS84_A  # equatorial radius (sphere radius when f == 0)
    f: float = WGS84_F  # flattening (0 = round Earth)
    gm: float = WGS84_GM
    omega: float = WGS84_OMEGA  # rotation rate, rad/s (0 = non-rotating)
    zonal: tuple[float, ...] = EGM96_J  # J2, J3, ... (empty = point mass)

    @property
    def b(self) -> float:
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        return self.f * (2.0 - self.f)

    @classmethod
    def round_earth(cls, radius: float, gm: float = WGS84_GM, omega: float = WGS84_OMEGA):
        return cls(a=radius, f=0.0, gm=gm, omega=omega, zonal=())


# ----------------------------------------------------------------------------- geodesy
def geodetic_to_ecef(lat: float, lon: float, h: float, ep: EarthParams) -> np.ndarray:
    """Geodetic latitude/longitude (rad) and height above the ellipsoid (m) -> ECEF (m)."""
    s, c = math.sin(lat), math.cos(lat)
    n = ep.a / math.sqrt(1.0 - ep.e2 * s * s)
    return np.array(
        [(n + h) * c * math.cos(lon), (n + h) * c * math.sin(lon), (n * (1.0 - ep.e2) + h) * s]
    )


def ecef_to_geodetic(r: np.ndarray, ep: EarthParams) -> tuple[float, float, float]:
    """ECEF (m) -> geodetic latitude, longitude (rad) and ellipsoidal height (m).

    Bowring's method with two Newton refinements: sub-millimetre for any altitude
    below the Moon's orbit.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if ep.f == 0.0:
        rn = math.sqrt(p * p + z * z)
        return math.atan2(z, p), lon, rn - ep.a
    e2 = ep.e2
    b = ep.b
    ep2 = (ep.a * ep.a - b * b) / (b * b)
    th = math.atan2(z * ep.a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(th) ** 3, p - e2 * ep.a * math.cos(th) ** 3)
    for _ in range(2):
        s = math.sin(lat)
        n = ep.a / math.sqrt(1.0 - e2 * s * s)
        h = p / math.cos(lat) - n if abs(lat) < 1.4 else z / s - n * (1.0 - e2)
        lat = math.atan2(z, p * (1.0 - e2 * n / (n + h)))
    s = math.sin(lat)
    n = ep.a / math.sqrt(1.0 - e2 * s * s)
    h = p / math.cos(lat) - n if abs(lat) < 1.4 else z / s - n * (1.0 - e2)
    return lat, lon, h


def enu_axes(lat: float, lon: float) -> np.ndarray:
    """3x3 matrix whose columns are local East, North, Up expressed in ECEF."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, -sl * co, cl * co],
            [co, -sl * so, cl * so],
            [0.0, cl, sl],
        ]
    )


# ----------------------------------------------------------------------------- gravity
def zonal_gravity_ecef(r: np.ndarray, ep: EarthParams) -> tuple[np.ndarray, float]:
    """Gravitational acceleration (m/s^2, ECEF) and potential energy per unit mass
    (J/kg, = -U) of a zonal-harmonic Earth.

    U(r, s) = GM/r [1 - sum_n J_n (a/r)^n P_n(s)],  s = z / r,  g = grad U.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    rn = math.sqrt(x * x + y * y + z * z)
    gm = ep.gm
    if not ep.zonal:
        k = -gm / rn**3
        return np.array([k * x, k * y, k * z]), -gm / rn
    s = z / rn
    # Legendre polynomials P_n(s) and derivatives up to the highest degree
    nmax = len(ep.zonal) + 1
    p = [1.0, s]
    dp = [0.0, 1.0]
    for n in range(2, nmax + 1):
        p.append(((2 * n - 1) * s * p[n - 1] - (n - 1) * p[n - 2]) / n)
        dp.append(dp[n - 2] + (2 * n - 1) * p[n - 1])
    ar = ep.a / rn
    sum_u = 0.0  # sum J_n (a/r)^n P_n
    sum_r = 0.0  # sum (n+1) J_n (a/r)^n P_n
    sum_s = 0.0  # sum J_n (a/r)^n P_n'
    arn = ar
    for n, jn in enumerate(ep.zonal, start=2):
        arn *= ar
        sum_u += jn * arn * p[n]
        sum_r += (n + 1) * jn * arn * p[n]
        sum_s += jn * arn * dp[n]
    u = gm / rn * (1.0 - sum_u)
    du_dr = -gm / rn**2 * (1.0 - sum_r)
    du_ds = -gm / rn * sum_s
    # grad s = (e_z - s r_hat) / r
    rx, ry, rz = x / rn, y / rn, z / rn
    gx = du_dr * rx + du_ds * (-s * rx) / rn
    gy = du_dr * ry + du_ds * (-s * ry) / rn
    gz = du_dr * rz + du_ds * (1.0 - s * rz) / rn
    return np.array([gx, gy, gz]), -u


# ----------------------------------------------------------------------------- world frame
@dataclass
class EarthGravity:
    """Gravity + geodesy in an Earth-fixed local ENU world frame at a geodetic origin.

    Interface-compatible with :class:`plume.physics.gravity.SphericalGravity` (``accel``,
    ``potential``, ``altitude``, ``up``, ``local_to_frame``, ``curved``), plus the
    rotating-frame terms the simulator adds when ``omega`` != 0.
    """

    lat0: float = 0.0  # rad
    lon0: float = 0.0  # rad
    h0: float = 0.0  # origin height above the ellipsoid, m
    params: EarthParams = field(default_factory=EarthParams)

    curved = True

    def __post_init__(self):
        ep = self.params
        self.origin_ecef = geodetic_to_ecef(self.lat0, self.lon0, self.h0, ep)
        self.R_we = enu_axes(self.lat0, self.lon0)  # world -> ECEF rotation
        self.R_ew = self.R_we.T
        # Earth's rotation vector and the ECEF origin expressed in the world frame
        self.omega_w = self.R_ew @ np.array([0.0, 0.0, ep.omega])
        self.center_w = self.R_ew @ (-self.origin_ecef)  # Earth's centre in world coords
        self.earth_radius = ep.a

    # aliases used by code written for SphericalGravity
    @property
    def center(self) -> np.ndarray:
        return self.center_w

    @property
    def mu(self) -> float:
        return self.params.gm

    # -- conversions
    def to_ecef(self, p_w: np.ndarray) -> np.ndarray:
        return self.origin_ecef + self.R_we @ p_w

    def from_ecef(self, r: np.ndarray) -> np.ndarray:
        return self.R_ew @ (r - self.origin_ecef)

    def geodetic(self, p_w: np.ndarray) -> tuple[float, float, float]:
        return ecef_to_geodetic(self.to_ecef(p_w), self.params)

    def world_point(self, lat: float, lon: float, h: float) -> np.ndarray:
        return self.from_ecef(geodetic_to_ecef(lat, lon, h, self.params))

    def enu_at(self, p_w: np.ndarray) -> np.ndarray:
        """Columns: local East, North, Up at p, expressed in world coordinates."""
        lat, lon, _ = self.geodetic(p_w)
        return self.R_ew @ enu_axes(lat, lon)

    # -- SphericalGravity-compatible interface
    def accel(self, p_w: np.ndarray) -> np.ndarray:
        g, _ = zonal_gravity_ecef(self.to_ecef(p_w), self.params)
        return self.R_ew @ g

    def potential(self, p_w: np.ndarray) -> float:
        """Gravitational potential energy per unit mass (J/kg)."""
        return zonal_gravity_ecef(self.to_ecef(p_w), self.params)[1]

    def centrifugal_potential(self, p_w: np.ndarray) -> float:
        """-1/2 |omega x r|^2 (J/kg): add to ``potential`` for the rotating-frame energy."""
        w = np.cross(self.omega_w, p_w - self.center_w)
        return -0.5 * float(w @ w)

    def altitude(self, p_w: np.ndarray) -> float:
        return self.geodetic(p_w)[2]

    def up(self, p_w: np.ndarray) -> np.ndarray:
        lat, lon, _ = self.geodetic(p_w)
        return self.R_ew @ enu_axes(lat, lon)[:, 2]

    def local_to_frame(self, p_w: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate a vector given in local ENU at ``p_w`` into the world frame."""
        return self.enu_at(p_w) @ vec

    # -- rotating-frame dynamics
    @property
    def rotating(self) -> bool:
        return self.params.omega != 0.0

    def fictitious_accel(self, p_w: np.ndarray, v_w: np.ndarray) -> np.ndarray:
        """Coriolis + centrifugal acceleration in the Earth-fixed world frame."""
        w = self.omega_w
        r = p_w - self.center_w
        return -2.0 * np.cross(w, v_w) - np.cross(w, np.cross(w, r))

    def attitude_torque(self, R_wb: np.ndarray, omega_rel_b: np.ndarray, inertia_b: np.ndarray):
        """Extra body torque so that MuJoCo (which treats the world frame as inertial)
        integrates the correct attitude dynamics in the rotating frame.

        With w_abs = w_rel + w_e:  I w_rel' = tau - w_abs x I w_abs + I (w_rel x w_e);
        MuJoCo already applies -w_rel x I w_rel, so the correction is
        -w_abs x I w_abs + w_rel x I w_rel + I (w_rel x w_e).
        """
        we_b = R_wb.T @ self.omega_w
        wa = omega_rel_b + we_b
        return (
            -np.cross(wa, inertia_b * wa)
            + np.cross(omega_rel_b, inertia_b * omega_rel_b)
            + inertia_b * np.cross(omega_rel_b, we_b)
        )

    def inertial_rates(self, R_wb: np.ndarray, omega_rel_b: np.ndarray) -> np.ndarray:
        return omega_rel_b + R_wb.T @ self.omega_w

    def surface_point(self, u: float, v: float, h: float = 0.0) -> np.ndarray:
        """Map coordinates (aeqd east/north from the origin) + height -> world point."""
        lat, lon = self.map_to_geodetic(u, v)
        return self.world_point(lat, lon, h)

    def map_to_geodetic(self, u: float, v: float) -> tuple[float, float]:
        """Inverse azimuthal-equidistant projection centred on the origin (ellipsoidal
        when pyproj is available, otherwise spherical with the mean radius)."""
        tr = _aeqd(self.lat0, self.lon0)
        if tr is not None:
            lon, lat = tr.transform(u, v, direction="INVERSE")
            return math.radians(lat), math.radians(lon)
        r = 6_371_008.8
        s = math.hypot(u, v)
        if s < 1e-9:
            return self.lat0, self.lon0
        az = math.atan2(u, v)
        d = s / r
        lat = math.asin(
            math.sin(self.lat0) * math.cos(d) + math.cos(self.lat0) * math.sin(d) * math.cos(az)
        )
        lon = self.lon0 + math.atan2(
            math.sin(az) * math.sin(d) * math.cos(self.lat0),
            math.cos(d) - math.sin(self.lat0) * math.sin(lat),
        )
        return lat, lon

    def geodetic_to_map(self, lat: float, lon: float) -> tuple[float, float]:
        tr = _aeqd(self.lat0, self.lon0)
        if tr is not None:
            return tr.transform(math.degrees(lon), math.degrees(lat))
        r = 6_371_008.8
        dlon = lon - self.lon0
        c = math.sin(self.lat0) * math.sin(lat) + math.cos(self.lat0) * math.cos(lat) * math.cos(
            dlon
        )
        d = math.acos(max(-1.0, min(1.0, c)))
        if d < 1e-12:
            return 0.0, 0.0
        az = math.atan2(
            math.sin(dlon) * math.cos(lat),
            math.cos(self.lat0) * math.sin(lat)
            - math.sin(self.lat0) * math.cos(lat) * math.cos(dlon),
        )
        return r * d * math.sin(az), r * d * math.cos(az)

    def map_coords(self, p_w: np.ndarray) -> tuple[float, float, float]:
        """World point -> (map east, map north, height above the ellipsoid)."""
        lat, lon, h = self.geodetic(p_w)
        u, v = self.geodetic_to_map(lat, lon)
        return u, v, h

    def surface_normal(self, u: float, v: float) -> np.ndarray:
        lat, lon = self.map_to_geodetic(u, v)
        return self.R_ew @ enu_axes(lat, lon)[:, 2]


_AEQD_CACHE: dict[tuple[float, float], object] = {}


def _aeqd(lat0: float, lon0: float):
    key = (round(lat0, 12), round(lon0, 12))
    if key in _AEQD_CACHE:
        return _AEQD_CACHE[key]
    try:
        from pyproj import Transformer

        tr = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={math.degrees(lat0)} +lon_0={math.degrees(lon0)} +ellps=WGS84 +units=m",
            always_xy=True,
        )
    except Exception:  # pyproj missing
        tr = None
    _AEQD_CACHE[key] = tr
    return tr
