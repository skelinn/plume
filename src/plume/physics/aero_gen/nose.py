"""Nose (forebody) pressure / wave drag at alpha = 0, all Mach numbers.

Supersonic, attached shock:

* Cones: Linnell & Bailey (J. Aero. Sci. 23(8), 1956) correlation of exact
  Taylor-Maccoll cone pressures, quoted in Jorgensen, NASA TN D-6996 eq. (10):
  CA_W = 4 sin^2(t) (2.5 + 8 b sin t) / (1 + 16 b sin t), b = sqrt(M^2 - 1);
  1.4 % rms accuracy for 0.05 <= b sin t <= 10 (Wittliff 1968).  Verified here
  against our own Taylor-Maccoll solver (``taylor_maccoll_cone``).
* Tangent ogives: Rossow's method-of-characteristics correlation (NACA TN 2399,
  1951) of the wave-drag parameter (gamma/2) M^2 CA_W vs K = M d / l_N, TN D-6996
  Fig. 6; we use the digitised ratio ogive/cone at equal K times the cone value.
* von Karman (LD-Haack) noses: the Newtonian minimum-drag power-law curve of the
  same figure (ratio ~0.83 to the cone) as a proxy (+-20 %).

Subsonic (M <= 0.8): cones 0.8 sin^2(t) (Hoerner 1965 via Niskanen 2009, sec.
3.4.4); smooth ogive / von Karman noses 0 (their pressure drag is within the
form factor applied to skin friction).
Transonic: monotone cubic (PCHIP) blend from M 0.8 to the first supersonic point
M_s (shock attached on the equivalent cone and b sin t >= 0.05), passing for cones
through the sonic value CA_W(M=1) = sin t (Hoerner via Niskanen 2009).  This is the
least certain part of the axial build-up (+-30 %).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from plume.physics.aero_gen.geometry import BodyGeometry

GAMMA = 1.4

#: Ratio of tangent-ogive to cone wave drag at equal K = M d / l_N (TN D-6996 Fig. 6,
#: digitised: Rossow ogive curve / Taylor-Maccoll cone curve).
OGIVE_CONE_RATIO = np.array(
    [
        [0.0, 1.0],
        [0.6, 1.0],
        [0.8, 1.05],
        [1.0, 1.09],
        [1.2, 1.12],
        [1.4, 1.145],
        [1.6, 1.16],
        [1.8, 1.17],
    ]
).T
VON_KARMAN_CONE_RATIO = 0.83


def linnell_bailey(mach, half_angle):
    """Cone wave-drag (pressure) coefficient on the base area, supersonic (vectorised)."""
    m = np.asarray(mach, dtype=float)
    b = np.sqrt(np.maximum(m * m - 1.0, 0.0))
    s = np.sin(half_angle)
    return 4 * s * s * (2.5 + 8 * b * s) / (1 + 16 * b * s)


def _oblique(mach: float, beta: np.ndarray, g: float = GAMMA):
    mn1 = mach * np.sin(beta)
    tan_d = 2 / np.tan(beta) * (mn1**2 - 1) / (mach**2 * (g + np.cos(2 * beta)) + 2)
    delta = np.arctan(tan_d)
    mn2 = np.sqrt((1 + 0.5 * (g - 1) * mn1**2) / (g * mn1**2 - 0.5 * (g - 1)))
    m2 = mn2 / np.sin(beta - delta)
    p21 = 1 + 2 * g / (g + 1) * (mn1**2 - 1)
    return delta, m2, p21


@functools.lru_cache(maxsize=256)
def _tm_curve(mach: float, n_beta: int = 240, n_steps: int = 800):
    """Taylor-Maccoll solutions for shock angles from the Mach angle to 89 deg.

    Returns arrays (shock angle, cone half-angle, surface Cp), attached weak branch
    (cone angle increasing with shock angle)."""
    g = GAMMA
    mu = math.asin(1.0 / mach)
    beta = mu + 1e-5 + (math.radians(89.0) - mu) * np.linspace(0.0, 1.0, n_beta) ** 2
    delta, m2, p21 = _oblique(mach, beta)
    v = (2.0 / ((g - 1) * m2**2) + 1.0) ** -0.5
    vr = v * np.cos(beta - delta)
    vt = -v * np.sin(beta - delta)
    th = beta.copy()
    h = -beta / n_steps
    cone = np.full_like(beta, np.nan)
    vrc = np.full_like(beta, np.nan)
    active = np.ones_like(beta, dtype=bool)

    def rhs(t, y0, y1):
        a = 0.5 * (g - 1) * (1 - y0 * y0 - y1 * y1)
        return y1, (y1 * y1 * y0 - a * (2 * y0 + y1 / np.tan(t))) / (a - y1 * y1)

    for _ in range(n_steps - 1):
        k1 = rhs(th, vr, vt)
        k2 = rhs(th + h / 2, vr + h / 2 * k1[0], vt + h / 2 * k1[1])
        k3 = rhs(th + h / 2, vr + h / 2 * k2[0], vt + h / 2 * k2[1])
        k4 = rhs(th + h, vr + h * k3[0], vt + h * k3[1])
        nvr = vr + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        nvt = vt + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        nth = th + h
        cross = active & (nvt >= 0) & (vt < 0)
        if np.any(cross):
            f = vt[cross] / (vt[cross] - nvt[cross])
            cone[cross] = th[cross] + f * h[cross]
            vrc[cross] = vr[cross] + f * (nvr[cross] - vr[cross])
            active &= ~cross
        vr, vt, th = (
            np.where(active, nvr, vr),
            np.where(active, nvt, vt),
            np.where(active, nth, th),
        )
        if not active.any():
            break
    ok = np.isfinite(cone)
    beta, cone, vrc = beta[ok], cone[ok], vrc[ok]
    m2, p21 = m2[ok], p21[ok]
    mc2 = 2.0 / (g - 1) * vrc**2 / (1 - vrc**2)
    pc_p2 = ((1 + 0.5 * (g - 1) * m2**2) / (1 + 0.5 * (g - 1) * mc2)) ** (g / (g - 1))
    cp = (p21 * pc_p2 - 1) / (0.5 * g * mach * mach)
    imax = int(np.argmax(cone))
    return beta[: imax + 1], cone[: imax + 1], cp[: imax + 1]


def taylor_maccoll_cone(mach: float, half_angle: float) -> dict[str, float] | None:
    """Exact inviscid cone flow (Taylor & Maccoll 1933): shock angle and surface Cp.

    Returns None if the shock is detached (half-angle above the maximum)."""
    beta, cone, cp = _tm_curve(round(float(mach), 6))
    if half_angle > cone[-1] or half_angle < cone[0]:
        return None
    return {
        "shock_angle": float(np.interp(half_angle, cone, beta)),
        "cp": float(np.interp(half_angle, cone, cp)),
    }


def max_cone_angle(mach: float) -> float:
    """Largest cone half-angle with an attached shock at ``mach`` (rad)."""
    if mach <= 1.0:
        return 0.0
    return float(_tm_curve(round(float(mach), 6))[1][-1])


@functools.lru_cache(maxsize=64)
def detachment_mach(half_angle: float) -> float:
    """Lowest Mach number with an attached conical shock for ``half_angle`` (rad)."""
    lo, hi = 1.0005, 12.0
    if max_cone_angle(hi) < half_angle:
        return math.inf
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if max_cone_angle(mid) >= half_angle:
            hi = mid
        else:
            lo = mid
        if hi - lo < 2e-3:
            break
    return hi


def supersonic_wave_drag(mach, body: BodyGeometry):
    """Attached-shock correlation for the nose shape (vectorised in Mach)."""
    t = body.nose_half_angle
    cone = linnell_bailey(mach, t)
    if body.nose_shape == "cone":
        return cone
    if body.nose_shape == "tangent_ogive":
        k = np.asarray(mach, dtype=float) / body.nose_fineness
        return cone * np.interp(k, OGIVE_CONE_RATIO[0], OGIVE_CONE_RATIO[1])
    if body.nose_shape == "von_karman":
        return cone * VON_KARMAN_CONE_RATIO
    raise ValueError(body.nose_shape)


def first_supersonic_mach(body: BodyGeometry) -> float:
    t = body.nose_half_angle
    m_lb = math.sqrt(1.0 + (0.05 / math.sin(t)) ** 2)
    return max(1.1, m_lb, detachment_mach(round(t, 6)) + 0.05)


def nose_pressure_drag(mach, body: BodyGeometry) -> np.ndarray:
    """Nose pressure/wave drag coefficient (on the base area) at alpha = 0 vs Mach.

    Flat-faced bodies return 0 here (their face drag is in ``base.flat_face_drag``)."""
    m = np.atleast_1d(np.asarray(mach, dtype=float))
    if body.nose_length <= 0 or body.nose_shape == "flat":
        return np.zeros_like(m)
    t = body.nose_half_angle
    ms = first_supersonic_mach(body)
    c_sub = 0.8 * math.sin(t) ** 2 if body.nose_shape == "cone" else 0.0
    xs = [0.0, 0.8]
    ys = [c_sub, c_sub]
    if body.nose_shape == "cone" and ms > 1.05:
        xs.append(1.0)
        ys.append(math.sin(t))
    for mm in (ms, ms + 0.15, ms + 0.3):
        xs.append(mm)
        ys.append(float(supersonic_wave_drag(mm, body)))
    blend = PchipInterpolator(xs, ys)
    out = np.where(m <= 0.8, c_sub, 0.0)
    trans = (m > 0.8) & (m < ms)
    out = np.where(trans, blend(np.clip(m, 0.8, ms)), out)
    sup = m >= ms
    if np.any(sup):
        out = np.where(sup, supersonic_wave_drag(np.maximum(m, ms), body), out)
    return out
