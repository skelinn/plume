"""Grid (lattice) fin Mach-number effects: normal-force slope and drag multipliers.

``gridfin_cn_alpha(mach)`` and ``gridfin_cd0(mach)`` return multipliers equal to 1
at low subsonic speed, so the YAML ``cn_alpha`` / ``cd0`` (low-speed values on the
fin area) are scaled as ``cn_alpha * gridfin_cn_alpha(M)``.

Physics (Washington & Miller, "Grid fins - a new concept for missile stability and
control", AIAA 93-0035, 1993; Washington & Miller, "Experimental investigations of
grid fin aerodynamics: a synopsis of nine wind tunnel and three flight tests",
AGARD/RTO-MP-5, 1998; Simpson & Sadler, RTO-MP-5 paper 9, 1998; Burkhalter,
Hartfield & Leleux, J. Aircraft 32(3), 1995):

1. Subsonic: each cell acts as a low-aspect-ratio biplane box; mild compressibility
   growth (1 - M^2)^-1/4 (weaker than Prandtl-Glauert because of the low cell aspect
   ratio).
2. Transonic choking: once the cells reach sonic speed at the web blockage the
   cells choke and a detached bow shock spills the flow around the lattice; the
   normal-force effectiveness drops (~40-50 %) and drag peaks.  Onset Mach from the
   isentropic area ratio A/A*(M_ch) = 1/phi (phi = open-area ratio of a cell);
   recovery when the normal shock is swallowed, from the Kantrowitz-Donaldson
   starting criterion A/A*(M_2) = 1/phi with M_2 the Mach behind a normal shock
   (Kantrowitz & Donaldson, NACA WR L-713, 1945).  For phi = 0.9: choking ~M 0.68,
   swallowing ~M 1.6 -- the literature range of 0.7-0.8 and 1.4-1.8.
3. Supersonic: each web pair behaves as a 2-D Ackeret plate, CN_alpha ~ 1/beta,
   continuous at the swallowing Mach.

Drag: subsonic constant; transonic peak of ~1.9x (Washington & Miller's data show
1.7-2.2x); supersonic decay ~ (beta_sw/beta)^0.5 toward friction + web wave drag,
floored at 0.6.

These are semi-empirical *shapes*: 1-sigma uncertainty +-15 % subsonic, +-35 %
transonic, +-20 % supersonic (``gridfin_uncertainty``).
"""

from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4


def _area_ratio(m):
    """Isentropic A/A* at Mach m."""
    g = GAMMA
    m = np.maximum(np.asarray(m, dtype=float), 1e-6)
    return 1 / m * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * m * m)) ** ((g + 1) / (2 * (g - 1)))


def _normal_shock_m2(m1: float) -> float:
    g = GAMMA
    return math.sqrt((1 + 0.5 * (g - 1) * m1 * m1) / (g * m1 * m1 - 0.5 * (g - 1)))


def choke_mach(open_area_ratio: float = 0.9) -> float:
    """Subsonic free-stream Mach at which the cells choke (A/A* = 1/phi)."""
    target = 1.0 / open_area_ratio
    lo, hi = 1e-3, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _area_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def swallow_mach(open_area_ratio: float = 0.9) -> float:
    """Supersonic Mach at which the bow shock is swallowed (Kantrowitz-Donaldson)."""
    target = 1.0 / open_area_ratio
    lo, hi = 1.0001, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _area_ratio(_normal_shock_m2(mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gridfin_cn_alpha(mach, open_area_ratio: float = 0.9, choke_loss: float = 0.45):
    """Normal-force-slope multiplier vs Mach (1 at M -> 0)."""
    m = np.atleast_1d(np.abs(np.asarray(mach, dtype=float)))
    mch, msw = choke_mach(open_area_ratio), swallow_mach(open_area_ratio)
    sub = (1 - np.minimum(m, mch) ** 2) ** -0.25
    m_ch_val = (1 - mch * mch) ** -0.25
    x = np.clip((m - mch) / (msw - mch), 0, 1)
    trans = m_ch_val * (1 - choke_loss * np.sin(math.pi * x))
    beta_sw = math.sqrt(msw * msw - 1)
    sup = m_ch_val * beta_sw / np.sqrt(np.maximum(m * m - 1, beta_sw**2 * 1e-6))
    out = np.where(m <= mch, sub, np.where(m >= msw, sup, trans))
    return out if np.ndim(mach) else float(out[0])


def gridfin_cd0(mach, open_area_ratio: float = 0.9, peak: float = 1.9, floor: float = 0.6):
    """Zero-deflection drag multiplier vs Mach (1 at M -> 0)."""
    m = np.atleast_1d(np.abs(np.asarray(mach, dtype=float)))
    mch, msw = choke_mach(open_area_ratio), swallow_mach(open_area_ratio)
    x = np.clip((m - mch) / (msw - mch), 0, 1)
    # rise to the peak at 40 % of the choked band, then down to the swallow value
    at_sw = 0.5 * (1 + peak)
    rise = 1 + (peak - 1) * np.sin(0.5 * math.pi * np.clip(x / 0.4, 0, 1))
    fall = peak + (at_sw - peak) * np.clip((x - 0.4) / 0.6, 0, 1)
    trans = np.where(x <= 0.4, rise, fall)
    beta_sw = math.sqrt(msw * msw - 1)
    sup = np.maximum(at_sw * np.sqrt(beta_sw / np.sqrt(np.maximum(m * m - 1, 1e-12))), floor)
    out = np.where(m <= mch, 1.0, np.where(m >= msw, sup, trans))
    return out if np.ndim(mach) else float(out[0])


def gridfin_uncertainty(mach, open_area_ratio: float = 0.9):
    """1-sigma relative uncertainty of both multipliers."""
    m = np.abs(np.asarray(mach, dtype=float))
    mch, msw = choke_mach(open_area_ratio), swallow_mach(open_area_ratio)
    return np.where(m < mch - 0.05, 0.15, np.where(m <= msw + 0.1, 0.35, 0.20))
