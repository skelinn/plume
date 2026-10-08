"""Planar fins (hobby-rocket fin sets): normal force, centre of pressure, drag, damping.

Normal-force slope of the fin set (per rad, on the body reference area):

* subsonic (M <= 0.9): Barrowman (M.S. thesis, Catholic Univ. of America, 1967;
  Barrowman & Barrowman, NARAM-8, 1966) with Diederich's compressibility form,
  CN_a1 = 2 pi AR (A_f/A_ref) / (2 + sqrt(4 + (beta AR / cos G_c)^2)), times n/2
  for n = 3, 4 fins (OpenRocket factors for 5-8, Niskanen 2009 Table 3.3);
* supersonic (M >= 1.2, or the Mach where beta AR >= 1 if higher): Ackeret linear
  theory (ZFM 16, 1925), 4/beta per unit panel area, with the rectangular-wing tip
  correction (1 - 1/(2 beta AR)) for beta AR >= 1 and a linear blend to the
  slender-wing value pi AR/2 (R.T. Jones, NACA Rep. 835, 1946) for beta AR < 1.
  Busemann's second-order terms cancel for a symmetric flat plate at incidence,
  so the second-order normal force equals Ackeret's;
* transonic: linear blend in Mach between the two;
* body-fin interference K_T(B) = 1 + R/(s + R) (Barrowman 1967).

Centre of pressure (aft of the root leading edge): MAC leading edge + 0.25 MAC for
M <= 0.5 (Barrowman), (AR beta - 0.67)/(2 AR beta - 1) MAC for M >= 2 (Niskanen
2009 eq. 3.35, after DATCOM), linear in Mach between.

High alpha / reversed flow: flat-plate model CN = CN_a sin a' cos a' + C_D90 sin^2 a'
per panel (a' = min(a, 180 - a); Hoerner 1965 ch. 3, C_D90 = 1.2 for low-aspect-
ratio plates); the crossflow term acts at the planform centroid.  In reversed flow
the linear term acts 0.25 MAC ahead of the (new) leading edge, i.e. at 0.75 MAC.

Drag (alpha 0): skin friction on both sides (x (1 + 2 t/c), Hoerner), rounded or
square leading edge and square trailing edge pressure drag (Hoerner via Niskanen
2009 eqs. 3.89-3.93).

Validity: thin flat-plate trapezoidal fins, n >= 3; +-15 % on CN_alpha subsonic and
supersonic, +-25 % transonic (0.8 < M < 1.3), CP +-0.1 MAC.
"""

from __future__ import annotations

import math

import numpy as np

from plume.physics.aero_gen.base import stagnation_cp
from plume.physics.aero_gen.friction import skin_friction
from plume.physics.aero_gen.geometry import FinPlanform, fin_count_factor, interference_factor

C_D90_PLATE = 1.2


def panel_slope(mach, fin: FinPlanform) -> np.ndarray:
    """Normal-force slope per rad per unit panel area of one panel (no interference)."""
    m = np.atleast_1d(np.asarray(mach, dtype=float))
    ar = fin.aspect_ratio
    cosg = math.cos(fin.midchord_sweep)

    def sub(mm):
        beta = np.sqrt(np.maximum(1 - mm * mm, 1e-9))
        return 2 * math.pi * ar / (2 + np.sqrt(4 + (beta * ar / cosg) ** 2))

    def sup(mm):
        beta = np.sqrt(np.maximum(mm * mm - 1, 1e-9))
        bar = beta * ar
        ack = 4 / beta * (1 - 1 / (2 * np.maximum(bar, 1e-9)))
        slender = math.pi * ar / 2
        low = slender + (2 * ar - slender) * np.clip(bar, 0, 1)
        return np.where(bar >= 1, ack, low)

    m_sub, m_sup = 0.9, 1.2
    a_sub = sub(np.minimum(m, m_sub))
    a_sup = sup(np.maximum(m, m_sup))
    w = np.clip((m - m_sub) / (m_sup - m_sub), 0, 1)
    out = np.where(
        m <= m_sub, a_sub, np.where(m >= m_sup, a_sup, (1 - w) * sub(m_sub) + w * sup(m_sup))
    )
    return out


def fin_cn_alpha(mach, fin: FinPlanform, ref_area: float, interference: bool = True):
    """CN_alpha of the whole fin set on ``ref_area`` (per rad), incl. K_T(B)."""
    k = interference_factor(fin) if interference else 1.0
    return panel_slope(mach, fin) * fin.area / ref_area * fin_count_factor(fin.count) * k


def _cp_frac_supersonic(mach, ar: float):
    beta = np.sqrt(np.maximum(np.asarray(mach, dtype=float) ** 2 - 1, 1e-9))
    den = 2 * ar * beta - 1
    frac = np.where(den > 0.1, (ar * beta - 0.67) / np.where(den > 0.1, den, 1.0), 0.5)
    return np.clip(frac, 0.25, 0.5)


def fin_cp_x(mach, fin: FinPlanform, reversed_flow: bool = False):
    """Centre of pressure of the linear normal force, distance aft of the root LE."""
    m = np.atleast_1d(np.asarray(mach, dtype=float))
    ar = fin.aspect_ratio
    at2 = float(_cp_frac_supersonic(2.0, ar))
    mid = 0.25 + (at2 - 0.25) * (m - 0.5) / 1.5
    frac = np.where(m <= 0.5, 0.25, np.where(m >= 2.0, _cp_frac_supersonic(m, ar), mid))
    if reversed_flow:
        frac = 1.0 - frac
    return fin.mac_x_le + frac * fin.mac


def fin_normal_force(alpha_deg, mach, fin: FinPlanform, ref_area: float):
    """(CN, x_aft_of_root_LE of the resultant) of the fin set for alpha_t 0-180 deg."""
    a = np.radians(np.asarray(alpha_deg, dtype=float))
    ap = np.where(a <= np.pi / 2, a, np.pi - a)
    rev = a > np.pi / 2
    cna = fin_cn_alpha(mach, fin, ref_area)
    lin = cna * np.sin(ap) * np.cos(ap)
    cross = fin_count_factor(fin.count) * fin.area / ref_area * C_D90_PLATE * np.sin(ap) ** 2
    x_lin = np.where(rev, fin_cp_x(mach, fin, True), fin_cp_x(mach, fin, False))
    x_cross = fin.centroid_x
    cn = lin + cross
    x = np.where(cn > 1e-12, (lin * x_lin + cross * x_cross) / np.where(cn > 1e-12, cn, 1.0), x_lin)
    return cn, x


def rounded_le_drag(mach):
    """Rounded leading-edge pressure drag on the LE frontal area (Hoerner/Niskanen)."""
    m = np.asarray(mach, dtype=float)
    sub = (1 - np.minimum(m, 0.9) ** 2) ** -0.417 - 1
    tra = 1 - 1.785 * (m - 0.9)
    msup = np.maximum(m, 1.0)
    sup = 1.214 - 0.502 / msup**2 + 0.1095 / msup**4
    return np.where(m < 0.9, sub, np.where(m < 1.0, tra, sup))


def square_te_drag(mach):
    """Square trailing-edge base drag on the TE area (0.12 + 0.13 M^2 / 0.25 / M)."""
    m = np.asarray(mach, dtype=float)
    return np.where(m < 1.0, 0.12 + 0.13 * m * m, 0.25 / np.maximum(m, 1.0))


def fin_axial_drag(
    mach,
    re_per_m,
    t_e,
    fin: FinPlanform,
    ref_area: float,
    roughness: float = 0.0,
    method: str = "van_driest_ii",
):
    """Zero-alpha axial-force coefficient of the fin set on ``ref_area`` (broadcasts)."""
    m = np.asarray(mach, dtype=float)
    re = np.asarray(re_per_m, dtype=float) * fin.mac
    cf = skin_friction(re, m, t_e, length=fin.mac, roughness=roughness, method=method)
    tc = fin.thickness / fin.mac
    n = fin.count
    friction = cf * (1 + 2 * tc) * 2 * fin.area * n / ref_area
    le_area = n * fin.span * fin.thickness
    if fin.le_shape == "rounded":
        le = rounded_le_drag(m) * math.cos(fin.le_sweep) ** 2
    else:
        le = 0.85 * stagnation_cp(m) * math.cos(fin.le_sweep) ** 2
    te = square_te_drag(m) if fin.te_shape == "square" else np.zeros_like(m)
    pressure = (le + te) * le_area / ref_area
    return friction + pressure


def fin_roll_damping(mach, fin: FinPlanform, ref_area: float, ref_length: float):
    """Clp of the fin set (strip theory, each panel at its local incidence p r / V)."""
    a_f = panel_slope(mach, fin)
    return -2 * fin.count * a_f * fin.roll_inertia_integral() / (ref_area * ref_length**2)
