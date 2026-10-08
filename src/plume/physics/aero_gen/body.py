"""Body-of-revolution normal force and pitching moment for alpha_t = 0 ... 180 deg.

Method: Allen's proposal (NACA RM A9I26, 1949; Allen & Perkins, NACA Rep. 1048,
1951) -- slender-body potential lift plus viscous crossflow -- in the 0-180 deg form
of Jorgensen, NASA TN D-6996 (1973), eqs. (1), (4), (5):

    CN = (Ab/A) sin(2a') cos(a'/2) + eta Cdn (Ap/A) sin^2 a'
    a' = a (a <= 90 deg),  a' = 180 deg - a (a > 90 deg)

The potential term acts at x = l - V/Ab from the nose tip for a <= 90 deg (e.g.
2/3 of a cone's length) and at x = V/Ab for a > 90 deg (base leading); the viscous
term acts at the planform centroid x_c.  ``Cdn(M_n, Re_n)`` is the crossflow drag
coefficient of a circular cylinder (TN D-6996 Figs. 1-2, digitised; crossflow Mach
M_n = M sin a, crossflow Reynolds Re_n = Re_d sin a) and ``eta`` the finite-length
factor (Fig. 4, from Goldstein 1938), taken as 1 for supersonic free streams as
recommended by Jorgensen.

Validity: bodies of revolution with fineness >~ 5, M 0-8 (Cdn tabulated to M_n 8),
any alpha.  Asymmetric vortex side forces (alpha 25-65 deg, low speed) are not
modelled.  Uncertainty: see ``generate`` (CN +-15 %, +-25 % in the critical
crossflow-Reynolds band).
"""

from __future__ import annotations

import numpy as np

from plume.physics.aero_gen.geometry import BodyGeometry

#: Crossflow drag coefficient of circular cylinders vs crossflow Mach number,
#: subcritical Reynolds number: faired curve of TN D-6996 Fig. 1 (digitised; the
#: dashed transonic part is "very limited data" per Jorgensen).
CDN_MACH = np.array(
    [
        [0.00, 1.20],
        [0.20, 1.20],
        [0.30, 1.22],
        [0.40, 1.28],
        [0.50, 1.38],
        [0.60, 1.55],
        [0.70, 1.61],
        [0.80, 1.55],
        [0.85, 1.60],
        [0.90, 1.85],
        [0.95, 2.10],
        [1.00, 2.05],
        [1.10, 1.83],
        [1.20, 1.67],
        [1.30, 1.58],
        [1.40, 1.56],
        [1.60, 1.49],
        [2.00, 1.41],
        [2.60, 1.36],
        [3.00, 1.345],
        [3.60, 1.33],
        [4.00, 1.31],
        [4.80, 1.29],
        [6.00, 1.285],
        [8.00, 1.28],
    ]
).T

#: Ratio Cdn(Re_n) / 1.2 at subcritical crossflow Mach (M_n <~ 0.4): middle of the
#: data band of TN D-6996 Fig. 2 (Wieselsberger, Stack, Polhamus, Roshko, Schmidt,
#: Jones et al.), digitised.  Columns: log10(Re_n), Cdn.
CDN_REYNOLDS = np.array(
    [
        [4.0, 1.12],
        [4.2, 1.18],
        [4.5, 1.21],
        [5.0, 1.22],
        [5.25, 1.20],
        [5.3, 1.15],
        [5.4, 1.05],
        [5.5, 0.88],
        [5.55, 0.78],
        [5.6, 0.55],
        [5.65, 0.32],
        [5.7, 0.28],
        [5.8, 0.27],
        [5.9, 0.28],
        [6.0, 0.32],
        [6.2, 0.40],
        [6.4, 0.46],
        [6.6, 0.52],
        [6.8, 0.58],
        [7.0, 0.65],
    ]
).T

#: Finite-length crossflow drag factor eta vs cylinder length/diameter, circular
#: cylinders at Re_n = 88 000 (Goldstein 1938 via TN D-6996 Fig. 4, digitised).
ETA_TABLE = np.array(
    [
        [0.0, 0.53],
        [1.0, 0.55],
        [2.0, 0.57],
        [3.0, 0.595],
        [5.0, 0.62],
        [10.0, 0.675],
        [20.0, 0.75],
        [30.0, 0.795],
        [40.0, 0.82],
        [1e9, 1.0],
    ]
).T


def crossflow_cd_mach(mach_n):
    """Subcritical-Re crossflow drag coefficient vs crossflow Mach (TN D-6996 Fig. 1)."""
    return np.interp(mach_n, CDN_MACH[0], CDN_MACH[1])


def crossflow_cd_reynolds_ratio(re_n):
    """Cdn(Re_n)/Cdn_subcritical for M_n <~ 0.4 (TN D-6996 Fig. 2)."""
    lre = np.log10(np.maximum(np.asarray(re_n, dtype=float), 1.0))
    return np.interp(lre, CDN_REYNOLDS[0], CDN_REYNOLDS[1]) / 1.22


def crossflow_cd(mach_n, re_n=None, reynolds_effect: bool = True):
    """Crossflow drag coefficient C_dn(M_n, Re_n) of a circular cylinder.

    The drag-crisis reduction of Fig. 2 applies fully for M_n <= 0.3 and fades out by
    M_n = 0.5, where compressibility suppresses it (Fig. 1 squares, Fig. 3)."""
    c = crossflow_cd_mach(mach_n)
    if re_n is None or not reynolds_effect:
        return c
    r = crossflow_cd_reynolds_ratio(re_n)
    w = np.clip((0.5 - np.asarray(mach_n, dtype=float)) / 0.2, 0.0, 1.0)
    return c * (1.0 - w * (1.0 - r))


def eta_factor(fineness, mach):
    """Finite-length factor eta(l/d) (subsonic) blended to 1 between M 0.8 and 1.2."""
    eta_sub = np.interp(fineness, ETA_TABLE[0], ETA_TABLE[1])
    w = np.clip((np.asarray(mach, dtype=float) - 0.8) / 0.4, 0.0, 1.0)
    return eta_sub + (1.0 - eta_sub) * w


def jorgensen_body(
    alpha_deg,
    mach,
    re_d,
    body: BodyGeometry,
    *,
    reynolds_effect: bool = True,
    reverse_potential: float = 1.0,
) -> dict[str, np.ndarray]:
    """Normal-force components and their stations for the body alone (broadcasts).

    Returns ``CN_pot``, ``CN_visc`` (on the body cross-section area) and ``x_pot``,
    ``x_visc`` (distance aft of the nose tip where each acts), plus ``Cdn``/``eta``.
    ``reverse_potential`` scales the potential term for alpha > 90 deg (1 =
    Jorgensen; 0 = viscous crossflow only, like Plume's fast strip model)."""
    a = np.radians(np.asarray(alpha_deg, dtype=float))
    mach = np.asarray(mach, dtype=float)
    ap = np.where(a <= np.pi / 2, a, np.pi - a)
    A = body.ref_area
    Ab = body.base_area
    Ap = body.planform_area
    V = body.volume
    L = body.length
    rev = a > np.pi / 2
    pot = (Ab / A) * np.sin(2 * ap) * np.cos(ap / 2)
    pot = np.where(rev, reverse_potential * pot, pot)
    s = np.sin(a)
    mn = mach * s
    re_n = None if re_d is None else np.asarray(re_d, dtype=float) * s
    cdn = crossflow_cd(mn, re_n, reynolds_effect)
    eta = eta_factor(body.fineness, mach)
    visc = eta * cdn * (Ap / A) * np.sin(ap) ** 2
    x_pot = np.where(rev, V / Ab, L - V / Ab)
    shape = np.broadcast_shapes(pot.shape, visc.shape, np.shape(eta))
    return {
        "CN_pot": np.broadcast_to(pot, shape),
        "CN_visc": np.broadcast_to(visc, shape),
        "x_pot": np.broadcast_to(x_pot, shape),
        "x_visc": np.full(shape, body.planform_centroid_x),
        "Cdn": np.broadcast_to(cdn, shape),
        "eta": np.broadcast_to(eta, shape),
    }


def body_cn_cm(
    alpha_deg, mach, re_d, body: BodyGeometry, x_ref_z: float, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """(CN, Cm about body-z ``x_ref_z``) for the body alone, Plume sign convention
    (Cm > 0 increases alpha_t; equals Jorgensen's Cm about the same point)."""
    j = jorgensen_body(alpha_deg, mach, re_d, body, **kw)
    d = body.diameter
    z_pot = body.length - j["x_pot"]
    z_visc = body.length - j["x_visc"]
    cn = j["CN_pot"] + j["CN_visc"]
    cm = (j["CN_pot"] * (z_pot - x_ref_z) + j["CN_visc"] * (z_visc - x_ref_z)) / d
    return cn, cm
