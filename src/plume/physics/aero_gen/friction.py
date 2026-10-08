"""Compressible turbulent skin friction on flat plates (applied to bodies and fins).

* Incompressible: Karman-Schoenherr mean skin friction, 0.242/sqrt(Cf) = log10(Re Cf)
  (Schoenherr 1932; the incompressible law used by van Driest II).
* Compressibility, default **van Driest II** (van Driest, Aero. Eng. Rev. 15(10),
  1956) in the Spalding-Chi form recommended by Hopkins & Inouye (AIAA J. 9(6),
  1971; cited by Jorgensen TN D-6996 for booster drag): Cf = Cf_inc(Re F_Rx) / F_c
  with F_c = (T_aw/T_e - 1) / (asin a + asin b)^2, F_Rtheta = mu_e/mu_w,
  F_Rx = F_Rtheta / F_c.  Adiabatic wall by default (recovery factor 0.89).
* Alternative: Eckert's reference-temperature method (J. Aero. Sci. 22(8), 1955),
  T* = T_e (1 + 0.032 M^2 + 0.58 (T_w/T_e - 1)).
* Roughness: Schlichting's fully-rough flat plate, Cf = (1.89 + 1.62 log10(L/k))^-2.5
  (Boundary-Layer Theory, ch. 21), with the same compressibility factor; the
  larger of smooth and rough is used (OpenRocket practice, Niskanen 2009 sec. 3.4.1).
* The boundary layer is assumed fully turbulent from the nose (the usual
  assumption for rocket bodies, and conservative); transition is not modelled.
  ``cf_laminar`` (Blasius) is provided for reference only.

Validity: M 0-10, Re 1e5-1e9, T_w/T_e near adiabatic; +-10 % (Hopkins & Inouye).
"""

from __future__ import annotations

import numpy as np

from plume.physics.aerodb import sutherland_viscosity

GAMMA = 1.4
RECOVERY = 0.89


def cf_karman_schoenherr(re) -> np.ndarray:
    """Mean incompressible turbulent skin friction (Newton iteration, vectorised)."""
    re = np.maximum(np.asarray(re, dtype=float), 1e3)
    cf = 0.074 / re**0.2
    for _ in range(30):
        # f(cf) = 0.242/sqrt(cf) - log10(re cf) = 0
        f = 0.242 / np.sqrt(cf) - np.log10(re * cf)
        df = -0.121 * cf**-1.5 - 1.0 / (cf * np.log(10.0))
        cf = np.clip(cf - f / df, 1e-5, 0.1)
    return cf


def cf_laminar(re) -> np.ndarray:
    return 1.328 / np.sqrt(np.maximum(np.asarray(re, dtype=float), 1.0))


def cf_fully_rough(length, roughness) -> np.ndarray:
    ratio = np.maximum(np.asarray(length, dtype=float) / max(roughness, 1e-12), 10.0)
    return (1.89 + 1.62 * np.log10(ratio)) ** -2.5


def van_driest_ii_factors(mach, t_e, t_w=None):
    """(F_c, F_Rx) of van Driest II; ``t_w`` None = adiabatic wall."""
    m = np.asarray(mach, dtype=float)
    t_e = np.asarray(t_e, dtype=float)
    t_aw = t_e * (1 + RECOVERY * 0.5 * (GAMMA - 1) * m * m)
    t_w = t_aw if t_w is None else np.asarray(t_w, dtype=float)
    a2 = RECOVERY * 0.5 * (GAMMA - 1) * m * m * t_e / t_w
    b = t_aw / t_w - 1.0
    den = np.sqrt(b * b + 4 * a2)
    small = den < 1e-9
    den = np.where(small, 1.0, den)
    alpha = np.clip((2 * a2 - b) / den, -1, 1)
    beta = np.clip(b / den, -1, 1)
    s = np.arcsin(alpha) + np.arcsin(beta)
    fc = np.where(small, 1.0, (t_aw / t_e - 1.0) / np.where(small, 1.0, s * s))
    frt = sutherland_viscosity(t_e) / sutherland_viscosity(t_w)
    return fc, frt / fc


def skin_friction(
    re_length,
    mach,
    t_e=288.15,
    length: float = 1.0,
    roughness: float = 0.0,
    method: str = "van_driest_ii",
):
    """Mean skin-friction coefficient of a plate of ``length`` (vectorised).

    ``re_length`` = Re based on the run length; ``t_e`` free-stream static
    temperature (K); ``roughness`` equivalent sand-grain height (m)."""
    re = np.maximum(np.asarray(re_length, dtype=float), 1.0)
    m = np.asarray(mach, dtype=float)
    if method == "van_driest_ii":
        fc, frx = van_driest_ii_factors(m, t_e)
        cf = cf_karman_schoenherr(re * frx) / fc
        comp = 1.0 / fc
    elif method == "eckert":
        t_e = np.asarray(t_e, dtype=float)
        t_aw = t_e * (1 + RECOVERY * 0.5 * (GAMMA - 1) * m * m)
        t_star = t_e * (1 + 0.032 * m * m + 0.58 * (t_aw / t_e - 1))
        ratio = t_e / t_star
        re_star = re * ratio * sutherland_viscosity(t_e) / sutherland_viscosity(t_star)
        cf = cf_karman_schoenherr(re_star) * ratio
        comp = ratio
    else:
        raise ValueError(f"unknown skin-friction method {method!r}")
    if roughness > 0:
        cf = np.maximum(cf, cf_fully_rough(length, roughness) * comp)
    return cf
