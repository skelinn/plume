"""Base drag (power off / power on) and blunt-face (engine-first) forebody drag.

Power-off base drag of a blunt (non-boattailed) base, coefficient on the base area:

* subsonic: Hoerner, *Fluid-Dynamic Drag* (1965) ch. 3: C_Db = 0.029 / sqrt(C_Df)
  with C_Df the forebody skin-friction drag referred to the base area, times the
  Mach growth (1 + 1.083 M^2) of the widely used fit 0.12 + 0.13 M^2 (Hoerner via
  Niskanen 2009, eq. 3.94) so that C_Db(M=1) ~ 2.1 C_Db(0);
* supersonic: Love's compilation for bodies of revolution with turbulent boundary
  layers (NACA TN 3819, 1957) as plotted in Jorgensen TN D-6996 Fig. 8 (digitised),
  M 1-8, with Gabeaud's formula (TN D-6996 eq. 11) beyond M 8 (-> vacuum limit);
* 0.9 < M < 1.1: linear blend.

Power on (engine running, nose first): see ``plume.physics.aerodb.base_drag_power_on_factor``.

Blunt face leading (engine-first flight, or a flat-nosed body): the face carries a
nearly uniform fraction of the stagnation pressure, CA_face = k(M) Cp_stag(M)
(TN D-6996 p. 10: "CA_W ~= Cp_stag" for a flat face).  ``k`` = 0.75 subsonic (sharp
edged flat-faced cylinder, Hoerner 1965 ch. 3) rising to 0.85 by M 1.2; the
supersonic value is calibrated on Jernell's flat-faced cylinders and base-first
cone-cylinders at M 2.86 (NASA TM X-1658 via TN D-6996 Figs. 10-11), i.e. it is not
an independent prediction there.  Uncertainty +-15 %.
"""

from __future__ import annotations

import numpy as np

from plume.physics.aero_gen.newtonian import cp_max_pitot

#: Love (NACA TN 3819) base pressure, bodies of revolution, turbulent BL, alpha 0,
#: as -Cp_B vs M (TN D-6996 Fig. 8 "Analysis of Love", digitised).
LOVE_BASE = np.array(
    [
        [1.0, 0.188],
        [1.25, 0.183],
        [1.5, 0.175],
        [1.75, 0.160],
        [2.0, 0.145],
        [2.5, 0.115],
        [3.0, 0.093],
        [3.5, 0.080],
        [4.0, 0.066],
        [4.5, 0.055],
        [5.0, 0.046],
        [6.0, 0.032],
        [7.0, 0.024],
        [8.0, 0.018],
    ]
).T


def gabeaud_base_cp(mach, gamma: float = 1.4):
    """Gabeaud (J. Aero. Sci. 17(8), 1950) base pressure coefficient, supersonic."""
    m = np.asarray(mach, dtype=float)
    return (
        2
        / (gamma * m * m)
        * (
            (2 / (gamma + 1)) ** gamma
            * (1 / m) ** (2 * gamma)
            * ((2 * gamma * m * m - (gamma - 1)) / (gamma + 1))
            - 1
        )
    )


def stagnation_cp(mach):
    """Stagnation pressure coefficient: isentropic (M <= 1) / Rayleigh pitot (M > 1)."""
    m = np.atleast_1d(np.asarray(mach, dtype=float))
    return np.array([cp_max_pitot(float(x)) for x in m.ravel()]).reshape(m.shape)


def base_drag_power_off(mach, cdf_forebody):
    """Power-off base drag coefficient on the base area (vectorised in both args)."""
    m = np.asarray(mach, dtype=float)
    cdf = np.maximum(np.asarray(cdf_forebody, dtype=float), 1e-3)
    sub = 0.029 / np.sqrt(cdf) * (1.0 + 1.083 * np.minimum(m, 1.0) ** 2)
    sup_love = np.interp(m, LOVE_BASE[0], LOVE_BASE[1])
    beyond = m > 8.0
    if np.any(beyond):
        scale = LOVE_BASE[1][-1] / -float(gabeaud_base_cp(8.0))
        sup_love = np.where(beyond, -gabeaud_base_cp(np.maximum(m, 8.0)) * scale, sup_love)
    w = np.clip((m - 0.9) / 0.2, 0.0, 1.0)
    return (1 - w) * sub + w * sup_love


def face_pressure_factor(mach):
    """Mean face pressure / stagnation pressure for a sharp-edged flat face."""
    return 0.75 + 0.10 * np.clip((np.asarray(mach, dtype=float) - 0.8) / 0.4, 0.0, 1.0)


def flat_face_drag(mach):
    """Pressure drag of a flat face leading (on the face area)."""
    return face_pressure_factor(mach) * stagnation_cp(mach)


def trailing_cone_drag(mach) -> np.ndarray:
    """Pressure drag of a pointed nose trailing in reversed (engine-first) flow.

    Treated as negligible (closed afterbody, attached or weakly separated flow);
    kept as a function so a better model can be dropped in.  Uncertainty: the face
    + friction build-up of engine-first CA is quoted at +-15 %."""
    return np.zeros_like(np.asarray(mach, dtype=float))


__all__ = [
    "base_drag_power_off",
    "face_pressure_factor",
    "flat_face_drag",
    "gabeaud_base_cp",
    "stagnation_cp",
    "trailing_cone_drag",
]
