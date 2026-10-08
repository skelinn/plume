"""Supersonic retro-propulsion (SRP): aerodynamic axial force during engine-first burns.

With a retro-rocket firing into the oncoming flow the plume displaces the bow shock
and shields the forebody, so the *aerodynamic* part of the axial force collapses
as the thrust coefficient C_T = T / (q S_ref) grows.  The survey of Korzun, Braun &
Cruz ("Survey of supersonic retropropulsion technology for Mars entry, descent, and
landing", J. Spacecraft & Rockets 46(5), 2009), drawing mainly on the wind-tunnel
data of Jarvinen & Adams (NASA CR-66918, 1970; 60 deg sphere-cone, M 2 and 4) and
Keyes & Hefner (NASA TN D-3919, 1967), reports that for a *single central nozzle*
the forebody pressure drag is largely lost by C_T ~ 1 and essentially gone for
C_T >~ 2, while peripheral multi-nozzle layouts preserve some drag at C_T < ~1.

Engineering fit (single central nozzle):

    CA_aero(C_T) / CA_aero(0) = exp(-C_T / C_T*),   C_T* = 0.4 (1-sigma +-50 %)

Validity: 1.5 <= M <= 4 (data range), C_T 0-10; outside that (e.g. subsonic
landing burns) the same trend is applied with the uncertainty doubled.  The
normal force under SRP is left unchanged (not modelled).  The runtime model uses
the copy in ``plume.physics.aerodb.srp_axial_factor``.
"""

from __future__ import annotations

import numpy as np

CT_DECAY = 0.4
CT_DECAY_SIGMA_REL = 0.5
VALID_MACH = (1.5, 4.0)


def srp_axial_factor(ct, ct_decay: float = CT_DECAY):
    """Fraction of the power-off engine-first aerodynamic axial force that remains."""
    ct = np.maximum(np.asarray(ct, dtype=float), 0.0)
    return np.exp(-ct / ct_decay)


def srp_axial_factor_sigma(ct, mach, ct_decay: float = CT_DECAY):
    """1-sigma absolute uncertainty of the factor (from +-50 % on C_T*; doubled
    outside the validated Mach range)."""
    ct = np.maximum(np.asarray(ct, dtype=float), 0.0)
    f = srp_axial_factor(ct, ct_decay)
    df = f * ct / ct_decay * CT_DECAY_SIGMA_REL  # |d f / d ln C_T*| * sigma_rel
    m = np.asarray(mach, dtype=float)
    inside = (m >= VALID_MACH[0]) & (m <= VALID_MACH[1])
    return np.where(inside, df, 2 * df)


def thrust_coefficient(thrust, q, ref_area):
    q = np.asarray(q, dtype=float)
    return np.where(
        q > 0, np.asarray(thrust, dtype=float) / (np.maximum(q, 1e-12) * ref_area), np.inf
    )
