"""Modified Newtonian impact theory for a body of revolution at any alpha_t.

Cp = Cp_max (V_hat . n)^2 on panels facing the flow, 0 in the shadow (Lees 1955's
modification of Newton's sine-squared law), with Cp_max the stagnation-pressure
coefficient behind a normal shock (Rayleigh pitot formula, NACA Rep. 1135 eq. 100).
The surface is the nose + cylinder + flat base face, panelled in x and azimuth.

Used for alpha_t-dependent normal force / pitching moment at hypersonic Mach
(blended in between M 4 and 6 by ``generate``).  Validity: M >~ 5 and surfaces
steeper than the Mach angle; it under-predicts slender-cone pressures (no
centrifugal correction) and ignores base pressure and viscous effects.
"""

from __future__ import annotations

import math

import numpy as np

from plume.physics.aero_gen.geometry import BodyGeometry


def cp_max_pitot(mach: float, gamma: float = 1.4) -> float:
    """Stagnation pressure coefficient behind a normal shock (Rayleigh pitot).

    For M <= 1 the isentropic stagnation value is returned."""
    m2 = mach * mach
    if mach <= 1.0:
        if mach <= 1e-6:
            return 1.0
        return ((1 + 0.5 * (gamma - 1) * m2) ** (gamma / (gamma - 1)) - 1) / (0.5 * gamma * m2)
    p02_p1 = ((gamma + 1) ** 2 * m2 / (4 * gamma * m2 - 2 * (gamma - 1))) ** (
        gamma / (gamma - 1)
    ) * ((1 - gamma + 2 * gamma * m2) / (gamma + 1))
    return (p02_p1 - 1) / (0.5 * gamma * m2)


def newtonian_body(
    body: BodyGeometry,
    alpha_deg,
    mach: float,
    x_ref_z: float,
    n_x: int = 240,
    n_phi: int = 96,
    cp_max: float | None = None,
) -> dict[str, np.ndarray]:
    """Pressure CA, CN and Cm (about body z ``x_ref_z``) by modified Newtonian theory.

    Sign conventions as in ``plume.physics.aerodb`` (CA > 0 pushes toward the base;
    CN > 0 opposes the lateral velocity; Cm > 0 increases alpha_t)."""
    cpm = cp_max_pitot(mach) if cp_max is None else cp_max
    alpha = np.atleast_1d(np.radians(np.asarray(alpha_deg, dtype=float)))
    L = body.length
    # lateral surface: x stations (aft of the tip) clustered at the nose
    s = np.linspace(0.0, 1.0, n_x + 1)
    xn = body.nose_length * s * s if body.nose_length > 0 else np.zeros(1)
    xs = np.unique(np.concatenate([xn, np.linspace(body.nose_length, L, 12)]))
    rs = body.radius_at(xs)
    xm = 0.5 * (xs[1:] + xs[:-1])
    rm = 0.5 * (rs[1:] + rs[:-1])
    dx = np.diff(xs)
    dr = np.diff(rs)
    ds = np.hypot(dx, dr)
    keep = ds > 0
    xm, rm, dx, dr, ds = xm[keep], rm[keep], dx[keep], dr[keep], ds[keep]
    phi = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
    dphi = 2 * math.pi / n_phi
    # outward normal (x aft, y, z): (-dr/ds, dx/ds cos phi, dx/ds sin phi)
    nx = (-dr / ds)[:, None] * np.ones_like(phi)[None, :]
    ny = (dx / ds)[:, None] * np.cos(phi)[None, :]
    nz = (dx / ds)[:, None] * np.sin(phi)[None, :]
    area = (rm * ds)[:, None] * dphi * np.ones_like(phi)[None, :]
    px = xm[:, None] * np.ones_like(phi)[None, :]
    py = rm[:, None] * np.cos(phi)[None, :]
    # faces: flat nose face (if no nose) at x = 0, base face at x = L
    faces = [(L, 1.0)]
    if body.nose_length <= 0:
        faces.append((0.0, -1.0))
    A = body.ref_area
    d = body.diameter
    x_r = L - x_ref_z
    out_ca, out_cn, out_cm = [], [], []
    for a in alpha:
        # air velocity direction relative to the body: aft (+x) at alpha 0, the
        # windward side is +y, so the vehicle's lateral velocity is +y.
        ux, uy = math.cos(a), -math.sin(a)
        un = ux * nx + uy * ny
        cp = np.where(un < 0, cpm * un * un, 0.0)
        f = -cp * area  # force per q along the outward normal
        fx = float(np.sum(f * nx))
        fy = float(np.sum(f * ny))
        mz = float(np.sum((px - x_r) * f * ny - py * f * nx))
        _ = nz  # z components cancel by symmetry
        for _xf, sign in faces:
            un_f = ux * sign
            if un_f < 0:
                cpf = cpm * un_f * un_f
                fxf = -cpf * A * sign
                fx += fxf
        out_ca.append(fx / A)
        out_cn.append(-fy / A)
        out_cm.append(mz / (A * d))
    return {"CA": np.array(out_ca), "CN": np.array(out_cn), "Cm": np.array(out_cm)}
