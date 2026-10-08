"""Assemble an :class:`~plume.physics.aerodb.AeroDatabase` from a ``VehicleSpec``.

Component build-up (each module documents its equations and references):

=====================  ======================================================
normal force / moment  body: ``body.jorgensen_body`` (Allen/Jorgensen 0-180 deg),
                       blended into modified Newtonian (``newtonian``) between
                       M 4 and 6; fins: ``fins.fin_normal_force``
axial, nose first      skin friction (``friction``, x Hoerner form factor) +
                       nose wave drag (``nose``) + base drag (``base``, kept
                       separately as ``CA_base`` for the power-on model) + fins
                       (``fins.fin_axial_drag``) + landing legs (drag areas)
axial, engine first    blunt-face pressure (``base.flat_face_drag``) + friction
                       + fins + legs (footpads leading)
axial vs alpha         Jorgensen TN D-6996 eqs. (2)-(3): CA(0) cos^2 a, CA(180) cos^2 a
pitch damping          lever-arm spread of the component normal-force slopes about
                       the centre of pressure (quasi-steady strip argument) with a
                       describing-function slope for the quadratic viscous
                       crossflow at ``damping_amplitude_deg``
roll damping           fin strip theory (``fins.fin_roll_damping``)
=====================  ======================================================

1-sigma uncertainty bands (``sigma`` tables) -- engineering judgement anchored on the
comparisons in ``docs/models/aero.md``:

* CA: 10 % subsonic (M < 0.8), 20 % transonic (0.8-1.2), 10 % supersonic, 15 %
  hypersonic (M > 5), at least 15 % engine first; CA_base 25 %.
* CN: 15 %; 25 % in the critical crossflow-Reynolds band (M_n < 0.5,
  1.5e5 < Re_n < 3e6); 30 % for alpha > 100 deg (Jorgensen over-predicts Jernell's
  base-first data by 20-35 %).
* Cm: sqrt((0.15 Cm)^2 + (0.25 CN)^2), i.e. +-0.25 d on the centre of pressure.
* Cmq: 50 %, Clp: 30 %.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from plume.physics.aero_gen import base as base_mod
from plume.physics.aero_gen import body as body_mod
from plume.physics.aero_gen import fins as fins_mod
from plume.physics.aero_gen.friction import skin_friction
from plume.physics.aero_gen.geometry import (
    BodyGeometry,
    FinPlanform,
    LegAeroGeometry,
    infer_fin_planform,
)
from plume.physics.aero_gen.newtonian import cp_max_pitot, newtonian_body
from plume.physics.aero_gen.nose import nose_pressure_drag
from plume.physics.aerodb import AeroDatabase

GENERATOR_VERSION = "1.0"

MACH_GRID = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5,
             1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0)  # fmt: skip
ALPHA_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0,
              50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 155.0,
              160.0, 165.0, 168.0, 170.0, 172.0, 174.0, 176.0, 177.0, 178.0, 179.0, 180.0)  # fmt: skip
LOG10_RE_GRID = (4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)

REFERENCES = {
    "jorgensen1973": "Jorgensen, L.H., Prediction of static aerodynamic characteristics for "
    "space-shuttle-like and other bodies at angles of attack from 0 to 180 deg, NASA TN D-6996, 1973.",
    "allen_perkins1951": "Allen, H.J., Perkins, E.W., A study of effects of viscosity on flow over "
    "slender inclined bodies of revolution, NACA Rep. 1048, 1951.",
    "jernell1968": "Jernell, L.S., Aerodynamic characteristics of bodies of revolution at Mach "
    "numbers from 1.50 to 2.86 and angles of attack to 180 deg, NASA TM X-1658, 1968.",
    "linnell_bailey1956": "Linnell, R.D., Bailey, J.Z., Similarity-rule estimation methods for cones "
    "and parabolic noses, J. Aero. Sci. 23(8), 1956.",
    "rossow1951": "Rossow, V.J., Applicability of the hypersonic similarity rule to pressure "
    "distributions ... bodies of revolution at zero angle of attack, NACA TN 2399, 1951.",
    "taylor_maccoll1933": "Taylor, G.I., Maccoll, J.W., The air pressure on a cone moving at high "
    "speeds, Proc. R. Soc. A 139, 1933.",
    "love1957": "Love, E.S., Base pressure at supersonic speeds on two-dimensional airfoils and on "
    "bodies of revolution with and without fins having turbulent boundary layers, NACA TN 3819, 1957.",
    "gabeaud1950": "Gabeaud, A., Base pressures at supersonic velocities, J. Aero. Sci. 17(8), 1950.",
    "hoerner1965": "Hoerner, S.F., Fluid-Dynamic Drag, 1965.",
    "vandriest1956": "van Driest, E.R., The problem of aerodynamic heating, Aero. Eng. Rev. 15(10), 1956.",
    "hopkins_inouye1971": "Hopkins, E.J., Inouye, M., An evaluation of theories for predicting "
    "turbulent skin friction and heat transfer on flat plates at supersonic and hypersonic Mach "
    "numbers, AIAA J. 9(6), 1971.",
    "eckert1955": "Eckert, E.R.G., Engineering relations for friction and heat transfer to surfaces "
    "in high velocity flow, J. Aero. Sci. 22(8), 1955.",
    "schlichting": "Schlichting, H., Boundary-Layer Theory (fully rough plate; Karman-Schoenherr).",
    "barrowman1967": "Barrowman, J.S., The practical calculation of the aerodynamic characteristics "
    "of slender finned vehicles, M.S. thesis, Catholic University of America, 1967.",
    "ackeret1925": "Ackeret, J., Luftkraefte auf Fluegel, die mit groesserer als "
    "Schallgeschwindigkeit bewegt werden, ZFM 16, 1925.",
    "niskanen2009": "Niskanen, S., Development of an open source model rocket simulation software "
    "(OpenRocket technical documentation), M.Sc. thesis, Helsinki Univ. of Technology, 2009.",
    "lees1955": "Lees, L., Hypersonic flow, Proc. 5th Int. Aero. Conf., 1955 (modified Newtonian).",
    "naca1135": "Ames Research Staff, Equations, tables, and charts for compressible flow, NACA "
    "Rep. 1135, 1953.",
    "korzun2009": "Korzun, A.M., Braun, R.D., Cruz, J.R., Survey of supersonic retropropulsion "
    "technology for Mars entry, descent, and landing, J. Spacecraft Rockets 46(5), 2009.",
    "sutton_graves1971": "Sutton, K., Graves, R.A., A general stagnation-point convective-heating "
    "equation for arbitrary gas mixtures, NASA TR R-376, 1971.",
}


@dataclass
class GeneratorOptions:
    nose_shape: str = "tangent_ogive"
    roughness: float = 20e-6  # equivalent sand-grain roughness, m (smooth paint)
    friction_method: str = "van_driest_ii"
    t_static: float = 250.0  # K, static temperature used for compressibility factors
    reynolds_effect: bool = True  # crossflow drag crisis (TN D-6996 Fig. 2)
    reverse_potential: float = 1.0  # potential CN term for alpha > 90 deg (0 = fast-model-like)
    newtonian_blend: tuple[float, float] = (4.0, 6.0)
    damping_amplitude_deg: float = 5.0
    x_ref: float = 0.0  # moment reference, body z (0 = base)
    include_legs: bool = True
    leg_strut_diameter: float | None = None
    footpad_wake_factor: float = 0.5  # nose-first: footpads sit in the base wake
    fin_planform: FinPlanform | None = None
    nominal_log10_re: float | None = None
    extra_meta: dict = field(default_factory=dict)


def _legs_drag_area(legs: LegAeroGeometry, nose_first: bool, wake: float) -> float:
    """Zero-alpha axial drag area (m^2, incompressible) of the legs (Hoerner 1965:
    inclined cylinder by the crossflow principle, C_D sin^3 psi; flat disk 1.17)."""
    if legs.count <= 0:
        return 0.0
    psi = legs.strut_inclination
    strut = 1.2 * legs.strut_diameter * legs.strut_length * math.sin(psi) ** 3
    pad = 1.17 * math.pi * legs.footpad_radius**2 * (wake if nose_first else 1.0)
    return legs.count * (strut + pad)


def generate_database(
    vehicle,
    mach_grid=MACH_GRID,
    alpha_grid=ALPHA_GRID,
    log10_re_grid=LOG10_RE_GRID,
    options: GeneratorOptions | None = None,
    **overrides,
) -> AeroDatabase:
    """Semi-empirical aerodynamic database for ``vehicle`` (a ``VehicleSpec``).

    ``overrides`` set ``GeneratorOptions`` fields directly, e.g.
    ``generate_database(v, reverse_potential=0.0)``."""
    opt = replace(options) if options is not None else GeneratorOptions()
    for k, v in overrides.items():
        if not hasattr(opt, k):
            raise TypeError(f"unknown generator option {k!r}")
        setattr(opt, k, v)
    body = BodyGeometry.from_spec(vehicle.geometry, nose_shape=opt.nose_shape)
    A = body.ref_area
    if vehicle.aero.reference_area:
        # Plume's AeroSpec allows another reference area; coefficients are scaled to it
        A_ref = float(vehicle.aero.reference_area)
    else:
        A_ref = A
    d = body.diameter
    L = body.length
    x_ref = opt.x_ref

    M = np.asarray(mach_grid, dtype=float)[:, None, None]
    alpha = np.asarray(alpha_grid, dtype=float)[None, :, None]
    lre = np.asarray(log10_re_grid, dtype=float)[None, None, :]
    re_m = 10.0**lre
    a_rad = np.radians(alpha)
    shape = (M.shape[0], alpha.shape[1], lre.shape[2])

    # ---------------------------------------------------------------- normal force
    jb = body_mod.jorgensen_body(
        alpha,
        M,
        re_m * d,
        body,
        reynolds_effect=opt.reynolds_effect,
        reverse_potential=opt.reverse_potential,
    )
    cn_pot = np.broadcast_to(jb["CN_pot"], shape)
    cn_visc = np.broadcast_to(jb["CN_visc"], shape)
    z_pot = L - np.broadcast_to(jb["x_pot"], shape)
    z_visc = L - np.broadcast_to(jb["x_visc"], shape)
    cn_body = cn_pot + cn_visc
    m_body = cn_pot * (z_pot - x_ref) + cn_visc * (z_visc - x_ref)  # CN * lever (m)

    # hypersonic blend toward modified Newtonian (pressure only)
    m0, m1 = opt.newtonian_blend
    w_n = np.clip((M - m0) / (m1 - m0), 0.0, 1.0)
    if np.any(w_n > 0):
        newt = newtonian_body(body, np.asarray(alpha_grid, dtype=float), 6.0, x_ref, cp_max=1.0)
        cpm = np.array([cp_max_pitot(float(m)) for m in M.ravel()])[:, None, None]
        cn_newt = cpm * newt["CN"][None, :, None]
        m_newt = cpm * newt["Cm"][None, :, None] * d
        cn_body = (1 - w_n) * cn_body + w_n * cn_newt
        m_body = (1 - w_n) * m_body + w_n * m_newt

    comps = [("body_pot", cn_pot, z_pot), ("body_visc", cn_visc, z_visc)]
    fin = None
    cn_fin = np.zeros(shape)
    if vehicle.aero.fins is not None and vehicle.aero.fins.count > 0:
        fin = opt.fin_planform or infer_fin_planform(vehicle.aero.fins, body)
        cnf, xf = fins_mod.fin_normal_force(alpha, M, fin, A)
        cn_fin = np.broadcast_to(cnf, shape)
        z_fin = np.broadcast_to(fin.z_root_le - xf, shape)
        comps.append(("fins", cn_fin, z_fin))
        m_body = m_body + cn_fin * (z_fin - x_ref)
    cn = cn_body + cn_fin
    cm = m_body / d

    # ---------------------------------------------------------------- axial force
    t_e = opt.t_static
    cf = skin_friction(
        re_m * L, M, t_e, length=L, roughness=opt.roughness, method=opt.friction_method
    )
    ld = body.fineness
    ff_sub = 1 + 1.5 * (1 / ld) ** 1.5 + 7 * (1 / ld) ** 3  # Hoerner body form factor
    ff = 1 + (ff_sub - 1) * np.clip((1.2 - M) / 0.4, 0, 1)
    ca_fric = cf * ff * body.wetted_area / A
    cdf_base = cf * body.wetted_area / body.base_area
    ca_base0 = base_mod.base_drag_power_off(M, cdf_base) * body.base_area / A
    if body.nose_length > 0:
        ca_nose = nose_pressure_drag(M.ravel(), body)[:, None, None]
    else:
        ca_nose = base_mod.flat_face_drag(M.ravel())[:, None, None]
    ca_fins = np.zeros_like(cf)
    if fin is not None:
        ca_fins = fins_mod.fin_axial_drag(
            M, re_m, t_e, fin, A, roughness=opt.roughness, method=opt.friction_method
        )
    q_ratio = base_mod.stagnation_cp(M.ravel())[:, None, None]  # bluff-body Mach growth
    ca_legs_n = ca_legs_t = np.zeros_like(cf)
    legs = None
    if opt.include_legs and vehicle.legs.count > 0:
        legs = LegAeroGeometry.from_spec(vehicle.legs, body, opt.leg_strut_diameter)
        ca_legs_n = _legs_drag_area(legs, True, opt.footpad_wake_factor) * q_ratio / A
        ca_legs_t = _legs_drag_area(legs, False, opt.footpad_wake_factor) * q_ratio / A
    ca0 = ca_fric + ca_nose + ca_base0 + ca_fins + ca_legs_n
    face = base_mod.flat_face_drag(M.ravel())[:, None, None] * body.base_area / A
    trailing = (
        base_mod.base_drag_power_off(M, cdf_base)
        if body.nose_length <= 0
        else base_mod.trailing_cone_drag(M)
    )
    ca180 = -(face + ca_fric + trailing + ca_fins + ca_legs_t)
    cos2 = np.cos(a_rad) ** 2
    nose_first = alpha <= 90.0
    ca = np.where(nose_first, ca0 * cos2, ca180 * cos2)
    ca_base = np.where(nose_first, ca_base0 * cos2, 0.0)
    ca = np.broadcast_to(ca, shape).copy()
    ca_base = np.broadcast_to(ca_base, shape).copy()

    # ---------------------------------------------------------------- damping
    zcp = np.where(np.abs(cn) > 1e-9, x_ref + cm * d / np.where(np.abs(cn) > 1e-9, cn, 1.0), np.nan)
    # nodes with CN ~ 0 (alpha 0/180): use the neighbouring alpha's centre of pressure
    for i in range(shape[0]):
        for k in range(shape[2]):
            col = zcp[i, :, k]
            ok = np.isfinite(col)
            if ok.any():
                zcp[i, :, k] = np.interp(
                    np.asarray(alpha_grid), np.asarray(alpha_grid)[ok], col[ok]
                )
            else:
                zcp[i, :, k] = x_ref
    da = 0.5
    slopes = _component_slopes(alpha_grid, da, M, re_m, d, body, fin, A, opt)
    cmq = np.zeros(shape)
    for name, _, z in comps:
        k = slopes[name]
        if name == "body_visc":
            k = np.maximum(k, slopes["body_visc_df"])
            rg = body.planform_radius_of_gyration
            cmq -= 2 * k * ((z - zcp) ** 2 + rg * rg) / (d * d)
        else:
            cmq -= 2 * k * ((z - zcp) / d) ** 2
    clp = np.zeros(shape)
    if fin is not None:
        clp = np.broadcast_to(fins_mod.fin_roll_damping(M, fin, A, d), shape).copy()

    # ---------------------------------------------------------------- uncertainty
    m_ = np.broadcast_to(M, shape)
    a_ = np.broadcast_to(alpha, shape)
    rel_ca = np.where(m_ < 0.8, 0.10, np.where(m_ <= 1.2, 0.20, np.where(m_ <= 5.0, 0.10, 0.15)))
    rel_ca = np.where(a_ > 90, np.maximum(rel_ca, 0.15), rel_ca)
    mn = m_ * np.sin(np.radians(a_))
    re_n = np.broadcast_to(re_m * d, shape) * np.sin(np.radians(a_))
    critical = opt.reynolds_effect & (mn < 0.5) & (re_n > 1.5e5) & (re_n < 3e6)
    rel_cn = np.where(critical, 0.25, 0.15)
    rel_cn = np.where(a_ > 100, 0.30, rel_cn)
    sigma = {
        "CA": rel_ca * np.abs(ca),
        "CA_base": 0.25 * np.abs(ca_base),
        "CN": rel_cn * np.abs(cn) + 0.005,
        "Cm": np.sqrt((0.15 * cm) ** 2 + (0.25 * cn) ** 2) + 0.005,
        "Cmq": 0.5 * np.abs(cmq),
        "Clp": 0.3 * np.abs(clp),
    }

    scale = A / A_ref
    coeffs = {
        "CA": ca * scale,
        "CN": cn * scale,
        "Cm": cm * scale,
        "Cmq": cmq * scale,
        "Clp": clp * scale,
        "CA_base": ca_base * scale,
    }
    sigma = {k: v * scale for k, v in sigma.items()}

    nominal = opt.nominal_log10_re
    if nominal is None:
        # sea level, M 0.3 (typical landing / low-speed flight)
        nominal = math.log10(1.225 * 0.3 * 340.3 / 1.789e-5)
    meta = {
        "name": f"{vehicle.name} (generated)",
        "vehicle": vehicle.name,
        "generator": f"plume.physics.aero_gen.generate {GENERATOR_VERSION}",
        "provenance": {k: "generated" for k in ("CA", "CN", "Cm", "Cmq", "Clp", "CA_base")},
        "methods": {
            "CN/Cm body": "Jorgensen TN D-6996 eqs. 1,4,5 (potential + viscous crossflow), "
            "Cdn(Mn,Re_n) Figs. 1-2, eta Fig. 4; modified Newtonian blend M "
            f"{m0}-{m1}",
            "CN/Cm fins": "Barrowman subsonic, Ackeret supersonic, K_T(B) interference, flat-plate "
            "crossflow at high alpha"
            if fin
            else "none",
            "CA nose first": f"skin friction ({opt.friction_method}, roughness "
            f"{opt.roughness:g} m, Hoerner form factor) + nose wave drag (Linnell-Bailey / Rossow, "
            "PCHIP transonic) + base drag (Hoerner subsonic, Love supersonic) + fins + legs",
            "CA engine first": "flat-face k*Cp_stag + friction + fins + legs (footpads leading)",
            "CA(alpha)": "CA(0) cos^2 a / CA(180) cos^2 a (TN D-6996 eqs. 2-3)",
            "Cmq": "lever-arm spread of component CN slopes about the CP (+ viscous planform "
            f"gyration), describing-function amplitude {opt.damping_amplitude_deg} deg",
            "Clp": "fin strip theory" if fin else "none (body roll damping neglected)",
            "power on": "CA_base x (1 - A_e/A_b) exp(-C_T/0.5) (runtime)",
            "SRP": "engine-first CA x exp(-C_T/0.4), Korzun et al. 2009 (runtime)",
        },
        "references": REFERENCES,
        "validity": {
            "mach": [float(M.min()), float(M.max())],
            "alpha_deg": [0.0, 180.0],
            "log10_re_per_m": [float(lre.min()), float(lre.max())],
            "geometry": "cylindrical body + nose, flat base, no boattail; >= 3 planar fins",
            "notes": "no asymmetric-vortex side forces; no control-surface increments; "
            "grid fins are modelled separately (physics/gridfins.py)",
        },
        "uncertainty": "1-sigma absolute in 'sigma' tables; see generate.py docstring",
        "geometry": {
            "length": L,
            "diameter": d,
            "nose_length": body.nose_length,
            "nose_shape": body.nose_shape,
            "planform_area": body.planform_area,
            "planform_centroid_z": L - body.planform_centroid_x,
            "volume": body.volume,
            "wetted_area": body.wetted_area,
            "fins": asdict(fin) if fin else None,
            "legs": asdict(legs) if legs else None,
        },
        "options": {
            k: (v if not isinstance(v, FinPlanform) else asdict(v))
            for k, v in asdict(opt).items()
            if k != "fin_planform"
        },
        "body_length": L,
        "planform_area": body.planform_area,
        "nominal_log10_re": nominal,
        **opt.extra_meta,
    }
    db = AeroDatabase(
        axes={
            "mach": M.ravel(),
            "alpha_deg": np.asarray(alpha_grid, float),
            "log10_re": lre.ravel(),
        },
        coeffs=coeffs,
        sigma=sigma,
        ref_area=A_ref,
        ref_length=d,
        x_ref=x_ref,
        meta=meta,
    )
    return db


def _component_slopes(alpha_grid, da, M, re_m, d, body, fin, A, opt) -> dict[str, np.ndarray]:
    """|dCN_i/dalpha| (per rad) of each component on the grid, by central differences,
    plus the describing-function slope of the quadratic viscous crossflow."""
    a = np.asarray(alpha_grid, dtype=float)[None, :, None]
    hi = np.minimum(a + da, 180.0)
    lo = np.maximum(a - da, 0.0)
    step = np.radians(hi - lo)

    def comp(alpha):
        j = body_mod.jorgensen_body(
            alpha, M, re_m * d, body,
            reynolds_effect=opt.reynolds_effect, reverse_potential=opt.reverse_potential,
        )  # fmt: skip
        out = {"body_pot": j["CN_pot"], "body_visc": j["CN_visc"]}
        if fin is not None:
            out["fins"] = fins_mod.fin_normal_force(alpha, M, fin, A)[0]
        return out, j

    c_hi, _ = comp(hi)
    c_lo, _ = comp(lo)
    shape = np.broadcast_shapes(M.shape, a.shape, re_m.shape)
    slopes = {k: np.broadcast_to(np.abs(c_hi[k] - c_lo[k]) / step, shape) for k in c_hi}
    # describing function of a u|u| law at amplitude A0: slope (8 / 3 pi) sin(A0)
    _, j = comp(a)
    amp = math.radians(opt.damping_amplitude_deg)
    k_df = (
        j["eta"]
        * j["Cdn"]
        * (body.planform_area / body.ref_area)
        * 8
        / (3 * math.pi)
        * math.sin(amp)
    )
    slopes["body_visc_df"] = np.broadcast_to(k_df, shape)
    return slopes
