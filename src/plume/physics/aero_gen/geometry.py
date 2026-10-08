"""Geometry of the aerodynamic configuration: body of revolution, fins, legs.

The vehicle hull is a cylinder of diameter ``d`` with a nose of length ``l_N`` and a
flat base (no boattail).  Lengths along the axis are measured from the nose tip
(``x``, aerodynamic convention) and converted to Plume body z (from the base,
+z toward the nose) with ``z = L - x``.

Nose shapes (radius ``r(x)``, ``R = d/2``):

* ``cone``           r = R x / l_N
* ``tangent_ogive``  r = sqrt(rho^2 - (l_N - x)^2) + R - rho,  rho = (R^2 + l_N^2) / 2R
* ``von_karman``     r = R/sqrt(pi) sqrt(t - sin(2t)/2),  t = acos(1 - 2x/l_N)  (LD-Haack)
* ``flat``           l_N = 0 (flat-faced cylinder)

Integral properties (volume, planform area and centroid, wetted area) are computed
by quadrature and verified against the closed-form tangent-ogive formulas and the
tabulated values of Jorgensen, NASA TN D-6996 (1973), Fig. 9 (see the tests).

``GeometrySpec`` has no nose-shape or fin-planform fields yet; the defaults here
(tangent ogive; fin planform inferred from ``FinSpec``) are documented in
``docs/models/aero.md`` together with the config fields we propose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201 (numpy < 2 compatibility)

NoseShape = Literal["cone", "tangent_ogive", "von_karman", "flat"]


def nose_radius_profile(x, nose_length: float, radius: float, shape: NoseShape) -> np.ndarray:
    """Local radius r(x) of the nose, x measured aft from the tip (array)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, nose_length)
    if nose_length <= 0 or shape == "flat":
        return np.full_like(x, radius)
    L, R = nose_length, radius
    if shape == "cone":
        return R * x / L
    if shape == "tangent_ogive":
        rho = (R * R + L * L) / (2 * R)
        return np.sqrt(np.maximum(rho * rho - (L - x) ** 2, 0.0)) + R - rho
    if shape == "von_karman":
        t = np.arccos(np.clip(1.0 - 2.0 * x / L, -1.0, 1.0))
        return R / math.sqrt(math.pi) * np.sqrt(np.maximum(t - np.sin(2 * t) / 2, 0.0))
    raise ValueError(f"unknown nose shape {shape!r}")


@dataclass(frozen=True)
class BodyGeometry:
    """Cylindrical body with a nose and a flat base."""

    length: float
    diameter: float
    nose_length: float = 0.0
    nose_shape: NoseShape = "tangent_ogive"
    nose_tip_radius: float | None = None  # stagnation-point radius (heating only)

    @classmethod
    def from_spec(
        cls, geometry, nose_shape: NoseShape = "tangent_ogive", nose_tip_radius: float | None = None
    ) -> BodyGeometry:
        """From a ``GeometrySpec`` (``length``, ``diameter``, ``nose_length``)."""
        shape = getattr(geometry, "nose_shape", None) or nose_shape
        tip = getattr(geometry, "nose_tip_radius", None) or nose_tip_radius
        return cls(
            float(geometry.length),
            float(geometry.diameter),
            float(geometry.nose_length),
            shape if geometry.nose_length > 0 else "flat",
            tip,
        )

    # ------------------------------------------------------------------ basics
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def ref_area(self) -> float:
        return math.pi * self.radius**2

    @property
    def base_area(self) -> float:
        return self.ref_area

    @property
    def cylinder_length(self) -> float:
        return self.length - self.nose_length

    @property
    def fineness(self) -> float:
        return self.length / self.diameter

    @property
    def nose_fineness(self) -> float:
        return self.nose_length / self.diameter

    @property
    def nose_half_angle(self) -> float:
        """Equivalent cone half-angle atan(R / l_N), rad (pi/2 for a flat face)."""
        return math.atan2(self.radius, self.nose_length) if self.nose_length > 0 else math.pi / 2

    def radius_at(self, x) -> np.ndarray:
        """Radius at distance x aft of the nose tip."""
        x = np.asarray(x, dtype=float)
        r = nose_radius_profile(x, self.nose_length, self.radius, self.nose_shape)
        return np.where(x >= self.nose_length, self.radius, r)

    def z_from_x(self, x):
        return self.length - np.asarray(x, dtype=float)

    # ------------------------------------------------------------------ integrals
    def _nose_quadrature(self, n: int = 4001):
        """Nodes/weights on the nose with clustering at the tip (x = l_N s^2)."""
        s = np.linspace(0.0, 1.0, n)
        x = self.nose_length * s * s
        r = nose_radius_profile(x, self.nose_length, self.radius, self.nose_shape)
        return x, r

    @property
    def nose_volume(self) -> float:
        if self.nose_length <= 0:
            return 0.0
        x, r = self._nose_quadrature()
        return float(_trapz(math.pi * r * r, x))

    @property
    def volume(self) -> float:
        return self.nose_volume + self.ref_area * self.cylinder_length

    def _planform_moments(self) -> tuple[float, float, float]:
        """(area, first moment about tip, second moment about tip) of the side view."""
        a0 = a1 = a2 = 0.0
        if self.nose_length > 0:
            x, r = self._nose_quadrature()
            a0 += float(_trapz(2 * r, x))
            a1 += float(_trapz(2 * r * x, x))
            a2 += float(_trapz(2 * r * x * x, x))
        x0, x1, d = self.nose_length, self.length, self.diameter
        a0 += d * (x1 - x0)
        a1 += d * (x1**2 - x0**2) / 2
        a2 += d * (x1**3 - x0**3) / 3
        return a0, a1, a2

    @property
    def planform_area(self) -> float:
        return self._planform_moments()[0]

    @property
    def planform_centroid_x(self) -> float:
        """Distance from the nose tip to the centroid of the planform (side-view) area."""
        a0, a1, _ = self._planform_moments()
        return a1 / a0

    @property
    def planform_radius_of_gyration(self) -> float:
        """sqrt(second moment about the centroid / area) of the planform, m."""
        a0, a1, a2 = self._planform_moments()
        xc = a1 / a0
        return math.sqrt(max(a2 / a0 - xc * xc, 0.0))

    @property
    def wetted_area(self) -> float:
        """Lateral (wetted) surface area, nose + cylinder, excluding the base."""
        side = math.pi * self.diameter * self.cylinder_length
        if self.nose_length <= 0:
            return side
        x, r = self._nose_quadrature()
        ds = np.hypot(np.diff(x), np.diff(r))
        rm = 0.5 * (r[1:] + r[:-1])
        return side + float(np.sum(2 * math.pi * rm * ds))

    @property
    def nose_wetted_area(self) -> float:
        return self.wetted_area - math.pi * self.diameter * self.cylinder_length


@dataclass(frozen=True)
class FinPlanform:
    """Trapezoidal fin set (``count`` identical panels equally spaced in roll).

    ``z_root_le`` is the body z of the root-chord leading edge (the root trailing
    edge is at ``z_root_le - root_chord``); ``sweep`` is the axial distance from the
    root leading edge to the tip leading edge (positive aft)."""

    count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep: float
    thickness: float
    z_root_le: float
    body_radius: float
    le_shape: Literal["rounded", "square"] = "rounded"
    te_shape: Literal["square", "tapered"] = "square"

    @property
    def area(self) -> float:
        """Exposed planform area of one panel."""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @property
    def aspect_ratio(self) -> float:
        """Barrowman's panel aspect ratio 2 s^2 / A (two panels form one wing)."""
        return 2.0 * self.span**2 / self.area

    @property
    def midchord_sweep(self) -> float:
        """Sweep angle of the mid-chord line, rad."""
        dx = self.sweep + 0.5 * self.tip_chord - 0.5 * self.root_chord
        return math.atan2(dx, self.span)

    @property
    def le_sweep(self) -> float:
        return math.atan2(self.sweep, self.span)

    @property
    def mac(self) -> float:
        cr, ct = self.root_chord, self.tip_chord
        return 2.0 / 3.0 * (cr + ct - cr * ct / (cr + ct))

    @property
    def mac_y(self) -> float:
        """Spanwise station of the mean aerodynamic chord, from the root."""
        cr, ct = self.root_chord, self.tip_chord
        return self.span / 3.0 * (cr + 2 * ct) / (cr + ct)

    @property
    def mac_x_le(self) -> float:
        """Axial distance from the root leading edge (aft) to the MAC leading edge."""
        return self.sweep * self.mac_y / self.span

    @property
    def centroid_x(self) -> float:
        """Axial distance from the root leading edge to the planform centroid."""
        y = np.linspace(0, self.span, 401)
        c = self.root_chord + (self.tip_chord - self.root_chord) * y / self.span
        xle = self.sweep * y / self.span
        return float(_trapz(c * (xle + 0.5 * c), y) / _trapz(c, y))

    def roll_inertia_integral(self) -> float:
        """Integral of c(y) (R + y)^2 dy over one panel (roll damping), m^4."""
        y = np.linspace(0, self.span, 401)
        c = self.root_chord + (self.tip_chord - self.root_chord) * y / self.span
        return float(_trapz(c * (self.body_radius + y) ** 2, y))

    def z_of(self, x_aft_of_root_le: float) -> float:
        return self.z_root_le - x_aft_of_root_le


#: Hobby-rocket reference fin (66 mm body, 3 fins): proportions used when a
#: ``FinSpec`` gives only ``count``/``cn_alpha``/``z``.  Ratios to the span.
DEFAULT_FIN_RATIOS = {"root_chord": 1.75, "tip_chord": 0.75, "sweep": 1.0, "thickness": 0.0375}


def barrowman_cn_alpha_subsonic(fin: FinPlanform, ref_area: float, mach: float = 0.0) -> float:
    """CN_alpha of the whole fin set (per rad, on ``ref_area``) *without* interference
    (Barrowman 1967 with Diederich's compressibility form)."""
    beta = math.sqrt(max(1.0 - mach * mach, 1e-6))
    ar = fin.aspect_ratio
    cosg = math.cos(fin.midchord_sweep)
    one = 2 * math.pi * ar * (fin.area / ref_area) / (2 + math.sqrt(4 + (beta * ar / cosg) ** 2))
    return fin_count_factor(fin.count) * one


def fin_count_factor(n: int) -> float:
    """Multiplier on the single-panel CN_alpha for n equally spaced fins.

    n/2 for 3 and 4 fins (Barrowman 1967); OpenRocket's interference-reduced values
    for 5-8 fins (Niskanen 2009, Table 3.3)."""
    table = {3: 1.5, 4: 2.0, 5: 2.5 * 0.948, 6: 3.0 * 0.913, 7: 3.5 * 0.854, 8: 4.0 * 0.810}
    if n in table:
        return table[n]
    if n > 8:
        return n / 2 * 0.75
    raise NotImplementedError(
        f"{n} fins: roll-dependent configuration (needs a phi axis); not generated"
    )


def interference_factor(fin: FinPlanform) -> float:
    """Barrowman's fin-body interference K_T(B) = 1 + R / (s + R)."""
    return 1.0 + fin.body_radius / (fin.span + fin.body_radius)


def infer_fin_planform(fin_spec, body: BodyGeometry, ratios: dict | None = None) -> FinPlanform:
    """Fin planform consistent with a ``FinSpec`` that has only count / cn_alpha / z.

    The planform keeps the hobby-rocket proportions (``DEFAULT_FIN_RATIOS``) and is
    scaled so that Barrowman's incompressible CN_alpha with interference K_T(B)
    equals ``fin_spec.cn_alpha`` (body reference area); it is then positioned so its
    Barrowman centre of pressure is at ``fin_spec.z``.  Add explicit planform
    fields to the config to override (see docs)."""
    ratios = {**DEFAULT_FIN_RATIOS, **(ratios or {})}
    R = body.radius
    S = body.ref_area

    def make(span: float, z_le: float = 0.0) -> FinPlanform:
        return FinPlanform(
            count=int(fin_spec.count),
            root_chord=ratios["root_chord"] * span,
            tip_chord=ratios["tip_chord"] * span,
            span=span,
            sweep=ratios["sweep"] * span,
            thickness=ratios["thickness"] * span,
            z_root_le=z_le,
            body_radius=R,
        )

    def cna(span: float) -> float:
        f = make(span)
        return barrowman_cn_alpha_subsonic(f, S) * interference_factor(f)

    target = float(fin_spec.cn_alpha)
    lo, hi = 1e-4 * body.diameter, 20.0 * body.diameter
    if target <= 0:
        raise ValueError("fin cn_alpha must be > 0 to infer a planform")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cna(mid) < target:
            lo = mid
        else:
            hi = mid
    span = 0.5 * (lo + hi)
    f = make(span)
    x_cp = f.mac_x_le + 0.25 * f.mac  # aft of the root LE
    return make(span, float(fin_spec.z) + x_cp)


@dataclass(frozen=True)
class LegAeroGeometry:
    """Landing legs as inclined circular struts with flat footpads (drag build-up)."""

    count: int
    span: float  # radial distance of footpad centres from the axis
    height: float  # footpad depth below the hull base
    attach_z: float  # strut attachment height on the hull
    footpad_radius: float
    body_radius: float
    strut_diameter: float

    @classmethod
    def from_spec(cls, legs, body: BodyGeometry, strut_diameter: float | None = None):
        d = strut_diameter if strut_diameter is not None else max(0.02, 0.06 * body.diameter)
        return cls(
            int(legs.count),
            float(legs.span),
            float(legs.height),
            float(legs.attach_z),
            float(legs.footpad_radius),
            body.radius,
            d,
        )

    @property
    def strut_length(self) -> float:
        return math.hypot(self.span - self.body_radius, self.attach_z + self.height)

    @property
    def strut_inclination(self) -> float:
        """Angle between the strut and the body axis, rad."""
        return math.atan2(self.span - self.body_radius, self.attach_z + self.height)
