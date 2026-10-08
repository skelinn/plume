"""Table-driven aerodynamic database and its runtime force/moment model.

``AeroDatabase`` holds aerodynamic coefficients on a rectilinear grid of

* ``mach``       free-stream Mach number (0 ... ~8),
* ``alpha_deg``  total angle of attack alpha_t in [0, 180] deg (0 = nose first,
                 180 = base / engine first),
* ``phi_deg``    aerodynamic roll angle (optional; omitted for axisymmetric
                 configurations),
* ``log10_re``   log10 of the unit Reynolds number per metre (optional),

with the coefficient set (reference area ``ref_area``, reference length
``ref_length`` = body diameter, moment reference point ``x_ref`` in body z):

=========  ================================================================
``CA``     axial force, body-axis, positive toward -z: F_z = -q S CA.  It is
           positive nose first and negative engine first.
``CN``     normal force in the alpha_t plane, positive when it opposes the
           lateral component of the vehicle velocity (the usual "lift" sense).
``Cm``     pitching moment about ``x_ref`` in the alpha_t plane, positive when it
           increases alpha_t.  Centre of pressure z_cp = x_ref + d Cm / CN.
``CY``     side force perpendicular to the alpha_t plane (roll-dependent
           configurations only; zero for bodies of revolution).
``Cn``     yawing moment about ``x_ref`` that goes with ``CY``.
``Cl``     static rolling moment.
``Cmq``    pitch damping (Cmq + Cm_alpha_dot) about the centre of pressure, *in
           excess of* the quasi-steady lever-arm damping of the normal force
           (which the runtime model produces by itself, see ``AeroModelHiFi``).
           Non-dimensionalised with q S d (q_rate d / 2V).  Must be <= 0.
``Clp``    roll damping, q S d (p d / 2V).  Must be <= 0.
``CA_base``  the part of ``CA`` that is power-off base drag (removed by the
           runtime model when the engine fills the base, nose-first flight).
=========  ================================================================

Every coefficient carries a 1-sigma absolute uncertainty on the same grid
(``sigma``) for Monte Carlo dispersions (:meth:`AeroDatabase.perturbed`).

Serialisation: ``.npz`` (arrays + JSON metadata: provenance, methods, references,
validity) with a human-readable ``.json`` sidecar, and a long-format CSV export.

The runtime model :class:`AeroModelHiFi` is a drop-in replacement for
:class:`plume.physics.aero.Aero` (same ``forces`` signature and attributes).  It
applies the aerodynamic force at the centre of pressure using the local air
velocity there, which makes the static aerodynamics strictly dissipative (with no
wind the aerodynamic power F . v_cp is never positive); rate damping is added
with non-positive derivatives.  See ``docs/models/aero.md``.
"""

from __future__ import annotations

import bisect
import csv
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GAMMA_AIR = 1.4
R_AIR = 287.052_87

#: Sutherland's law constants for air (White, *Viscous Fluid Flow*, 3rd ed., Table 1-2).
SUTHERLAND_MU0 = 1.716e-5  # Pa s at T0
SUTHERLAND_T0 = 273.15  # K
SUTHERLAND_S = 110.4  # K

AXIS_ORDER = ("mach", "alpha_deg", "phi_deg", "log10_re")
COEFFICIENTS = ("CA", "CN", "Cm", "CY", "Cn", "Cl", "Cmq", "Clp", "CA_base")
FORMAT_VERSION = 1


def sutherland_viscosity(T):
    """Dynamic viscosity of air (Pa s) from Sutherland's law; ``T`` in K (scalar or array).

    Valid ~100-1900 K for air (White 2006).  Below 50 K the value is clamped."""
    T = np.maximum(np.asarray(T, dtype=float), 50.0)
    mu = (
        SUTHERLAND_MU0
        * (T / SUTHERLAND_T0) ** 1.5
        * (SUTHERLAND_T0 + SUTHERLAND_S)
        / (T + SUTHERLAND_S)
    )
    return float(mu) if mu.ndim == 0 else mu


def reynolds_per_metre(rho, V, T):
    """Unit Reynolds number rho V / mu(T), 1/m."""
    out = np.asarray(rho, dtype=float) * np.asarray(V, dtype=float) / sutherland_viscosity(T)
    return float(out) if np.ndim(out) == 0 else out


def _mu_scalar(T: float) -> float:
    T = max(T, 50.0)
    return (
        SUTHERLAND_MU0
        * (T / SUTHERLAND_T0) ** 1.5
        * (SUTHERLAND_T0 + SUTHERLAND_S)
        / (T + SUTHERLAND_S)
    )


# --------------------------------------------------------------------------- interpolation
class GridInterpolator:
    """Lean multilinear interpolation of a vector of values on a rectilinear grid.

    ``axes`` are ascending 1-D arrays; ``data`` has shape ``(*map(len, axes), C)``.
    Axes of length 1 are collapsed.  Queries outside the grid are clamped to the
    edge (no extrapolation).  Cost is a few microseconds per call (pure Python index
    search + one fancy-index gather + one dot product), far below
    ``scipy.interpolate.RegularGridInterpolator`` for single points.
    """

    def __init__(self, axes: list[np.ndarray], data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != len(axes) + 1:
            raise ValueError("data must have one trailing value axis")
        keep = [k for k, a in enumerate(axes) if len(a) > 1]
        squeeze = tuple(k for k, a in enumerate(axes) if len(a) == 1)
        if squeeze:
            data = data.reshape([s for k, s in enumerate(data.shape) if k not in squeeze])
        self.keep = keep
        self.axes = [[float(v) for v in axes[k]] for k in keep]
        for ax in self.axes:
            if any(b <= a for a, b in itertools.pairwise(ax)):
                raise ValueError("grid axes must be strictly increasing")
        self.ncomp = data.shape[-1]
        self.flat = np.ascontiguousarray(data.reshape(-1, self.ncomp))
        n = [len(a) for a in self.axes]
        strides = [int(np.prod(n[k + 1 :])) for k in range(len(n))]
        self.strides = strides
        d = len(n)
        self.offsets = np.array(
            [sum(((j >> k) & 1) * strides[k] for k in range(d)) for j in range(1 << d)],
            dtype=np.intp,
        )

    def __call__(self, *x: float) -> np.ndarray:
        base = 0
        w = [1.0]
        for k, ax in enumerate(self.axes):
            v = x[self.keep[k]]
            n = len(ax)
            if v <= ax[0]:
                i, t = 0, 0.0
            elif v >= ax[-1]:
                i, t = n - 2, 1.0
            else:
                i = bisect.bisect_right(ax, v) - 1
                t = (v - ax[i]) / (ax[i + 1] - ax[i])
            base += i * self.strides[k]
            s = 1.0 - t
            w = [wi * s for wi in w] + [wi * t for wi in w]
        if len(w) == 1:
            return self.flat[base].copy()
        return np.dot(w, self.flat[base + self.offsets])


# --------------------------------------------------------------------------- database
@dataclass
class AeroDatabase:
    """Gridded aerodynamic coefficients + 1-sigma uncertainties + provenance."""

    axes: dict[str, np.ndarray]
    coeffs: dict[str, np.ndarray]
    sigma: dict[str, np.ndarray] = field(default_factory=dict)
    ref_area: float = 1.0
    ref_length: float = 1.0
    x_ref: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        axes = {}
        for name in AXIS_ORDER:
            if name in self.axes:
                a = np.atleast_1d(np.asarray(self.axes[name], dtype=float))
                if a.ndim != 1 or (len(a) > 1 and np.any(np.diff(a) <= 0)):
                    raise ValueError(f"axis {name!r} must be strictly increasing 1-D")
                axes[name] = a
        unknown = set(self.axes) - set(AXIS_ORDER)
        if unknown:
            raise ValueError(f"unknown axes {sorted(unknown)}; allowed {AXIS_ORDER}")
        for req in ("mach", "alpha_deg"):
            if req not in axes:
                raise ValueError(f"database needs a {req!r} axis")
        self.axes = axes
        shape = self.shape
        coeffs = {}
        for name in COEFFICIENTS:
            c = self.coeffs.get(name)
            coeffs[name] = (
                np.zeros(shape)
                if c is None
                else np.broadcast_to(np.asarray(c, dtype=float), shape).copy()
            )
        unknown = set(self.coeffs) - set(COEFFICIENTS)
        if unknown:
            raise ValueError(f"unknown coefficients {sorted(unknown)}")
        self.coeffs = coeffs
        self.sigma = {
            k: np.broadcast_to(np.abs(np.asarray(v, dtype=float)), shape).copy()
            for k, v in self.sigma.items()
            if k in COEFFICIENTS
        }
        for k in COEFFICIENTS:
            self.sigma.setdefault(k, np.zeros(shape))
        if not all(np.all(np.isfinite(v)) for v in self.coeffs.values()):
            raise ValueError("non-finite coefficient values")
        self.meta.setdefault("format_version", FORMAT_VERSION)
        self._interp: GridInterpolator | None = None
        self._zcp_interp: GridInterpolator | None = None

    # ------------------------------------------------------------------ basics
    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes.values())

    @property
    def roll_dependent(self) -> bool:
        return "phi_deg" in self.axes and len(self.axes["phi_deg"]) > 1

    @property
    def phi_period(self) -> float:
        return float(self.meta.get("phi_period_deg", 360.0))

    def _full_axes(self) -> list[np.ndarray]:
        """Axes in AXIS_ORDER with singleton placeholders for missing ones."""
        return [self.axes.get(n, np.array([0.0])) for n in AXIS_ORDER]

    def _full_shape(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape([len(a) for a in self._full_axes()])

    def interpolator(self, names: tuple[str, ...] = COEFFICIENTS) -> GridInterpolator:
        data = np.stack([self._full_shape(self.coeffs[n]) for n in names], axis=-1)
        return GridInterpolator(self._full_axes(), data)

    def _query(self, mach, alpha_deg, phi_deg, log10_re):
        phi = float(phi_deg) % self.phi_period if self.roll_dependent else 0.0
        lre = 0.0 if log10_re is None else float(log10_re)
        if log10_re is None and "log10_re" in self.axes:
            lre = float(self.meta.get("nominal_log10_re", np.median(self.axes["log10_re"])))
        return float(mach), min(max(float(alpha_deg), 0.0), 180.0), phi, lre

    def evaluate(
        self,
        mach: float,
        alpha_deg: float,
        phi_deg: float = 0.0,
        log10_re: float | None = None,
    ) -> dict[str, float]:
        """All coefficients at one flight condition (alpha_t in [0, 180] deg)."""
        if self._interp is None:
            self._interp = self.interpolator()
        vals = self._interp(*self._query(mach, alpha_deg, phi_deg, log10_re))
        return dict(zip(COEFFICIENTS, map(float, vals), strict=True))

    def evaluate_signed(self, mach: float, alpha_deg: float, **kw) -> dict[str, float]:
        """Coefficients for a *signed* pitch-plane angle in (-180, 180]: CN and Cm are
        odd in alpha, CA even (convenience for plotting and symmetry checks)."""
        c = self.evaluate(mach, abs(alpha_deg), **kw)
        if alpha_deg < 0:
            c["CN"], c["Cm"] = -c["CN"], -c["Cm"]
        return c

    # ------------------------------------------------------------------ centre of pressure
    def zcp_table(self) -> np.ndarray:
        """Centre of pressure (body z) on the grid: x_ref + d Cm / CN.

        Where |CN| is negligible (alpha_t = 0 or 180 deg) the limit is taken from the
        nearest alpha nodes with a usable CN; the result is clipped to the body extent
        (+-50 % of the length) given in ``meta['body_length']`` if present."""
        cn = self.coeffs["CN"]
        cm = self.coeffs["Cm"]
        d = self.ref_length
        ia = self.axis_names.index("alpha_deg")
        cn_m = np.moveaxis(cn, ia, -1)
        cm_m = np.moveaxis(cm, ia, -1)
        alpha = self.axes["alpha_deg"]
        out = np.empty_like(cn_m)
        scale = max(1e-9, float(np.max(np.abs(cn))))
        for idx in np.ndindex(cn_m.shape[:-1]):
            c = cn_m[idx]
            m = cm_m[idx]
            ok = np.abs(c) > 1e-6 * scale
            if ok.sum() == 0:
                out[idx] = self.x_ref
                continue
            z = self.x_ref + d * m[ok] / c[ok]
            out[idx] = np.interp(alpha, alpha[ok], z)
        out = np.moveaxis(out, -1, ia)
        L = self.meta.get("body_length")
        if L:
            out = np.clip(out, -0.5 * L, 1.5 * L)
        return out

    def centre_of_pressure(self, mach, alpha_deg, phi_deg=0.0, log10_re=None) -> float:
        if self._zcp_interp is None:
            self._zcp_interp = GridInterpolator(
                self._full_axes(), self._full_shape(self.zcp_table())[..., None]
            )
        return float(self._zcp_interp(*self._query(mach, alpha_deg, phi_deg, log10_re))[0])

    # ------------------------------------------------------------------ Monte Carlo
    def perturbed(self, rng: np.random.Generator, scale: float = 1.0) -> AeroDatabase:
        """A dispersed copy: each coefficient table is shifted by z * sigma with one
        standard-normal draw z per coefficient (fully correlated over the grid, which
        keeps the tables smooth).  Damping derivatives are clipped to stay <= 0."""
        new = {}
        draws = {}
        for k in COEFFICIENTS:
            z = float(rng.standard_normal()) * scale
            draws[k] = z
            new[k] = self.coeffs[k] + z * self.sigma[k]
        new["Cmq"] = np.minimum(new["Cmq"], 0.0)
        new["Clp"] = np.minimum(new["Clp"], 0.0)
        meta = json.loads(json.dumps(self.meta, default=str))
        meta["dispersion"] = {"draws": draws, "scale": scale}
        return AeroDatabase(
            axes={k: v.copy() for k, v in self.axes.items()},
            coeffs=new,
            sigma={k: v.copy() for k, v in self.sigma.items()},
            ref_area=self.ref_area,
            ref_length=self.ref_length,
            x_ref=self.x_ref,
            meta=meta,
        )

    # ------------------------------------------------------------------ checks
    def check(self) -> list[str]:
        """Physical sanity checks; returns a list of human-readable warnings."""
        warn = []
        alpha = self.axes["alpha_deg"]
        ia = self.axis_names.index("alpha_deg")
        a = np.radians(alpha).reshape([-1 if k == ia else 1 for k in range(len(self.shape))])
        drag = self.coeffs["CN"] * np.sin(a) + self.coeffs["CA"] * np.cos(a)
        if np.any(drag < -1e-6):
            warn.append(f"negative drag (CN sin a + CA cos a) at {int(np.sum(drag < -1e-6))} nodes")
        if np.any(self.coeffs["Cmq"] > 1e-12):
            warn.append("Cmq > 0 (anti-damping) at some nodes; the runtime clips it to 0")
        if np.any(self.coeffs["Clp"] > 1e-12):
            warn.append("Clp > 0 at some nodes; the runtime clips it to 0")
        if alpha[0] > 0 or alpha[-1] < 180:
            warn.append(f"alpha grid covers only {alpha[0]}..{alpha[-1]} deg (clamped outside)")
        return warn

    # ------------------------------------------------------------------ IO
    def _meta_full(self) -> dict:
        m = dict(self.meta)
        m.update(
            ref_area=self.ref_area,
            ref_length=self.ref_length,
            x_ref=self.x_ref,
            axes=list(self.axes),
            coefficients=list(COEFFICIENTS),
            format_version=FORMAT_VERSION,
        )
        return m

    def save(self, path: str | Path) -> Path:
        """Write ``<path>.npz`` (arrays + metadata) and a ``<path>.json`` sidecar."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(self._meta_full(), indent=2, default=str)
        arrays = {f"axis__{k}": v for k, v in self.axes.items()}
        arrays.update({f"coef__{k}": v for k, v in self.coeffs.items()})
        arrays.update({f"sigma__{k}": v for k, v in self.sigma.items()})
        np.savez_compressed(path, meta=np.array(meta), **arrays)
        path.with_suffix(".json").write_text(meta, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> AeroDatabase:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            axes = {k[6:]: z[k] for k in z.files if k.startswith("axis__")}
            coeffs = {k[6:]: z[k] for k in z.files if k.startswith("coef__")}
            sigma = {k[7:]: z[k] for k in z.files if k.startswith("sigma__")}
        ref_area = float(meta.pop("ref_area"))
        ref_length = float(meta.pop("ref_length"))
        x_ref = float(meta.pop("x_ref"))
        for k in ("axes", "coefficients"):
            meta.pop(k, None)
        return cls(axes, coeffs, sigma, ref_area, ref_length, x_ref, meta)

    def to_csv(self, path: str | Path) -> Path:
        """Long-format CSV: one row per grid node (axes, coefficients, sigma_*)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.axes)
        grids = np.meshgrid(*self.axes.values(), indexing="ij")
        cols = [g.ravel() for g in grids]
        cols += [self.coeffs[k].ravel() for k in COEFFICIENTS]
        cols += [self.sigma[k].ravel() for k in COEFFICIENTS]
        header = names + list(COEFFICIENTS) + [f"sigma_{k}" for k in COEFFICIENTS]
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(f"# Plume aerodynamic database; ref_area={self.ref_area!r} ")
            f.write(f"ref_length={self.ref_length!r} x_ref={self.x_ref!r}\n")
            w = csv.writer(f)
            w.writerow(header)
            for row in zip(*cols, strict=True):
                w.writerow([f"{v:.8g}" for v in row])
        return path

    def summary(self) -> str:
        lines = [
            f"AeroDatabase {self.meta.get('name', '')}: axes "
            + ", ".join(f"{k}[{len(v)}] {v[0]:g}..{v[-1]:g}" for k, v in self.axes.items()),
            f"  S_ref={self.ref_area:.6g} m^2  d_ref={self.ref_length:.6g} m  x_ref={self.x_ref:.6g} m",
        ]
        for k, v in self.meta.get("methods", {}).items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- power effects
def base_drag_power_on_factor(ct: float, exit_area_ratio: float, ct_ref: float = 0.5) -> float:
    """Fraction of the power-off base drag that remains with the engine running.

    The nozzle exit area carries the exit pressure (already in the thrust), so only
    the annulus ``1 - A_e/A_b`` is subject to base pressure, and the plume raises the
    annulus pressure toward ambient as the thrust coefficient grows (plume-induced
    base pressure recovery).  ``exp(-C_T/ct_ref)`` is an engineering fit; see
    ``docs/models/aero.md`` (uncertainty +-50 % on ``ct_ref``)."""
    if ct <= 0.0:
        return 1.0
    return max(0.0, 1.0 - exit_area_ratio) * math.exp(-ct / ct_ref)


def srp_axial_factor(ct: float, ct_decay: float = 0.4) -> float:
    """Remaining fraction of the engine-first (retro-propulsive) aerodynamic axial
    force for a single central nozzle, as a function of C_T = T / (q S_ref).

    Trend from the supersonic-retropropulsion survey of Korzun, Braun & Cruz (JSR
    46(5), 2009; data of Jarvinen & Adams 1970, M 1.5-4): the plume shields the
    forebody and the aerodynamic drag collapses for C_T >~ 1.  Mirrors
    :func:`plume.physics.aero_gen.srp.srp_axial_factor`."""
    if ct <= 0.0:
        return 1.0
    return math.exp(-ct / ct_decay)


# --------------------------------------------------------------------------- runtime model
_ZERO3 = np.zeros(3)
_RUNTIME = ("CA", "CN", "CY", "Cl", "Cmq", "Clp", "CA_base")


class AeroModelHiFi:
    """Database-driven aerodynamics with the same interface as :class:`plume.physics.aero.Aero`.

    Parameters
    ----------
    db:
        the coefficient database.
    geometry:
        hull geometry (``GeometrySpec`` or anything with ``length``, ``diameter``,
        ``nose_length``); used for the compatibility attributes ``z`` /
        ``strip_area`` and for clipping the centre of pressure.
    enabled, cd_scale:
        as in ``AeroSpec`` (``cd_scale`` multiplies every coefficient).
    nozzle_exit_area:
        engine exit area for the power-on base-drag model (m^2).
    stations:
        number of hull strips for the compatibility attributes.

    Runtime inputs besides ``forces``' arguments: set ``thrust`` (N, >= 0) every
    step so the power-on base drag (nose-first) and the supersonic-retropropulsion
    drag reduction (engine-first) are applied.
    """

    def __init__(
        self,
        db: AeroDatabase,
        geometry,
        *,
        enabled: bool = True,
        cd_scale: float = 1.0,
        nozzle_exit_area: float = 0.0,
        stations: int = 10,
        srp_ct_decay: float = 0.4,
        base_ct_ref: float = 0.5,
        spec=None,
    ):
        self.db = db
        self.spec = spec
        self.enabled = enabled
        self._cd_scale = float(cd_scale)
        self.ref_area = float(db.ref_area)
        self.ref_length = float(db.ref_length)
        self.thrust = 0.0
        self.srp_ct_decay = srp_ct_decay
        self.base_ct_ref = base_ct_ref
        self.exit_area_ratio = min(max(nozzle_exit_area / self.ref_area, 0.0), 1.0)
        L = float(geometry.length)
        self.length = L
        if "body_length" not in db.meta:
            db.meta["body_length"] = L
        self._coef = db.interpolator(_RUNTIME)
        zcp = db.zcp_table()
        self._zcp = GridInterpolator(db._full_axes(), db._full_shape(zcp)[..., None])
        self._zlo, self._zhi = -0.5 * L, 1.5 * L
        self._phi_period = db.phi_period
        self._roll = db.roll_dependent
        self._has_re = "log10_re" in db.axes
        self._lre_nominal = float(db.meta.get("nominal_log10_re", 6.5))
        # compatibility with the strip model (used by plotting / diagnostics)
        n = stations
        dz = L / n
        self.z = (np.arange(n) + 0.5) * dz
        width = np.full(n, geometry.diameter)
        if geometry.nose_length > 0:
            nose_start = L - geometry.nose_length
            in_nose = self.z > nose_start
            width[in_nose] = geometry.diameter * (L - self.z[in_nose]) / geometry.nose_length
        self.strip_area = width * dz
        self.planform_area = float(db.meta.get("planform_area", float(self.strip_area.sum())))

    @classmethod
    def from_vehicle(cls, vehicle, db: AeroDatabase | str | Path | None = None, **gen_kw):
        """Build from a ``VehicleSpec``: load ``db`` (path) or use it, or generate one
        with :func:`plume.physics.aero_gen.generate.generate_database` (``gen_kw``)."""
        if db is None:
            from plume.physics.aero_gen.generate import generate_database

            db = generate_database(vehicle, **gen_kw)
        elif not isinstance(db, AeroDatabase):
            db = AeroDatabase.load(db)
        aero = vehicle.aero
        exit_area = math.pi * vehicle.engine.nozzle_radius**2
        return cls(
            db,
            vehicle.geometry,
            enabled=aero.enabled,
            cd_scale=aero.cd_scale,
            nozzle_exit_area=exit_area,
            stations=aero.stations,
            spec=aero,
        )

    # ------------------------------------------------------------------ compat API
    @property
    def cd_scale(self) -> float:
        return self._cd_scale

    @cd_scale.setter
    def cd_scale(self, value: float) -> None:
        self._cd_scale = float(value)

    def set_thrust(self, thrust: float) -> None:
        """Current engine thrust (N) for the power-on / retro-propulsion corrections."""
        self.thrust = max(float(thrust), 0.0)

    def coefficients(
        self, mach: float, alpha_deg: float, phi_deg: float = 0.0, log10_re: float | None = None
    ) -> dict[str, float]:
        return self.db.evaluate(mach, alpha_deg, phi_deg, log10_re)

    def axial_coefficient(self, mach: float, nose_first: bool) -> float:
        """|CA| at alpha_t = 0 (nose first) or 180 deg (engine first), power off, x cd_scale."""
        c = self._coef(float(mach), 0.0 if nose_first else 180.0, 0.0, self._lre_nominal)
        return abs(float(c[0])) * self._cd_scale

    def crossflow_coefficient(self, mach_cross: float) -> float:
        """Effective crossflow drag coefficient eta*C_dn on the planform area, taken from
        CN at alpha_t = 90 deg (where the normal force is all viscous crossflow)."""
        c = self._coef(float(mach_cross), 90.0, 0.0, self._lre_nominal)
        return float(c[1]) * self.ref_area / self.planform_area * self._cd_scale

    def static_margin(self, mach: float, alpha_deg: float, cg_z: float) -> float:
        """(z_cg - z_cp) / d for nose-first flight; positive = statically stable.
        For engine-first flight (alpha > 90) stability needs z_cp > z_cg, so the sign
        convention flips: the returned value is positive when stable in either case."""
        zcp = self.db.centre_of_pressure(mach, alpha_deg)
        m = (cg_z - zcp) / self.ref_length
        return m if alpha_deg <= 90.0 else -m

    # ------------------------------------------------------------------ forces
    def forces(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        cg_z: float,
        rho: float,
        sound_speed: float,
    ):
        """Aerodynamic force and torque about the CG in body axes.

        Returns ``(force, torque, dynamic pressure, Mach)`` like ``Aero.forces``."""
        vx, vy, vz = float(v_air_body[0]), float(v_air_body[1]), float(v_air_body[2])
        wx, wy, wz = float(omega_body[0]), float(omega_body[1]), float(omega_body[2])
        speed2 = vx * vx + vy * vy + vz * vz
        q = 0.5 * rho * speed2
        V = math.sqrt(speed2)
        mach = V / sound_speed if sound_speed > 0 else 0.0
        if (
            not self.enabled
            or rho <= 0.0
            or (speed2 == 0.0 and wx == 0.0 and wy == 0.0 and wz == 0.0)
        ):
            return _ZERO3.copy(), _ZERO3.copy(), q, mach

        if self._has_re:
            T = sound_speed * sound_speed / (GAMMA_AIR * R_AIR) if sound_speed > 0 else 288.15
            re_m = rho * V / _mu_scalar(T)
            lre = math.log10(re_m) if re_m > 1.0 else 0.0
        else:
            lre = 0.0

        # pass 1: centre of pressure from the flow at the CG
        vlat = math.sqrt(vx * vx + vy * vy)
        phi = 0.0
        if self._roll and vlat > 0.0:
            phi = math.degrees(math.atan2(vy, vx)) % self._phi_period
        alpha = math.degrees(math.atan2(vlat, vz)) if speed2 > 0.0 else 0.0
        zcp = float(self._zcp(mach, alpha, phi, lre)[0])
        zcp = min(max(zcp, self._zlo), self._zhi)
        lever = zcp - cg_z

        # pass 2: coefficients at the local flow angle at the centre of pressure
        ux = vx + wy * lever
        uy = vy - wx * lever
        uz = vz
        ul2 = ux * ux + uy * uy
        U2 = ul2 + uz * uz
        fx = fy = fz = 0.0
        tz = 0.0
        scale = self._cd_scale
        U = math.sqrt(U2)
        if U2 > 0.0:
            ul = math.sqrt(ul2)
            a2 = math.degrees(math.atan2(ul, uz))
            if self._roll and ul > 0.0:
                phi = math.degrees(math.atan2(uy, ux)) % self._phi_period
            ca, cn, cy, cl, cmq, clp, ca_base = self._coef(mach, a2, phi, lre)
            thrust = self.thrust
            if thrust > 0.0 and q > 0.0:
                ct = thrust / (q * self.ref_area)
                if a2 <= 90.0:
                    ca -= ca_base * (
                        1.0 - base_drag_power_on_factor(ct, self.exit_area_ratio, self.base_ct_ref)
                    )
                else:
                    ca *= srp_axial_factor(ct, self.srp_ct_decay)
            k = 0.5 * rho * U2 * self.ref_area * scale
            if ul > 0.0:
                ex, ey = ux / ul, uy / ul
            else:
                ex = ey = 0.0
            # normal force opposes the local lateral velocity; side force is +90 deg
            fx = k * (-cn * ex - cy * ey)
            fy = k * (-cn * ey + cy * ex)
            fz = -k * ca
            p = fx * ux + fy * uy + fz * uz
            if p > 0.0:  # guard: interpolated tables must never feed energy in
                s = p / U2
                fx -= s * ux
                fy -= s * uy
                fz -= s * uz
            tz = k * self.ref_length * cl
        else:
            _, _, _, _, cmq, clp, _ = self._coef(mach, alpha, phi, lre)

        # rate damping beyond the lever-arm term: q S d (d / 2U) C omega
        kd = 0.25 * rho * U * self.ref_area * self.ref_length * self.ref_length * scale
        cmq = min(cmq, 0.0)
        clp = min(clp, 0.0)
        force = np.array([fx, fy, fz])
        torque = np.array(
            [-lever * fy + kd * cmq * wx, lever * fx + kd * cmq * wy, tz + kd * clp * wz]
        )
        return force, torque, q, mach
