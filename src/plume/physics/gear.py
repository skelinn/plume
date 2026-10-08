"""Landing gear: crushable-core shock absorbers, soil sinkage and tip-over margin.

Two leg models (``LegsSpec.model``):

* ``rigid`` (fast fidelity, RL): the legs are geoms fixed to the hull; a touchdown above
  ``max_touchdown_speed`` breaks them. This is the original model, unchanged.
* ``crush`` (high fidelity): each leg's lower strut and footpad are a separate body on a
  *slide joint* along the leg (the stroke ``s``) and the footpad on a second slide joint
  along the body axis (the soil sinkage ``z``). Both are plastic elements implemented
  with MuJoCo's dry-friction loss on the joint, so they are resolved implicitly by the
  constraint solver:

  - crushable core: the joint moves only when the axial load exceeds the crush force
    ``F_c(s)`` (onset -> plateau -> densification), and never springs back
    (non-recoverable); the joint range is the available stroke (bottoming out is a hard
    stop). A parallel spring ``k s`` and damper ``c ds/dt`` are MuJoCo joint
    stiffness/damping.
  - soil: the pad sinks while the bearing pressure exceeds the Bekker pressure-sinkage
    curve ``p(z) = (k_c / b + k_phi) z^n`` (capped by the ultimate bearing strength).

Equations, sources and limitations: ``docs/models/landing_gear.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from plume.config import LegsSpec, SoilSpec, VehicleSpec, WorldSpec
from plume.constants import G0

# Bekker parameters from Wong, *Theory of Ground Vehicles* (4th ed., 2008), Table 2.3
# (k_c in N/m^(n+1), k_phi in N/m^(n+2)); lunar regolith after Bekker / Apollo LRV data.
SOIL_PRESETS: dict[str, dict] = {
    "rigid": {"rigid": True},
    "concrete": {"rigid": True, "friction": 0.8},
    "steel_deck": {"rigid": True, "friction": 0.6},
    "compacted_gravel": {"k_c": 0.0, "k_phi": 6.0e7, "n": 1.0, "friction": 0.7},
    "dry_sand": {"k_c": 0.99e3, "k_phi": 1528.43e3, "n": 1.1, "friction": 0.55},
    "sandy_loam": {"k_c": 5.27e3, "k_phi": 1515.04e3, "n": 0.7, "friction": 0.6},
    "clay": {
        "k_c": 13.19e3,
        "k_phi": 692.15e3,
        "n": 0.5,
        "bearing_strength": 150e3,
        "friction": 0.45,
    },
    "lunar_regolith": {"k_c": 1.4e3, "k_phi": 820e3, "n": 1.0, "friction": 0.65},
}


def soil_spec(soil: SoilSpec | str | None) -> SoilSpec:
    """Resolve a preset name (or None = rigid) into a :class:`SoilSpec`."""
    if soil is None:
        return SoilSpec(name="rigid", rigid=True)
    if isinstance(soil, SoilSpec):
        return soil
    if soil not in SOIL_PRESETS:
        raise ValueError(f"unknown soil {soil!r}; presets: {', '.join(SOIL_PRESETS)}")
    return SoilSpec(name=soil, **SOIL_PRESETS[soil])


def soil_force(soil: SoilSpec, z: float, pad_radius: float) -> float:
    """Plastic soil resistance of one circular pad at sinkage ``z`` (N)."""
    if soil.rigid:
        return math.inf
    b = pad_radius  # Bekker's b is the smaller contact dimension: the radius of a round plate
    p = (soil.k_c / b + soil.k_phi) * max(z, 0.0) ** soil.n
    if soil.bearing_strength is not None:
        p = min(p, soil.bearing_strength)
    return p * math.pi * pad_radius * pad_radius


def static_sinkage(soil: SoilSpec, load: float, pad_radius: float) -> float:
    """Sinkage at which the soil carries ``load`` (inf if beyond the bearing strength)."""
    if soil.rigid or load <= 0:
        return 0.0
    area = math.pi * pad_radius * pad_radius
    p = load / area
    if soil.bearing_strength is not None and p >= soil.bearing_strength:
        return math.inf
    k = soil.k_c / pad_radius + soil.k_phi
    return (p / k) ** (1.0 / soil.n) if k > 0 else math.inf


def legs_model(legs: LegsSpec, world: WorldSpec) -> str:
    if legs.count == 0:
        return "rigid"
    if legs.model == "auto":
        return "crush" if world.fidelity == "high" else "rigid"
    return legs.model


def landing_mass(vehicle: VehicleSpec) -> float:
    """Nominal landing mass used to size the crush core: dry + cargo + RCS gas + 10 %
    of the propellant capacity (landing reserve)."""
    gas = vehicle.rcs.gas if vehicle.rcs.enabled else 0.0
    return vehicle.mass.dry + vehicle.cargo.mass + gas + 0.1 * vehicle.prop_capacity


def crush_plateau(vehicle: VehicleSpec) -> float:
    """Plateau crush force per leg (N): given, or sized so that all legs stop the design
    touchdown speed within 75 % of the stroke: ``n F = m (g + v^2 / (1.5 s))``."""
    legs = vehicle.legs
    if legs.crush_force is not None:
        return legs.crush_force
    m = landing_mass(vehicle)
    v = legs.max_touchdown_speed
    return m * (G0 + v * v / (1.5 * legs.stroke)) / max(legs.count, 1)


def crush_breakout(vehicle: VehicleSpec) -> float:
    """Minimum force per leg before the core starts to crush (N): 1.2 x the fully
    fuelled weight per leg, so a vehicle standing on its legs on the pad (cargo hop
    launch) does not crush them (a preloaded core / break-out stage)."""
    gas = vehicle.rcs.gas if vehicle.rcs.enabled else 0.0
    m = vehicle.mass.dry + vehicle.cargo.mass + gas + vehicle.prop_capacity
    return 1.2 * m * G0 / max(vehicle.legs.count, 1)


def crush_params(vehicle: VehicleSpec) -> tuple[float, float]:
    """(plateau force, onset fraction) honouring the break-out load."""
    p = crush_plateau(vehicle)
    b = crush_breakout(vehicle)
    p = max(p, b)
    return p, max(vehicle.legs.crush_onset, min(b / p, 1.0))


def crush_curve(legs: LegsSpec, plateau: float, s: float, onset: float | None = None) -> float:
    """Crush force at stroke ``s``: linear rise from ``onset`` (default ``crush_onset``)
    x plateau over ``elastic_stroke``, plateau, then linear densification to
    ``densified_ratio`` x plateau at full stroke."""
    s = max(s, 0.0)
    f0 = legs.crush_onset if onset is None else onset
    if s < legs.elastic_stroke:
        f = f0 + (1.0 - f0) * s / legs.elastic_stroke
    else:
        f = 1.0
    s_d = legs.densification * legs.stroke
    if s > s_d and legs.stroke > s_d:
        f += (legs.densified_ratio - 1.0) * min((s - s_d) / (legs.stroke - s_d), 1.0)
    return plateau * f


def crush_energy(
    legs: LegsSpec, plateau: float, s: float, n: int = 400, onset: float | None = None
) -> float:
    """Energy absorbed crushing the core from 0 to ``s`` (integral of the curve, J)."""
    if s <= 0:
        return 0.0
    xs = np.linspace(0.0, s, n + 1)
    fs = np.array([crush_curve(legs, plateau, x, onset) for x in xs])
    return float(np.sum(0.5 * (fs[1:] + fs[:-1]) * np.diff(xs)))


# --------------------------------------------------------------------------- geometry
@dataclass
class LegGeometry:
    """Body-frame leg geometry shared by the MJCF builder and the runtime model."""

    attach: np.ndarray  # (n, 3) hinge on the hull
    foot: np.ndarray  # (n, 3) footpad sphere centre
    axis: np.ndarray  # (n, 3) stroke (compression) direction, body frame


def leg_geometry(vehicle: VehicleSpec) -> LegGeometry:
    from plume.physics.mjcf import leg_angles

    legs = vehicle.legs
    r = vehicle.geometry.radius
    fr = legs.footpad_radius
    att, ft = [], []
    for th in leg_angles(legs.count):
        c, s = math.cos(th), math.sin(th)
        att.append((r * c, r * s, legs.attach_z))
        ft.append((legs.span * c, legs.span * s, -legs.height + fr))
    att = np.array(att, dtype=float).reshape(-1, 3)
    ft = np.array(ft, dtype=float).reshape(-1, 3)
    # the footpad strokes along the body axis: an idealised leg linkage that moves the pad
    # straight up relative to the hull, so a level touchdown does not scrub the pads
    # sideways (a pad sliding along an inclined telescoping leg would dissipate a large,
    # unphysical share of the impact energy in friction)
    axis = np.tile([0.0, 0.0, 1.0], (len(ft), 1))
    return LegGeometry(att, ft, axis)


def dry_body_split(vehicle: VehicleSpec, dry_inertia) -> tuple[float, float, np.ndarray]:
    """Mass, CG height and principal inertia of the hull body once the stroking leg
    masses are moved to their own bodies, such that hull + legs (at zero stroke) has
    exactly the vehicle's dry mass properties."""
    legs = vehicle.legs
    geo = leg_geometry(vehicle)
    mf = legs.foot_mass
    n = legs.count
    md = vehicle.mass.dry
    zd = vehicle.mass.dry_cg_z
    m_hull = md - n * mf
    if m_hull <= 0.2 * md:
        raise ValueError("legs.foot_mass is too large for the dry mass")
    z_hull = (md * zd - mf * float(geo.foot[:, 2].sum())) / m_hull
    I = np.asarray(dry_inertia, dtype=float).copy()
    # parallel-axis: I_dry(about zd) = I_hull(about z_hull) + m_hull dz^2 + feet(about zd)
    x, y, zf = geo.foot[:, 0], geo.foot[:, 1], geo.foot[:, 2] - zd
    feet = mf * np.array([np.sum(y * y + zf * zf), np.sum(x * x + zf * zf), np.sum(x * x + y * y)])
    dz2 = (z_hull - zd) ** 2
    I_hull = I - feet - m_hull * np.array([dz2, dz2, 0.0])
    I_hull = np.maximum(I_hull, 0.05 * I)
    return m_hull, z_hull, I_hull


# soft-constraint parameters of the plastic joints: stiff (impedance close to 1) so the
# light stroking bodies do not creep under loads below the crush force
SOLIMP_PLASTIC = "0.9999 0.9999 0.001"
SOLREF_PLASTIC = "0.01 1"


def needs_sinkage(world: WorldSpec, tile_soils=()) -> bool:
    """Whether any ground can sink (otherwise the sinkage joints are omitted)."""
    return any(not soil_spec(s).rigid for s in (world.soil, *[t for t in tile_soils if t]))


def legs_mjcf_crush(vehicle: VehicleSpec, fmt, sink: bool = True) -> tuple[list[str], list[str]]:
    """Leg bodies (children of the hull) for the crush model, and contact exclusions.
    ``fmt`` = (scalar formatter, vector formatter); ``sink`` adds the soil-sinkage joint."""
    _f, _v = fmt
    legs = vehicle.legs
    geo = leg_geometry(vehicle)
    fr = legs.footpad_radius
    plateau, onset = crush_params(vehicle)
    f0 = crush_curve(legs, plateau, 0.0, onset)
    mh = 0.5 * legs.foot_mass if sink else legs.foot_mass
    rod = 0.35 * fr
    plastic = f'solreffriction="{SOLREF_PLASTIC}" solimpfriction="{SOLIMP_PLASTIC}" '
    foot = (
        '<geom name="foot{k}" type="sphere" size="{fr}" friction="0.3 0.005 0.0001" '
        'solref="0.01 1" rgba="0.1 0.1 0.1 1"/>'
    )
    bodies, excl = [], []
    for k in range(legs.count):
        a, f, u = geo.attach[k], geo.foot[k], geo.axis[k]
        length = float(np.linalg.norm(a - f))
        knee = f + u * min(legs.stroke + 0.15, 0.7 * length)
        piston = u * min(legs.stroke + 0.1, 0.65 * length)
        pad = foot.format(k=k, fr=_f(fr))
        if sink:
            pad = (
                f'<body name="pad{k}" pos="0 0 0">'
                f'<joint name="sink{k}" type="slide" axis="0 0 1" range="0 1" frictionloss="1e7" '
                f'{plastic}solreflimit="0.01 1"/>'
                f'<inertial pos="0 0 0" mass="{_f(mh)}" diaginertia="0.05 0.05 0.05"/>'
                f"{pad}</body>"
            )
            excl.append(f'<exclude body1="rocket" body2="pad{k}"/>')
        bodies.append(
            f'<geom name="leg{k}" type="capsule" fromto="{_v(*a, *knee)}" size="{_f(rod)}" '
            'rgba="0.15 0.15 0.15 1"/>'
            f'<body name="strut{k}" pos="{_v(*f)}">'
            f'<joint name="stroke{k}" type="slide" axis="{_v(*u)}" range="0 {_f(legs.stroke)}" '
            f'frictionloss="{_f(f0)}" stiffness="{_f(legs.spring_rate)}" '
            f'damping="{_f(legs.damping)}" {plastic}solreflimit="0.01 1"/>'
            f'<inertial pos="0 0 0" mass="{_f(mh)}" diaginertia="0.05 0.05 0.05"/>'
            f'<geom name="leg{k}_piston" type="capsule" fromto="0 0 0 {_v(*piston)}" '
            f'size="{_f(0.7 * rod)}" rgba="0.6 0.6 0.6 1"/>'
            f"{pad}</body>"
        )
    return bodies, excl


# --------------------------------------------------------------------------- runtime
@dataclass
class GearReport:
    """Touchdown metrics for one landing (per leg lists, SI units)."""

    model: str
    stroke_used_m: list[float] = field(default_factory=list)
    stroke_fraction: list[float] = field(default_factory=list)
    peak_load_n: list[float] = field(default_factory=list)
    energy_crush_j: list[float] = field(default_factory=list)
    energy_damper_j: list[float] = field(default_factory=list)
    sinkage_m: list[float] = field(default_factory=list)
    energy_soil_j: list[float] = field(default_factory=list)
    tipover_margin_deg: float = float("nan")
    min_tipover_margin_deg: float = float("nan")
    failure: str = ""

    def metrics(self) -> dict:
        """Flat scalar metrics for mission results / Monte Carlo summaries."""
        mx = lambda xs: float(max(xs)) if xs else float("nan")  # noqa: E731
        return {
            "gear_model": self.model,
            "stroke_used_max_m": mx(self.stroke_used_m),
            "stroke_fraction_max": mx(self.stroke_fraction),
            "leg_load_peak_kn": mx(self.peak_load_n) / 1e3 if self.peak_load_n else float("nan"),
            "gear_energy_kj": (sum(self.energy_crush_j) + sum(self.energy_damper_j)) / 1e3
            if self.energy_crush_j
            else float("nan"),
            "sinkage_max_m": mx(self.sinkage_m),
            "tipover_margin_deg": self.tipover_margin_deg,
            "min_tipover_margin_deg": self.min_tipover_margin_deg,
        }


class LandingGear:
    """Runtime state of the legs: stroke, sinkage, loads, energy, failure, tip-over.

    In ``rigid`` mode only the tip-over margin is tracked (and failure is the touchdown
    speed check). In ``crush`` mode :meth:`after_step` updates the plastic resistances
    of the stroke/sinkage joints, integrates the absorbed energy and checks for bottoming
    out and overload."""

    def __init__(self, vehicle: VehicleSpec, world: WorldSpec):
        self.vehicle = vehicle
        self.legs = vehicle.legs
        self.model_name = legs_model(vehicle.legs, world)
        self.crush = self.model_name == "crush"
        self.n = vehicle.legs.count
        self.geo = leg_geometry(vehicle) if self.n else None
        self.plateau, self.onset = crush_params(vehicle) if self.crush else (0.0, 0.0)
        self.max_load = (self.legs.max_load or 1.6 * self.plateau) if self.crush else float("inf")
        self.default_soil = soil_spec(world.soil)
        self.geom_soil: dict[int, SoilSpec] = {}
        self.pad_r = vehicle.legs.footpad_radius
        self.reset()

    # ------------------------------------------------------------------ binding
    def bind(self, model, ground_geoms: set[int], soils: dict[int, SoilSpec]) -> None:
        self.geom_soil = {g: soils.get(g, self.default_soil) for g in ground_geoms}
        if not self.crush:
            return
        jid = [model.joint(f"stroke{k}").id for k in range(self.n)]
        self.s_q = np.array([model.jnt_qposadr[j] for j in jid])
        self.s_v = np.array([model.jnt_dofadr[j] for j in jid])
        names = {model.joint(i).name for i in range(model.njnt)}
        self.has_sink = "sink0" in names
        if self.has_sink:
            sid = [model.joint(f"sink{k}").id for k in range(self.n)]
            self.z_q = np.array([model.jnt_qposadr[j] for j in sid])
            self.z_v = np.array([model.jnt_dofadr[j] for j in sid])
        self.foot_gids = [model.geom(f"foot{k}").id for k in range(self.n)]
        self.foot_set = {g: k for k, g in enumerate(self.foot_gids)}
        self.strut_bids = [model.body(f"strut{k}").id for k in range(self.n)]

    def reset(self) -> None:
        n = self.n
        self.stroke = np.zeros(n)
        self.sink = np.zeros(n)
        self.peak_load = np.zeros(n)
        self.load = np.zeros(n)
        self.e_crush = np.zeros(n)
        self.e_damper = np.zeros(n)
        self.e_soil = np.zeros(n)
        self.foot_soil: list[SoilSpec | None] = [None] * n
        self.failure = ""
        self.margin = float("nan")
        self.min_margin = float("nan")

    def init_model(self, model) -> None:
        """Reset the joint resistances (after mj_resetData)."""
        if not self.crush:
            return
        f0 = crush_curve(self.legs, self.plateau, 0.0, self.onset)
        for k in range(self.n):
            model.dof_frictionloss[self.s_v[k]] = f0
            if self.has_sink:
                model.dof_frictionloss[self.z_v[k]] = 1e7

    # ------------------------------------------------------------------ stepping
    def after_step(self, sim, contacts: dict[int, int]) -> None:
        """Update after a physics step. ``contacts`` maps foot index -> ground geom."""
        if not self.crush:
            return
        model, data, dt = sim.model, sim.data, sim.dt
        s = data.qpos[self.s_q].copy()
        z = data.qpos[self.z_q].copy() if self.has_sink else np.zeros(self.n)
        sdot = data.qvel[self.s_v]
        legs = self.legs
        for k in range(self.n):
            # work against the resistances that acted during the step (set after the
            # previous step from the stroke / sinkage reached so far)
            ds = s[k] - self.stroke[k]
            if ds > 0:
                self.e_crush[k] += model.dof_frictionloss[self.s_v[k]] * ds
            self.e_damper[k] += legs.damping * float(sdot[k]) ** 2 * dt
            dz = z[k] - self.sink[k]
            if dz > 0 and self.has_sink:
                self.e_soil[k] += min(model.dof_frictionloss[self.z_v[k]], 1e6) * dz
            g = contacts.get(k)
            if g is not None:
                self.foot_soil[k] = self.geom_soil.get(g, self.default_soil)
        self.stroke = np.maximum(self.stroke, s)
        self.sink = np.maximum(self.sink, z)
        # plastic resistances for the next step (non-recoverable: functions of the max)
        for k in range(self.n):
            model.dof_frictionloss[self.s_v[k]] = crush_curve(
                legs, self.plateau, self.stroke[k], self.onset
            )
            if not self.has_sink:
                continue
            soil = self.foot_soil[k]
            if soil is None or soil.rigid:
                fz = 1e7
            else:
                # a small floor keeps an unloaded pad from drifting
                fz = max(soil_force(soil, self.sink[k], self.pad_r), 50.0)
            model.dof_frictionloss[self.z_v[k]] = fz
        if contacts or self.load.any():
            self.load = self.axial_loads(sim)
            self.peak_load = np.maximum(self.peak_load, self.load)
        if not self.failure:
            if np.any(self.stroke >= legs.stroke - 1e-3):
                self.failure = "gear_bottomed"
            elif np.any(self.peak_load > self.max_load):
                self.failure = "gear_overload"
            elif np.any(self.sink >= 0.98 * min(s.max_sinkage for s in self._soils()) - 1e-9):
                self.failure = "pad_buried"

    def axial_loads(self, sim) -> np.ndarray:
        """Axial (compressive) leg loads: the ground contact force on each footpad
        projected on the leg axis, at the end-of-step state."""
        import mujoco

        model, data = sim.model, sim.data
        sim.forward()
        R = sim.rot
        axes = self.geo.axis @ R.T  # world-frame leg axes (compression direction)
        load = np.zeros(self.n)
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            k = self.foot_set.get(int(c.geom1), self.foot_set.get(int(c.geom2)))
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]  # force on geom2 from geom1, world
            if int(c.geom1) == self.foot_gids[k]:
                fw = -fw
            load[k] += float(fw @ axes[k])
        return np.maximum(load, 0.0)

    def _soils(self):
        out = [s for s in self.foot_soil if s is not None and not s.rigid]
        return out or [SoilSpec(rigid=True)]

    def check_touchdown(self, td) -> str:
        """Leg failure for a touchdown: rigid legs break above ``max_touchdown_speed``;
        crush legs fail by bottoming out, overload or burying a pad."""
        if self.crush:
            return self.failure
        if td is not None and td.vertical_speed > self.legs.max_touchdown_speed:
            return "crash_legs"
        return ""

    # ------------------------------------------------------------------ tip-over
    def foot_points(self, sim) -> np.ndarray:
        """World positions of the footpad contact points (bottom of the pads)."""
        R = sim.rot
        pos = sim.data.qpos[:3]
        if self.crush:
            down = -self.pad_r * R[:, 2]
            return np.array([sim.data.geom_xpos[g] + down for g in self.foot_gids])
        return pos + sim.foot_body @ R.T

    def tipover_margin(self, sim, up: np.ndarray | None = None) -> float:
        """Static tip-over margin (deg): the smallest rotation about a support-polygon edge
        (the line between adjacent footpads) that brings the CG over that edge,
        ``atan(d_edge / h_cg)``. Negative when the CG is already outside."""
        if self.n < 3:
            return float("nan")
        st = sim.state
        up = st.up if up is None else up
        P = self.foot_points(sim)
        c = st.com
        # project onto the plane normal to 'up'
        e1 = np.cross(up, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(up, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        p2 = np.column_stack([P @ e1, P @ e2])
        c2 = np.array([c @ e1, c @ e2])
        centre = p2.mean(axis=0)
        ang = np.arctan2(p2[:, 1] - centre[1], p2[:, 0] - centre[0])
        order = np.argsort(ang)
        p2, P = p2[order], P[order]
        best = math.inf
        for i in range(len(p2)):
            a, b = p2[i], p2[(i + 1) % len(p2)]
            t = b - a
            nrm = np.array([t[1], -t[0]]) / max(float(np.linalg.norm(t)), 1e-9)
            if float((centre - a) @ nrm) < 0:
                nrm = -nrm
            d = float((c2 - a) @ nrm)  # >0 inside
            h = float(c @ up) - 0.5 * float((P[i] + P[(i + 1) % len(P)]) @ up)
            best = min(best, math.degrees(math.atan2(d, max(h, 1e-6))))
        return best

    def track_tipover(self, sim) -> None:
        m = self.tipover_margin(sim)
        self.margin = m
        if not math.isfinite(self.min_margin) or m < self.min_margin:
            self.min_margin = m

    def report(self) -> GearReport:
        r = GearReport(
            model=self.model_name,
            tipover_margin_deg=self.margin,
            min_tipover_margin_deg=self.min_margin,
            failure=self.failure,
        )
        if self.crush:
            r.stroke_used_m = self.stroke.tolist()
            r.stroke_fraction = (self.stroke / self.legs.stroke).tolist()
            r.peak_load_n = self.peak_load.tolist()
            r.energy_crush_j = self.e_crush.tolist()
            r.energy_damper_j = self.e_damper.tolist()
            r.sinkage_m = self.sink.tolist()
            r.energy_soil_j = self.e_soil.tolist()
        return r
