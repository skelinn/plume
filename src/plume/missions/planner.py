"""Quick 3-DOF feasibility for point-to-point cargo hops (the browser mission planner).

The 6-DOF mission (:func:`plume.missions.hop.run_mission`) takes minutes; this module answers
"can this vehicle carry this cargo this far, and with how much propellant to spare?" in a few
seconds, with the same models the flight software plans with:

* the 3-DOF point-mass integrator (:class:`plume.physics.pointmass.PointMassSim`) with the
  vehicle's engine, mass and axial aerodynamics, US76 atmosphere, calm air;
* the hop ascent law of :func:`plume.missions.hop.plan_ascent`: vertical rise, pitch kick,
  hold, gravity turn, cargo-g-limited throttle;
* MECO placed where the drag-aware impact prediction *with the planned entry burn* (the logic
  of :class:`plume.missions.targeting.ImpactPredictor`) reaches the landing site;
* the entry burn to ``entry_speed`` below ``entry_altitude``, the unpowered aero descent with
  grid fins deployed, and a landing burn costed as the flight software's landing reserve
  (``1.5 * terminal speed + 150 m/s``; the 6-DOF real_hop flight used 195 kg of a 197 kg
  reserve).

Fast fidelity only: non-rotating spherical Earth, so the result depends on the range, not on
the direction or the sites themselves. Earth rotation (high fidelity) changes the achievable
range by a few per cent with azimuth; the planner says so in its notes. The answers are
screening estimates; the 6-DOF flight and the Monte Carlo campaign are the reference.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np

from plume.config import HopGuidanceSpec, VehicleSpec, WorldSpec, config_root, load_vehicle
from plume.constants import G0
from plume.missions.targeting import kepler_impact
from plume.physics.gravity import SphericalGravity
from plume.physics.pointmass import PointMassSim

ASCENT_DT = 0.25  # s, 3-DOF RK4 step for the ascent family
RISES = (4.0, 7.0)  # vertical rise times tried (plan_ascent picks 4 s for both bundled missions)
KICKS_DEG = tuple(float(k) for k in np.arange(6.0, 26.01, 2.0))
LANDING_EXTRA_S = 12.0  # landing burn duration after the unpowered descent reaches the ground
# Peak cargo load allowed over the guidance limit. The 6-DOF reference flight (real_hop, 761 km)
# peaks at 6.06 g against the 6 g limit in the unpowered descent, where no throttle logic can
# limit it (docs/models/guidance.md); beyond 5 % the route is reported as not feasible.
G_TOLERANCE = 0.05


def reference_guidance() -> HopGuidanceSpec:
    """Guidance settings of the bundled real-route mission (entry burn, g-limit, arc cap)."""
    from plume.config import load_mission

    return load_mission("real_hop").guidance


def planner_world() -> WorldSpec:
    return WorldSpec(gravity="spherical", ground="none", dt=0.01)


def planner_vehicles() -> list[dict]:
    """Bundled vehicle presets that can fly a cargo hop (liquid engine, cargo bay)."""
    out = []
    for p in sorted((config_root() / "vehicles").glob("*.yaml")):
        try:
            v = load_vehicle(p)
        except Exception:
            continue
        if v.engine.type != "liquid" or v.cargo.max_mass is None:
            continue
        out.append(
            {
                "name": v.name,
                "description": v.description,
                "cargo_nominal_kg": v.cargo.mass,
                "cargo_max_kg": v.cargo.max_mass,
                "dry_mass_kg": v.mass.dry,
                "propellant_kg": v.prop_capacity,
                "thrust_kn": v.engine.thrust_vac / 1000.0,
                "grid_fins": v.grid_fins is not None,
                "hash": vehicle_hash(v),
            }
        )
    return out


def vehicle_hash(v: VehicleSpec) -> str:
    """Short hash of a preset (with its cargo zeroed), to tell when a precomputed table is
    stale."""
    v0 = v.model_copy(update={"cargo": v.cargo.model_copy(update={"mass": 0.0})})
    return hashlib.sha1(v0.model_dump_json().encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- 3-DOF pieces
class _Model:
    """Point-mass model of one vehicle + cargo in the planner frame: launch site at the map
    origin, the target due east (the frame is non-rotating, so direction does not matter)."""

    def __init__(self, vehicle: VehicleSpec, cargo: float, guidance: HopGuidanceSpec):
        self.vehicle = vehicle.with_cargo(cargo)
        self.cargo = float(cargo)
        self.g = guidance
        self.gravity = SphericalGravity()
        world = planner_world()
        self.pm = PointMassSim(self.vehicle, world, cargo_mass=cargo)
        self.pm.accel_limit = guidance.cargo_g_limit * 0.92 * G0
        gf = self.vehicle.grid_fins
        self.fin_cda = 0.0
        if gf is not None:
            area = gf.area if gf.area is not None else gf.span * gf.chord
            self.fin_cda = gf.count * area * gf.cd0
        self.pad = self.gravity.surface_point(0.0, 0.0, 0.0)
        self.up0 = self.gravity.up(self.pad)
        self.east = np.array([1.0, 0.0, 0.0])
        self.m_dry = self.pm.m_dry
        self.landing_prop, self.v_term = self._landing_reserve()

    def _landing_reserve(self) -> tuple[float, float]:
        """The flight software's landing reserve (HopAutopilot._landing_reserve), at sea level."""
        rho = self.pm.atmosphere.density(0.0)
        cda = self.pm.aero.axial_coefficient(0.3, nose_first=False) * self.pm.aero.ref_area
        cda += self.fin_cda
        v_term = math.sqrt(2.0 * self.m_dry * G0 / max(rho * cda, 1e-9))
        dv = 1.5 * v_term + 150.0
        isp = self.vehicle.engine.isp_sea_level
        return self.m_dry * (math.exp(dv / (isp * G0)) - 1.0), v_term

    def ascent(self, rise: float, kick_deg: float):
        """Fly the ascent law to propellant depletion; returns the recorded trajectory."""
        g = self.g
        kick = math.radians(kick_deg)
        up0, dr = self.up0, self.east
        kick_axis = math.cos(kick) * up0 + math.sin(kick) * dr
        hold = {"aligned": False}
        pm = self.pm
        pm.extra_cda = 0.0  # grid fins stowed for ascent
        m0 = pm.m_dry + pm.prop0
        thr = min(1.0, g.cargo_g_limit * G0 * m0 * 0.92 / pm.engine.max_thrust(0.0))

        def direction(t, r, v):
            if t < rise:
                return up0
            if t < rise + g.kick_time:
                a = (t - rise) / g.kick_time * kick
                return math.cos(a) * up0 + math.sin(a) * dr
            vhat = v / np.linalg.norm(v)
            if not hold["aligned"]:
                if (
                    float(vhat @ kick_axis) > math.cos(math.radians(0.5))
                    or t > rise + g.kick_time + 40
                ):
                    hold["aligned"] = True
                else:
                    return kick_axis
            return vhat

        return pm.run(
            self.pad + up0 * 5.0,
            np.zeros(3),
            t_end=600.0,
            dt=ASCENT_DT,
            throttle=thr,
            direction=direction,
            stop_on_ground=True,
            ground_altitude=-10.0,
            stop_fn=lambda t, r, v, prop: prop <= 0.0,
        )

    def downrange(self, p: np.ndarray) -> float:
        return float(self.gravity.map_coords(p)[0])

    def vacuum_range(self, r, v) -> float:
        imp = kepler_impact(r, v, self.gravity, self.gravity.earth_radius)
        return math.nan if imp is None else self.downrange(imp)

    def entry_prop_estimate(self, r, v, prop: float) -> float:
        """Entry-burn propellant from vis-viva (vacuum above the entry altitude) and the
        rocket equation at vacuum Isp."""
        rn = float(np.linalg.norm(r - self.gravity.center))
        r60 = self.gravity.earth_radius + self.g.entry_altitude
        v2 = float(v @ v) + 2.0 * self.gravity.mu * (1.0 / r60 - 1.0 / rn)
        if v2 <= 0:
            return 0.0
        dv = max(math.sqrt(v2) - self.g.entry_speed, 0.0)
        m = self.m_dry + prop
        return m * (1.0 - math.exp(-dv / (self.vehicle.engine.isp_vac * G0)))

    def descent(self, r, v, prop: float, t0: float, record: bool = False):
        """Coast, entry burn (g-limited, retrograde, to entry_speed) and unpowered aero
        descent with grid fins to the ground (the ImpactPredictor flight plan)."""
        g = self.g
        gravity = self.gravity
        pm = self.pm
        pm.extra_cda = self.fin_cda
        state = {"on": False, "done": False, "prop_after": None, "t_on": None, "t_off": None}

        def throttle(t, rr, vv):
            if state["done"]:
                return 0.0
            alt = gravity.altitude(rr)
            descending = float(vv @ gravity.up(rr)) < 0
            speed = float(np.linalg.norm(vv))
            if not state["on"] and descending and alt < g.entry_altitude:
                if speed > g.entry_speed:
                    state["on"], state["t_on"] = True, t
                else:
                    state["done"] = True
            if state["on"] and speed <= g.entry_speed:
                state["on"], state["done"], state["t_off"] = False, True, t
            return 1.0 if state["on"] else 0.0

        def direction(t, rr, vv):
            sp = float(np.linalg.norm(vv))
            return -vv / sp if sp > 1e-6 else gravity.up(rr)

        def dt_fn(t, rr, vv):
            if state["on"]:
                return 0.1
            return 1.0 if gravity.altitude(rr) > 70_000 else 0.25

        traj = pm.run(
            r,
            v,
            t_end=t0 + 3000.0,
            t0=t0,
            throttle=throttle,
            direction=direction,
            tail_first=True,
            stop_on_ground=True,
            ground_altitude=0.0,
            dt_fn=dt_fn,
            prop0=prop,
            record_every=1,
        )
        pm.extra_cda = 0.0
        prop_end = traj.mass[-1] - self.m_dry
        peak_g = float(np.max(np.linalg.norm(traj.accel, axis=1))) / G0
        out = {
            "impact": traj.pos[-1],
            "downrange": self.downrange(traj.pos[-1]),
            "prop_after_entry": float(prop_end),
            "entry_burn_s": (state["t_off"] or traj.t[-1]) - state["t_on"]
            if state["t_on"] is not None
            else 0.0,
            # complete = braked to entry_speed with the engine (not by drag after running dry)
            "entry_complete": state["t_on"] is None
            or (state["t_off"] is not None and prop_end > 1.0),
            "peak_g": peak_g,
            "t_ground": float(traj.t[-1]),
        }
        if record:
            out["traj"] = traj
        return out


@dataclass
class _Arc:
    rise: float
    kick_deg: float
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    prop: np.ndarray
    accel: np.ndarray
    alt: np.ndarray
    vac_range: np.ndarray  # vacuum impact downrange of a cutoff at each sample, m
    gamma_deg: np.ndarray  # flight-path angle at each sample
    margin_est: np.ndarray  # propellant left after the landing burn (analytic estimate), kg

    def state(self, t: float):
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = (t - self.t[k]) / max(self.t[k + 1] - self.t[k], 1e-9)
        a = float(np.clip(a, 0.0, 1.0))

        def lerp(x):
            return x[k] + a * (x[k + 1] - x[k])

        return lerp(self.pos), lerp(self.vel), float(lerp(self.prop))


@functools.lru_cache(maxsize=32)
def _family_cached(vehicle_json: str, cargo: float, guidance_json: str):
    vehicle = VehicleSpec.model_validate_json(vehicle_json)
    guidance = HopGuidanceSpec.model_validate_json(guidance_json)
    model = _Model(vehicle, cargo, guidance)
    arcs = []
    for rise in RISES:
        for kick in KICKS_DEG:
            tr = model.ascent(rise, kick)
            prop = tr.mass - model.m_dry
            n = len(tr.t)
            vac = np.full(n, math.nan)
            gam = np.full(n, math.nan)
            marg = np.full(n, math.nan)
            for k in range(n):
                if tr.t[k] < rise + guidance.kick_time:
                    continue
                r, v = tr.pos[k], tr.vel[k]
                sp = float(np.linalg.norm(v))
                vac[k] = model.vacuum_range(r, v)
                gam[k] = math.degrees(math.asin(float(v @ model.gravity.up(r)) / sp))
                marg[k] = prop[k] - model.entry_prop_estimate(r, v, prop[k]) - model.landing_prop
            arcs.append(
                _Arc(rise, kick, tr.t, tr.pos, tr.vel, prop, tr.accel, tr.altitude, vac, gam, marg)
            )
    return model, arcs


def _family(vehicle: VehicleSpec, cargo: float, guidance: HopGuidanceSpec):
    return _family_cached(
        vehicle.model_dump_json(), round(float(cargo), 3), guidance.model_dump_json()
    )


def _cross_time(arc: _Arc, target: float) -> float | None:
    """First time the vacuum impact range of a cutoff reaches ``target`` (interpolated)."""
    x = arc.vac_range
    ok = np.isfinite(x)
    idx = np.where(ok[1:] & ok[:-1] & (x[:-1] < target) & (x[1:] >= target))[0]
    if not len(idx):
        return None
    k = int(idx[0])
    a = (target - x[k]) / (x[k + 1] - x[k])
    return float(arc.t[k] + a * (arc.t[k + 1] - arc.t[k]))


def _interp(arc: _Arc, field: np.ndarray, t: float) -> float:
    return float(np.interp(t, arc.t, field))


def _meco_for_range(model: _Model, arc: _Arc, target: float, t_guess: float):
    """Secant search for the cutoff time whose drag-aware impact (with the entry burn) lands
    at ``target`` downrange. Returns (t_meco, descent result) or None if out of reach."""
    t_max = float(arc.t[-1])

    def f(t):
        r, v, prop = arc.state(t)
        d = model.descent(r, v, prop, t)
        return d["downrange"] - target, d

    t0 = min(t_guess, t_max)
    f0, d0 = f(t0)
    if abs(f0) < 50.0:
        return t0, d0
    t1 = min(t0 + (0.5 if f0 < 0 else -0.5), t_max)
    best = (abs(f0), t0, d0)
    for _ in range(8):
        f1, d1 = f(t1)
        if abs(f1) < best[0]:
            best = (abs(f1), t1, d1)
        if abs(f1) < 50.0 or t1 == t0:
            break
        if f1 == f0:
            break
        t2 = t1 - f1 * (t1 - t0) / (f1 - f0)
        t2 = float(np.clip(t2, t1 - 10.0, t1 + 10.0))
        if t2 > t_max:
            if t1 >= t_max:  # still short at depletion
                return None
            t2 = t_max
        t0, f0, t1 = t1, f1, t2
    if best[0] > 2_000.0:
        f_end, _ = f(t_max)
        if f_end < 0:
            return None
    return best[1], best[2]


def _summary(model: _Model, arc: _Arc, t_meco: float, target: float | None) -> dict:
    r, v, prop = arc.state(t_meco)
    d = model.descent(r, v, prop, t_meco, record=True)
    traj = d.pop("traj")
    k = int(np.searchsorted(arc.t, t_meco))
    g_asc = np.linalg.norm(arc.accel[: max(k, 1)], axis=1) / G0
    g_des = np.linalg.norm(traj.accel, axis=1) / G0
    apogee = max(float(np.max(arc.alt[: max(k, 1)])), float(np.max(traj.altitude)))
    margin = d["prop_after_entry"] - model.landing_prop
    lim = model.g.cargo_g_limit
    peak_g = float(max(g_asc.max(), g_des.max()))
    return {
        "range_km": d["downrange"] / 1000.0,
        "target_km": None if target is None else target / 1000.0,
        "kick_deg": arc.kick_deg,
        "rise_time_s": arc.rise,
        "meco_s": t_meco,
        "meco_speed_m_s": float(np.linalg.norm(v)),
        "meco_altitude_km": float(model.gravity.altitude(r)) / 1000.0,
        "meco_flight_path_deg": _interp(arc, arc.gamma_deg, t_meco),
        "propellant_at_meco_kg": prop,
        "entry_burn_s": d["entry_burn_s"],
        "entry_burn_complete": d["entry_complete"],
        "propellant_after_entry_kg": d["prop_after_entry"],
        "landing_burn_kg": model.landing_prop,
        "fuel_margin_kg": margin,
        "fuel_margin_pct": 100.0 * margin / model.vehicle.prop_capacity,
        "apogee_km": apogee / 1000.0,
        "flight_time_s": d["t_ground"] + LANDING_EXTRA_S,
        "peak_cargo_g": peak_g,
        "peak_cargo_g_ascent": float(g_asc.max()),
        "peak_cargo_g_descent": float(g_des.max()),
        "cargo_g_limit": lim,
        "g_limit_exceeded": peak_g > lim,
        "terminal_speed_m_s": model.v_term,
    }


# ----------------------------------------------------------------------------- public API
def _refine_cutoff(model: _Model, arc: _Arc, t_est: float):
    """Latest cutoff time whose *simulated* descent keeps the landing reserve (the analytic
    estimate that seeded ``t_est`` ignores drag): bracket, then bisect."""
    t_min = arc.rise + model.g.kick_time + 1.0
    t_end = float(arc.t[-1])
    cache: dict[float, dict] = {}

    g_max = model.g.cargo_g_limit * (1.0 + G_TOLERANCE)

    def margin(t):  # >= 0: propellant for the landing burn left and cargo load acceptable
        if t not in cache:
            r, v, prop = arc.state(t)
            cache[t] = model.descent(r, v, prop, t)
        d = cache[t]
        m = d["prop_after_entry"] - model.landing_prop
        if not d["entry_complete"]:
            m = min(m, -1.0)
        return min(m, 100.0 * (g_max - d["peak_g"]))

    t_est = float(np.clip(t_est, t_min, t_end))
    step = 2.0
    if margin(t_est) >= 0:
        lo, hi = t_est, None
        while hi is None:
            t = min(lo + step, t_end)
            if margin(t) >= 0:
                lo = t
                if t >= t_end:
                    return lo, cache[lo]
                step *= 2
            else:
                hi = t
    else:
        lo, hi = None, t_est
        while lo is None:
            t = max(hi - step, t_min)
            if margin(t) >= 0:
                lo = t
            else:
                if t <= t_min:
                    return None
                hi = t
                step *= 2
    while hi - lo > 0.05:
        mid = 0.5 * (lo + hi)
        if margin(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo, cache[lo]


@functools.lru_cache(maxsize=64)
def _max_range_cached(vehicle_json: str, cargo: float, guidance_json: str) -> dict:
    vehicle = VehicleSpec.model_validate_json(vehicle_json)
    guidance = HopGuidanceSpec.model_validate_json(guidance_json)
    model, arcs = _family(vehicle, cargo, guidance)
    cands = []
    for arc in arcs:
        ok = np.isfinite(arc.margin_est) & (arc.margin_est >= 0)
        ok &= arc.gamma_deg <= guidance.max_flight_path_deg
        ok &= np.isfinite(arc.vac_range)
        if ok.any():
            k = int(np.where(ok)[0][np.argmax(arc.vac_range[ok])])
            cands.append((float(arc.vac_range[k]), arc, float(arc.t[k])))
    if not cands:
        return {"range_km": 0.0, "cargo_kg": cargo, "feasible": False}
    cands.sort(key=lambda c: -c[0])
    best = None
    for _, arc, t in cands[:3]:
        hit = _refine_cutoff(model, arc, t)
        if hit is None:
            continue
        t_cut, d = hit
        # the flight-path cap still applies at the refined cutoff
        if _interp(arc, arc.gamma_deg, t_cut) > guidance.max_flight_path_deg + 0.5:
            continue
        if best is None or d["downrange"] > best[0]:
            best = (d["downrange"], arc, t_cut)
    if best is None:
        return {"range_km": 0.0, "cargo_kg": cargo, "feasible": False}
    out = _summary(model, best[1], best[2], None)
    out["cargo_kg"] = cargo
    out["feasible"] = True
    return out


def max_range(
    vehicle: VehicleSpec | str, cargo: float, guidance: HopGuidanceSpec | None = None
) -> dict:
    """Longest range the vehicle reaches with ``cargo`` while keeping the landing reserve
    (drag-aware, with the entry burn); ``range_km`` = 0 if it cannot fly a hop at all."""
    vehicle = load_vehicle(vehicle) if isinstance(vehicle, str) else vehicle
    guidance = guidance or reference_guidance()
    return dict(
        _max_range_cached(
            vehicle.model_dump_json(), round(float(cargo), 3), guidance.model_dump_json()
        )
    )


def range_curve(
    vehicle: VehicleSpec | str,
    cargos=None,
    guidance: HopGuidanceSpec | None = None,
) -> list[dict]:
    """Max range vs cargo mass (the planner's capability curve)."""
    vehicle = load_vehicle(vehicle) if isinstance(vehicle, str) else vehicle
    if cargos is None:
        top = vehicle.cargo.max_mass or 2 * vehicle.cargo.mass or 500.0
        cargos = np.linspace(0.0, top, 10)
    rows = []
    for c in cargos:
        m = max_range(vehicle, float(c), guidance)
        rows.append(
            {
                "cargo_kg": float(c),
                "max_range_km": m["range_km"],
                "apogee_km": m.get("apogee_km"),
                "flight_time_s": m.get("flight_time_s"),
                "peak_cargo_g": m.get("peak_cargo_g"),
            }
        )
    return rows


def cargo_for_range(curve: list[dict], range_km: float) -> float | None:
    """Largest cargo whose max range reaches ``range_km`` (linear interpolation of the
    curve, which decreases with cargo); None if even an empty vehicle cannot."""
    pts = sorted((r["cargo_kg"], r["max_range_km"]) for r in curve)
    # a lighter cargo can always fly at least as far: monotone envelope (the kick-angle grid
    # leaves wiggles of a few km)
    for k in range(len(pts) - 2, -1, -1):
        pts[k] = (pts[k][0], max(pts[k][1], pts[k + 1][1]))
    if not pts or pts[0][1] < range_km:
        return None
    for (c0, r0), (c1, r1) in itertools.pairwise(pts):
        if r1 < range_km <= r0:
            return c0 + (c1 - c0) * (r0 - range_km) / (r0 - r1)
    return pts[-1][0]


def range_feasibility(
    vehicle: VehicleSpec | str,
    cargo: float,
    range_m: float,
    guidance: HopGuidanceSpec | None = None,
) -> dict:
    """Fly the planner's 3-DOF mission over ``range_m`` and report feasibility, fuel margin,
    apogee, flight time and peak cargo load; always includes the max range for this cargo."""
    vehicle = load_vehicle(vehicle) if isinstance(vehicle, str) else vehicle
    guidance = guidance or reference_guidance()
    model, arcs = _family(vehicle, cargo, guidance)
    mr = max_range(vehicle, cargo, guidance)
    best = None
    for arc in arcs:
        t = _cross_time(arc, range_m)
        if t is None:
            continue
        gam = _interp(arc, arc.gamma_deg, t)
        score = _interp(arc, arc.margin_est, t) - 50.0 * max(
            gam - guidance.max_flight_path_deg, 0.0
        )
        if best is None or score > best[0]:
            best = (score, arc, t)
    base = {
        "range_requested_km": range_m / 1000.0,
        "cargo_kg": cargo,
        "vehicle": vehicle.name,
        "max_range_km": mr["range_km"],
        "fidelity": "fast (3-DOF point mass, calm air, non-rotating spherical Earth)",
    }
    if best is None or range_m < 20_000.0:
        reason = (
            "too short for a ballistic hop (under 20 km)"
            if range_m < 20_000.0
            else "beyond the vehicle's reach even burning all propellant"
        )
        return base | {"feasible": False, "reason": reason, "result": None}
    _, arc, t_vac = best
    hit = _meco_for_range(model, arc, range_m, t_vac + 0.5)
    if hit is None:
        return base | {
            "feasible": False,
            "reason": "beyond the vehicle's reach even burning all propellant",
            "result": None,
        }
    res = _summary(model, arc, hit[0], range_m)
    g_max = guidance.cargo_g_limit * (1.0 + G_TOLERANCE)
    fuel_ok = res["fuel_margin_kg"] >= 0 and res["entry_burn_complete"]
    g_ok = res["peak_cargo_g"] <= g_max
    feasible = fuel_ok and g_ok
    if not fuel_ok:
        reason = "not enough propellant for the entry and landing burns"
    elif not g_ok:
        reason = (
            f"peak cargo load {res['peak_cargo_g']:.1f} g exceeds the {guidance.cargo_g_limit:g} g "
            "limit (the re-entry is too fast for this range)"
        )
    else:
        reason = "ok"
    if feasible and res["meco_flight_path_deg"] > guidance.max_flight_path_deg + 0.5:
        reason = "ok (lofted arc above the flight-path cap)"
    return base | {"feasible": feasible, "reason": reason, "result": res}


TABLE_FIELDS = (
    "fuel_margin_kg",
    "apogee_km",
    "flight_time_s",
    "peak_cargo_g",
    "kick_deg",
    "propellant_at_meco_kg",
)


def _table_column(args) -> dict:
    """One cargo mass of the capability table (top level so it pickles for workers)."""
    vehicle_json, cargo, ranges_km = args
    vehicle = VehicleSpec.model_validate_json(vehicle_json)
    mr = max_range(vehicle, cargo)
    col = {k: [] for k in TABLE_FIELDS}
    for rk in ranges_km:
        res = None
        if rk <= mr["range_km"] + 60.0:  # beyond that nothing is reachable
            res = range_feasibility(vehicle, cargo, rk * 1000.0)["result"]
        for k in TABLE_FIELDS:
            v = None if res is None else res[k]
            col[k].append(None if v is None else round(float(v), 2))
    return {
        "cargo_kg": cargo,
        "max_range_km": round(mr["range_km"], 2),
        "max_range_apogee_km": round(mr.get("apogee_km") or 0.0, 2),
        "max_range_flight_time_s": round(mr.get("flight_time_s") or 0.0, 1),
        "max_range_peak_cargo_g": round(mr.get("peak_cargo_g") or 0.0, 3),
        **col,
    }


def capability_table(
    vehicle: VehicleSpec | str,
    ranges_km=None,
    cargos_kg=None,
    workers: int = 1,
) -> dict:
    """Precomputed feasibility over a (range, cargo) grid plus the max-range curve: the data
    behind the static (server-less) planner and the 'cargo that would make it feasible'
    answer. ``workers`` processes (one cargo mass each)."""
    vehicle = load_vehicle(vehicle) if isinstance(vehicle, str) else vehicle
    top = vehicle.cargo.max_mass or 500.0
    cargos = [float(c) for c in (cargos_kg if cargos_kg is not None else np.arange(0, top + 1, 50))]
    if ranges_km is None:
        r0 = max_range(vehicle, min(cargos))["range_km"]
        ranges_km = np.arange(50.0, 50.0 * math.ceil(r0 / 50.0) + 1, 50.0)
    ranges = [float(r) for r in ranges_km]
    jobs = [(vehicle.model_dump_json(), c, ranges) for c in cargos]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, 4)) as pool:
            cols = list(pool.map(_table_column, jobs))
    else:
        cols = [_table_column(j) for j in jobs]
    g = reference_guidance()
    return {
        "vehicle": vehicle.name,
        "description": vehicle.description,
        "hash": vehicle_hash(vehicle),
        "cargo_nominal_kg": vehicle.cargo.mass,
        "cargo_max_kg": vehicle.cargo.max_mass,
        "cargo_g_limit": g.cargo_g_limit,
        "ranges_km": ranges,
        "cargos_kg": cargos,
        "curve": [
            {
                "cargo_kg": c["cargo_kg"],
                "max_range_km": c["max_range_km"],
                "apogee_km": c["max_range_apogee_km"],
                "flight_time_s": c["max_range_flight_time_s"],
                "peak_cargo_g": c["max_range_peak_cargo_g"],
            }
            for c in cols
        ],
        # grid[field][i_cargo][i_range]; null = not reachable
        "grid": {k: [c[k] for c in cols] for k in TABLE_FIELDS},
    }


def route_feasibility(
    launch: tuple[float, float],
    landing: tuple[float, float],
    vehicle: VehicleSpec | str,
    cargo: float,
    guidance: HopGuidanceSpec | None = None,
    curve: list[dict] | None = None,
) -> dict:
    """Feasibility of a hop between two (lat, lon) sites on WGS-84. ``curve`` (from
    :func:`range_curve`) is used to suggest the cargo that would make an infeasible route
    feasible."""
    from plume.terrain.dem import geodesic_inverse

    dist, az = geodesic_inverse(launch[0], launch[1], landing[0], landing[1])
    out = range_feasibility(vehicle, cargo, dist, guidance)
    out["launch"] = {"lat": launch[0], "lon": launch[1]}
    out["landing"] = {"lat": landing[0], "lon": landing[1]}
    out["azimuth_deg"] = az
    out["geodesic_km"] = dist / 1000.0
    if not out["feasible"] and curve:
        c = cargo_for_range(curve, dist / 1000.0)
        out["cargo_for_range_kg"] = None if c is None else max(0.0, math.floor(c))
    out["notes"] = [
        "Screening estimate: 3-DOF point mass with the flight software's planning models, "
        "calm air, nominal vehicle.",
        "Earth rotation is not modelled here; it changes the achievable range by a few per "
        "cent depending on direction (east-bound gains).",
        "Run the full 6-DOF flight and a Monte Carlo campaign before relying on a route.",
    ]
    return out
