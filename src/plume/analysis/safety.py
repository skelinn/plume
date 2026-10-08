"""Flight-safety analysis export: ground track, instantaneous impact point (IIP) trace,
landing dispersion, failure impact points, failure probability by flight phase and hazard
areas, on WGS-84 latitude/longitude, as GeoJSON, KML and a monochrome HTML summary.

Inputs: a flight replay (``*.plume.json[.gz]``) and/or a Monte Carlo campaign directory
(``runs.jsonl`` + ``dispersion.json``, written by ``plume mc``). ``plume safety`` is the CLI.

This is an *engineering input* to a licence application (for example the flight safety
analysis of FAA 14 CFR 450), not a certified analysis: there is no breakup or debris model
(the intact vehicle is propagated), no population or casualty-expectation (Ec) computation,
and the probabilities carry the Monte Carlo sampling uncertainty that is reported with them.
See docs/models/flight_safety.md.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from plume.analysis.montecarlo import INTACT_REASONS, wilson_interval

PHASES = (
    ("ascent", "Powered ascent", "liftoff to main-engine cutoff"),
    ("descent", "Coast, entry and aero descent", "MECO to landing-burn ignition"),
    ("landing", "Landing burn and touchdown", "landing burn, touchdown, settling"),
)
LOST_REASONS = frozenset(
    {"terrain_impact", "crash_hull", "crash_legs", "tipped_over", "timeout", "no_touchdown"}
)
DISCLAIMER = (
    "Engineering input to a licence application (e.g. the flight safety analysis of FAA "
    "14 CFR 450), not a certified analysis. The intact vehicle is propagated: there is no "
    "breakup, debris-fragment, population or casualty-expectation model. Probabilities are "
    "simulation estimates with the stated sampling intervals; model-form error is not "
    "included beyond the dispersed parameters."
)


# ----------------------------------------------------------------------------- geodesy
class GeoFrame:
    """Simulation world frame <-> WGS-84 latitude/longitude (degrees).

    * ``wgs84`` (high fidelity): the world frame is East-North-Up at the geodetic origin on
      the rotating WGS-84 ellipsoid; conversions are exact (:class:`EarthGravity`).
    * ``spherical`` (fast fidelity): a non-rotating sphere of radius ``R_EARTH`` whose map
      coordinates are the spherical azimuthal-equidistant projection about the launch site,
      so a point's lat/lon is the inverse projection of its map coordinates.
    """

    def __init__(self, kind: str, lat0: float, lon0: float, h0: float = 0.0, radius=None):
        from plume.physics.gravity import SphericalGravity

        self.kind = kind
        self.lat0, self.lon0 = float(lat0), float(lon0)
        if kind == "wgs84":
            from plume.config import EarthSpec, WorldSpec
            from plume.physics.gravity import gravity_from_world

            earth = EarthSpec(origin_lat_deg=lat0, origin_lon_deg=lon0, origin_height=h0)
            self.world = WorldSpec(fidelity="high", gravity="wgs84", earth=earth, ground="none")
            self.gravity = gravity_from_world(self.world)
        else:
            from plume.config import WorldSpec
            from plume.terrain.dem import site_projection

            self.world = WorldSpec(gravity="spherical", ground="none")
            self.gravity = SphericalGravity() if radius is None else SphericalGravity(radius=radius)
            self.proj = site_projection(lat0, lon0, backend="sphere")

    # -- conversions
    def latlon(self, p) -> tuple[float, float, float]:
        p = np.asarray(p, dtype=float)
        if self.kind == "wgs84":
            lat, lon, h = self.gravity.geodetic(p)
            return math.degrees(lat), (math.degrees(lon) + 180.0) % 360.0 - 180.0, float(h)
        u, v, h = self.gravity.map_coords(p)
        lat, lon = self.proj.inverse(u, v)
        return float(lat), float(lon), float(h)

    def world_point(self, lat: float, lon: float, h: float = 0.0) -> np.ndarray:
        if self.kind == "wgs84":
            return self.gravity.world_point(math.radians(lat), math.radians(lon), h)
        u, v = self.proj.forward(lat, lon)
        return self.gravity.surface_point(float(u), float(v), h)

    def enu(self, p) -> np.ndarray:
        """Columns: local east, north, up at ``p`` (world coordinates); for the sphere the
        map-grid axes, as the mission scoring uses (MissionWorld.local_frame)."""
        p = np.asarray(p, dtype=float)
        if self.kind == "wgs84":
            return self.gravity.enu_at(p)
        g = self.gravity
        u, v, _ = g.map_coords(p)
        n = g.surface_normal(u, v)
        e = g.surface_point(u + 1.0, v) - g.surface_point(u - 1.0, v)
        e -= (e @ n) * n
        e /= np.linalg.norm(e)
        return np.column_stack([e, np.cross(n, e), n])

    def offset_latlon(self, ref, east: float, north: float) -> tuple[float, float]:
        """Ground point whose offset from ``ref`` in the local tangent plane at ``ref`` is
        (east, north) metres: how ``plume mc`` records landing and impact points. Iterated
        so that it also holds for failures hundreds of kilometres away."""
        ref = np.asarray(ref, dtype=float)
        E = self.enu(ref)
        h_ref = self.latlon(ref)[2]
        q = ref + east * E[:, 0] + north * E[:, 1]
        lat = lon = 0.0
        for _ in range(8):
            lat, lon, _ = self.latlon(q)
            p = self.world_point(lat, lon, h_ref)
            d = p - ref
            de, dn = east - float(d @ E[:, 0]), north - float(d @ E[:, 1])
            if abs(de) + abs(dn) < 0.01:
                break
            q = q + de * E[:, 0] + dn * E[:, 1]
        return lat, lon

    # -- construction
    @classmethod
    def from_replay_meta(cls, meta: dict, origin: tuple[float, float] | None = None) -> GeoFrame:
        scene = meta.get("scene") or {}
        kind = "wgs84" if scene.get("frame") == "wgs84" else "spherical"
        o = scene.get("origin") or {}
        geo = meta.get("geo") or {}
        if origin is not None:
            lat0, lon0 = origin
        elif o.get("lat_deg") is not None:
            lat0, lon0 = o["lat_deg"], o["lon_deg"]
        elif geo.get("origin_lat_deg") is not None:
            lat0, lon0 = geo["origin_lat_deg"], geo["origin_lon_deg"]
        else:
            lat0 = lon0 = None
            name = (meta.get("mission") or {}).get("name")
            if name:
                try:
                    from plume.config import load_mission

                    spec = load_mission(name)
                    lat0, lon0 = spec.launch.lat, spec.launch.lon
                except FileNotFoundError:
                    pass
            if lat0 is None:
                raise ValueError("the replay has no geodetic origin: pass origin=(lat, lon)")
        if kind != "wgs84":
            return cls(kind, lat0, lon0, radius=scene.get("earth_radius"))
        return cls(kind, lat0, lon0, h0=float(o.get("height", 0.0)))


def mission_geo(mission: str, fidelity: str, origin=None):
    """(GeoFrame, target pad position, launch pad position) for a mission as flown at
    ``fidelity`` (the frame ``plume mc`` measured its landing offsets in)."""
    from plume.config import load_mission
    from plume.missions.hop import MissionWorld, mission_frame

    spec = load_mission(mission)
    spec2, _world, gravity = mission_frame(spec, fidelity)
    mw = MissionWorld(spec2, gravity)
    if fidelity == "high" or spec.world.gravity == "wgs84":
        geo = GeoFrame("wgs84", spec.launch.lat, spec.launch.lon)
    else:
        lat0, lon0 = origin or (spec.launch.lat, spec.launch.lon)
        if lat0 is None:
            raise ValueError(f"mission {mission!r} has no launch lat/lon: pass origin=(lat, lon)")
        geo = GeoFrame("spherical", lat0, lon0)
    return geo, mw.pad_b, mw.pad_a, spec2


# ----------------------------------------------------------------------------- geometry
def _local(points_ll: np.ndarray, center: tuple[float, float]):
    from plume.terrain.dem import site_projection

    pr = site_projection(center[0], center[1], backend="sphere")
    x, y = pr.forward(points_ll[:, 0], points_ll[:, 1])
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]), pr


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; counter-clockwise, without the closing point."""
    p = sorted({(float(a), float(b)) for a, b in pts})
    if len(p) < 3:
        return np.array(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def _circle(n: int = 32) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)])


def buffered_hull(points_ll: np.ndarray, buffer_m: float) -> tuple[list, float]:
    """Convex hull of lat/lon points grown by ``buffer_m``: ring [[lat, lon], ...] (closed,
    counter-clockwise) and its area in km^2 (local azimuthal-equidistant plane)."""
    pts = np.asarray(points_ll, dtype=float).reshape(-1, 2)
    c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    xy, pr = _local(pts, c)
    grown = (xy[:, None, :] + buffer_m * _circle()[None, :, :]).reshape(-1, 2)
    hull = convex_hull(grown)
    return _to_ring(hull, pr), _area_km2(hull)


def corridor(trace_ll: np.ndarray, half_width_m: float) -> tuple[list, float]:
    """Polygon of all points within ``half_width_m`` of a (gently curving) polyline, e.g.
    the IIP trace: left/right offsets plus round end caps."""
    pts = np.asarray(trace_ll, dtype=float).reshape(-1, 2)
    c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    xy, pr = _local(pts, c)
    keep = [xy[0]]
    for q in xy[1:]:  # drop points closer than a quarter width (noise, slow IIP at liftoff)
        if np.linalg.norm(q - keep[-1]) > 0.25 * half_width_m:
            keep.append(q)
    xy = np.array(keep)
    if len(xy) < 2:
        return buffered_hull(pts, half_width_m)
    seg = np.diff(xy, axis=0)
    seg /= np.linalg.norm(seg, axis=1)[:, None]
    nrm = np.column_stack([-seg[:, 1], seg[:, 0]])
    vn = np.vstack([nrm[:1], nrm[:-1] + nrm[1:], nrm[-1:]])
    vn /= np.linalg.norm(vn, axis=1)[:, None]
    left = xy + half_width_m * vn
    right = xy - half_width_m * vn

    def cap(p, start_angle):  # half circle, clockwise from start_angle (endpoints excluded)
        ang = start_angle - np.linspace(0.0, math.pi, 13)[1:-1]
        return p + half_width_m * np.column_stack([np.cos(ang), np.sin(ang)])

    a_end = math.atan2(vn[-1][1], vn[-1][0])  # left normal at the end
    a_start = math.atan2(vn[0][1], vn[0][0]) + math.pi  # right normal at the start
    ring = np.vstack([left, cap(xy[-1], a_end), right[::-1], cap(xy[0], a_start)])
    if _signed_area(ring) < 0:
        ring = ring[::-1]
    return _to_ring(ring, pr), _area_km2(ring)


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _area_km2(xy: np.ndarray) -> float:
    return abs(_signed_area(np.asarray(xy))) / 1e6 if len(xy) >= 3 else 0.0


def _to_ring(xy: np.ndarray, pr) -> list:
    lat, lon = pr.inverse(xy[:, 0], xy[:, 1])
    ring = [
        [float(a), float(b)] for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lon), strict=True)
    ]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def ellipse_ring(geo: GeoFrame, ref, ell: dict, n: int = 72) -> list:
    """Dispersion ellipse (target-local east/north metres, from montecarlo.error_ellipse) as
    a closed lat/lon ring."""
    a, b = ell["semi_axes_m"]
    th = ell["angle_rad"]
    cx, cy = ell["center"][:2]
    ring = []
    for s in np.linspace(0.0, 2 * math.pi, n, endpoint=False):
        x, y = a * math.cos(s), b * math.sin(s)
        e = cx + x * math.cos(th) - y * math.sin(th)
        nn = cy + x * math.sin(th) + y * math.cos(th)
        ring.append(list(geo.offset_latlon(ref, e, nn)))
    ring.append(ring[0])
    return ring


def clusters(points: np.ndarray, link_m: float) -> list[np.ndarray]:
    """Single-linkage clusters of lat/lon points (indices)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return []
    from plume.terrain.dem import geodesic_inverse

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if geodesic_inverse(*pts[i], *pts[j])[0] <= link_m:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g) for g in groups.values()]


# ----------------------------------------------------------------------------- replay
def load_replay(path: str | Path) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def iip_trace(
    replay: dict,
    geo: GeoFrame,
    step_s: float = 2.0,
    drag: bool = True,
) -> dict:
    """Instantaneous impact points along the powered flight: where the vehicle would hit
    the ground if thrust stopped at that instant. ``vacuum``: two-body (Kepler) arc in
    inertial space, Earth rotation undone over the fall (:func:`kepler_impact`). ``drag``:
    the 3-DOF point mass with the vehicle's axial aerodynamics (engine-first, grid fins
    deployed), standard atmosphere, calm air (:class:`ImpactPredictor`)."""
    from plume.missions.targeting import kepler_impact

    f = replay["frames"]
    meta = replay["meta"]
    t = np.asarray(f["t"], dtype=float)
    pos = np.asarray(f["pos"], dtype=float)
    vel = np.asarray(f["vel"], dtype=float)
    thrust = np.asarray(f.get("thrust", np.zeros(len(t))), dtype=float)
    prop = np.asarray(f.get("prop_mass", np.zeros(len(t))), dtype=float)
    phase = f.get("phase") or [""] * len(t)
    target = ((meta.get("scene") or {}).get("target") or {}).get("pos")
    g = geo.gravity
    ref = np.asarray(target, dtype=float) if target is not None else pos[0]
    r_target = float(np.linalg.norm(ref - g.center))
    h_ground = geo.latlon(ref)[2]
    predictor = None
    if drag:
        predictor = _drag_predictor(meta, geo, h_ground)
    out = {"t": [], "phase": [], "vacuum": [], "drag": [], "vehicle_alt_km": []}
    last = -1e9
    for k in range(len(t)):
        if thrust[k] <= 0 or t[k] - last < step_s:
            continue
        if phase[k] in ("landing_burn", "landed"):
            continue  # terminal landing burn: the IIP is the vehicle's own position
        last = t[k]
        r, v = pos[k], vel[k]
        vac = kepler_impact(r, v, g, r_target)
        if vac is None:
            continue
        out["t"].append(float(t[k]))
        out["phase"].append(phase[k])
        out["vehicle_alt_km"].append(geo.latlon(r)[2] / 1000.0)
        out["vacuum"].append(list(geo.latlon(vac)[:2]))
        if predictor is not None:
            imp = predictor.predict(r, v, float(prop[k]), entry_burn=False, t0=float(t[k]))
            out["drag"].append(None if imp is None else list(geo.latlon(imp)[:2]))
    if not out["drag"]:
        out.pop("drag")
    return out


def _drag_predictor(meta: dict, geo: GeoFrame, h_ground: float):
    from plume.config import load_mission, load_vehicle
    from plume.missions.targeting import ImpactPredictor

    name = (meta.get("vehicle") or {}).get("name")
    try:
        vehicle = load_vehicle(name)
    except Exception:
        return None
    cargo = (meta.get("mission") or {}).get("cargo_kg", vehicle.cargo.mass)
    try:
        guidance = load_mission("real_hop").guidance
    except Exception:  # pragma: no cover
        from plume.config import HopGuidanceSpec

        guidance = HopGuidanceSpec()
    world = geo.world.model_copy(update={"wind": geo.world.wind.model_copy(update={"speed": 0.0})})
    return ImpactPredictor(vehicle, world, guidance, ground_altitude=h_ground, cargo_mass=cargo)


def ground_track(replay: dict, geo: GeoFrame, step_s: float = 1.0) -> list[list[float]]:
    """[[lat, lon, height m], ...] of the flight, decimated to ``step_s``."""
    f = replay["frames"]
    out, last = [], -1e9
    t = f["t"]
    for k in range(len(t)):
        if t[k] - last >= step_s or k == len(t) - 1:
            last = t[k]
            out.append(list(geo.latlon(f["pos"][k])))
    return out


def nominal_timeline(replay: dict) -> dict:
    ev = replay.get("events") or []

    def first(label):
        return next((e["t"] for e in ev if e.get("label") == label), None)

    alt = np.asarray(replay["frames"].get("alt") or [0.0], dtype=float)
    return {
        "meco_s": first("MECO"),
        "entry_burn_s": first("Entry burn"),
        "entry_cutoff_s": first("Entry burn cutoff"),
        "landing_burn_s": first("Landing burn"),
        "touchdown_s": first("Touchdown"),
        "apogee_km": float(alt.max()) / 1000.0,
    }


# ----------------------------------------------------------------------------- Monte Carlo
def failure_phase(result: dict, nominal: dict) -> str | None:
    """Flight phase in which a failed run went wrong (None = success). Uses
    ``result['failure_phase']`` when the record has one; otherwise a documented heuristic:
    an apogee off the nominal by more than max(15 km, 10 %) means the ascent went wrong
    (the arc is fixed at MECO); a loss before the nominal landing-burn time is a descent
    failure; everything else, including off-target and tipped-over landings, is the
    landing phase."""
    if result.get("success"):
        return None
    if result.get("failure_phase") in {p[0] for p in PHASES}:
        return result["failure_phase"]
    reason = result.get("reason", "")
    if reason in INTACT_REASONS or reason in ("crash_legs", "tipped_over"):
        return "landing"
    apo, apo_nom = result.get("apogee_km"), nominal.get("apogee_km")
    if apo is not None and apo_nom and abs(apo - apo_nom) > max(15.0, 0.1 * apo_nom):
        return "ascent"
    t_lb = nominal.get("landing_burn_s")
    ft = result.get("flight_time_s")
    if t_lb is not None and ft is not None and ft < t_lb - 5.0:
        return "descent"
    return "landing"


def phase_table(records: list[dict], nominal: dict) -> list[dict]:
    valid = [r for r in records if r["result"].get("reason") != "sim_error"]
    n = len(valid)
    rows = []
    for key, label, span in PHASES:
        fails = [r for r in valid if failure_phase(r["result"], nominal) == key]
        lost = [r for r in fails if r["result"].get("reason") in LOST_REASONS]
        lo, hi = wilson_interval(len(fails), n)
        llo, lhi = wilson_interval(len(lost), n)
        rows.append(
            {
                "phase": key,
                "label": label,
                "span": span,
                "failures": len(fails),
                "probability": len(fails) / n if n else 0.0,
                "ci95": [lo, hi],
                "vehicle_lost": len(lost),
                "loss_probability": len(lost) / n if n else 0.0,
                "loss_ci95": [llo, lhi],
                "modes": dict(_count(r["result"].get("reason", "?") for r in fails)),
            }
        )
    return rows


def _count(items):
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return sorted(out.items(), key=lambda kv: -kv[1])


def _read_mc(mc_dir: Path):
    from plume.analysis.montecarlo import DispersionSpec

    records = []
    for line in (mc_dir / "runs.jsonl").read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    records.sort(key=lambda r: r["run"])
    ds = DispersionSpec(**json.loads((mc_dir / "dispersion.json").read_text()))
    return records, ds


# ----------------------------------------------------------------------------- analysis
@dataclass
class SafetyAnalysis:
    name: str
    sources: dict
    launch: list | None = None
    landing: list | None = None
    ground_track: list = field(default_factory=list)
    iip: dict | None = None
    landing_points: list = field(default_factory=list)  # [lat, lon, run, reason]
    impact_points: list = field(default_factory=list)  # [lat, lon, run, reason, phase]
    ellipses: dict = field(default_factory=dict)  # "50"/"99" -> ring
    hazard_areas: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    nominal: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"disclaimer": DISCLAIMER}


def analyze(
    mc_dir: str | Path | None = None,
    replay: str | Path | dict | None = None,
    origin: tuple[float, float] | None = None,
    iip_step_s: float = 2.0,
    drag: bool = True,
    corridor_half_width_m: float = 5_000.0,
    impact_buffer_m: float = 5_000.0,
    landing_buffer_m: float = 500.0,
    cluster_m: float = 50_000.0,
) -> SafetyAnalysis:
    """Build the safety analysis from a replay, a Monte Carlo directory, or both."""
    if mc_dir is None and replay is None:
        raise ValueError("need a replay and/or a Monte Carlo directory")
    rep = None
    if replay is not None:
        rep = replay if isinstance(replay, dict) else load_replay(replay)
    name = Path(mc_dir).name if mc_dir else (rep["meta"].get("title") or "flight")
    sa = SafetyAnalysis(name=name, sources={})
    geo = target = pad_a = None
    if mc_dir is not None:
        mc_dir = Path(mc_dir)
        records, ds = _read_mc(mc_dir)
        geo, target, pad_a, _spec = mission_geo(ds.mission, ds.fidelity, origin)
        sa.sources["monte_carlo"] = {
            "dir": mc_dir.name,
            "mission": Path(ds.mission).stem,
            "fidelity": ds.fidelity,
            "runs": len(records),
        }
    if rep is not None:
        rgeo = GeoFrame.from_replay_meta(rep["meta"], origin)
        if geo is None:
            geo = rgeo
            tgt = ((rep["meta"].get("scene") or {}).get("target") or {}).get("pos")
            target = np.asarray(tgt, dtype=float) if tgt else None
            pad_a = np.asarray(rep["frames"]["pos"][0], dtype=float)
        sa.sources["replay"] = {
            "title": rep["meta"].get("title"),
            "fidelity": rep["meta"].get("fidelity"),
            "outcome": (rep["meta"].get("outcome") or {}).get("reason"),
        }
        sa.ground_track = ground_track(rep, rgeo)
        sa.iip = iip_trace(rep, rgeo, step_s=iip_step_s, drag=drag)
        sa.nominal = nominal_timeline(rep)
    if pad_a is not None:
        sa.launch = list(geo.latlon(pad_a)[:2])
    if target is not None:
        sa.landing = list(geo.latlon(target)[:2])

    hazards = []
    if sa.iip and sa.iip.get("t"):
        trace = [p for p in (sa.iip.get("drag") or sa.iip["vacuum"]) if p is not None]
        if len(trace) >= 2:
            ring, area = corridor(np.array(trace), corridor_half_width_m)
            hazards.append(
                {
                    "kind": "iip_corridor",
                    "label": "Powered-flight IIP corridor",
                    "basis": (
                        f"{'drag-aware' if sa.iip.get('drag') else 'vacuum'} IIP trace "
                        f"+/- {corridor_half_width_m / 1000:.1f} km"
                    ),
                    "ring": ring,
                    "area_km2": area,
                }
            )

    if mc_dir is not None:
        if not sa.nominal:
            ok = [r["result"] for r in records if r["result"].get("success")]
            if ok:
                sa.nominal = {
                    "apogee_km": float(np.median([x.get("apogee_km", np.nan) for x in ok])),
                    "landing_burn_s": float(
                        np.median([x.get("flight_time_s", np.nan) for x in ok]) - 15.0
                    ),
                }
        from plume.analysis.montecarlo import summarize

        summary = summarize(records, ds)
        for r in records:
            res = r["result"]
            if res.get("reason") == "sim_error" or "landing_east_m" not in res:
                continue
            lat, lon = geo.offset_latlon(target, res["landing_east_m"], res["landing_north_m"])
            if res.get("success") or res.get("reason") in INTACT_REASONS:
                sa.landing_points.append([lat, lon, r["run"], res.get("reason")])
            else:
                ph = failure_phase(res, sa.nominal)
                sa.impact_points.append([lat, lon, r["run"], res.get("reason"), ph])
        land = summary.get("landing") or {}
        for key in ("50", "99"):
            e = land.get(f"ellipse{key}")
            if e and e["semi_axes_m"][0] > 0:
                sa.ellipses[key] = ellipse_ring(geo, target, e)
        sa.phases = phase_table(records, sa.nominal)
        valid = [r for r in records if r["result"].get("reason") != "sim_error"]
        k = sum(1 for r in valid if r["result"].get("success"))
        lost = sum(1 for r in valid if r["result"].get("reason") in LOST_REASONS)
        sa.stats = {
            "runs": len(records),
            "valid_runs": len(valid),
            "sim_errors": len(records) - len(valid),
            "success_probability": k / len(valid) if valid else 0.0,
            "success_ci95": list(wilson_interval(k, len(valid))),
            "vehicle_loss_probability": lost / len(valid) if valid else 0.0,
            "vehicle_loss_ci95": list(wilson_interval(lost, len(valid))),
            "cep50_m": land.get("cep50_m"),
            "cep90_m": land.get("cep90_m"),
        }
        # landing-zone hazard: every touchdown plus the 99 % ellipse, buffered
        # (vehicles lost *at* the landing site - tipped over, broken legs - belong here too)
        at_lz = [p for p in sa.impact_points if p[4] == "landing"]
        lz = [p[:2] for p in sa.landing_points + at_lz] + (sa.ellipses.get("99") or [])
        if lz:
            ring, area = buffered_hull(np.array(lz), landing_buffer_m)
            hazards.append(
                {
                    "kind": "landing_zone",
                    "label": "Landing hazard area",
                    "basis": (
                        f"hull of {len(sa.landing_points) + len(at_lz)} touchdowns and the "
                        f"99 % ellipse + {landing_buffer_m:.0f} m"
                    ),
                    "ring": ring,
                    "area_km2": area,
                }
            )
        away = [p for p in sa.impact_points if p[4] != "landing"]
        if away:
            pts = np.array([p[:2] for p in away], dtype=float)
            for gi, idx in enumerate(clusters(pts, cluster_m)):
                ring, area = buffered_hull(pts[idx], impact_buffer_m)
                phases = sorted({away[i][4] for i in idx})
                hazards.append(
                    {
                        "kind": "failure_impacts",
                        "label": f"Failure impact area {gi + 1}",
                        "basis": (
                            f"hull of {len(idx)} failed-run impact point(s) "
                            f"+ {impact_buffer_m / 1000:.1f} km"
                        ),
                        "phases": phases,
                        "runs": [away[i][2] for i in idx],
                        "ring": ring,
                        "area_km2": area,
                    }
                )
    sa.hazard_areas = hazards
    return sa


# ----------------------------------------------------------------------------- GeoJSON
def _ll(p):  # [lat, lon(, h)] -> GeoJSON [lon, lat(, h)]
    return [round(p[1], 7), round(p[0], 7), *([round(p[2], 1)] if len(p) > 2 else [])]


def _ring_ccw(ring: list) -> list:
    """GeoJSON rings: exterior counter-clockwise in lon/lat (RFC 7946 right-hand rule)."""
    xy = np.array([[p[1], p[0]] for p in ring])
    pts = [_ll(p) for p in ring]
    return pts if _signed_area(xy) >= 0 else pts[::-1]


def to_geojson(sa: SafetyAnalysis) -> dict:
    feats = []

    def add(geom, **props):
        feats.append({"type": "Feature", "geometry": geom, "properties": props})

    if sa.launch:
        add(
            {"type": "Point", "coordinates": _ll(sa.launch)}, kind="launch_site", name="Launch site"
        )
    if sa.landing:
        add(
            {"type": "Point", "coordinates": _ll(sa.landing)},
            kind="landing_site",
            name="Landing site",
        )
    if sa.ground_track:
        add(
            {"type": "LineString", "coordinates": [_ll(p) for p in sa.ground_track]},
            kind="ground_track",
            name="Nominal ground track",
            note="third coordinate: height above the ellipsoid, m",
        )
    if sa.iip and sa.iip.get("t"):
        for key, label in (("vacuum", "IIP trace (vacuum)"), ("drag", "IIP trace (drag-aware)")):
            pts = [p for p in sa.iip.get(key) or [] if p is not None]
            if len(pts) >= 2:
                add(
                    {"type": "LineString", "coordinates": [_ll(p) for p in pts]},
                    kind=f"iip_{key}",
                    name=label,
                    t_start_s=sa.iip["t"][0],
                    t_end_s=sa.iip["t"][-1],
                )
    for key, ring in sa.ellipses.items():
        add(
            {"type": "Polygon", "coordinates": [_ring_ccw(ring)]},
            kind=f"landing_ellipse_{key}",
            name=f"Landing dispersion {key} % ellipse",
            probability=int(key) / 100,
        )
    for lat, lon, run, reason in sa.landing_points:
        add(
            {"type": "Point", "coordinates": _ll([lat, lon])},
            kind="landing_point",
            run=run,
            reason=reason,
        )
    for lat, lon, run, reason, phase in sa.impact_points:
        add(
            {"type": "Point", "coordinates": _ll([lat, lon])},
            kind="impact_point",
            run=run,
            reason=reason,
            phase=phase,
        )
    for h in sa.hazard_areas:
        props = {k: v for k, v in h.items() if k != "ring"}
        add({"type": "Polygon", "coordinates": [_ring_ccw(h["ring"])]}, **props, name=h["label"])
    return {
        "type": "FeatureCollection",
        "name": sa.name,
        "properties": {
            "phases": sa.phases,
            "stats": sa.stats,
            "sources": sa.sources,
            "disclaimer": DISCLAIMER,
            "crs": "WGS-84 longitude/latitude, degrees",
        },
        "features": feats,
    }


# ----------------------------------------------------------------------------- KML
_KML_STYLES = {
    # aabbggrr; monochrome
    "track": ("ffffffff", 2.0, None),
    "iip_vacuum": ("ff9a9a9a", 1.5, None),
    "iip_drag": ("ffffffff", 1.5, None),
    "ellipse": ("ffffffff", 1.5, "00ffffff"),
    "hazard": ("ffffffff", 2.0, "33ffffff"),
    "point": ("ffffffff", 1.0, None),
    "impact": ("ff000000", 1.0, None),
}


def _kml_coords(pts, alt: bool = False) -> str:
    if alt:
        return " ".join(f"{p[1]:.7f},{p[0]:.7f},{p[2]:.1f}" for p in pts)
    return " ".join(f"{p[1]:.7f},{p[0]:.7f},0" for p in pts)


def to_kml(sa: SafetyAnalysis) -> str:
    styles = []
    for sid, (line, width, fill) in _KML_STYLES.items():
        poly = f"<PolyStyle><color>{fill}</color></PolyStyle>" if fill else ""
        icon = ""
        if sid in ("point", "impact"):
            icon = (
                f"<IconStyle><color>{'ffffffff' if sid == 'point' else 'ff000000'}</color>"
                "<scale>0.6</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/"
                f"{'placemark_circle' if sid == 'point' else 'cross-hairs'}.png</href></Icon>"
                "</IconStyle>"
            )
        styles.append(
            f'<Style id="{sid}">{icon}<LineStyle><color>{line}</color><width>{width}</width>'
            f"</LineStyle>{poly}</Style>"
        )

    def pm(name, style, geom, desc=""):
        d = f"<description>{escape(desc)}</description>" if desc else ""
        return f"<Placemark><name>{escape(name)}</name>{d}<styleUrl>#{style}</styleUrl>{geom}</Placemark>"

    def line(pts, alt=False):
        mode = "absolute" if alt else "clampToGround"
        return (
            f"<LineString><tessellate>1</tessellate><altitudeMode>{mode}</altitudeMode>"
            f"<coordinates>{_kml_coords(pts, alt)}</coordinates></LineString>"
        )

    def poly(ring):
        return (
            "<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>"
            f"{_kml_coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        )

    def point(p):
        return f"<Point><coordinates>{p[1]:.7f},{p[0]:.7f},0</coordinates></Point>"

    folders = []
    sites = []
    if sa.launch:
        sites.append(pm("Launch site", "point", point(sa.launch)))
    if sa.landing:
        sites.append(pm("Landing site", "point", point(sa.landing)))
    if sa.ground_track:
        sites.append(pm("Nominal trajectory", "track", line(sa.ground_track, alt=True)))
    if sites:
        folders.append(("Nominal flight", sites))
    if sa.iip and sa.iip.get("t"):
        items = []
        for key in ("vacuum", "drag"):
            pts = [p for p in sa.iip.get(key) or [] if p is not None]
            if len(pts) >= 2:
                items.append(pm(f"IIP trace ({key})", f"iip_{key}", line(pts)))
        folders.append(("Instantaneous impact point", items))
    if sa.ellipses or sa.landing_points:
        items = [pm(f"{k} % landing ellipse", "ellipse", poly(r)) for k, r in sa.ellipses.items()]
        items += [
            pm(f"run {run}: {reason}", "point", point([lat, lon]))
            for lat, lon, run, reason in sa.landing_points
        ]
        folders.append(("Landing dispersion", items))
    if sa.impact_points:
        folders.append(
            (
                "Failure impact points",
                [
                    pm(f"run {run}: {reason} ({phase})", "impact", point([lat, lon]))
                    for lat, lon, run, reason, phase in sa.impact_points
                ],
            )
        )
    if sa.hazard_areas:
        folders.append(
            (
                "Hazard areas",
                [
                    pm(
                        h["label"],
                        "hazard",
                        poly(h["ring"]),
                        f"{h['basis']}; {h['area_km2']:,.1f} km2",
                    )
                    for h in sa.hazard_areas
                ],
            )
        )
    body = "".join(
        f"<Folder><name>{escape(n)}</name>{''.join(items)}</Folder>" for n, items in folders
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape('Plume flight safety: ' + sa.name)}</name>"
        f"<description>{escape(DISCLAIMER)}</description>"
        f"{''.join(styles)}{body}</Document></kml>\n"
    )


# ----------------------------------------------------------------------------- HTML
def _map_svg(sa: SafetyAnalysis, w: int = 880, h: int = 520) -> str:
    """Equirectangular sketch (longitude scaled by cos(latitude)) with Natural Earth
    coastlines and borders when the vendored map is available."""
    pts = []
    pts += [p[:2] for p in sa.ground_track]
    for h_ in sa.hazard_areas:
        pts += h_["ring"]
    pts += [p[:2] for p in sa.impact_points] + [p[:2] for p in sa.landing_points]
    if sa.launch:
        pts.append(sa.launch)
    if sa.landing:
        pts.append(sa.landing)
    if not pts:
        return ""
    a = np.array(pts, dtype=float)
    lat0 = float(a[:, 0].mean())
    k = math.cos(math.radians(lat0))
    x0, x1 = a[:, 1].min() * k, a[:, 1].max() * k
    y0, y1 = a[:, 0].min(), a[:, 0].max()
    span = max(x1 - x0, (y1 - y0) * w / h, 0.05) * 1.15
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    s = w / span

    def X(lon):
        return w / 2 + (lon * k - cx) * s

    def Y(lat):
        return h / 2 - (lat - cy) * s

    def path(points, close=False):
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{X(p[1]):.1f},{Y(p[0]):.1f}" for i, p in enumerate(points)
        )
        return d + (" Z" if close else "")

    out = [
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="map">',
        f'<rect width="{w}" height="{h}" fill="none" stroke="var(--rule)"/>',
    ]
    world = Path(__file__).resolve().parents[1] / "viz" / "static" / "planner" / "world.json"
    lon_lo, lon_hi = cx / k - span / 2 / k, cx / k + span / 2 / k
    lat_lo, lat_hi = cy - h / 2 / s, cy + h / 2 / s
    if world.exists():
        data = json.loads(world.read_text(encoding="utf-8"))
        for layer, sw, dash in (("coast", 1.0, ""), ("borders", 0.8, ""), ("states", 0.6, "2 3")):
            for flat in data.get(layer, []):
                ll = np.asarray(flat, dtype=float).reshape(-1, 2)
                inb = (ll[:, 0] > lon_lo - 1) & (ll[:, 0] < lon_hi + 1)
                inb &= (ll[:, 1] > lat_lo - 1) & (ll[:, 1] < lat_hi + 1)
                if not inb.any():
                    continue
                d = path([[q[1], q[0]] for q in ll])
                da = f' stroke-dasharray="{dash}"' if dash else ""
                out.append(
                    f'<path d="{d}" fill="none" stroke="var(--mute)" stroke-width="{sw}"{da} '
                    'opacity="0.6"/>'
                )
    for hz in sa.hazard_areas:
        out.append(
            f'<path d="{path(hz["ring"], True)}" fill="var(--fg)" fill-opacity="0.08" '
            'stroke="var(--fg)" stroke-width="1"/>'
        )
    if sa.ground_track:
        out.append(
            f'<path d="{path(sa.ground_track)}" fill="none" stroke="var(--fg)" stroke-width="1.5"/>'
        )
    if sa.iip:
        for key, dash in (("vacuum", "2 3"), ("drag", "6 3")):
            pts = [p for p in sa.iip.get(key) or [] if p is not None]
            if len(pts) >= 2:
                out.append(
                    f'<path d="{path(pts)}" fill="none" stroke="var(--fg)" stroke-dasharray="{dash}"/>'
                )
    for p in sa.landing_points:
        out.append(f'<circle cx="{X(p[1]):.1f}" cy="{Y(p[0]):.1f}" r="1.5" fill="var(--fg)"/>')
    for p in sa.impact_points:
        x, y = X(p[1]), Y(p[0])
        out.append(
            f'<path d="M{x - 4:.1f},{y - 4:.1f} L{x + 4:.1f},{y + 4:.1f} M{x - 4:.1f},{y + 4:.1f} '
            f'L{x + 4:.1f},{y - 4:.1f}" stroke="var(--fg)" stroke-width="1.5"/>'
        )
    for p, lbl in ((sa.launch, "LAUNCH"), (sa.landing, "LANDING")):
        if p:
            x, y = X(p[1]), Y(p[0])
            out.append(
                f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="var(--bg)" '
                f'stroke="var(--fg)"/><text x="{x + 8:.1f}" y="{y - 6:.1f}" class="ax">{lbl}</text>'
            )
    km = (w / s) * 111.32 / 5  # one fifth of the width, km
    nice = 10 ** math.floor(math.log10(km))
    nice *= 5 if km / nice >= 5 else 2 if km / nice >= 2 else 1
    px = nice / 111.32 * s
    out.append(
        f'<path d="M16,{h - 16} h{px:.1f}" stroke="var(--fg)"/><text x="16" y="{h - 22}" '
        f'class="ax">{nice:,.0f} km</text>'
    )
    out.append("</svg>")
    return "".join(out)


def to_html(sa: SafetyAnalysis) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f} %"

    def ci(c):
        return "-" if not c else f"{100 * c[0]:.1f}–{100 * c[1]:.1f} %"

    st = sa.stats
    kpis = []
    if st:
        kpis += [
            (pct(st.get("success_probability")), "mission success"),
            (ci(st.get("success_ci95")), "95 % interval"),
            (pct(st.get("vehicle_loss_probability")), "vehicle lost"),
            (ci(st.get("vehicle_loss_ci95")), "95 % interval"),
        ]
    kpis.append((f"{len(sa.hazard_areas)}", "hazard areas"))
    if sa.iip and sa.iip.get("t"):
        kpis.append((f"{sa.iip['t'][-1] - sa.iip['t'][0]:.0f} s", "IIP trace span"))
    kp = "".join(f'<div class="kpi"><b>{v}</b><span>{lbl}</span></div>' for v, lbl in kpis)
    phase_rows = "".join(
        f"<tr><td>{p['label']}<br><small>{p['span']}</small></td><td class=n>{p['failures']}</td>"
        f"<td class=n>{pct(p['probability'])}</td><td class=n>{ci(p['ci95'])}</td>"
        f"<td class=n>{p['vehicle_lost']}</td><td class=n>{pct(p['loss_probability'])}</td>"
        f"<td>{', '.join(f'{k} {v}' for k, v in p['modes'].items())}</td></tr>"
        for p in sa.phases
    )
    phase_tbl = (
        "<table><tr><th>phase</th><th>failed runs</th><th>P(failure)</th><th>95 % interval</th>"
        f"<th>vehicle lost</th><th>P(loss)</th><th>modes</th></tr>{phase_rows}</table>"
        if sa.phases
        else '<p class="note">No Monte Carlo campaign given: no failure probabilities.</p>'
    )
    hz_rows = "".join(
        f"<tr><td>{h['label']}</td><td>{escape(h['basis'])}</td>"
        f"<td class=n>{h['area_km2']:,.1f}</td></tr>"
        for h in sa.hazard_areas
    )
    imp_rows = "".join(
        f"<tr><td class=n>{run}</td><td>{reason}</td><td>{phase}</td>"
        f"<td class=n>{lat:.4f}</td><td class=n>{lon:.4f}</td></tr>"
        for lat, lon, run, reason, phase in sa.impact_points
    )
    src = " · ".join(
        f"{k}: {', '.join(f'{a} {b}' for a, b in v.items() if b is not None)}"
        for k, v in sa.sources.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flight safety: {escape(sa.name)}</title>
<style>
:root{{--bg:#fff;--fg:#0a0a0a;--mute:#666;--rule:#d0d0d0}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0a0a0a;--fg:#f2f2f2;--mute:#8a8a8a;--rule:#2a2a2a}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:32px 16px}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}} h2{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:32px 0 8px;font-weight:500}}
p.sub{{color:var(--mute);margin:0 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid var(--rule);border-left:1px solid var(--rule)}}
.kpi{{padding:12px 16px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.kpi b{{display:block;font:500 20px "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.kpi span{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}}
th{{font-weight:500;color:var(--mute);font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
td.n{{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
small{{color:var(--mute)}} svg{{max-width:100%;height:auto;display:block}}
.ax{{font:10px "IBM Plex Mono",monospace;fill:var(--mute)}}
.note{{color:var(--mute);font-size:12px}}
.warn{{border:1px solid var(--fg);padding:12px 16px;margin:0 0 24px;font-size:13px}}
.warn b{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;display:block;margin-bottom:4px}}
</style></head><body><main>
<h1>Flight safety: {escape(sa.name)}</h1>
<p class="sub">{escape(src)}</p>
<div class="warn"><b>Not a certified analysis</b>{escape(DISCLAIMER)}</div>
<div class="kpis">{kp}</div>
<h2>Map</h2>
{_map_svg(sa)}
<p class="note">Solid line: nominal ground track. Dashed: drag-aware IIP trace; dotted: vacuum IIP
trace. Shaded: hazard areas. Dots: Monte Carlo touchdowns; crosses: impact points of failed runs.
Coastlines and borders: Natural Earth (public domain).</p>
<h2>Failure probability by flight phase</h2>
{phase_tbl}
<p class="note">Phase assignment: a recorded failure phase when the run has one; otherwise an apogee
more than max(15 km, 10 %) off nominal marks an ascent failure (the ballistic arc is fixed at
MECO), a loss before the nominal landing-burn time a descent failure, and the rest (including
off-target and tipped-over landings) the landing phase. Simulation errors are excluded.</p>
<h2>Hazard areas</h2>
<table><tr><th>area</th><th>basis</th><th>km&sup2;</th></tr>{hz_rows}</table>
<h2>Failed-run impact points</h2>
{"<table><tr><th>run</th><th>reason</th><th>phase</th><th>lat</th><th>lon</th></tr>" + imp_rows + "</table>" if imp_rows else '<p class="note">None.</p>'}
<h2>Method</h2>
<p class="note">IIP: for each powered-flight instant, the impact point if thrust stopped then.
Vacuum: two-body arc (inertial, Earth rotation undone over the fall). Drag-aware: 3-DOF point mass
with the vehicle's axial aerodynamics (engine-first with grid fins deployed), standard atmosphere,
calm air. Landing and impact points come from the Monte Carlo records (local east/north offsets at
the target, converted to WGS-84). Ellipses: bivariate-normal fit to the touchdowns. Hazard areas:
convex hulls with the stated buffers; they are not containment probabilities. Files: GeoJSON
(RFC 7946, lon/lat) and KML for Google Earth. See docs/models/flight_safety.md.</p>
</main></body></html>"""


def write_outputs(sa: SafetyAnalysis, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "geojson": out / "safety.geojson",
        "kml": out / "safety.kml",
        "html": out / "safety.html",
        "json": out / "safety.json",
    }
    paths["geojson"].write_text(json.dumps(to_geojson(sa)), encoding="utf-8")
    paths["kml"].write_text(to_kml(sa), encoding="utf-8")
    paths["html"].write_text(to_html(sa), encoding="utf-8")
    paths["json"].write_text(json.dumps(sa.to_json(), default=float), encoding="utf-8")
    return paths


def viewer_overlay(sa: SafetyAnalysis, geo: GeoFrame) -> dict:
    """``meta.safety`` block for a replay: IIP traces and hazard rings in world (ENU frame)
    coordinates on the ground, for the viewer's engineering view."""

    def w(points):
        return [[round(float(x), 1) for x in geo.world_point(p[0], p[1], 0.0)] for p in points]

    out: dict = {"hazards": []}
    if sa.iip:
        for key in ("vacuum", "drag"):
            pts = [p for p in sa.iip.get(key) or [] if p is not None]
            if len(pts) >= 2:
                out[f"iip_{key}"] = w(pts)
    for h in sa.hazard_areas:
        out["hazards"].append({"label": h["label"], "kind": h["kind"], "ring": w(h["ring"])})
    return out
