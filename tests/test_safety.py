"""Flight-safety export: geodesy, IIP, phases, hazard geometry, GeoJSON/KML/HTML, CLI."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plume.analysis import safety
from plume.analysis.safety import GeoFrame


@pytest.mark.parametrize("kind", ["wgs84", "spherical"])
def test_latlon_round_trip(kind):
    geo = GeoFrame(kind, 32.99, -106.986)
    rng = np.random.default_rng(3)
    for _ in range(20):
        lat = 32.99 + rng.uniform(-8, 8)
        lon = -106.986 + rng.uniform(-12, 12)
        h = rng.uniform(-100, 200_000)
        p = geo.world_point(lat, lon, h)
        la, lo, hh = geo.latlon(p)
        assert la == pytest.approx(lat, abs=1e-8)
        assert lo == pytest.approx(lon, abs=1e-8)
        assert hh == pytest.approx(h, abs=1e-3)
    o = geo.latlon(np.zeros(3))
    assert o[0] == pytest.approx(32.99, abs=1e-9) and o[1] == pytest.approx(-106.986, abs=1e-9)


@pytest.mark.parametrize("kind", ["wgs84", "spherical"])
def test_target_offsets_convert_back(kind):
    """plume mc stores landing/impact points as local east/north offsets at the target;
    offset_latlon must invert that even hundreds of km away."""
    geo = GeoFrame(kind, 32.99, -106.986)
    ref = geo.world_point(35.335, -99.225, 450.0)
    E = geo.enu(ref)
    for lat, lon in [(35.336, -99.224), (34.0, -101.0), (33.1, -106.9), (27.8, -119.5)]:
        p = geo.world_point(lat, lon, 450.0)
        d = p - ref
        la, lo = geo.offset_latlon(ref, float(d @ E[:, 0]), float(d @ E[:, 1]))
        assert la == pytest.approx(lat, abs=1e-7) and lo == pytest.approx(lon, abs=1e-7)


def test_vacuum_iip_matches_kepler_and_integration():
    """IIP of a ballistic state: the analytic two-body impact point agrees with a numerical
    vacuum integration of the same state (non-rotating sphere)."""
    from plume.config import WorldSpec, load_vehicle
    from plume.missions.targeting import kepler_impact
    from plume.physics.pointmass import PointMassSim

    geo = GeoFrame("spherical", 32.99, -106.986)
    g = geo.gravity
    r0 = g.surface_point(0.0, 0.0, 40_000.0)
    v0 = np.array([1500.0, 600.0, 900.0])
    imp = kepler_impact(r0, v0, g, g.earth_radius)
    world = WorldSpec(gravity="spherical", ground="none", atmosphere=False)
    pm = PointMassSim(load_vehicle("cargo_hopper"), world)
    traj = pm.run(r0, v0, t_end=2000.0, dt=0.05, throttle=0.0, prop0=0.0, ground_altitude=0.0)
    end = traj.pos[-1]
    # the integrator stops within one step below the surface: compare on the ground
    assert np.linalg.norm(end - imp) < 150.0
    # and the replay-based trace reproduces kepler_impact exactly
    rep = {
        "meta": {"scene": {"frame": "spherical", "target": {"pos": g.surface_point(0, 0, 0).tolist()}}},
        "frames": {"t": [0.0], "pos": [r0.tolist()], "vel": [v0.tolist()], "thrust": [1.0],
                   "prop_mass": [0.0], "phase": ["ascent"]},
    }  # fmt: skip
    tr = safety.iip_trace(rep, geo, drag=False)
    la, lo, _ = geo.latlon(imp)
    assert tr["vacuum"][0] == pytest.approx([la, lo], abs=1e-9)


def test_rotating_earth_iip_is_east_shifted():
    """On the rotating WGS-84 Earth the ground moves east under a falling vehicle: a
    vertical shot comes down west of the pad."""
    from plume.missions.targeting import kepler_impact

    geo = GeoFrame("wgs84", 0.0, 0.0)
    g = geo.gravity
    r0 = geo.world_point(0.0, 0.0, 50_000.0)
    up = g.up(r0)
    imp = kepler_impact(
        r0, 1000.0 * up, g, float(np.linalg.norm(geo.world_point(0, 0, 0) - g.center))
    )
    lat, lon, _ = geo.latlon(imp)
    west_m = -math.radians(lon) * 6_378_137.0
    # Coriolis: (4/3) w v0^3 / g^2 ~ 1 km for a 1 km/s vertical shot
    assert 300.0 < west_m < 3000.0 and abs(lat) < 0.001


def test_hull_and_corridor_areas():
    ring, area = safety.buffered_hull(np.array([[35.0, -100.0]]), 1000.0)
    assert area == pytest.approx(math.pi, rel=0.02)  # km^2
    assert ring[0] == ring[-1]
    # a ~111 km north-south line with a 5 km half-width
    line = np.column_stack([np.linspace(35.0, 36.0, 40), np.full(40, -100.0)])
    ring, area = safety.corridor(line, 5000.0)
    length = 111.0
    assert area == pytest.approx(2 * 5 * length + math.pi * 25, rel=0.03)


def test_failure_phase_assignment():
    nominal = {"apogee_km": 186.0, "landing_burn_s": 575.0}
    fp = safety.failure_phase
    assert fp({"success": True, "reason": "landed"}, nominal) is None
    assert (
        fp({"reason": "terrain_impact", "apogee_km": 10.9, "flight_time_s": 209}, nominal)
        == "ascent"
    )
    assert (
        fp({"reason": "terrain_impact", "apogee_km": 185.0, "flight_time_s": 480}, nominal)
        == "descent"
    )
    assert (
        fp({"reason": "crash_legs", "apogee_km": 186.0, "flight_time_s": 590}, nominal) == "landing"
    )
    assert fp({"reason": "missed_target", "apogee_km": 186.0}, nominal) == "landing"
    assert fp({"reason": "terrain_impact", "failure_phase": "descent"}, nominal) == "descent"


def _synthetic_mc(tmp_path):
    """A small fake campaign on the demo route: 12 landings, an ascent loss, a descent
    loss and a tip-over."""
    from plume.analysis.montecarlo import DispersionSpec

    d = tmp_path / "mc"
    d.mkdir()
    ds = DispersionSpec(name="demo_hop", mission="demo_hop", runs=15, fidelity="fast")
    (d / "dispersion.json").write_text(ds.model_dump_json())
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(12):
        e, n = rng.normal(0, 15, 2)
        rows.append({"success": True, "reason": "landed", "landing_east_m": e, "landing_north_m": n,
                     "apogee_km": 180.0 + rng.normal(), "flight_time_s": 566.0, "touchdown_vz_mps": 1.0})  # fmt: skip
    rows.append({"success": False, "reason": "terrain_impact", "landing_east_m": -720e3,
                 "landing_north_m": -165e3, "apogee_km": 8.0, "flight_time_s": 120.0})  # fmt: skip
    rows.append({"success": False, "reason": "terrain_impact", "landing_east_m": -40e3,
                 "landing_north_m": -9e3, "apogee_km": 179.0, "flight_time_s": 470.0})  # fmt: skip
    rows.append({"success": False, "reason": "tipped_over", "landing_east_m": 6.0,
                 "landing_north_m": -3.0, "apogee_km": 180.0, "flight_time_s": 570.0,
                 "touchdown_vz_mps": 3.0})  # fmt: skip
    with (d / "runs.jsonl").open("w") as f:
        for i, res in enumerate(rows):
            f.write(
                json.dumps({"run": i, "values": {"x": float(i)}, "result": res, "error": None})
                + "\n"
            )
    return d


def test_mc_analysis_and_exports(tmp_path):
    d = _synthetic_mc(tmp_path)
    sa = safety.analyze(mc_dir=d)
    assert sa.landing and sa.launch
    assert len(sa.landing_points) == 12 and len(sa.impact_points) == 3
    phases = {p["phase"]: p for p in sa.phases}
    assert phases["ascent"]["failures"] == 1 and phases["descent"]["failures"] == 1
    assert phases["landing"]["failures"] == 1 and phases["landing"]["vehicle_lost"] == 1
    assert phases["ascent"]["probability"] == pytest.approx(1 / 15)
    lo, hi = phases["ascent"]["ci95"]
    assert lo < 1 / 15 < hi
    kinds = [h["kind"] for h in sa.hazard_areas]
    assert kinds.count("landing_zone") == 1 and kinds.count("failure_impacts") == 2
    # the ascent failure lands near the pad (720 km back along the route)
    asc = next(p for p in sa.impact_points if p[4] == "ascent")
    from plume.terrain.dem import geodesic_inverse

    assert geodesic_inverse(asc[0], asc[1], *sa.launch)[0] < 60_000

    paths = safety.write_outputs(sa, tmp_path / "out")
    gj = json.loads(paths["geojson"].read_text())
    assert gj["type"] == "FeatureCollection" and gj["features"]
    for f in gj["features"]:
        g = f["geometry"]
        assert f["type"] == "Feature" and "kind" in f["properties"]
        coords = {"Point": lambda c: [c], "LineString": lambda c: c, "Polygon": lambda c: c[0]}[
            g["type"]
        ](g["coordinates"])
        for lon, lat, *_ in coords:
            assert -180 <= lon <= 180 and -90 <= lat <= 90
        if g["type"] == "Polygon":
            ring = np.array(g["coordinates"][0])
            assert ring[0].tolist() == ring[-1].tolist() and len(ring) >= 4
            x, y = ring[:, 0], ring[:, 1]
            assert np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0  # counter-clockwise exterior
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"landing_ellipse_50", "landing_ellipse_99", "hazard_area", "impact_point"} <= kinds
    hz = [
        f["properties"]["hazard"]
        for f in gj["features"]
        if f["properties"]["kind"] == "hazard_area"
    ]
    assert sorted(hz) == ["failure_impacts", "failure_impacts", "landing_zone"]
    root = ET.fromstring(paths["kml"].read_text(encoding="utf-8"))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    assert root.tag == "{http://www.opengis.net/kml/2.2}kml"
    assert len(root.findall(".//k:Placemark", ns)) >= 15 + 2 + len(sa.hazard_areas)
    for c in root.findall(".//k:coordinates", ns):
        for tup in c.text.split():
            lon, lat, _ = (float(x) for x in tup.split(","))
            assert -180 <= lon <= 180 and -90 <= lat <= 90
    html = paths["html"].read_text(encoding="utf-8")
    assert "Not a certified analysis" in html and "14 CFR 450" in html
    assert "<svg" in html


def test_replay_analysis_and_cli(tmp_path):
    """A recorded flight: ground track, IIP trace and the corridor, through the CLI."""
    from typer.testing import CliRunner

    from plume.cli import app

    rep = tmp_path / "flight.plume.json"
    geo = GeoFrame("spherical", 31.0, -104.0)
    g = geo.gravity
    pad = g.surface_point(0, 0, 0)
    frames = {"t": [], "pos": [], "vel": [], "thrust": [], "prop_mass": [], "phase": []}
    r, v = pad.copy(), np.zeros(3)
    for k in range(400):  # crude powered ascent pitching east, then coast
        t = 0.25 * k
        burn = t < 60.0
        dirn = np.array(
            [math.sin(math.radians(min(t, 45.0))), 0.0, math.cos(math.radians(min(t, 45.0)))]
        )
        a = (25.0 * dirn if burn else 0.0) + g.accel(r)
        v = v + 0.25 * a
        r = r + 0.25 * v
        for key, val in (("t", t), ("pos", r.tolist()), ("vel", v.tolist()), ("thrust", 1e5 if burn else 0.0),
                         ("prop_mass", 3000.0), ("phase", "ascent" if burn else "coast")):  # fmt: skip
            frames[key].append(val)
    target = g.surface_point(300_000.0, 0.0, 0.0)
    meta = {"title": "test flight", "vehicle": {"name": "cargo_hopper"}, "mission": {"name": "demo_hop", "cargo_kg": 250},
            "scene": {"frame": "spherical", "earth_radius": g.earth_radius, "target": {"pos": target.tolist()}}}  # fmt: skip
    rep.write_text(json.dumps({"meta": meta, "frames": frames, "events": []}))
    out = tmp_path / "out"
    res = CliRunner().invoke(
        app,
        [
            "safety",
            str(rep),
            "--out",
            str(out),
            "--iip-step",
            "5",
            "--embed",
            str(tmp_path / "e.plume.json"),
        ],
    )
    assert res.exit_code == 0, res.output
    gj = json.loads((out / "safety.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"ground_track", "iip_vacuum", "iip_drag", "hazard_area"} <= kinds
    vac = next(f for f in gj["features"] if f["properties"]["kind"] == "iip_vacuum")["geometry"][
        "coordinates"
    ]
    drag = next(f for f in gj["features"] if f["properties"]["kind"] == "iip_drag")["geometry"][
        "coordinates"
    ]
    assert vac[-1][0] > -104.0 and drag[-1][0] <= vac[-1][0] + 1e-6  # drag shortens the fall
    emb = json.loads((tmp_path / "e.plume.json").read_text())
    assert emb["meta"]["safety"]["iip_vacuum"] and emb["meta"]["safety"]["hazards"]
