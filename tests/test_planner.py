"""Mission planner: 3-DOF feasibility, route missions and the planner endpoints."""

from __future__ import annotations

import json
import time
import warnings

import pytest
from fastapi.testclient import TestClient

from plume.config import load_vehicle
from plume.missions import planner
from plume.missions.planner import cargo_for_range, range_feasibility
from plume.viz.server import STATIC_DIR, create_app

PAD_A = {"lat": 32.990, "lon": -106.986}  # Spaceport America (real_hop)
SITE_B = {"lat": 35.335, "lon": -99.225}  # Burns Flat (real_hop)
CAPE = {"lat": 28.608, "lon": -80.604}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planner")
    app = create_app(replay_dirs=[tmp / "replays"], terrain_dirs=[tmp / "terrain"],
                     planner_dir=tmp / "planner")  # fmt: skip
    return TestClient(app)


def test_planner_vehicles_are_hop_capable():
    names = {v["name"] for v in planner.planner_vehicles()}
    assert "cargo_hopper" in names
    assert "lander_small" not in names and "hobby_rocket" not in names


def test_real_route_matches_the_6dof_reference():
    """761 km, 250 kg: the 6-DOF high-fidelity flight lands with 132 kg of propellant,
    186 km apogee, 589 s, 6.06 g peak cargo load (data/replays/real_hop_high)."""
    out = range_feasibility("cargo_hopper", 250.0, 761_164.0)
    assert out["feasible"], out["reason"]
    r = out["result"]
    assert abs(r["range_km"] - 761.164) < 1.0
    assert 60.0 < r["fuel_margin_kg"] < 220.0
    assert 160.0 < r["apogee_km"] < 210.0
    assert 540.0 < r["flight_time_s"] < 640.0
    assert 5.5 < r["peak_cargo_g"] < 6.3
    assert out["max_range_km"] > 761.0


def test_cargo_for_range_uses_a_monotone_envelope():
    curve = [
        {"cargo_kg": 0.0, "max_range_km": 800.0},
        {"cargo_kg": 100.0, "max_range_km": 790.0},
        {"cargo_kg": 200.0, "max_range_km": 795.0},  # grid wiggle
        {"cargo_kg": 300.0, "max_range_km": 700.0},
    ]
    assert cargo_for_range(curve, 900.0) is None
    assert cargo_for_range(curve, 750.0) == pytest.approx(200.0 + 100.0 * 45.0 / 95.0)
    assert cargo_for_range(curve, 600.0) == 300.0


def test_bundled_capability_table():
    data = json.loads((STATIC_DIR / "planner" / "capability.json").read_text())
    for name, tab in data["vehicles"].items():
        v = load_vehicle(name)
        if tab["hash"] != planner.vehicle_hash(v):
            warnings.warn(
                f"{name}: planner table is stale; run scripts/build_planner_table.py",
                stacklevel=1,
            )
        n_c, n_r = len(tab["cargos_kg"]), len(tab["ranges_km"])
        for field in planner.TABLE_FIELDS:
            assert len(tab["grid"][field]) == n_c
            assert all(len(row) == n_r for row in tab["grid"][field])
        assert len(tab["curve"]) == n_c
        assert all(c["max_range_km"] > 300.0 for c in tab["curve"])


def test_feasibility_endpoint(client):
    t0 = time.perf_counter()
    r = client.post(
        "/api/planner/feasibility",
        json={"launch": PAD_A, "landing": SITE_B, "vehicle": "cargo_hopper", "cargo_kg": 250},
    )
    assert r.status_code == 200
    out = r.json()
    assert time.perf_counter() - t0 < 60.0
    assert out["feasible"] and out["reason"] == "ok"
    assert abs(out["geodesic_km"] - 761.2) < 1.0
    assert abs(out["azimuth_deg"] - 67.9) < 0.5
    assert out["result"]["fuel_margin_kg"] > 0
    assert any("Earth rotation" in n for n in out["notes"])


def test_out_of_range_is_reported_honestly(client):
    far = client.post(
        "/api/planner/feasibility",
        json={"launch": PAD_A, "landing": CAPE, "vehicle": "cargo_hopper", "cargo_kg": 250},
    ).json()
    assert not far["feasible"]
    assert far["geodesic_km"] > 2000 and far["max_range_km"] < far["geodesic_km"]
    assert far["cargo_for_range_kg"] is None  # not even an empty vehicle gets there
    heavy = client.post(
        "/api/planner/feasibility",
        json={"launch": PAD_A, "landing": SITE_B, "vehicle": "cargo_hopper", "cargo_kg": 440},
    ).json()
    assert not heavy["feasible"]
    assert heavy["max_range_km"] < 761.2
    assert 250 <= heavy["cargo_for_range_kg"] < 440


def test_planner_validation_and_pages(client):
    bad = client.post(
        "/api/planner/feasibility",
        json={"launch": PAD_A, "landing": SITE_B, "vehicle": "nope", "cargo_kg": 250},
    )
    assert bad.status_code == 404
    assert (
        client.post("/api/planner/feasibility", json={"launch": {"lat": 95, "lon": 0}}).status_code
        == 422
    )
    page = client.get("/planner")
    assert page.status_code == 200 and "planner.js" in page.text
    assert client.get("/static/planner/world.json").status_code == 200
    sites = client.get("/static/planner/sites.json").json()["sites"]
    assert all(-90 <= s["lat"] <= 90 and -180 <= s["lon"] <= 180 and s["source"] for s in sites)
    curve = client.get("/api/planner/curve", params={"vehicle": "cargo_hopper"}).json()
    assert curve["curve"] and curve["source"] in ("bundled table", "computed")


def test_jobs_run_in_the_background(client, monkeypatch):
    from plume.viz.planner_api import JobManager

    def fake_run(self, job):
        job.result = {"flight": {"replay_id": "x.plume.json.gz", "success": True}}

    monkeypatch.setattr(JobManager, "run", fake_run)
    body = {"launch": PAD_A, "landing": SITE_B, "vehicle": "cargo_hopper", "cargo_kg": 250}
    job = client.post("/api/planner/jobs", json=body | {"kind": "flight"}).json()
    for _ in range(100):
        state = client.get(f"/api/planner/jobs/{job['id']}").json()
        if state["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert state["state"] == "done" and state["result"]["flight"]["success"]
    assert any(j["id"] == job["id"] for j in client.get("/api/planner/jobs").json())
    assert client.get("/api/planner/jobs/nope").status_code == 404
    assert client.get(f"/api/planner/jobs/{job['id']}/files/safety.kml").status_code == 404
    assert client.post("/api/planner/jobs", json=body | {"runs": 1000}).status_code == 422


def test_route_mission_build(tmp_path):
    from plume.config import load_mission
    from plume.missions.hop import MissionWorld, mission_frame
    from plume.missions.route import build_route_mission
    from plume.terrain.dem import geodesic_inverse

    path = build_route_mission(
        "t1", (PAD_A["lat"], PAD_A["lon"]), (SITE_B["lat"], SITE_B["lon"]), "cargo_hopper",
        300.0, tmp_path, tmp_path / "terrain",
    )  # fmt: skip
    spec = load_mission(path)
    assert spec.cargo_mass == 300.0 and spec.vehicle == "cargo_hopper"
    spec2, _world, gravity = mission_frame(spec, "fast")
    mw = MissionWorld(spec2, gravity)
    dist, _ = geodesic_inverse(PAD_A["lat"], PAD_A["lon"], SITE_B["lat"], SITE_B["lon"])
    assert abs(mw.range - dist) / dist < 0.005  # sphere vs ellipsoid
    assert mw.terrain_height(*mw.site_b) == pytest.approx(0.0)
    assert any(t.contains(*mw.site_b) for t in mw.tiles)
