"""Cargo-hop missions: terrain tiles, targeting, scoring and the full demo flight."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import WorldSpec, load_mission, load_vehicle
from plume.missions.hop import MissionWorld, run_mission
from plume.missions.targeting import kepler_impact
from plume.physics.gravity import SphericalGravity
from plume.physics.mjcf import GroundTile
from plume.physics.pointmass import PointMassSim
from plume.physics.sim import RocketSim


@pytest.fixture(scope="module")
def mission_world():
    spec = load_mission("demo_hop")
    return spec, MissionWorld(spec, SphericalGravity())


def test_mission_geometry(mission_world):
    _, mw = mission_world
    assert mw.range == pytest.approx(749_500, rel=1e-3)
    # pads sit on the terrain
    for site, pad in ((mw.site_a, mw.pad_a), (mw.site_b, mw.pad_b)):
        assert mw.gravity.altitude(pad) == pytest.approx(mw.terrain_height(*site), abs=1e-6)
    along, cross = mw.along_cross(mw.pad_b)
    assert along == pytest.approx(mw.range, rel=1e-6)
    assert abs(cross) < 1e-3
    # local frame is orthonormal and right-handed with z = up
    R = mw.local_frame(*mw.site_b)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(R[:, 2], mw.gravity.up(mw.pad_b), atol=1e-9)


def test_hfield_orientation_matches_heightmap(lander):
    """A sloped tile: the vehicle rests at the heightmap's height (row 0 = -y, col 0 = -x)."""
    n, half = 81, 40.0
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    heights = 0.05 * i * (2 * half / (n - 1)) + 0.02 * j * (2 * half / (n - 1))
    tile = GroundTile("t", heights, origin=np.zeros(3), half_x=half, half_y=half)
    for px, py in [(20.0, -20.0), (-20.0, 20.0)]:
        expected = 0.05 * (px + half) + 0.02 * (py + half)
        sim = RocketSim(lander, WorldSpec(ground="none"), tiles=[tile])
        sim.reset(pos=(px, py, expected + lander.legs.height + 0.5))
        sim.step(800)
        base = sim.state.pos[2] - lander.legs.height
        assert base == pytest.approx(expected, abs=0.6)


def test_kepler_impact_matches_numerical_ballistic(mission_world):
    _, mw = mission_world
    g = mw.gravity
    vehicle = load_vehicle("cargo_hopper")
    pm = PointMassSim(vehicle, WorldSpec(gravity="spherical", atmosphere=False), prop_mass=0.0)
    r0 = g.surface_point(0, 0, 80_000.0)
    v0 = np.array([1700.0, 300.0, 1500.0])
    traj = pm.run(r0, v0, t_end=2000, dt=0.05, ground_altitude=0.0, record_every=1000)
    imp = kepler_impact(r0, v0, g, g.earth_radius)
    assert np.linalg.norm(traj.pos[-1] - imp) < 200.0  # ~0.05% of the ~400 km range


def test_spherical_energy_over_hop_arc():
    vehicle = load_vehicle("cargo_hopper")
    sim = RocketSim(
        vehicle, WorldSpec(gravity="spherical", atmosphere=False, ground="none", dt=0.01)
    )
    sim.reset(pos=(0, 0, 70_000), vel=(1800.0, 400.0, 1700.0))
    e0 = sim.energy()["total"]
    ke0 = sim.energy()["kinetic"]
    for _ in range(30):
        sim.step(1000)
        assert abs(sim.energy()["total"] - e0) / ke0 < 1e-6


def test_scoring_and_cargo_sweep(mission_world):
    """Heavier cargo burns more fuel and still flies (sweep through the CLI-equivalent API)."""
    spec, _ = mission_world
    light = run_mission(spec, seed=1, cargo_mass=100.0)
    heavy = run_mission(spec, seed=1, cargo_mass=250.0)
    for run in (light, heavy):
        r = run.result
        assert math.isfinite(r.score)
        assert r.apogee_km > 80
        assert 0 < r.max_cargo_g < 15

    # more cargo -> more ascent propellant spent reaching the same range
    def ascent_fuel(run):
        rec = run.recorder.to_dict()
        t = np.asarray(rec["frames"]["t"])
        prop = np.asarray(rec["frames"]["prop_mass"])
        meco = next(e["t"] for e in rec["events"] if e["type"] == "cutoff")
        return prop[0] - prop[np.searchsorted(t, meco)]

    assert ascent_fuel(heavy) > ascent_fuel(light)


@pytest.mark.slow
def test_demo_hop_lands_on_target():
    run = run_mission("demo_hop", seed=0)
    r = run.result
    assert r.success, r
    assert r.landing_error_m < 50.0
    assert r.max_cargo_g < 6.5
    assert r.fuel_remaining_kg > 0
    assert 400 < r.flight_time_s < 900
    rec = run.recorder.to_dict()
    assert rec["meta"]["scene"]["frame"] == "spherical"
    labels = [e["label"] for e in rec["events"]]
    for needed in ("MECO", "Entry burn", "Landing burn", "Touchdown"):
        assert any(needed in label for label in labels), labels
