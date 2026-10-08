"""Analytic verification of the high-fidelity Earth and dynamics models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import EarthSpec, WorldSpec
from plume.physics.earth import (
    WGS84_A,
    EarthParams,
    ecef_to_geodetic,
    geodetic_to_ecef,
    zonal_gravity_ecef,
)
from plume.physics.sim import RocketSim
from tests.conftest import make_test_vehicle

pytestmark = pytest.mark.vv


def test_geodetic_roundtrip():
    ep = EarthParams()
    for lat, lon, h in [(0, 0, 0), (45, 120, 1e4), (-89.9, -170, 5e5), (32.99, -106.97, 1401.0)]:
        r = geodetic_to_ecef(math.radians(lat), math.radians(lon), h, ep)
        la, lo, hh = ecef_to_geodetic(r, ep)
        assert math.degrees(la) == pytest.approx(lat, abs=1e-10)
        assert math.degrees(lo) == pytest.approx(lon, abs=1e-10)
        assert hh == pytest.approx(h, abs=1e-6)


def test_gravity_matches_potential_gradient_and_known_values():
    ep = EarthParams()
    r = np.array([4.0e6, 3.0e6, 4.2e6])
    g, pe = zonal_gravity_ecef(r, ep)
    eps = 1.0
    grad = np.array(
        [(zonal_gravity_ecef(r + eps * e, ep)[1] - zonal_gravity_ecef(r - eps * e, ep)[1]) / (2 * eps) for e in np.eye(3)]
    )
    np.testing.assert_allclose(g, -grad, rtol=1e-7)
    # WGS-84 normal gravity at the equator is 9.7803 m/s^2 *including* centrifugal (0.0339)
    g_eq, _ = zonal_gravity_ecef(np.array([WGS84_A, 0.0, 0.0]), ep)
    assert np.linalg.norm(g_eq) - WGS84_A * ep.omega**2 == pytest.approx(9.7803, abs=5e-4)
    # J2 makes polar gravity stronger than equatorial
    g_pole, _ = zonal_gravity_ecef(np.array([0.0, 0.0, ep.b]), ep)
    assert np.linalg.norm(g_pole) > np.linalg.norm(g_eq)


def _world(rotating=True, zonal=6, shape="wgs84"):
    return WorldSpec(
        fidelity="high",
        gravity="wgs84",
        atmosphere=False,
        ground="none",
        dt=0.01,
        earth=EarthSpec(rotating=rotating, zonal_degree=zonal, shape=shape, origin_lat_deg=28.5, origin_lon_deg=-80.6),
    )


def test_point_mass_kepler_orbit_closes():
    """Non-rotating, point-mass Earth: a circular orbit returns to its start after one period."""
    vehicle = make_test_vehicle(rcs={"enabled": False, "propellant": 0.0})
    sim = RocketSim(vehicle, _world(rotating=False, zonal=0))
    g = sim.gravity
    p0 = g.world_point(math.radians(28.5), math.radians(-80.6), 400e3)
    r = p0 - g.center_w
    rn = float(np.linalg.norm(r))
    v_circ = math.sqrt(g.params.gm / rn)
    east = g.enu_at(p0)[:, 0]
    sim.reset(pos=p0, vel=v_circ * east)
    period = 2 * math.pi * math.sqrt(rn**3 / g.params.gm)
    n = round(period / sim.dt)
    sim.step(n)
    err = float(np.linalg.norm(sim.state.com - p0))
    assert err < 50.0  # metres after ~5,550 s / 40,000 km travelled (dt = 10 ms)


def test_rotating_frame_jacobi_energy_conserved():
    """Vacuum, rotating J6 Earth: the Jacobi integral (KE + gravity + centrifugal) is constant."""
    vehicle = make_test_vehicle(rcs={"enabled": False, "propellant": 0.0})
    sim = RocketSim(vehicle, _world())
    p0 = sim.gravity.world_point(math.radians(28.5), math.radians(-80.6), 80e3)
    sim.reset(pos=p0, vel=(1500.0, 800.0, 900.0), omega=(0.2, 0.1, 0.05))
    e0 = sim.energy()["total"]
    ke0 = sim.energy()["kinetic"]
    for _ in range(20):
        sim.step(1000)  # 200 s
        assert abs(sim.energy()["total"] - e0) / ke0 < 1e-6


def test_convergence_with_timestep():
    """High fidelity evaluates every force at each RK4 stage: a tumbling, dragged
    trajectory converges at second order or better and the default step (5 ms) is
    within a centimetre of the converged answer after 30 s."""
    vehicle = make_test_vehicle(aero={"enabled": True}, rcs={"enabled": False, "propellant": 0.0})

    def final(dt):
        w = _world().model_copy(update={"dt": dt, "atmosphere": True})
        sim = RocketSim(vehicle, w)
        p0 = sim.gravity.world_point(math.radians(28.5), math.radians(-80.6), 5000.0)
        sim.reset(pos=p0, vel=(150.0, 40.0, 120.0))
        sim.step(round(30.0 / dt))
        return sim.state.com

    ref = final(0.00125)
    errs = [float(np.linalg.norm(final(dt) - ref)) for dt in (0.02, 0.01, 0.005)]
    assert errs[1] < 0.35 * errs[0] and errs[2] < 0.35 * errs[1]  # ~4x per halving
    assert errs[2] < 0.01
