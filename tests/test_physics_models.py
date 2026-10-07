"""Unit tests for the individual physics models."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from plume.config import WindSpec, WorldSpec
from plume.constants import G0, P0
from plume.physics.atmosphere import Atmosphere
from plume.physics.gravity import FlatGravity, SphericalGravity
from plume.physics.massprops import MassModel
from plume.physics.pointmass import PointMassSim
from plume.physics.propulsion import RCS, Engine
from plume.physics.sim import RocketSim
from plume.physics.wind import WindModel
from tests.conftest import make_test_vehicle

# --------------------------------------------------------------------------- atmosphere


@pytest.mark.parametrize(
    ("alt", "T", "P", "rho"),
    [
        (0.0, 288.15, 101_325.0, 1.2250),
        (5_000.0, 255.68, 54_048.0, 0.73643),
        (11_000.0, 216.77, 22_700.0, 0.36480),
        (20_000.0, 216.65, 5_529.3, 0.088910),
        (32_000.0, 228.49, 889.06, 0.013555),
        (50_000.0, 270.65, 79.779, 0.0010269),
        (80_000.0, 198.64, 1.0524, 1.8458e-5),
    ],
)
def test_us_standard_atmosphere(alt, T, P, rho):
    """Reference values from the 1976 U.S. Standard Atmosphere tables (geometric altitude)."""
    s = Atmosphere().at(alt)
    assert s.temperature == pytest.approx(T, rel=2e-3)
    assert s.pressure == pytest.approx(P, rel=5e-3)
    assert s.density == pytest.approx(rho, rel=5e-3)


def test_atmosphere_monotonic_and_tail():
    atm = Atmosphere()
    alts = np.linspace(0, 200_000, 400)
    rho = np.array([atm.density(a) for a in alts])
    assert np.all(np.diff(rho) < 0)
    assert 0 < atm.density(150_000) < 1e-8
    assert Atmosphere(enabled=False).density(0) == 0.0


# --------------------------------------------------------------------------- gravity


def test_gravity_models():
    flat = FlatGravity()
    np.testing.assert_allclose(flat.accel(np.array([1.0, 2.0, 3.0])), [0, 0, -G0])
    sph = SphericalGravity()
    assert np.linalg.norm(sph.accel(np.zeros(3))) == pytest.approx(G0)
    # potential gradient equals acceleration
    p = np.array([2.0e5, -1.0e5, 3.0e4])
    eps = 1.0
    grad = np.array(
        [(sph.potential(p + eps * e) - sph.potential(p - eps * e)) / (2 * eps) for e in np.eye(3)]
    )
    np.testing.assert_allclose(-grad, sph.accel(p), rtol=1e-6)


def test_spherical_map_roundtrip():
    sph = SphericalGravity()
    for u, v, h in [(0, 0, 0), (750e3, 40e3, 1200), (-3e5, 2e5, 50)]:
        p = sph.surface_point(u, v, h)
        uu, vv, hh = sph.map_coords(p)
        assert (uu, vv, hh) == pytest.approx((u, v, h), abs=1e-4)
        assert sph.altitude(p) == pytest.approx(h, abs=1e-6)
    # 750 km downrange the surface has dropped ~44 km below the launch tangent plane
    assert sph.surface_point(750e3, 0, 0)[2] == pytest.approx(-44_100, rel=0.01)


# --------------------------------------------------------------------------- mass properties


def test_mass_properties_match_brute_force():
    v = make_test_vehicle(cargo={"mass": 120.0, "cg_z": 6.0})
    mm = MassModel(v)
    tanks = np.array([350.0])
    mp = mm.evaluate(tanks, rcs_mass=7.0)
    assert mp.mass == pytest.approx(800 + 120 + 350 + 7)
    # brute force: CG from discrete components
    t = v.tanks[0]
    h = (t.z_top - t.z_bottom) * 350 / t.capacity
    zs = {800: 3.0, 120: 6.0, 350: t.z_bottom + h / 2, 7: v.rcs.z}
    cg = sum(m * z for m, z in zs.items()) / mp.mass
    assert mp.cg_z == pytest.approx(cg)
    assert mp.inertia[0] > v.mass.dry_inertia[0]


def test_cg_drops_as_tanks_drain(lander):
    mm = MassModel(lander)
    cap = mm.tank_capacity
    cgs = [mm.evaluate(cap * f).cg_z for f in (1.0, 0.75, 0.5)]
    # the upper part of the tank drains first, pulling the CG down
    assert all(a > b for a, b in itertools.pairwise(cgs))
    # once the liquid level is below the dry CG the vehicle CG rises again
    assert mm.evaluate(cap * 0.0).cg_z > mm.evaluate(cap * 0.25).cg_z


# --------------------------------------------------------------------------- propulsion


def test_engine_isp_and_throttle_limits(lander):
    e = Engine(lander.engine, lander.prop_capacity)
    assert e.isp_at(0.0) == pytest.approx(lander.engine.isp_vac)
    assert e.isp_at(P0) == pytest.approx(lander.engine.isp_sl)
    e.command(0.25)  # below min throttle but above off threshold -> clamps to min
    for _ in range(400):
        thrust, mdot = e.update(0.005, 0.0, 100.0)
    assert e.throttle == pytest.approx(lander.engine.throttle_min, rel=1e-3)
    assert thrust / mdot == pytest.approx(lander.engine.isp_vac * G0, rel=1e-9)
    e.command(0.1)  # below half of min -> off
    for _ in range(400):
        thrust, _ = e.update(0.005, 0.0, 100.0)
    assert thrust == 0.0 and not e.on


def test_engine_ignition_limit(lander):
    spec = lander.engine.model_copy(update={"max_ignitions": 1, "throttle_tau": 0.0})
    e = Engine(spec, lander.prop_capacity)
    e.command(1.0)
    assert e.update(0.01, 0.0, 100.0)[0] > 0
    e.command(0.0)
    e.update(0.01, 0.0, 100.0)
    e.command(1.0)
    assert e.update(0.01, 0.0, 100.0)[0] == 0.0  # no relight left


def test_engine_flameout_when_dry(lander):
    e = Engine(lander.engine, lander.prop_capacity)
    e.command(1.0)
    thrust, mdot = e.update(0.01, 0.0, 0.0)
    assert thrust == 0.0 and mdot == 0.0


def test_gimbal_rate_and_angle_limits(lander):
    e = Engine(lander.engine, lander.prop_capacity)
    e.command(1.0, (1.0, 1.0))
    e.update(0.01, 0.0, 100.0)
    rate = math.radians(lander.engine.gimbal_rate_deg_s)
    assert np.all(np.abs(e.gimbal) <= rate * 0.01 + 1e-12)
    for _ in range(500):
        e.update(0.01, 0.0, 100.0)
    assert np.hypot(*e.gimbal) == pytest.approx(math.radians(lander.engine.gimbal_max_deg))


def test_gimbal_torque_signs(test_vehicle):
    """Thrust direction d = (cos a sin b, -sin a, cos a cos b) applied below the CG."""
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(test_vehicle, world)
    sim.reset(pos=(0, 0, 0))
    sim.set_controls(throttle=1.0, gimbal=(1.0, 0.0))
    sim.step(100)
    st = sim.state
    assert st.omega[0] < -1e-3  # a > 0 pitches about -x
    assert abs(st.omega[1]) < 1e-9
    sim.reset(pos=(0, 0, 0))
    sim.set_controls(throttle=1.0, gimbal=(0.0, 1.0))
    sim.step(100)
    assert sim.state.omega[1] < -1e-3  # b > 0 pitches about -y


def test_rcs_allocation_produces_requested_axis(test_vehicle):
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    for axis in range(3):
        cmd = np.zeros(3)
        cmd[axis] = 1.0
        sim = RocketSim(test_vehicle, world)
        sim.reset(pos=(0, 0, 0))
        sim.set_controls(rcs=cmd)
        sim.step(100)
        w = sim.state.omega
        assert w[axis] > 1e-3
        others = [w[i] for i in range(3) if i != axis]
        assert max(abs(o) for o in others) < 1e-6 * max(1.0, abs(w[axis]) * 1e3)


def test_rcs_capability_positive(lander):
    rcs = RCS(lander.rcs, lander, 3.5)
    assert np.all(rcs.torque_cap > 0)


# --------------------------------------------------------------------------- wind


def test_wind_profile_and_direction():
    w = WindModel(speed=10.0, from_deg=270.0)  # from the west -> blows east (+x)
    v = w.at(10.0)
    assert v[0] == pytest.approx(10.0)
    assert abs(v[1]) < 1e-9
    assert np.linalg.norm(w.at(1000.0)) > 10.0  # shear
    assert np.linalg.norm(w.at(40_000.0)) == 0.0


def test_wind_spec_roundtrip():
    spec = WindSpec(speed=5.0, turbulence=1.0)
    m = WindModel.from_spec(spec, seed=1)
    assert m.enabled


# --------------------------------------------------------------------------- contact


def test_lander_rests_on_legs(lander):
    sim = RocketSim(lander, WorldSpec())
    sim.reset(pos=(0, 0, lander.legs.height + 0.05))
    sim.step(600)
    st = sim.state
    assert st.legs_down == lander.legs.count
    assert not st.body_contact
    assert st.pos[2] == pytest.approx(lander.legs.height, abs=0.01)
    assert st.g_load == pytest.approx(1.0, abs=0.02)
    assert sim.touchdown is not None and sim.touchdown.vertical_speed < 1.5


def test_hard_impact_reports_touchdown_speed(lander):
    sim = RocketSim(lander, WorldSpec())
    sim.reset(pos=(0, 0, 30.0), vel=(0, 0, -20.0))
    sim.step(800)
    td = sim.touchdown
    assert td is not None
    assert td.vertical_speed > 20.0


# --------------------------------------------------------------------------- 3-DOF vs 6-DOF


def test_pointmass_agrees_with_6dof_vertical_flight():
    vehicle = make_test_vehicle(aero={"enabled": True}, rcs={"enabled": False, "propellant": 0.0})
    world = WorldSpec(ground="none")
    sim = RocketSim(vehicle, world)
    sim.reset(pos=(0, 0, 0))
    sim.set_controls(throttle=1.0)
    sim.step(4000)  # 20 s
    st6 = sim.state

    pm = PointMassSim(vehicle, world)
    traj = pm.run(
        r0=[0, 0, vehicle.mass.dry_cg_z],
        v0=[0, 0, 0],
        t_end=20.0,
        dt=0.005,
        direction=lambda t, r, v: np.array([0.0, 0.0, 1.0]),
        stop_on_ground=False,
    )
    z3 = traj.pos[-1, 2] - vehicle.mass.dry_cg_z
    z6 = st6.com[2] - sim.mass_model.evaluate(sim.tank_init, 0.0).cg_z
    assert z3 == pytest.approx(z6, rel=2e-3)
    assert traj.vel[-1, 2] == pytest.approx(st6.vel_com[2], rel=2e-3)
    assert traj.mass[-1] == pytest.approx(st6.mass, rel=1e-6)
