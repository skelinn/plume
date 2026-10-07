"""Conservation checks for the 6-DOF simulator: energy, momentum, mass, Tsiolkovsky."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import WindSpec, WorldSpec
from plume.constants import G0
from plume.physics.sim import RocketSim, quat_from_axis_angle
from tests.conftest import make_test_vehicle


def test_free_flight_energy_conserved(test_vehicle, vacuum_world):
    """Gravity only, tumbling body: total mechanical energy is constant."""
    sim = RocketSim(test_vehicle, vacuum_world)
    sim.reset(
        pos=(0, 0, 5000),
        vel=(40.0, -15.0, 120.0),
        quat=quat_from_axis_angle([1, 1, 0], 0.4),
        omega=(0.3, -0.2, 0.5),
    )
    e0 = sim.energy()["total"]
    ke0 = sim.energy()["kinetic"]
    for _ in range(100):
        sim.step(200)  # 100 s total
        e = sim.energy()["total"]
        assert abs(e - e0) / ke0 < 1e-6
    # it actually moved a lot
    assert sim.state.pos[2] < 0


def test_torque_free_angular_momentum_conserved(test_vehicle):
    """Asymmetric tumbling body in zero-g: |L| and the L vector are conserved (Euler's equations)."""
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(test_vehicle, world)
    sim.reset(pos=(0, 0, 0), omega=(0.4, 1.2, 0.3))
    L0 = sim.angular_momentum()
    w0 = sim.state.omega.copy()
    sim.step(4000)  # 20 s
    L1 = sim.angular_momentum()
    assert np.linalg.norm(L1 - L0) / np.linalg.norm(L0) < 1e-6
    # the body rates themselves must change (intermediate-axis tumbling), so this is non-trivial
    assert np.linalg.norm(sim.state.omega - w0) > 0.05


def test_linear_momentum_conserved_without_forces(test_vehicle):
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(test_vehicle, world)
    sim.reset(pos=(0, 0, 0), vel=(3.0, -2.0, 1.0), omega=(0.2, 0.1, 0.05))
    p0 = sim.state.mass * sim.state.vel_com
    sim.step(2000)
    p1 = sim.state.mass * sim.state.vel_com
    np.testing.assert_allclose(p1, p0, rtol=1e-8, atol=1e-8)


def test_spherical_gravity_ballistic_arc_conserves_energy(test_vehicle):
    """A 10-minute suborbital arc under inverse-square gravity."""
    world = WorldSpec(gravity="spherical", atmosphere=False, ground="none")
    sim = RocketSim(test_vehicle, world)
    sim.reset(pos=(0, 0, 1000), vel=(1500.0, 300.0, 1500.0))
    e0 = sim.energy()["total"]
    ke0 = sim.energy()["kinetic"]
    apogee = 0.0
    for _ in range(25):
        sim.step(2000)  # 250 s
        apogee = max(apogee, sim.state.altitude)
        assert abs(sim.energy()["total"] - e0) / ke0 < 1e-6
    assert apogee > 100_000
    assert sim.state.altitude > 10_000  # still on the arc


def test_drag_work_accounts_for_energy_loss(test_vehicle):
    """E(t) - W_aero = E0, and aerodynamic work is never positive with no wind."""
    vehicle = make_test_vehicle(aero={"enabled": True})
    world = WorldSpec(atmosphere=True, ground="none")
    sim = RocketSim(vehicle, world)
    sim.reset(
        pos=(0, 0, 3000),
        vel=(180.0, 0.0, -250.0),
        quat=quat_from_axis_angle([0, 1, 0], 2.5),
        omega=(0.0, 0.3, 0.0),
    )
    e0 = sim.energy()["total"]
    prev_w = 0.0
    for _ in range(40):
        sim.step(100)
        w = sim.work["aero"]
        assert w <= prev_w + 1e-9  # monotonically dissipative
        prev_w = w
    e1 = sim.energy()["total"]
    lost = e0 - e1
    assert lost > 1e5  # drag did real work
    assert abs(lost + sim.work["aero"]) / lost < 2e-3


def test_mass_conservation_and_total_impulse(test_vehicle, vacuum_world):
    sim = RocketSim(test_vehicle, vacuum_world)
    sim.reset(pos=(0, 0, 1000))
    m0 = sim.state.mass
    sim.set_controls(throttle=0.8, gimbal=(0.3, -0.2), rcs=(0.5, 0.0, -0.4))
    sim.step(3000)  # 15 s
    st = sim.state
    assert sim.prop_used > 50
    assert sim.rcs_used > 0
    # mass is exactly accounted for
    assert st.mass + sim.prop_used + sim.rcs_used == pytest.approx(m0, rel=0, abs=1e-9)
    # in vacuum, total impulse = Isp_vac * g0 * propellant used
    isp = test_vehicle.engine.isp_vac
    assert sim.impulse_total == pytest.approx(isp * G0 * sim.prop_used, rel=1e-9)


def test_tsiolkovsky_delta_v():
    """Zero gravity, vacuum, axial thrust: delta-v = Isp g0 ln(m0/m1)."""
    vehicle = make_test_vehicle(rcs={"enabled": False, "propellant": 0.0})
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(vehicle, world)
    sim.reset(pos=(0, 0, 0))
    m0 = sim.state.mass
    sim.set_controls(throttle=1.0)
    sim.step(12000)  # 60 s, ~2/3 of the propellant
    st = sim.state
    m1 = st.mass
    assert m0 / m1 > 1.35
    dv_expected = vehicle.engine.isp_vac * G0 * math.log(m0 / m1)
    dv = float(np.linalg.norm(st.vel_com))
    assert dv == pytest.approx(dv_expected, rel=1e-4)


def test_burn_to_depletion_matches_rocket_equation():
    vehicle = make_test_vehicle(rcs={"enabled": False, "propellant": 0.0})
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(vehicle, world)
    sim.reset(pos=(0, 0, 0))
    m0 = sim.state.mass
    sim.set_controls(throttle=1.0)
    sim.step(20000)  # longer than the ~91 s burn
    assert sim.prop_mass == pytest.approx(0.0, abs=1e-9)
    assert sim.state.thrust == 0.0
    m1 = sim.state.mass
    dv = float(np.linalg.norm(sim.state.vel_com))
    assert dv == pytest.approx(vehicle.engine.isp_vac * G0 * math.log(m0 / m1), rel=1e-4)


def test_work_energy_with_variable_mass():
    """dKE = W_thrust + (1/2) v^2 dm  (exhaust carries away kinetic energy)."""
    vehicle = make_test_vehicle(rcs={"enabled": False, "propellant": 0.0})
    world = WorldSpec(atmosphere=False, ground="none", g=0.0)
    sim = RocketSim(vehicle, world)
    sim.reset(pos=(0, 0, 0), vel=(0, 0, 50.0))
    ke0 = sim.energy()["kinetic"]
    sim.set_controls(throttle=0.7)
    sim.step(3000)
    ke1 = sim.energy()["kinetic"]
    predicted = sim.work["thrust"] + sim.work["mass_loss_ke"]
    assert ke1 - ke0 == pytest.approx(predicted, rel=1e-3)


def test_gravity_free_fall_matches_analytic(test_vehicle, vacuum_world):
    sim = RocketSim(test_vehicle, vacuum_world)
    sim.reset(pos=(0, 0, 1000), vel=(0, 0, 0))
    z0 = sim.state.com[2]
    sim.step(1000)  # 5 s
    st = sim.state
    assert st.com[2] - z0 == pytest.approx(-0.5 * G0 * 25.0, rel=1e-9)
    assert st.vel_com[2] == pytest.approx(-G0 * 5.0, rel=1e-9)


def test_wind_is_deterministic_per_seed(test_vehicle):
    world = WorldSpec(
        atmosphere=True,
        ground="none",
        wind=WindSpec(speed=8.0, turbulence=2.0, gust_rate=0.2, gust_max=6.0),
    )
    vehicle = make_test_vehicle(aero={"enabled": True})

    def run(seed):
        sim = RocketSim(vehicle, world, seed=seed)
        sim.reset(pos=(0, 0, 800), vel=(0, 0, -30), seed=seed)
        sim.step(1000)
        return sim.state.pos

    np.testing.assert_array_equal(run(3), run(3))
    assert np.linalg.norm(run(3) - run(4)) > 1e-3
