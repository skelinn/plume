"""Grid fins: physics, control allocation, closed-loop use and the mission payoff."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import VehicleSpec, WorldSpec, load_vehicle
from plume.control.attitude import AttitudeController
from plume.physics.sim import RocketSim, quat_from_axis_angle, quat_from_z_axis


@pytest.fixture(scope="module")
def hopper():
    return load_vehicle("cargo_hopper")


def deployed(sim):
    sim.grid_fins.deploy(True)
    return sim


def test_stowed_fins_do_nothing_until_deployed(hopper):
    sim = RocketSim(hopper, WorldSpec(ground="none"))
    sim.reset(pos=(0, 0, 3000), vel=(0, 0, -150))
    gf = sim.grid_fins
    assert not gf.deployed  # deploy: on_command
    f, tau = gf.forces(np.array([5.0, 0, -150.0]), np.zeros(3), 5.0, 1.0)
    assert not f.any() and not tau.any()
    gf.deploy(True)
    f, _ = gf.forces(np.array([5.0, 0, -150.0]), np.zeros(3), 5.0, 1.0)
    assert f[2] > 0  # lattice drag opposes the (downward) motion


def test_passive_fins_only_dissipate(hopper):
    """Undeflected fins oppose the local crossflow, so aerodynamics still only remove energy."""
    world = WorldSpec(ground="none", g=0.0)
    sim = deployed(RocketSim(hopper, world))
    sim.reset(
        pos=(0, 0, 4000),
        vel=(40.0, 0, -200.0),
        quat=quat_from_axis_angle([0, 1, 0], 2.8),
        omega=(0, 0.3, 0),
    )
    sim.grid_fins.deploy(True)
    prev = 0.0
    for _ in range(30):
        sim.step(40)
        assert sim.work["aero"] <= prev + 1e-6
        prev = sim.work["aero"]
    assert prev < 0


def test_control_sign_flips_with_flow_direction(hopper):
    sim = deployed(RocketSim(hopper, WorldSpec(ground="none")))
    gf = sim.grid_fins
    B_tail = gf.torque_matrix(np.array([0, 0, -150.0]), 5.0, 1.0)  # engine-first descent
    B_nose = gf.torque_matrix(np.array([0, 0, 150.0]), 5.0, 1.0)  # nose-first
    np.testing.assert_allclose(B_tail, -B_nose)
    assert np.linalg.matrix_rank(B_tail) == 3  # pitch, yaw and roll all controllable


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_allocation_produces_requested_torque(hopper, axis):
    sim = deployed(RocketSim(hopper, WorldSpec(ground="none")))
    gf = sim.grid_fins
    v_b = np.array([0, 0, -150.0])
    cap = gf.capability(v_b, 5.0, 1.0)
    want = np.zeros(3)
    want[axis] = 0.3 * cap[axis]
    cmd, achieved = gf.allocate(want, v_b, 5.0, 1.0)
    assert np.all(np.abs(cmd) <= 1.0)
    np.testing.assert_allclose(achieved, want, atol=1e-6 * cap.max())


def test_fins_hold_a_tilt_without_rcs(hopper):
    """Closed loop: with the RCS disabled, the fins alone hold a 5 deg tilt in a descent."""
    data = hopper.model_dump()
    data["rcs"]["enabled"] = False
    v = VehicleSpec.model_validate(data)
    sim = deployed(RocketSim(v, WorldSpec(ground="none")))
    th = math.radians(5)
    axis = np.array([math.sin(th), 0, math.cos(th)])
    sim.reset(pos=(0, 0, 5000), vel=(0, 0, -160), quat=quat_from_z_axis([0, 0, 1]), prop=300)
    sim.grid_fins.deploy(True)
    att = AttitudeController(sim)
    for _ in range(80):  # 4 s
        gimbal, rcs = att(sim.state, axis, 0.0)
        sim.set_controls(0.0, gimbal, rcs)
        sim.step(10)
    held = math.degrees(math.atan2(sim.state.axis[0], sim.state.axis[2]))
    assert held == pytest.approx(5.0, abs=1.0)
    assert np.abs(sim.grid_fins.delta).max() > math.radians(1)


def test_replay_carries_fin_state(hopper):
    sim = RocketSim(hopper, WorldSpec(ground="none"))
    sim.reset(pos=(0, 0, 1000))
    f = sim.frame()
    assert f["fins_out"] == 0.0 and len(f["fins"]) == 4
    assert sim.replay_meta("x")["vehicle"]["grid_fins"]["count"] == 4


@pytest.mark.slow
def test_grid_fins_improve_the_hop():
    from plume.config import load_mission
    from plume.missions.hop import run_mission

    spec = load_mission("demo_hop")
    with_fins = run_mission(spec, seed=0).result
    spec.vehicle = "cargo_hopper_finless"
    without = run_mission(spec, seed=0).result
    assert with_fins.success
    assert with_fins.fuel_remaining_kg > without.fuel_remaining_kg + 50
    assert with_fins.landing_error_m < 30
