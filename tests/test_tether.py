"""Tether (tension-only rope) force model: slack/taut, damping sign, torque, energy."""

from __future__ import annotations

import numpy as np
import pytest

from plume.config import WorldSpec
from plume.physics.sim import RocketSim, quat_from_axis_angle
from plume.physics.tether import Tether, TetherSpec
from tests.conftest import make_test_vehicle

G = 9.80665


def _sim(fidelity="fast", **kw):
    v = make_test_vehicle(engine={"gimbal_max_deg": 0.0}, rcs={"enabled": False}, **kw)
    world = WorldSpec(atmosphere=False, ground="none", fidelity=fidelity)
    return RocketSim(v, world)


def _eval(tether, sim):
    st = sim.state
    return tether(sim, st.vel_com, None, st.rot, st.omega, sim.mp)


def test_slack_rope_exerts_nothing():
    sim = _sim()
    sim.reset(pos=(0.0, 0.0, 5.0), vel=(0.0, 0.0, -3.0))
    t = Tether(anchor=(0.0, 0.0, 0.0), attach=(0.0, 0.0, 0.0), length=6.0)
    f, tau = _eval(t, sim)
    assert not f.any() and not tau.any()
    assert t.tension == 0.0 and t.stretch < -1.0


def test_taut_rope_pulls_toward_anchor_with_k_times_stretch():
    sim = _sim()
    sim.reset(pos=(3.0, 0.0, 4.0))  # attach point (hull base) is 5 m from the anchor
    t = Tether(anchor=(0.0, 0.0, 0.0), attach=(0.0, 0.0, 0.0), length=4.8, stiffness=1e4)
    f, tau = _eval(t, sim)
    assert t.tension == pytest.approx(1e4 * 0.2, rel=1e-9)
    assert np.allclose(f, -2000.0 * np.array([0.6, 0.0, 0.8]))
    # torque about the CG (body z = cg_z) of a force at the hull base
    r = np.array([0.0, 0.0, -sim.mp.cg_z])
    assert np.allclose(tau, np.cross(r, f))  # body = world axes here (upright)
    assert tau[1] > 0  # pulling the base toward -x pitches the nose toward +x


def test_damping_only_while_taut_and_never_pushes():
    t = Tether(length=1.0, stiffness=1e4, damping=1e4)
    assert t.tension_for(-0.1, 5.0) == 0.0  # slack: nothing, whatever the rate
    assert t.tension_for(0.1, 0.5) == pytest.approx(1e4 * 0.1 + 1e4 * 0.5)
    assert t.tension_for(0.1, -0.5) == 0.0  # recoiling faster than the rope: no push


def test_rotated_attach_point_and_point_velocity():
    sim = _sim("high")  # no mid-step extrapolation: exact point kinematics
    q = quat_from_axis_angle([0.0, 1.0, 0.0], np.radians(90.0))  # nose toward +x
    sim.reset(pos=(0.0, 0.0, 0.0), quat=q, omega=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    t = Tether(anchor=(0.0, 0.0, 0.0), attach=(0.0, 0.0, 8.0), length=7.0, stiffness=1e3)
    f, _ = _eval(t, sim)
    assert np.allclose(t.attach_point(sim), [8.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(f, [-1e3 * 1.0 - t.spec.damping * 1.0, 0.0, 0.0])


def test_spec_validation_and_critical_damping():
    with pytest.raises(ValueError):
        TetherSpec(length=0.0)
    assert TetherSpec.critical_damping(1e4, 100.0) == pytest.approx(2000.0)


def _drop_on_rope(fidelity: str, damping: float, t_end: float = 8.0):
    """Hang the vehicle from a crane hook by its nose and drop it 1 m onto the rope."""
    sim = _sim(fidelity)
    nose = 8.0
    hook = np.array([0.0, 0.0, 30.0])
    length = 5.0
    # nose starts 1 m above the point where the rope goes taut
    sim.reset(pos=(0.0, 0.0, hook[2] - length + 1.0 - nose))
    t = Tether(
        anchor=tuple(hook), attach=(0.0, 0.0, nose), length=length, stiffness=5e4, damping=damping
    ).install(sim)
    energies = []
    while sim.t < t_end:
        sim.step(4)
        energies.append(sim.energy()["total"] + t.elastic_energy(sim))
    return sim, t, np.array(energies)


def test_undamped_rope_conserves_energy_high_fidelity():
    sim, t, e = _drop_on_rope("high", damping=0.0)
    assert t.max_tension > 0  # it bounced on the rope
    scale = sim.mp.mass * G * 1.0  # the 1 m drop
    assert np.ptp(e) / scale < 2e-3


def test_undamped_rope_energy_drift_small_fast_fidelity():
    # forces are held over a step in fast fidelity: only first-order accurate
    sim, _, e = _drop_on_rope("fast", damping=0.0)
    assert np.ptp(e) / (sim.mp.mass * G) < 0.05


def test_damped_rope_dissipates_and_settles_at_static_stretch():
    sim, t, e = _drop_on_rope(
        "high", damping=TetherSpec.critical_damping(5e4, 1400.0, 0.3), t_end=12.0
    )
    assert np.all(np.diff(e) < 1e-6 * abs(e[0]) + 1e-6)  # never gains energy
    assert e[-1] < e[0] - 0.5 * sim.mp.mass * G
    assert t.stretch == pytest.approx(sim.mp.mass * G / 5e4, rel=0.02)
