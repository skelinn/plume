"""Propellant slosh (first lateral mode, spring-mass analogy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import WorldSpec, load_vehicle
from plume.physics.sim import RocketSim, quat_from_z_axis
from plume.physics.slosh import XI1, Slosh, slosh_parameters


def test_first_mode_frequency_and_mass():
    # deep tank: omega^2 -> a xi / R (classic f = sqrt(1.841 g / R) / 2 pi)
    m1, w, depth = slosh_parameters(1000.0, 1.0, 5.0, 9.80665)
    assert w == pytest.approx(math.sqrt(9.80665 * XI1 / 1.0), rel=1e-3)
    assert w / (2 * math.pi) == pytest.approx(0.676, abs=0.002)
    assert m1 / 1000.0 == pytest.approx(2.0 / (XI1 * (XI1**2 - 1.0) * 5.0), rel=1e-3)
    assert depth == pytest.approx(1.0 / XI1, rel=1e-3)
    # shallow fill: a larger share of the liquid sloshes, at a lower frequency
    m1s, ws, _ = slosh_parameters(1000.0, 1.0, 0.3, 9.80665)
    assert m1s / 1000.0 > m1 / 1000.0 and ws < w
    assert slosh_parameters(1000.0, 1.0, 1.0, 0.0)[0] == 0.0  # unsettled: no slosh


def test_free_oscillation_matches_natural_frequency():
    v = load_vehicle("cargo_hopper")
    tanks = [t.model_copy(update={"slosh": True, "slosh_damping": 0.0}) for t in v.tanks]
    v = v.model_copy(update={"tanks": tanks})
    sl = Slosh(v)
    masses = np.array([t.capacity * 0.5 for t in v.tanks])
    f = np.array([0.0, 0.0, 9.80665])
    sl.x[0] = [0.05, 0.0]
    dt, xs = 0.001, []
    for _ in range(6000):
        sl.step(dt, masses, f, np.zeros(3), 4.0)
        xs.append(sl.x[0, 0])
    xs = np.array(xs)
    crossings = np.where(np.diff(np.sign(xs)) != 0)[0]
    period = 2 * np.mean(np.diff(crossings)) * dt
    t0 = v.tanks[0]
    h = 0.5 * (t0.z_top - t0.z_bottom)
    _, w, _ = slosh_parameters(masses[0], t0.radius, h, 9.80665)
    assert period == pytest.approx(2 * math.pi / w, rel=0.01)
    assert np.abs(xs).max() == pytest.approx(0.05, rel=0.02)  # undamped: amplitude kept


def test_hover_with_slosh_stays_controlled():
    """Slosh couples into attitude: a hovering vehicle kicked sideways still settles."""
    from plume.control.attitude import AttitudeController

    v = load_vehicle("cargo_hopper")
    tanks = [t.model_copy(update={"slosh": True}) for t in v.tanks]
    v = v.model_copy(update={"tanks": tanks})
    sim = RocketSim(v, WorldSpec(fidelity="high", ground="none", dt=0.005), seed=0)
    sim.reset(pos=(0, 0, 500.0), quat=quat_from_z_axis(np.array([0.05, 0.0, 1.0])), seed=0)
    att = AttitudeController(sim)
    hover = sim.state.mass * 9.80665 / sim.engine.max_thrust(101325.0)
    for _ in range(400):  # 20 s
        st = sim.state
        gim, rcs = att(st, np.array([0.0, 0.0, 1.0]), sim.thrust)
        sim.set_controls(hover, gim, rcs)
        sim.step(10)
    assert np.any(sim.slosh.m1 > 0)  # settled by thrust: slosh active
    assert math.degrees(sim.state.tilt) < 1.0
