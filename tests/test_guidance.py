"""Cargo-hop guidance building blocks."""

from __future__ import annotations

import math

import numpy as np

from plume.config import load_mission, load_vehicle
from plume.missions.hop import MissionWorld, ascent_profile, mission_frame


def test_ascent_profile_is_a_monotone_gravity_turn():
    spec, world, gravity = mission_frame(load_mission("demo_hop"))
    mw = MissionWorld(spec, gravity)
    vehicle = load_vehicle(spec.vehicle).with_cargo(spec.cargo_mass)
    prof = ascent_profile(spec, mw, vehicle, vehicle.cargo.mass, world, 4.0, 17.5)
    assert prof is not None
    speed, gamma = prof
    assert np.all(np.diff(speed) > 0)
    g = np.degrees(gamma)
    assert 60.0 < g[0] < 90.0  # velocity still lagging the kick attitude
    assert 0.0 < g[-1] < g[0]  # pitches over through the burn
    assert speed[-1] > 1500.0 and math.isfinite(float(g.mean()))
