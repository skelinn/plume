"""Drone ship: sea-state spectrum, deck motion statistics, kinematics, ship landing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import SeaStateSpec, ShipSpec, WorldSpec, load_vehicle
from plume.constants import G0
from plume.physics.ship import DeckLink, ShipModel, barge_heave_pitch, jonswap
from plume.physics.sim import RocketSim


@pytest.mark.parametrize("gamma", [1.0, 3.3])
def test_spectrum_zeroth_moment_gives_hs(gamma):
    hs, tp = 3.0, 10.0
    w = np.linspace(0.05, 6.0, 20000)
    m0 = np.trapezoid(jonswap(w, hs, tp, gamma), w)
    assert 4 * math.sqrt(m0) == pytest.approx(hs, rel=0.02)
    wp = w[np.argmax(jonswap(w, hs, tp, gamma))]
    assert wp == pytest.approx(2 * math.pi / tp, rel=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthesised_sea_significant_wave_height(seed):
    """4 x the standard deviation of a one-hour elevation record = Hs (within the
    sampling scatter of a finite record)."""
    sea = SeaStateSpec(hs=2.5, tp=9.0, spreading_s=4.0, directions=5)
    ship = ShipModel(ShipSpec(sea=sea), seed=seed)
    assert ship.sea.hs_spectral == pytest.approx(2.5, rel=0.03)
    t = np.arange(0.0, 3600.0, 0.5)
    eta = np.array([ship.sea.elevation(0.0, 0.0, ti) for ti in t])
    assert 4 * eta.std() == pytest.approx(2.5, rel=0.05)
    assert abs(eta.mean()) < 0.1


def test_barge_transfer_functions_long_wave_limit():
    w = np.array([0.05])
    heave, pitch = barge_heave_pitch(w, math.pi, 91.0, 30.0, 4.0)
    k = w[0] ** 2 / G0
    assert heave[0] == pytest.approx(1.0, abs=0.01)  # follows the surface
    assert pitch[0] == pytest.approx(k, rel=0.02)  # follows the slope
    # short waves average out over the hull
    h2, p2 = barge_heave_pitch(np.array([1.5]), math.pi, 91.0, 30.0, 4.0)
    assert abs(h2[0]) < 0.05 and abs(p2[0]) < 0.05 * (1.5**2 / G0)


def test_deck_heave_statistics_match_transfer_function():
    """The heave record's variance equals the spectral prediction sum(a^2 H^2) / 2."""
    ship = ShipModel(ShipSpec(sea=SeaStateSpec(hs=2.0, tp=8.0, direction_deg=180.0)), seed=4)
    t = np.arange(0.0, 3600.0, 1.0)
    heave = np.array([ship._channels(ti)[0][2] for ti in t])
    pitch = np.array([ship._channels(ti)[0][4] for ti in t])
    assert heave.std() == pytest.approx(math.sqrt(0.5 * np.sum(ship.h_amp**2)), rel=0.12)
    assert pitch.std() == pytest.approx(math.sqrt(0.5 * np.sum(ship.p_amp**2)), rel=0.12)
    # head seas: no roll, no sway
    roll = np.array([ship._channels(ti)[0][3] for ti in t])
    assert np.abs(roll).max() < 1e-9


def test_ship_velocity_is_derivative_of_pose():
    ship = ShipModel(ShipSpec(), seed=2)
    h = 1e-5
    for t in (10.0, 123.4):
        p0, q0, v0, w0 = ship.motion(t)
        p1, q1, _, _ = ship.motion(t + h)
        np.testing.assert_allclose((p1 - p0) / h, v0, atol=1e-3)
        # body rate from the quaternion derivative: omega = 2 q* dq/dt
        dq = (q1 - q0) / h
        w, x, y, z = q0
        conj = np.array([w, -x, -y, -z])
        a, b = conj, dq
        prod = np.array(
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
            ]
        )
        np.testing.assert_allclose(2 * prod[1:], w0, atol=1e-4)


def test_deck_link_latency_and_noise():
    ship = ShipModel(ShipSpec(), seed=1)
    link = DeckLink(ship, seed=0)
    errs = []
    for t in np.arange(1.0, 60.0, 0.05):
        ship._cache(t)
        p_true, _ = ship.target()
        p, _ = link.estimate(t)
        errs.append(np.linalg.norm(p - p_true))
    # latency-compensated: errors of the order of the noise and the extrapolation
    assert np.median(errs) < 0.3


def test_deck_carries_the_vehicle_and_clamp_holds_it():
    """A lander set down on the moving deck rides with it (friction) and stays upright;
    the hold-down clamp then welds it to the deck."""
    spec = ShipSpec(sea=SeaStateSpec(hs=3.0, tp=8.0, direction_deg=120.0))
    ship = ShipModel(spec, seed=5)
    v = load_vehicle("lander_small")
    sim = RocketSim(v, WorldSpec(ground="none", dt=0.005), ship=ship)
    p, _ = ship.target()
    up = ship.deck_normal()
    # set down on the rolling deck (roll up to ~11 deg in these quartering seas)
    sim.reset(pos=p + up * (v.legs.height + 0.05), quat=ship.quat, prop=200.0)
    sim.step(int(20.0 / sim.dt))
    st = sim.state
    rel = st.vel_com - ship.point_velocity(st.com)
    assert np.linalg.norm(rel) < 0.2
    assert ship.on_deck(st.pos)
    assert math.degrees(math.acos(st.axis @ ship.deck_normal())) < 3.0
    assert sim.gear.min_margin > 5.0  # ~21 deg on level ground, less the deck roll
    lp0 = ship.R.T @ (st.pos - ship.pos)
    sim.clamp_to_deck()
    sim.step(int(10.0 / sim.dt))
    lp1 = ship.R.T @ (sim.state.pos - ship.pos)
    assert np.linalg.norm(lp1 - lp0) < 0.02


def test_touchdown_speed_is_relative_to_the_deck():
    spec = ShipSpec(sea=SeaStateSpec(hs=3.0, tp=8.0, direction_deg=180.0))
    ship = ShipModel(spec, seed=3)
    v = load_vehicle("lander_small")
    sim = RocketSim(v, WorldSpec(ground="none", dt=0.005), ship=ship)
    p, vd = ship.target()
    up = ship.deck_normal()
    # a vehicle moving with the deck, falling 1 m/s relative to it
    sim.reset(pos=p + up * (v.legs.height + 0.03), quat=ship.quat, vel=vd - up * 1.0, prop=200.0)
    sim.step(40)
    td = sim.touchdown
    assert td is not None
    # 1 m/s plus the 3 cm fall (the deck heave acceleration is small over 30 ms)
    assert td.vertical_speed == pytest.approx(math.sqrt(1.0 + 2 * G0 * 0.03), abs=0.08)
    assert td.horizontal_speed < 0.3


def test_nominal_ship_landing_succeeds():
    pytest.importorskip("gymnasium")
    from plume.envs.landing_env import AutopilotPolicy, LandingEnv
    from plume.rl.evaluate import run_episodes

    env = LandingEnv("ship_landing", fixed_stage=True)
    res = run_episodes(env, AutopilotPolicy, stage=0, episodes=3, seed=11)
    assert res.success_rate == 1.0, res.reasons
    assert res.landing_error_mean < 6.0


@pytest.mark.slow
def test_ship_landing_high_fidelity_crush_legs():
    pytest.importorskip("gymnasium")
    from plume.envs.landing_env import AutopilotPolicy, LandingEnv
    from plume.rl.evaluate import run_episodes

    env = LandingEnv("ship_landing", fixed_stage=True, fidelity="high", record=True)
    res = run_episodes(env, AutopilotPolicy, stage=0, episodes=1, seed=11)
    assert res.success_rate == 1.0, res.reasons
    m = env.last_replay["meta"]["outcome"]["metrics"]
    assert 0 <= m["stroke_fraction_max"] < 0.8
    assert m["min_tipover_margin_deg"] > 10
