"""Multi-stage vehicles, staging, orbital elements and orbit guidance."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from plume.config import WorldSpec
from plume.launcher.guidance import UpperStageGuidance
from plume.launcher.orbit import (
    InertialFrame,
    elements,
    state_from_elements,
    target_plane_normal,
)
from plume.launcher.spec import (
    LauncherSpec,
    OrbitTargetSpec,
    UpperGuidanceSpec,
    load_launch_mission,
    load_launcher,
)
from plume.launcher.stack import (
    momentum,
    separation_impulse,
    shell_part,
    spawn_from,
    stack_vehicle,
    stage_vehicle,
)
from plume.physics.earth import EarthGravity, EarthParams
from plume.physics.gravity import SphericalGravity
from plume.physics.massprops import MassModel
from plume.physics.sim import RocketSim, quat_from_axis_angle, quat_from_z_axis
from plume.recording import Recorder
from plume.recording.recorder import SCHEMA_PATH

MU = 3.986004418e14


@pytest.fixture(scope="module")
def launcher() -> LauncherSpec:
    return load_launcher("launcher_two_stage")


def _full(v):
    mm = MassModel(v)
    tanks = np.array([t.initial_mass for t in v.tanks])
    return mm.evaluate(tanks, v.rcs.gas, v.cargo.mass)


# --------------------------------------------------------------------------- configuration
def test_launcher_preset_and_mission_load(launcher):
    assert [s.name for s in launcher.stages] == ["booster", "upper"]
    assert launcher.stage_base(1) == launcher.stages[0].vehicle.geometry.length
    mission = load_launch_mission("demo_orbit")
    assert mission.vehicle == launcher.name
    with pytest.raises(ValueError):
        load_launcher("cargo_hopper")  # single-stage vehicles are not launchers


def test_stack_mass_properties_match_brute_force(launcher):
    """Stack = booster (active) + upper stage, payload and fairing as rigid dead mass."""
    stack = stack_vehicle(launcher, 0, fairing=True)
    booster = launcher.stages[0].vehicle
    upper = stage_vehicle(launcher, 1, fairing=False)
    f = launcher.fairing
    base_u = launcher.stage_base(1)
    parts = [(p.mass, p.cg_z, p.inertia) for p in (_full(booster),)]
    pu = _full(upper)
    parts.append((pu.mass, pu.cg_z + base_u, pu.inertia))
    parts.append(shell_part(f.mass, 0.5 * booster.geometry.diameter, f.length, launcher.top))
    m = sum(p[0] for p in parts)
    z = sum(p[0] * p[1] for p in parts) / m
    i_lat = sum(p[2][0] + p[0] * (p[1] - z) ** 2 for p in parts)
    i_ax = sum(p[2][2] for p in parts)

    mp = _full(stack)
    assert mp.mass == pytest.approx(m, rel=1e-12)
    assert mp.cg_z == pytest.approx(z, rel=1e-12)
    assert mp.inertia[0] == pytest.approx(i_lat, rel=1e-9)
    assert mp.inertia[2] == pytest.approx(i_ax, rel=1e-9)
    # the stack keeps the booster's engine, tanks and RCS; the geometry is the whole stack
    assert stack.engine == booster.engine
    assert stack.prop_capacity == booster.prop_capacity
    assert stack.geometry.length == pytest.approx(launcher.top + f.length)
    expect = (
        booster.mass.dry
        + booster.prop_initial
        + booster.rcs.gas
        + upper.mass.dry
        + upper.prop_initial
        + upper.rcs.gas
        + launcher.payload.mass
        + f.mass
    )
    assert mp.mass == pytest.approx(expect, rel=1e-12)


@pytest.mark.parametrize("fidelity", ["fast", "high"])
def test_separation_conserves_mass_and_momentum(launcher, fidelity):
    """Splitting the rigid stack into per-stage simulators conserves mass, linear and
    angular momentum exactly; the spring impulse conserves momentum and gives the
    requested relative speed along the axis."""
    lau = launcher.model_copy(deep=True)
    for s in lau.stages:  # strip aero: no database generation needed in high fidelity
        s.vehicle.aero.model = "strip"
    world = WorldSpec(fidelity=fidelity, gravity="flat", g=0.0, atmosphere=False, ground="none")
    stack = RocketSim(stack_vehicle(lau, 0, fairing=False), world)
    tilt = quat_from_axis_angle([1.0, 0.3, 0.0], math.radians(35.0))
    stack.reset(
        pos=(1000.0, -500.0, 60_000.0),
        vel=(900.0, 400.0, 1100.0),
        quat=tilt,
        omega=(0.02, -0.015, 0.01),
        prop=3000.0,
    )
    stack.step(3)
    m0, p0, h0 = momentum(stack)

    booster = RocketSim(stage_vehicle(lau, 0), world)
    upper = RocketSim(stage_vehicle(lau, 1, fairing=False), world)
    spawn_from(stack, booster, 0.0, tanks=stack.tanks.copy(), rcs_prop=stack.rcs_prop)
    spawn_from(stack, upper, lau.stage_base(1))
    mb, pb, hb = momentum(booster)
    mu, pu, hu = momentum(upper)
    assert mb + mu == pytest.approx(m0, rel=1e-12)
    np.testing.assert_allclose(pb + pu, p0, rtol=1e-10, atol=1e-6 * np.linalg.norm(p0))
    np.testing.assert_allclose(hb + hu, h0, rtol=1e-9, atol=1e-9 * np.linalg.norm(h0))
    assert booster.t == stack.t and upper.t == stack.t

    axis = stack.state.axis
    dv_before = float((upper.state.vel_com - booster.state.vel_com) @ axis)
    impulse = separation_impulse(booster, upper, 1.0)
    assert impulse > 0
    dv_after = float((upper.state.vel_com - booster.state.vel_com) @ axis)
    assert dv_after - dv_before == pytest.approx(1.0, rel=1e-9)
    _, pb, hb = momentum(booster)
    _, pu, hu = momentum(upper)
    np.testing.assert_allclose(pb + pu, p0, rtol=1e-10, atol=1e-6 * np.linalg.norm(p0))
    np.testing.assert_allclose(hb + hu, h0, rtol=1e-9, atol=1e-9 * np.linalg.norm(h0))

    # free flight afterwards (no forces): each part keeps its momentum
    booster.step(20)
    upper.step(20)
    _, pb2, _ = momentum(booster)
    _, pu2, _ = momentum(upper)
    np.testing.assert_allclose(pb2 + pu2, p0, rtol=1e-9)


# --------------------------------------------------------------------------- orbital elements
@pytest.mark.parametrize(
    "el",
    [
        (7_000e3, 0.01, 0.9, 1.2, 0.4, 2.5),
        (6_600e3, 0.0035, math.radians(40.0), 4.0, 2.0, 0.3),
        (26_560e3, 0.7, math.radians(63.4), 0.5, math.radians(270.0), 3.0),
    ],
)
def test_elements_roundtrip_analytic(el):
    a, e, i, raan, argp, nu = el
    r, v = state_from_elements(a, e, i, raan, argp, nu, MU)
    # vis-viva and angular momentum of the analytic two-body orbit
    assert float(v @ v) == pytest.approx(MU * (2 / np.linalg.norm(r) - 1 / a), rel=1e-12)
    out = elements(r, v, MU)
    assert out.a == pytest.approx(a, rel=1e-10)
    assert out.e == pytest.approx(e, abs=1e-10)
    assert out.i == pytest.approx(i, abs=1e-10)
    assert out.raan == pytest.approx(raan, abs=1e-9)
    assert out.argp == pytest.approx(argp, abs=1e-7)
    assert out.nu == pytest.approx(nu, abs=1e-7)
    assert out.rp == pytest.approx(a * (1 - e), rel=1e-10)
    assert out.ra == pytest.approx(a * (1 + e), rel=1e-10)
    assert out.h == pytest.approx(math.sqrt(MU * a * (1 - e * e)), rel=1e-10)


def test_circular_orbit_elements():
    r = 6_678_137.0
    out = elements(np.array([r, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / r), 0.0]), MU)
    assert out.a == pytest.approx(r, rel=1e-12)
    assert out.e < 1e-12
    assert out.i == pytest.approx(0.0, abs=1e-12)
    assert out.period(MU) == pytest.approx(2 * math.pi * math.sqrt(r**3 / MU), rel=1e-12)


def _world_from_inertial(g: EarthGravity, r_i, v_i, t):
    """Inverse of InertialFrame.to_inertial for the rotating WGS-84 frame."""
    th = -g.params.omega * t
    c, s = math.cos(th), math.sin(th)
    Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_e = Q @ r_i
    v_e = Q @ v_i - np.cross([0.0, 0.0, g.params.omega], r_e)
    return g.from_ecef(r_e), g.R_ew @ v_e


def test_elements_from_rotating_world_frame():
    """Rotating Earth: elements recovered from the Earth-fixed world state at any time,
    and a point at rest on the ground has i = its geocentric latitude."""
    lat = math.radians(31.0)
    g = EarthGravity(lat0=lat, lon0=math.radians(-104.0), params=EarthParams())
    frame = InertialFrame(g)
    a, e, i, raan, argp, nu = 6_650e3, 0.004, math.radians(40.0), 1.0, 0.5, 2.0
    r_i, v_i = state_from_elements(a, e, i, raan, argp, nu, frame.mu)
    for t in (0.0, 437.0, 5000.0):
        p_w, v_w = _world_from_inertial(g, r_i, v_i, t)
        out = frame.elements(p_w, v_w, t)
        assert out.a == pytest.approx(a, rel=1e-9)
        assert out.e == pytest.approx(e, abs=1e-9)
        assert out.i == pytest.approx(i, abs=1e-9)
        assert out.raan == pytest.approx(raan, abs=1e-8)
    pad = np.zeros(3)
    _r, v = frame.to_inertial(pad, np.zeros(3), 0.0)
    r_e = g.to_ecef(pad)
    assert float(np.linalg.norm(v)) == pytest.approx(
        g.params.omega * math.hypot(r_e[0], r_e[1]), rel=1e-12
    )
    geocentric = math.atan2(r_e[2], math.hypot(r_e[0], r_e[1]))
    assert frame.elements(pad, np.zeros(3), 0.0).i == pytest.approx(geocentric, abs=1e-9)


def test_non_rotating_frame_inclination_and_target_plane():
    g = SphericalGravity()
    frame = InertialFrame(g, launch_lat_deg=31.0)
    p = np.array([0.0, 0.0, 200e3])
    v_east = np.array([7800.0, 0.0, 0.0])
    assert math.degrees(frame.elements(p, v_east).i) == pytest.approx(31.0, abs=1e-9)
    r_i, v_i = frame.to_inertial(p, np.zeros(3))
    for inc in (31.0, 40.0, 97.0):
        n = target_plane_normal(r_i, v_i, math.radians(inc))
        assert math.degrees(math.acos(n[2])) == pytest.approx(inc, abs=1e-9)
        assert abs(float(n @ r_i)) < 1e-6 * np.linalg.norm(r_i)  # plane through the position


# --------------------------------------------------------------------------- guidance
def test_upper_stage_guidance_reaches_target_orbit(launcher):
    """Upper stage alone from a typical staging state (53 km, 1.74 km/s, 35 deg): the
    closed-loop guidance inserts it into the 200 x 250 km, 40 deg target orbit."""
    world = WorldSpec(gravity="spherical", ground="none", dt=0.025)
    v = stage_vehicle(launcher, 1, fairing=False)
    sim = RocketSim(v, world)
    frame = InertialFrame(sim.gravity, launch_lat_deg=31.0)
    target = OrbitTargetSpec(perigee_altitude=200e3, apogee_altitude=250e3, inclination_deg=40.0)
    up = np.array([0.0, 0.0, 1.0])
    heading = np.array([0.894, 0.448, 0.0])
    gam = math.radians(35.0)
    vel = 1740.0 * (math.cos(gam) * heading + math.sin(gam) * up)
    sim.reset(pos=(0.0, 0.0, 53_600.0), vel=vel, quat=quat_from_z_axis(vel))
    r_i, v_i = frame.to_inertial(sim.state.com, sim.state.vel_com)
    n = target_plane_normal(r_i, v_i, math.radians(40.0))
    gd = UpperStageGuidance(sim, frame, target, n, UpperGuidanceSpec(), t_ignite=1.0)
    steps = round(0.05 / world.dt)
    while sim.t < 600.0 and gd.phase != "coast":
        throttle, gimbal, rcs, _ = gd.act(sim.state)
        sim.set_controls(throttle, gimbal, rcs)
        sim.step(steps)
    assert gd.phase == "coast", "no cutoff"
    sim.set_controls(0.0)
    sim.step(40)  # thrust tail-off
    st = sim.state
    el = frame.elements(st.com, st.vel_com)
    r_eq = frame.r_eq
    assert el.perigee_altitude(r_eq) == pytest.approx(200e3, abs=3e3)
    assert el.apogee_altitude(r_eq) == pytest.approx(250e3, abs=6e3)
    assert math.degrees(el.i) == pytest.approx(40.0, abs=0.05)
    assert sim.prop_mass > 0.0


# --------------------------------------------------------------------------- replays
def test_multi_vehicle_replay_validates(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    veh = {"name": "s", "length": 5.0, "diameter": 1.0}
    rec = Recorder(
        {
            "title": "two vehicles",
            "source": "sim",
            "vehicle": veh,
            "scene": {"frame": "flat", "ground": {"type": "plane"}},
            "vehicles": [
                {
                    "id": "upper",
                    "name": "Upper",
                    "role": "upper_stage",
                    "primary": True,
                    "vehicle": veh,
                },
                {"id": "booster", "name": "Booster", "role": "booster", "vehicle": veh},
            ],
        }
    )
    tr = rec.add_track("booster", Recorder({}))
    for i in range(4):
        rec.record({"t": 0.1 * i, "pos": [0, 0, 10.0 * i], "quat": [1, 0, 0, 0]})
        tr.record({"t": 0.1 * i, "pos": [0, 0, -5.0], "quat": [1, 0, 0, 0]})
    rec.event(0.1, "separation", "Stage separation", vehicle="booster")
    data = json.loads(json.dumps(rec.to_dict()))
    assert data["tracks"]["booster"]["frames"]["pos"][0] == [0, 0, -5.0]
    assert data["events"][0]["vehicle"] == "booster"
    jsonschema.validate(data, json.loads(SCHEMA_PATH.read_text()))
    # single-vehicle replays are unchanged (no tracks key)
    assert "tracks" not in Recorder({"title": "x"}).to_dict()


@pytest.mark.slow
def test_demo_orbit_coarse_flight(tmp_path):
    """Coarse (dt = 20 ms) fast-fidelity demo: orbit reached, booster lands on LZ-1."""
    from plume.launcher.mission import run_launch

    run = run_launch("demo_orbit", dt=0.02)
    o, b = run.metrics["orbit"], run.metrics["booster"]
    assert o["reached"], o
    assert o["perigee_alt_km"] == pytest.approx(200.0, abs=5.0)
    assert o["apogee_alt_km"] == pytest.approx(250.0, abs=10.0)
    assert o["inclination_deg"] == pytest.approx(40.0, abs=0.1)
    assert b["result"] == "landed" and b["landing_error_m"] < 30.0, b
    assert run.success
    data = run.recorder.to_dict()
    assert set(data["tracks"]) == {"booster", "fairing_a", "fairing_b"}
    ids = [v["id"] for v in data["meta"]["vehicles"]]
    assert ids[0] == "upper" and "booster" in ids
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads(json.dumps(data)), json.loads(SCHEMA_PATH.read_text()))
