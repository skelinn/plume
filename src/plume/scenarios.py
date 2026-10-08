"""Scripted demo flights for the 6-DOF core (``plume sim``)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from plume.config import VehicleSpec, WorldSpec
from plume.control.attitude import AttitudeController
from plume.control.guidance import WaypointGuidance, thrust_command
from plume.physics.sim import RocketSim
from plume.recording import Recorder

CONTROL_DT = 0.05


def _control_step(sim, att, guid, target_pos, target_vel, max_tilt_deg=15.0):
    st = sim.state
    a_des = guid(st.com, st.vel_com, target_pos, target_vel, dt=CONTROL_DT, up=st.up)
    g = sim.gravity.accel(st.com)
    p_amb = sim.atmosphere.at(st.altitude).pressure
    eng = sim.vehicle.engine
    throttle, axis, _ = thrust_command(
        a_des,
        g,
        st.mass,
        sim.engine.max_thrust(p_amb),
        st.up,
        math.radians(max_tilt_deg),
        eng.throttle_min,
        eng.throttle_max,
    )
    gimbal, rcs = att(st, axis, sim.engine.max_thrust(p_amb) * sim.engine.throttle)
    sim.set_controls(throttle, gimbal, rcs)


def hop_test(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    altitude: float = 150.0,
    distance: float = 60.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """Lift off, climb, translate sideways, descend and land on a second pad."""
    sim = RocketSim(vehicle, world, seed=seed)
    sim.reset(pos=(0.0, 0.0, vehicle.legs.height + 0.01), seed=seed)
    att = AttitudeController(sim)
    guid = WaypointGuidance(bandwidth=0.45, damping=1.0, accel_limit=4.0, integral_gain=0.03)
    pad_b = np.array([distance, 0.0, 0.0])
    rec = Recorder(
        sim.replay_meta(
            f"Hop test: {vehicle.name}",
            controller="waypoint-pd",
            seed=seed,
            scene={
                "frame": "flat",
                "ground": {"type": "plane"},
                "pads": [
                    {"name": "Pad A", "pos": [0.0, 0.0, 0.0], "radius": 6.0},
                    {"name": "Pad B", "pos": pad_b.tolist(), "radius": 6.0},
                ],
                "target": {"pos": pad_b.tolist(), "radius": 6.0},
            },
        )
    )
    steps = round(CONTROL_DT / sim.dt)
    phase = "ascent"
    rec.event(0.0, "phase", "Ascent")
    landed_at = None
    cg0 = sim.state.cg_z
    while sim.t < 120.0:
        st = sim.state
        if phase == "ascent":
            target, tv = np.array([0.0, 0.0, altitude + cg0]), None
            if st.com[2] > altitude * 0.9 + cg0 and sim.t > 8:
                phase = "translate"
                rec.event(sim.t, "phase", "Translate")
        elif phase == "translate":
            target, tv = pad_b + np.array([0.0, 0.0, altitude + cg0]), None
            if np.linalg.norm(st.com[:2] - pad_b[:2]) < 2.0 and st.horizontal_speed < 0.8:
                phase = "descent"
                rec.event(sim.t, "phase", "Descent")
        if phase == "descent":
            vz = -float(np.clip(0.25 * st.agl + 0.5, 0.6, 6.0))
            target = np.array([pad_b[0], pad_b[1], st.com[2]])
            tv = np.array([0.0, 0.0, vz])
            if st.legs_down >= 1 or st.body_contact:
                phase = "landed"
                landed_at = sim.t
                rec.event(sim.t, "touchdown", "Touchdown")
        if phase == "landed":
            sim.set_controls(0.0)
            if sim.t - landed_at > 3.0:
                break
        else:
            _control_step(sim, att, guid, target, tv)
        sim.step(steps)
        frame = sim.frame(phase)
        rec.record(frame)
        if on_frame:
            on_frame(frame)

    st = sim.state
    td = sim.touchdown
    err = float(np.linalg.norm(st.pos[:2] - pad_b[:2]))
    ok = (
        phase == "landed"
        and not sim.ever_body_contact
        and td is not None
        and st.tilt < math.radians(10)
    )
    rec.set_outcome(
        ok,
        "landed" if ok else "failed",
        {
            "landing_error_m": err,
            "touchdown_speed_mps": td.vertical_speed if td else float("nan"),
            "fuel_used_kg": sim.prop_used,
            "flight_time_s": sim.t,
            "max_g": sim.max_g,
        },
    )
    return rec


def drop(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    altitude: float = 2000.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """Uncontrolled tail-first fall from altitude with a small initial tilt (aero demo)."""
    sim = RocketSim(vehicle, world, seed=seed)
    q = np.array([math.cos(0.05), math.sin(0.05), 0.0, 0.0])
    sim.reset(pos=(0.0, 0.0, altitude), vel=(15.0, 0.0, -40.0), quat=q, seed=seed)
    rec = Recorder(
        sim.replay_meta(f"Uncontrolled drop: {vehicle.name}", controller="none", seed=seed)
    )
    while sim.t < 200.0 and sim.touchdown is None:
        sim.step(10)
        frame = sim.frame("freefall")
        rec.record(frame)
        if on_frame:
            on_frame(frame)
    rec.set_outcome(False, "impact", {"impact_speed_mps": sim.state.speed})
    return rec


SCENARIOS = {"hop_test": hop_test, "drop": drop}

# hop-test rig scenarios (tethered hover, tether catch, translation step, free hop)
from plume.hoprig.scenarios import HOP_RIG_SCENARIOS  # noqa: E402

SCENARIOS.update(HOP_RIG_SCENARIOS)


def run_scenario(name: str, vehicle: VehicleSpec, world: WorldSpec, out: Path, seed: int = 0, **kw):
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    rec = SCENARIOS[name](vehicle, world, seed=seed, **kw)
    return rec, rec.save(out)
