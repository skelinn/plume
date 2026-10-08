"""Orbital launch missions: two-stage ascent to orbit with booster return (RTLS).

Sequence (``run_launch``)::

    stack (one RocketSim)   liftoff -> pitch kick -> gravity turn -> MECO at the return reserve
    separation              coast, then one RocketSim per stage (state copied), spring impulse
    upper stage             RCS slew, ignition, closed-loop guidance, fairing jettison, SECO, coast
    booster                 flip, boost-back, coast, entry burn, aero descent, landing burn
    fairing halves          ballistic, tracked for a short while for the replay

All simulators advance in lock-step on the 20 Hz flight-software cycle. The replay holds
the upper stage as its primary track (``frames``) and the booster and fairing halves as
extra tracks (``tracks``), described in ``meta.vehicles``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plume.config import MissionSpec, VehicleSpec
from plume.launcher.guidance import BoosterReturn, StackAscent, UpperStageGuidance
from plume.launcher.orbit import InertialFrame, OrbitalElements, target_plane_normal
from plume.launcher.spec import (
    LauncherSpec,
    LaunchMissionSpec,
    load_launch_mission,
    load_launcher,
)
from plume.launcher.stack import (
    FairingHalf,
    jettison_fairing,
    separation_impulse,
    spawn_from,
    stack_vehicle,
    stage_vehicle,
)
from plume.missions.hop import HopAutopilot, MissionWorld, mission_frame
from plume.physics.propulsion import Engine
from plume.physics.sim import RocketSim, quat_from_z_axis, quat_mul
from plume.recording import Recorder

CONTROL_DT = 0.05
FAIRING_TRACK_S = 40.0


def vehicle_meta(v: VehicleSpec, **extra) -> dict:
    """Replay description of a vehicle (same layout as ``RocketSim.replay_meta``)."""
    out = {
        "name": v.name,
        "length": v.geometry.length,
        "diameter": v.geometry.diameter,
        "nose_length": v.geometry.nose_length,
        "legs": {
            "count": v.legs.count,
            "span": v.legs.span,
            "height": v.legs.height,
            "attach_z": v.legs.attach_z,
        },
        "engine": {
            "nozzle_radius": v.engine.nozzle_radius,
            "gimbal_z": v.engine.gimbal_z,
            "thrust_max": Engine(v.engine, v.prop_capacity).max_thrust(0.0),
        },
        "rcs_z": v.rcs.z,
        "cargo_mass": v.cargo.mass,
        "dry_mass": v.mass.dry,
        "prop_mass_initial": v.prop_initial,
    }
    gf = v.grid_fins
    if gf is not None:
        out["grid_fins"] = {
            "count": gf.count,
            "z": gf.z,
            "span": gf.span,
            "chord": gf.chord,
            "depth": gf.depth,
            "radius": gf.radius if gf.radius is not None else v.geometry.radius + 0.5 * gf.span,
            "max_deflection": math.radians(gf.max_deflection_deg),
        }
    out.update(extra)
    return out


@dataclass
class Flight:
    """One independently simulated vehicle after separation."""

    vid: str
    sim: RocketSim
    ap: object
    done: bool = False
    result: str = ""
    t_end: float | None = None
    info: dict = field(default_factory=dict)


@dataclass
class LaunchRun:
    success: bool
    reason: str
    metrics: dict
    recorder: Recorder
    elements: OrbitalElements | None = None


def _derived(frame: dict, st, z_off: float, keep: set[str], **values) -> dict:
    """Frame of a passive part riding on the stack (offset ``z_off`` along the axis)."""
    out = {k: v for k, v in frame.items() if k in keep}
    out["pos"] = st.pos + st.rot @ np.array([0.0, 0.0, z_off])
    if "alt" in out:
        out["alt"] = st.agl + z_off * max(float(st.axis @ st.up), 0.0)
    for k in ("throttle", "thrust"):
        if k in out:
            out[k] = 0.0
    if "gimbal" in out:
        out["gimbal"] = np.zeros(2)
    if "rcs" in out:
        out["rcs"] = np.zeros(3)
    if "f_thrust" in out:
        out["f_thrust"] = np.zeros(3)
    out.update(values)
    return out


UPPER_KEYS = {
    "t", "pos", "quat", "vel", "omega", "alt", "throttle", "thrust", "gimbal", "rcs",
    "prop_mass", "mass", "g_load", "mach", "q_dyn", "wind", "v_air", "f_thrust", "f_aero",
    "alpha", "beta", "aoa_total", "phase",
}  # fmt: skip
FAIRING_KEYS = {"t", "pos", "quat", "vel", "alt", "throttle", "mass", "phase"}


def run_launch(
    spec: LaunchMissionSpec | str,
    seed: int = 0,
    fidelity: str | None = None,
    payload_mass: float | None = None,
    on_frame=None,
    dt: float | None = None,
    launcher: LauncherSpec | None = None,
    record_every: int | None = None,
    stop_after_orbit: bool = False,
    on_meta=None,
) -> LaunchRun:
    """Fly an orbital launch mission. ``dt`` overrides the physics step (coarse runs);
    ``stop_after_orbit`` ends the run once the upper stage's orbit is known (the booster
    return is then not flown to the ground)."""
    spec = load_launch_mission(spec) if isinstance(spec, str) else spec
    launcher = launcher or load_launcher(spec.vehicle)
    pay = payload_mass if payload_mass is not None else spec.payload_mass
    if pay is not None:
        launcher = launcher.with_payload(pay)
    hop_spec = MissionSpec(
        name=spec.name,
        vehicle=launcher.stages[0].vehicle.name,
        terrain=spec.terrain,
        launch=spec.launch,
        target=spec.landing,
        target_radius=spec.landing_radius,
        world=spec.world,
        guidance=spec.booster_guidance,
        max_time=spec.max_time,
    )
    hop_spec, world, gravity = mission_frame(hop_spec, fidelity)
    if dt is not None:
        world = world.model_copy(update={"dt": dt})
    # flight software flies on the true state in launch missions (no EKF per vehicle yet)
    world = world.model_copy(update={"navigation": "truth"})
    mw = MissionWorld(hop_spec, gravity)
    frame = InertialFrame(gravity, spec.launch.lat)
    tiles = mw.ground_tiles()
    last = len(launcher.stages) - 1
    if last != 1:
        raise NotImplementedError("run_launch flies two-stage vehicles (stack + upper stage)")
    fair = launcher.fairing

    # ---------------------------------------------------------------- stack on the pad
    sv = stack_vehicle(launcher, 0, fairing=fair is not None)
    stack = RocketSim(sv, world, seed=seed, tiles=tiles, ground_height=mw.ground_height)
    stack.nominal_world = world
    up = gravity.up(mw.pad_a)
    stack.reset(pos=mw.pad_a + up * (sv.legs.height - 0.005), quat=quat_from_z_axis(up), seed=seed)
    r_i, v_i = frame.to_inertial(stack.state.com, np.zeros(3), 0.0)
    incl = spec.orbit.inclination_deg
    n_target = target_plane_normal(r_i, v_i, None if incl is None else math.radians(incl))
    ascent = StackAscent(stack, spec.ascent, frame, n_target, launcher.stages[0].reserve)

    booster_v = stage_vehicle(launcher, 0)
    upper_fair_v = stage_vehicle(launcher, last, fairing=fair is not None)
    upper_v = stage_vehicle(launcher, last, fairing=False)
    z_upper = launcher.stage_base(last)
    z_fair = launcher.top

    # ---------------------------------------------------------------- replay layout
    f_dia = (fair.diameter if fair and fair.diameter else upper_v.geometry.diameter) if fair else 0
    vehicles = [
        {
            "id": "upper",
            "name": "Upper stage",
            "role": "upper_stage",
            "primary": True,
            "attach_z": z_upper,
            "vehicle": vehicle_meta(upper_v),
        },
        {
            "id": "booster",
            "name": "Booster",
            "role": "booster",
            "attach_z": 0.0,
            "vehicle": vehicle_meta(booster_v),
        },
    ]
    if fair is not None:
        for k in ("a", "b"):
            vehicles.append(
                {
                    "id": f"fairing_{k}",
                    "name": f"Fairing half {k.upper()}",
                    "role": "fairing_half",
                    "attach_z": z_fair,
                    "persist": False,
                    "vehicle": {
                        "name": "fairing",
                        "kind": "fairing_half",
                        "length": fair.length,
                        "diameter": f_dia,
                        "nose_length": fair.nose_length,
                        "legs": {"count": 0},
                        "dry_mass": 0.5 * fair.mass,
                    },
                }
            )
    meta = stack.replay_meta(
        f"Orbital launch: {spec.name}",
        controller="launch-guidance",
        seed=seed,
        scene=mw.scene_meta(),
        mission={
            "name": spec.name,
            "kind": "launch",
            "payload_kg": launcher.payload.mass,
            "target_orbit": spec.orbit.model_dump(),
            "landing_zone": spec.landing.name,
        },
    )
    meta["vehicle"] = vehicles[0]["vehicle"]
    meta["vehicles"] = vehicles
    meta["launcher"] = {
        "name": launcher.name,
        "stages": [s.name for s in launcher.stages],
        "liftoff_mass": sv.mass.dry + sv.prop_initial + sv.rcs.gas,
    }
    rec = Recorder(meta)
    if on_meta is not None:  # live streaming: the primary (upper-stage) track only
        on_meta({k: v for k, v in rec.meta.items() if k != "vehicles"})
    tracks = {"upper": rec, "booster": rec.add_track("booster", Recorder({}))}
    if fair is not None:
        tracks["fairing_a"] = rec.add_track("fairing_a", Recorder({}))
        tracks["fairing_b"] = rec.add_track("fairing_b", Recorder({}))
    ended: set[str] = set()
    events: list[tuple[float, str, str, str]] = []

    def ev(t, kind, label, vid):
        events.append((t, kind, label, vid))

    upper_mass0 = upper_fair_v.mass.dry + upper_fair_v.prop_initial + upper_fair_v.rcs.gas
    upper_mass0 += upper_fair_v.cargo.mass
    half_mass = 0.5 * fair.mass if fair is not None else 0.0
    flip = np.array([0.0, 0.0, 0.0, 1.0])

    def record_stack(phase: str, force: bool = False):
        f = stack.frame(phase, {"impact": mw.pad_b})
        st = stack.state
        tracks["booster"].record(f, force=force)
        up_f = _derived(
            f,
            st,
            z_upper,
            UPPER_KEYS,
            mass=upper_mass0,
            prop_mass=upper_fair_v.prop_initial,
            phase="stacked",
        )
        rec.record(up_f, force=force)
        if on_frame:
            on_frame(up_f)
        if fair is not None:
            for k, q in (("a", st.quat), ("b", quat_mul(st.quat, flip))):
                ff = _derived(f, st, z_fair, FAIRING_KEYS, mass=half_mass, phase="stacked")
                ff["quat"] = q
                tracks[f"fairing_{k}"].record(ff, force=force)

    # ---------------------------------------------------------------- stacked flight
    steps = max(1, round(CONTROL_DT / world.dt))
    rec_every = record_every or spec.record_every
    k = 0
    failure = ""
    meco_t: float | None = None
    staging: dict = {}
    record_stack("prelaunch", force=True)
    ev(0.0, "ignition", "Liftoff", "booster")
    while stack.t < spec.max_time:
        st = stack.state
        throttle, gimbal, rcs, phase = ascent.act(st)
        stack.set_controls(throttle, gimbal, rcs)
        stack.step(steps)
        k += 1
        st = stack.state
        if ascent.done and meco_t is None:
            meco_t = stack.t
        if k % rec_every == 0:
            record_stack(phase)
        if stack.ever_body_contact or (stack.t > 5 and st.agl < -2.0):
            failure = "stack_crash"
            break
        if meco_t is not None and stack.t >= meco_t + launcher.separation.coast:
            break
    else:
        failure = "timeout"
    for t, kind, label in ascent.events:
        ev(t, kind, label, "booster")

    flights: dict[str, Flight] = {}
    halves: dict[str, FairingHalf] = {}
    elements_final: OrbitalElements | None = None
    if not failure:
        record_stack("coast", force=True)
        st = stack.state
        t_sep = stack.t
        staging = {
            "t": t_sep,
            "altitude_m": st.altitude,
            "speed_m_s": st.speed,
            "flight_path_deg": math.degrees(
                math.asin(float(np.clip(st.vertical_speed / max(st.speed, 1e-9), -1, 1)))
            ),
            "booster_prop_kg": stack.prop_mass,
            "max_q_pa": stack.max_q,
        }
        # ------------------------------------------------------------ separation
        booster = RocketSim(
            booster_v, world, seed=seed, tiles=tiles, ground_height=mw.ground_height
        )
        booster.nominal_world = world
        booster.engine = stack.engine  # same engine: keeps its state and ignition count
        spawn_from(
            stack, booster, 0.0, tanks=stack.tanks.copy(), rcs_prop=stack.rcs_prop, seed=seed
        )
        booster.set_legs(False)
        upper = RocketSim(upper_fair_v, world, seed=seed + 1)
        upper.nominal_world = world
        spawn_from(stack, upper, z_upper, seed=seed + 1)
        separation_impulse(booster, upper, launcher.separation.delta_v)
        ev(t_sep, "separation", "Stage separation", "booster")
        hop_ap = HopAutopilot(booster, mw, spec.booster_guidance, kick_deg=0.0, rise_time=0.0)
        hop_ap.phase = "coast"
        flights["booster"] = Flight(
            "booster", booster, BoosterReturn(booster, mw, hop_ap, spec.boostback, t_sep)
        )
        flights["upper"] = Flight(
            "upper",
            upper,
            UpperStageGuidance(
                upper,
                frame,
                spec.orbit,
                n_target,
                spec.upper_guidance,
                t_sep + launcher.separation.ignition_delay,
            ),
        )
        flights["upper"].info["fairing"] = fair is not None
        del stack

    def fr_record(force=False):
        for vid, fl in flights.items():
            if vid in ended:
                continue
            if vid == "booster":
                imp = fl.ap.predicted_impact
                if fl.ap.phase == "handover":
                    imp = fl.ap.hop.predicted_impact
                f = fl.sim.frame(
                    fl.info.get("phase", "coast"), {"impact": imp if imp is not None else mw.pad_b}
                )
                tracks["booster"].record(f, force=force)
            else:
                f = fl.sim.frame(fl.info.get("phase", "coast"))
                rec.record(f, force=force)
                if on_frame:
                    on_frame(f)
                if fl.info.get("fairing") and not halves:  # fairing still on the stage
                    st = fl.sim.state
                    z_top = launcher.stages[last].vehicle.geometry.length
                    for k, q in (("a", st.quat), ("b", quat_mul(st.quat, flip))):
                        ff = _derived(f, st, z_top, FAIRING_KEYS, mass=half_mass, phase="stacked")
                        ff["quat"] = q
                        tracks[f"fairing_{k}"].record(ff, force=force)
            if fl.done:
                ended.add(vid)
        for vid, h in halves.items():
            if vid in ended:
                continue
            q = h.quat
            tracks[vid].record(
                {
                    "t": h.t,
                    "pos": h.pos,
                    "quat": q,
                    "vel": h.vel,
                    "alt": gravity.altitude(h.pos),
                    "throttle": 0.0,
                    "mass": half_mass,
                    "phase": "jettisoned",
                },
                force=force,
            )
            if h.t - h.log[0] >= FAIRING_TRACK_S:
                ended.add(vid)

    # ---------------------------------------------------------------- independent flights
    max_alt_b = 0.0
    landed_t = None
    while flights and not all(f.done for f in flights.values()):
        t_now = next(f.sim.t for f in flights.values())
        if t_now >= spec.max_time:
            for f in flights.values():
                if not f.done:
                    f.done, f.result = True, "timeout"
            break
        for fl in flights.values():
            if fl.done:
                continue
            st = fl.sim.state
            throttle, gimbal, rcs, phase = fl.ap.act(st)
            fl.info["phase"] = phase
            if fl.vid == "booster" and landed_t is not None:
                throttle = 0.0
            fl.sim.set_controls(throttle, gimbal, rcs)
        for fl in flights.values():
            if not fl.done:
                fl.sim.step(steps)
        for h in halves.values():
            for _ in range(5):
                h.step(CONTROL_DT / 5, gravity)
        k += 1
        # ---- upper stage: fairing jettison, orbit
        up_f = flights.get("upper")
        if up_f is not None and not up_f.done:
            us = up_f.sim.state
            ap = up_f.ap
            if (
                up_f.info.get("fairing")
                and ap.phase == "burn"
                and us.altitude > fair.jettison_altitude
            ):
                old = up_f.sim
                hs = jettison_fairing(
                    old,
                    launcher.stages[last].vehicle.geometry.length,
                    fair.jettison_speed,
                    fair.jettison_rate_deg_s,
                )
                new = RocketSim(upper_v, world, seed=seed + 2)
                new.nominal_world = world
                new.engine = old.engine  # burning engine carries on
                spawn_from(
                    old, new, 0.0, tanks=old.tanks.copy(), rcs_prop=old.rcs_prop, seed=seed + 2
                )
                new.thrust, new.mdot = old.thrust, old.mdot
                up_f.sim = new
                ap.bind(new)
                up_f.info["fairing"] = False
                for name, h in zip(("fairing_a", "fairing_b"), hs, strict=True):
                    h.log.append(h.t)
                    halves[name] = h
                ev(new.t, "separation", "Fairing jettison", "upper")
            if ap.phase == "coast" and ap.cutoff_t is not None:
                if "seco" not in up_f.info:
                    el = frame.elements(us.com, us.vel_com, up_f.sim.t)
                    up_f.info["seco"] = el
                    up_f.info["prop_at_seco"] = up_f.sim.prop_mass
                if up_f.sim.t - ap.cutoff_t >= spec.upper_guidance.coast_after_cutoff:
                    up_f.done = True
                    up_f.t_end = up_f.sim.t
            if us.altitude < 20_000.0 and us.vertical_speed < 0:
                up_f.done, up_f.result = True, "suborbital"
        # ---- booster: landing / crash checks (as in the cargo-hop runner)
        bo = flights.get("booster")
        if bo is not None and not bo.done:
            bs = bo.sim.state
            max_alt_b = max(max_alt_b, bs.altitude)
            sim_b = bo.sim
            if sim_b.ever_body_contact:
                bo.done, bo.result = True, "crash_hull"
            elif (
                sim_b.touchdown is not None
                and sim_b.touchdown.vertical_speed > booster_v.legs.max_touchdown_speed
            ):
                bo.done, bo.result = True, "crash_legs"
            elif bs.agl < 0.0 and sim_b.touchdown is None:
                u, v, _ = mw.gravity.map_coords(bs.pos)
                if not any(tile.contains(u, v) for tile in mw.tiles):
                    gentle = (
                        -bs.vertical_speed <= booster_v.legs.max_touchdown_speed
                        and bs.horizontal_speed < 2.0
                        and bs.tilt < math.radians(10.0)
                    )
                    bo.done, bo.result = True, ("landed_off_site" if gentle else "terrain_impact")
                elif bs.agl < -2.0:
                    bo.done, bo.result = True, "terrain_impact"
            if landed_t is None and sim_b.touchdown is not None:
                landed_t = sim_b.t
            if landed_t is not None and sim_b.t - landed_t > 5.0 and not bo.done:
                bo.done, bo.result = True, "landed"
            if bo.done:
                bo.t_end = sim_b.t
            if stop_after_orbit and up_f is not None and up_f.done:
                bo.done, bo.result = True, bo.result or "not_flown"
        if k % rec_every == 0:
            fr_record()
    if flights:
        fr_record(force=True)
        for vid, h in halves.items():
            if vid not in ended:
                tracks[vid].record(
                    {
                        "t": h.t,
                        "pos": h.pos,
                        "quat": h.quat,
                        "vel": h.vel,
                        "alt": gravity.altitude(h.pos),
                        "throttle": 0.0,
                        "mass": half_mass,
                        "phase": "jettisoned",
                    },
                    force=True,
                )

    # ---------------------------------------------------------------- outcome
    metrics: dict = {"staging": staging} if staging else {}
    r_eq = frame.r_eq
    orbit_ok = False
    up_f = flights.get("upper")
    if up_f is not None:
        for t, kind, label in up_f.ap.events:
            ev(t, kind, label, "upper")
        us = up_f.sim.state
        el = frame.elements(us.com, us.vel_com, up_f.sim.t)
        elements_final = el
        orbit_ok = el.perigee_altitude(r_eq) > spec.orbit.min_perigee_altitude and el.e < 1.0
        tgt = spec.orbit
        metrics["orbit"] = {
            "reached": orbit_ok,
            "perigee_alt_km": el.perigee_altitude(r_eq) / 1e3,
            "apogee_alt_km": el.apogee_altitude(r_eq) / 1e3,
            "inclination_deg": math.degrees(el.i),
            "semi_major_axis_km": el.a / 1e3,
            "eccentricity": el.e,
            "perigee_error_km": (el.perigee_altitude(r_eq) - tgt.perigee_altitude) / 1e3,
            "apogee_error_km": (el.apogee_altitude(r_eq) - tgt.apogee_altitude) / 1e3,
            "inclination_error_deg": (
                math.degrees(el.i) - tgt.inclination_deg
                if tgt.inclination_deg is not None
                else None
            ),
            "seco_t": up_f.ap.cutoff_t,
            "upper_prop_left_kg": up_f.info.get("prop_at_seco", up_f.sim.prop_mass),
            "upper_prop_left_pct": 100.0
            * up_f.info.get("prop_at_seco", up_f.sim.prop_mass)
            / max(upper_v.prop_initial, 1e-9),
            "result": up_f.result or ("orbit" if orbit_ok else "no_orbit"),
        }
    bo = flights.get("booster")
    booster_ok = False
    if bo is not None:
        for t, kind, label in bo.ap.all_events:
            ev(t, kind, label, "booster")
        sim_b = bo.sim
        bs = sim_b.state
        err = mw.miss(bs.pos)
        booster_ok = bo.result == "landed" and err <= spec.landing_radius
        td = sim_b.touchdown
        metrics["booster"] = {
            "result": bo.result,
            "landed": booster_ok,
            "landing_error_m": err,
            "touchdown_speed_m_s": td.vertical_speed if td else None,
            "touchdown_horizontal_m_s": td.horizontal_speed if td else None,
            "prop_left_kg": sim_b.prop_mass,
            "boostback_prop_kg": bo.ap.boostback_used,
            "apogee_km": max_alt_b / 1e3,
            "max_g": sim_b.max_g,
        }
        if td is not None:
            ev(td.t, "touchdown", "Booster touchdown", "booster")
    if failure:
        reason = failure
    elif orbit_ok and booster_ok:
        reason = "orbit_and_booster_landed"
    elif orbit_ok:
        reason = f"orbit; booster {metrics.get('booster', {}).get('result', 'lost')}"
    else:
        reason = metrics.get("orbit", {}).get("result", "no_orbit")
    success = bool(orbit_ok and booster_ok and not failure)
    for t, kind, label, vid in sorted(events, key=lambda e: e[0]):
        rec.event(t, kind, label, vehicle=vid)
    for vid, tr in tracks.items():
        for v in vehicles:
            if v["id"] == vid and tr.frames.get("t"):
                v["t_start"], v["t_end"] = tr.frames["t"][0], tr.frames["t"][-1]
    rec.set_outcome(success, reason, _flatten(metrics))
    return LaunchRun(success, reason, metrics, rec, elements_final)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif v is not None:
            out[key] = v
    return out


def summary_rows(run: LaunchRun, spec: LaunchMissionSpec) -> list[tuple[str, str]]:
    """Human-readable result lines (CLI table)."""
    m = run.metrics
    rows: list[tuple[str, str]] = []
    s = m.get("staging")
    if s:
        rows.append(
            (
                "staging",
                f"T+{s['t']:.0f} s at {s['altitude_m'] / 1e3:.1f} km, {s['speed_m_s']:,.0f} m/s, "
                f"flight path {s['flight_path_deg']:.0f} deg (max q {s['max_q_pa'] / 1e3:.1f} kPa)",
            )
        )
    o = m.get("orbit")
    if o:
        tgt = spec.orbit
        inc_t = f"{tgt.inclination_deg:g}" if tgt.inclination_deg is not None else "-"
        rows.append(("orbit reached", "yes" if o["reached"] else f"no ({o['result']})"))
        rows.append(
            (
                "orbit",
                f"{o['perigee_alt_km']:.1f} x {o['apogee_alt_km']:.1f} km, i = "
                f"{o['inclination_deg']:.2f} deg (target {tgt.perigee_altitude / 1e3:g} x "
                f"{tgt.apogee_altitude / 1e3:g} km, {inc_t} deg)",
            )
        )
        rows.append(
            (
                "upper stage propellant left",
                f"{o['upper_prop_left_kg']:.0f} kg ({o['upper_prop_left_pct']:.1f} %)",
            )
        )
    b = m.get("booster")
    if b:
        rows.append(("booster", b["result"]))
        rows.append(
            (
                "booster landing error",
                f"{b['landing_error_m']:,.1f} m (radius {spec.landing_radius:g} m)",
            )
        )
        if b.get("touchdown_speed_m_s") is not None:
            rows.append(
                (
                    "booster touchdown",
                    f"{b['touchdown_speed_m_s']:.2f} m/s down, {b['touchdown_horizontal_m_s']:.2f} m/s across",
                )
            )
        rows.append(("booster propellant left", f"{b['prop_left_kg']:.0f} kg"))
    return rows


def save_launch(run: LaunchRun, out: str | Path) -> Path:
    return run.recorder.save(out)


__all__ = ["LaunchRun", "run_launch", "save_launch", "vehicle_meta"]
