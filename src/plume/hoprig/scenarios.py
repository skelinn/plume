"""Scripted hop-rig test flights (``plume sim hop_rig --script <name>``).

* ``tethered_hover``   lift off on a ground tether, hover 1.5 m up, step 0.5 m, land.
* ``tether_catch``     as above, but the throttle sticks at 100 % for 2 s: the tether
                       catches the rig, the controller recovers and lands.
* ``translation_step`` free flight: hover at 10 m, 5 m sideways step, land at the new
                       spot. Reports rise time, overshoot and settling time (the numbers
                       to compare with the rig's logged response).
* ``free_hop``         free flight: climb to 50 m, move 15 m to a second pad, land.

The controller is the same waypoint-guidance + attitude-control pair as the lander's
``hop_test`` scenario (``plume.control``). These are demonstrations of the test
programme in docs/hop_rig.md on a *representative* vehicle, not predictions for a real rig.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from plume.config import VehicleSpec, WorldSpec
from plume.constants import G0
from plume.control.attitude import AttitudeController
from plume.control.guidance import WaypointGuidance, thrust_command
from plume.physics.sim import RocketSim
from plume.physics.tether import Tether, TetherSpec
from plume.recording import Recorder

CONTROL_DT = 0.05
TETHER_RATED_LOAD = 15_000.0  # N, working load limit of the default rope and anchor


def default_tether(vehicle: VehicleSpec, length: float = 3.0) -> TetherSpec:
    """Rope from a ground anchor at the pad centre to the hull base (engine end).

    Stiffness ~2e4 N/m is a few metres of 6-8 mm Dyneema or steel cable with a
    shock-absorbing link; damping is ~0.4 of critical for the rig's wet mass.
    """
    m = vehicle.mass.dry + vehicle.prop_initial + vehicle.rcs.gas
    k = 2.0e4
    return TetherSpec(
        anchor=(0.0, 0.0, 0.0),
        attach=(0.0, 0.0, 0.0),
        length=length,
        stiffness=k,
        damping=TetherSpec.critical_damping(k, m, 0.4),
        breaking_load=TETHER_RATED_LOAD,
    )


@dataclass
class RigFlight:
    """Small wrapper: simulator, controllers, recorder and the per-step control law."""

    sim: RocketSim
    rec: Recorder
    on_frame: Callable[[dict], None] | None = None
    tether: Tether | None = None
    max_tilt_deg: float = 15.0
    guid: WaypointGuidance = field(
        default_factory=lambda: WaypointGuidance(
            bandwidth=0.6, damping=1.0, accel_limit=3.0, integral_gain=0.0
        )
    )

    def __post_init__(self) -> None:
        self.att = AttitudeController(self.sim)
        self.steps = round(CONTROL_DT / self.sim.dt)
        self.cg0 = self.sim.state.cg_z
        self.base_height = self.sim.vehicle.legs.height  # hull base above the footpads
        self.override_throttle: float | None = None

    def com_target(self, x: float, y: float, agl: float) -> np.ndarray:
        """CG position for the footpads ``agl`` metres above flat ground at (x, y)."""
        return np.array([x, y, agl + self.base_height + self.sim.state.cg_z])

    def control(self, target: np.ndarray, target_vel: np.ndarray | None = None) -> float:
        sim = self.sim
        st = sim.state
        a_des = self.guid(st.com, st.vel_com, target, target_vel, dt=CONTROL_DT, up=st.up)
        g = sim.gravity.accel(st.com)
        p_amb = sim.atmosphere.at(st.altitude).pressure
        eng = sim.vehicle.engine
        t_max = sim.engine.max_thrust(p_amb)
        _, axis, f_des = thrust_command(
            a_des,
            g,
            st.mass,
            t_max,
            st.up,
            math.radians(self.max_tilt_deg),
            eng.throttle_min,
            eng.throttle_max,
        )
        # invert the engine's thrust law F(u) = u F_vac - p A_e (the back-pressure loss
        # is fixed, so thrust is not proportional to throttle on a sea-level rig)
        f_vac = sim.engine.mdot_max * eng.isp_vac * G0
        throttle = (f_des + p_amb * sim.engine.exit_area) / max(f_vac, 1e-9)
        throttle = min(max(throttle, eng.throttle_min), eng.throttle_max)
        if self.override_throttle is not None:
            throttle = self.override_throttle
        gimbal, rcs = self.att(st, axis, t_max * sim.engine.throttle)
        sim.set_controls(throttle, gimbal, rcs)
        return throttle

    def advance(self, phase: str) -> None:
        self.sim.step(self.steps)
        extra = None
        if self.tether is not None:
            extra = {"tether_tension": self.tether.tension, "tether_stretch": self.tether.stretch}
        frame = self.sim.frame(phase, extra)
        self.rec.record(frame)
        if self.on_frame:
            self.on_frame(frame)

    def descend(self, x: float, y: float, max_rate: float = 2.5) -> np.ndarray:
        """Vertical-descent target: sink rate tapering with height (soft touchdown)."""
        st = self.sim.state
        vz = -float(np.clip(0.4 * st.agl + 0.4, 0.5, max_rate))
        return np.array([x, y, st.com[2]]), np.array([0.0, 0.0, vz])

    def touched_down(self) -> bool:
        st = self.sim.state
        return st.legs_down >= 1 or st.body_contact


def _setup(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None,
    title: str,
    pads: list[tuple[str, np.ndarray]],
    target: np.ndarray | None,
    tether: TetherSpec | None = None,
    on_frame=None,
) -> RigFlight:
    sim = RocketSim(vehicle, world, seed=seed)
    sim.reset(pos=(0.0, 0.0, vehicle.legs.height + 0.01), seed=seed)
    teth = Tether(tether).install(sim) if tether is not None else None
    scene = {
        "frame": "flat",
        "ground": {"type": "plane"},
        "pads": [{"name": n, "pos": p.tolist(), "radius": 2.0} for n, p in pads],
        "target": {"pos": target.tolist(), "radius": 2.0} if target is not None else None,
    }
    extra = {}
    if tether is not None:
        extra["tether"] = {
            "anchor": list(tether.anchor),
            "attach": list(tether.attach),
            "length": tether.length,
            "stiffness": tether.stiffness,
            "damping": tether.damping,
        }
    rec = Recorder(
        sim.replay_meta(title, controller="waypoint-pd", seed=seed, scene=scene, **extra)
    )
    return RigFlight(sim=sim, rec=rec, on_frame=on_frame, tether=teth)


def _land_outcome(f: RigFlight, landed: bool, pad: np.ndarray, extra: dict) -> Recorder:
    sim = f.sim
    st = sim.state
    td = sim.touchdown
    err = float(np.linalg.norm(st.pos[:2] - pad[:2]))
    ok = (
        landed
        and not sim.ever_body_contact
        and td is not None
        and td.vertical_speed < sim.vehicle.legs.max_touchdown_speed
        and st.tilt < math.radians(10)
    )
    metrics = {
        "landing_error_m": err,
        "touchdown_speed_mps": td.vertical_speed if td else float("nan"),
        "fuel_used_kg": sim.prop_used,
        "rcs_gas_used_kg": sim.rcs_used,
        "flight_time_s": sim.t,
        "max_g": sim.max_g,
    }
    if f.tether is not None:
        metrics["max_tether_tension_n"] = f.tether.max_tension
    metrics.update(extra)
    reason = "landed" if ok else ("no touchdown" if not landed else "hard or tipped landing")
    if ok and f.tether is not None and f.tether.overloaded:
        ok, reason = False, "tether overloaded"
    f.rec.set_outcome(ok, reason, metrics)
    return f.rec


def _hover_profile(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None,
    on_frame,
    *,
    title: str,
    hover_agl: float,
    step: float,
    hold: float,
    tether_length: float,
    fault: tuple[float, float] | None,
) -> Recorder:
    tether = default_tether(vehicle, tether_length)
    pad = np.zeros(3)
    f = _setup(vehicle, world, seed, title, [("Rig pad", pad)], None, tether, on_frame)
    sim = f.sim
    phase = "ascent"
    f.rec.event(0.0, "phase", "Lift-off on tether")
    max_agl = 0.0
    landed = False
    t_land = None
    t_hover = None
    while sim.t < 90.0:
        st = sim.state
        if fault is not None:
            on = fault[0] <= sim.t < fault[0] + fault[1]
            if on and f.override_throttle is None:
                f.rec.event(sim.t, "fault", "Throttle stuck at 100 %")
            if not on and f.override_throttle is not None:
                f.rec.event(sim.t, "phase", "Fault cleared")
            f.override_throttle = vehicle.engine.throttle_max if on else None
        if phase == "ascent":
            target, tv = f.com_target(0.0, 0.0, hover_agl), None
            if st.agl > 0.85 * hover_agl:
                phase, t_hover = "hover", sim.t
                f.rec.event(sim.t, "phase", "Hover")
        elif phase == "hover":
            dt_h = sim.t - t_hover
            h = hover_agl + (step if hold / 3 <= dt_h < 2 * hold / 3 else 0.0)
            target, tv = f.com_target(0.0, 0.0, h), None
            if dt_h > hold:
                phase = "descent"
                f.rec.event(sim.t, "phase", "Descent")
        if phase == "descent":
            target, tv = f.descend(0.0, 0.0, max_rate=1.0)
            if f.touched_down():
                phase, landed, t_land = "landed", True, sim.t
                f.rec.event(sim.t, "touchdown", "Touchdown")
        if phase == "landed":
            sim.set_controls(0.0)
            if sim.t - t_land > 2.0:
                break
        else:
            f.control(target, tv)
        f.advance(phase)
        if sim.airborne:
            max_agl = max(max_agl, sim.state.agl)
    # height the tether allows: rope length minus the footpad depth below the attach point
    limit = tether_length - vehicle.legs.height
    return _land_outcome(
        f,
        landed,
        pad,
        {"max_footpad_agl_m": max_agl, "tether_limit_agl_m": limit},
    )


def tethered_hover(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    hover_agl: float = 1.5,
    tether_length: float = 3.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """Tethered hover: lift off, hover with a +0.5 m altitude step, land."""
    return _hover_profile(
        vehicle,
        world,
        seed,
        on_frame,
        title=f"Tethered hover: {vehicle.name}",
        hover_agl=hover_agl,
        step=0.5,
        hold=15.0,
        tether_length=tether_length,
        fault=None,
    )


def tether_catch(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    hover_agl: float = 1.0,
    tether_length: float = 3.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """Tethered hover with a 2 s stuck-throttle fault: the tether must arrest the climb."""
    rec = _hover_profile(
        vehicle,
        world,
        seed,
        on_frame,
        title=f"Tether catch (stuck throttle): {vehicle.name}",
        hover_agl=hover_agl,
        step=0.0,
        hold=14.0,
        tether_length=tether_length,
        fault=(8.0, 2.0),
    )
    m = rec.meta["outcome"]["metrics"]
    caught = m["max_footpad_agl_m"] < m["tether_limit_agl_m"] + 0.5
    m["caught_by_tether"] = bool(caught)
    if not caught:
        rec.set_outcome(False, "tether did not arrest the climb", m)
    return rec


def step_metrics(t: np.ndarray, y: np.ndarray, t0: float, y0: float, y1: float) -> dict:
    """Rise time (10-90 %), overshoot (%) and 5 % settling time of a step y0 -> y1 at t0."""
    m = t >= t0
    t, y = t[m] - t0, y[m]
    span = y1 - y0
    frac = (y - y0) / span
    i10 = np.nonzero(frac >= 0.1)[0]
    i90 = np.nonzero(frac >= 0.9)[0]
    rise = float(t[i90[0]] - t[i10[0]]) if len(i10) and len(i90) else float("nan")
    overshoot = max(float(frac.max()) - 1.0, 0.0) * 100.0
    outside = np.nonzero(np.abs(frac - 1.0) > 0.05)[0]
    settle = (
        float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else float("nan")
    )
    if not len(outside):
        settle = 0.0
    return {"rise_time_s": rise, "overshoot_pct": overshoot, "settling_time_s": settle}


def translation_step(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    hover_agl: float = 10.0,
    step: float = 5.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """Free-flight lateral step response at constant height, then land at the new spot."""
    pad_b = np.array([step, 0.0, 0.0])
    f = _setup(
        vehicle,
        world,
        seed,
        f"Translation step: {vehicle.name}",
        [("Pad A", np.zeros(3)), ("Pad B", pad_b)],
        pad_b,
        None,
        on_frame,
    )
    sim = f.sim
    phase = "ascent"
    f.rec.event(0.0, "phase", "Ascent")
    t_hover = t_step = t_land = None
    landed = False
    ts, xs = [], []
    while sim.t < 90.0:
        st = sim.state
        if phase == "ascent":
            target, tv = f.com_target(0.0, 0.0, hover_agl), None
            if abs(st.agl - hover_agl) < 0.3 and abs(st.vertical_speed) < 0.3:
                phase, t_hover = "hover", sim.t
                f.rec.event(sim.t, "phase", "Hover")
        elif phase == "hover":
            target, tv = f.com_target(0.0, 0.0, hover_agl), None
            if sim.t - t_hover > 5.0:
                phase, t_step = "step", sim.t
                f.rec.event(sim.t, "phase", f"Step {step:g} m east")
        elif phase == "step":
            target, tv = f.com_target(step, 0.0, hover_agl), None
            ts.append(sim.t)
            xs.append(float(st.com[0]))
            if sim.t - t_step > 15.0:
                phase = "descent"
                f.rec.event(sim.t, "phase", "Descent")
        if phase == "descent":
            target, tv = f.descend(step, 0.0)
            if f.touched_down():
                phase, landed, t_land = "landed", True, sim.t
                f.rec.event(sim.t, "touchdown", "Touchdown")
        if phase == "landed":
            sim.set_controls(0.0)
            if sim.t - t_land > 2.0:
                break
        else:
            f.control(target, tv)
        f.advance(phase)
    resp = (
        step_metrics(np.array(ts), np.array(xs), t_step, xs[0], step)
        if ts
        else {"rise_time_s": float("nan"), "overshoot_pct": float("nan")}
    )
    return _land_outcome(f, landed, pad_b, resp)


def free_hop(
    vehicle: VehicleSpec,
    world: WorldSpec,
    seed: int | None = 0,
    altitude: float = 50.0,
    distance: float = 15.0,
    on_frame: Callable[[dict], None] | None = None,
) -> Recorder:
    """50 m free hop: climb, move to a second pad, descend and land."""
    pad_b = np.array([distance, 0.0, 0.0])
    f = _setup(
        vehicle,
        world,
        seed,
        f"Free hop to {altitude:g} m: {vehicle.name}",
        [("Pad A", np.zeros(3)), ("Pad B", pad_b)],
        pad_b,
        None,
        on_frame,
    )
    f.guid = WaypointGuidance(bandwidth=0.5, damping=1.0, accel_limit=3.0, integral_gain=0.03)
    sim = f.sim
    phase = "ascent"
    f.rec.event(0.0, "phase", "Ascent")
    landed = False
    t_land = None
    apogee = 0.0
    while sim.t < 120.0:
        st = sim.state
        apogee = max(apogee, st.agl)
        if phase == "ascent":
            # move over while climbing once clear of the pad
            x = distance if st.agl > 0.5 * altitude else 0.0
            target, tv = f.com_target(x, 0.0, altitude), None
            if st.agl > 0.93 * altitude:
                phase = "translate"
                f.rec.event(sim.t, "phase", "Translate")
        elif phase == "translate":
            target, tv = f.com_target(distance, 0.0, altitude), None
            if np.linalg.norm(st.com[:2] - pad_b[:2]) < 1.0 and st.horizontal_speed < 0.5:
                phase = "descent"
                f.rec.event(sim.t, "phase", "Descent")
        if phase == "descent":
            target, tv = f.descend(distance, 0.0, max_rate=5.0)
            if f.touched_down():
                phase, landed, t_land = "landed", True, sim.t
                f.rec.event(sim.t, "touchdown", "Touchdown")
        if phase == "landed":
            sim.set_controls(0.0)
            if sim.t - t_land > 2.0:
                break
        else:
            f.control(target, tv)
        f.advance(phase)
    return _land_outcome(
        f,
        landed,
        pad_b,
        {"max_agl_m": apogee, "propellant_left_kg": sim.prop_mass},
    )


HOP_RIG_SCENARIOS = {
    "tethered_hover": tethered_hover,
    "tether_catch": tether_catch,
    "translation_step": translation_step,
    "free_hop": free_hop,
}
