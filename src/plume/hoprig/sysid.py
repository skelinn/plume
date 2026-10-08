"""System identification for the hop-test rig: test inputs, a rig-log format and the
estimators that turn rig logs into a calibrated vehicle file.

Workflow (docs/hop_rig.md):

1. ``MANOEUVRES`` lists which test excites which model parameter (``plume rig plan``).
2. On the rig, the flight software injects the inputs below (throttle doublets, gimbal
   chirps) during a tethered hover and logs the columns in ``RIG_LOG_COLUMNS``.
   ``fly_sysid`` produces the same log from the simulator (``plume rig sysid``) so the
   pipeline can be rehearsed - and checked - before real data exists.
3. ``calibrate_rig`` fits, from such a log:

   * engine: ``thrust_vac``, ``isp_vac``/``isp_sl`` and ``throttle_tau``. Thrust is
     measured as m a_z + tether tension (accelerometer x weighed mass), mass flow from the
     tank load cells; the model is F = u F_vac - p_amb A_e with u the first-order-lagged
     throttle, the same law as ``plume.physics.propulsion.Engine``;
   * gimbal actuator: first-order time constant (+ rate limit) and second-order natural
     frequency, damping and delay, from commanded vs measured nozzle angle;
   * pitch/yaw inertia: a scale on the model's inertia from the gimbal torque
     (arm x thrust x angle) and the gyro rate change, in short windows (integral form,
     so no differentiation of noisy gyro data is needed).

The estimators assume what a real rig log would provide; the mass model (dry mass, CG
and inertia vs propellant load, from weighing and CAD) comes from the vehicle file.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from plume.config import VehicleSpec, WorldSpec
from plume.constants import G0, P0
from plume.physics.massprops import MassModel

# ----------------------------------------------------------------------------- inputs


def doublet(t: float, t0: float, width: float, amp: float) -> float:
    """+amp for ``width`` seconds, then -amp for ``width`` seconds, from ``t0``."""
    if t0 <= t < t0 + width:
        return amp
    if t0 + width <= t < t0 + 2 * width:
        return -amp
    return 0.0


def multistep_3211(t: float, t0: float, unit: float, amp: float) -> float:
    """3-2-1-1 multistep (+, -, +, -) with steps of 3, 2, 1, 1 ``unit`` seconds."""
    edges = np.cumsum([0, 3, 2, 1, 1]) * unit + t0
    for k in range(4):
        if edges[k] <= t < edges[k + 1]:
            return amp if k % 2 == 0 else -amp
    return 0.0


def chirp(t: float, t0: float, duration: float, f0: float, f1: float, amp: float) -> float:
    """Logarithmic frequency sweep f0 -> f1 (Hz) with a 1 s cosine fade in and out."""
    tau = t - t0
    if tau < 0 or tau > duration:
        return 0.0
    k = (f1 / f0) ** (1.0 / duration)
    phase = 2 * math.pi * f0 * (k**tau - 1.0) / math.log(k)
    fade = min(1.0, tau, duration - tau)
    w = 0.5 - 0.5 * math.cos(math.pi * max(fade, 0.0))
    return amp * w * math.sin(phase)


# ----------------------------------------------------------------------------- test plan
@dataclass(frozen=True)
class Manoeuvre:
    name: str
    where: str  # ground / test stand / tethered / free flight
    inputs: str
    excites: str  # model parameters (vehicle YAML keys)
    measure: str
    notes: str = ""


MANOEUVRES: list[Manoeuvre] = [
    Manoeuvre(
        "Weighing and CG",
        "ground",
        "weigh on three scales, dry and with known propellant/water loads",
        "mass.dry, mass.dry_cg_z, tanks (capacity, z_bottom/z_top)",
        "scale readings, tank fill levels",
        "repeat after every hardware change",
    ),
    Manoeuvre(
        "Swing test (bifilar/trifilar pendulum)",
        "ground",
        "small free swings about each axis",
        "mass.dry_inertia",
        "swing period (stopwatch or IMU)",
        "I = m g r^2 T^2 / (4 pi^2 L) for a bifilar pendulum",
    ),
    Manoeuvre(
        "Static hot fire, throttle steps",
        "test stand",
        "throttle steps 30-100 % and back, 3-5 s each",
        "engine.thrust_vac, engine.isp_vac/isp_sl, engine.throttle_tau, throttle_min",
        "load cell thrust, tank load cells / flow meter, chamber pressure",
        "the most direct engine calibration; do it before any flight",
    ),
    Manoeuvre(
        "Gimbal bench sweep",
        "test stand",
        "chirp 0.2-5 Hz, +/-1 deg and +/-4 deg; steps; slow triangle",
        "engine.gimbal_tau (fast), gimbal_wn_hz, gimbal_zeta, gimbal_delay_s, "
        "gimbal_rate_deg_s, gimbal_backlash_deg",
        "commanded vs measured nozzle angle (LVDT / encoder)",
        "repeat under thrust: hinge friction and loads change the dynamics",
    ),
    Manoeuvre(
        "Tethered hover: throttle doublets",
        "tethered",
        "+/-0.10 throttle doublets (1 s) and a 3-2-1-1 on the hover throttle",
        "engine.thrust_vac, engine.throttle_tau, isp (installed)",
        "accelerometer, tether load cell, tank load cells, baro/laser altitude",
        "plume rig calibrate fits these from the log",
    ),
    Manoeuvre(
        "Tethered hover: gimbal chirps",
        "tethered",
        "+/-1 deg chirp 0.2-3 Hz on one gimbal axis at a time, RCS pitch/yaw off",
        "mass.dry_inertia (pitch/yaw), gimbal dynamics under thrust",
        "gyro rates, measured gimbal angles, throttle",
        "keep the rope slack; abort on attitude > 10 deg",
    ),
    Manoeuvre(
        "RCS pulse train",
        "tethered",
        "single-thruster pulses 20-200 ms, engine at hover or off on the stand",
        "rcs.thrust, rcs.min_on_time_s, rcs.valve_delay_s",
        "gyro (roll), valve commands, bottle pressure",
    ),
    Manoeuvre(
        "Translation step",
        "free flight (or long tether)",
        "5 m horizontal waypoint step at constant height",
        "closed-loop guidance + attitude response (rise time, overshoot, settling)",
        "GNSS/RTK position, attitude",
        "compare with `plume sim hop_rig --script translation_step`",
    ),
    Manoeuvre(
        "Leg drop test",
        "ground",
        "drop the rig from increasing heights onto the landing surface",
        "legs.max_touchdown_speed, leg stroke/stiffness",
        "accelerometer peak, leg stroke, high-speed video",
    ),
    Manoeuvre(
        "Free hop",
        "free flight",
        "the planned hop profile",
        "everything together: propellant used, landing accuracy, touchdown speed",
        "full log; predicted beforehand with plume sim",
        "the validation flight - record the prediction before flying",
    ),
]


# ----------------------------------------------------------------------------- rig log
RIG_LOG_COLUMNS: dict[str, str] = {
    "time_s": "time since logging started, s",
    "manoeuvre": "test-sequencer step label (hover, throttle_doublet, gimbal_chirp_x, ...)",
    "throttle_cmd": "commanded throttle, 0-1 (as sent to the valves)",
    "gimbal_cmd_x_deg": "commanded nozzle angle about body x, deg",
    "gimbal_cmd_y_deg": "commanded nozzle angle about body y, deg",
    "gimbal_x_deg": "measured nozzle angle about body x (LVDT/encoder), deg",
    "gimbal_y_deg": "measured nozzle angle about body y, deg",
    "gyro_x_dps": "body rate about x, deg/s",
    "gyro_y_dps": "body rate about y, deg/s",
    "gyro_z_dps": "body rate about z (roll), deg/s",
    "accel_x_mps2": "specific force along body x at the CG, m/s^2",
    "accel_y_mps2": "specific force along body y at the CG, m/s^2",
    "accel_z_mps2": "specific force along body z (reads +9.8 at rest), m/s^2",
    "prop_mass_kg": "main propellant on board (tank load cells or integrated flow), kg",
    "tether_tension_n": "tether load cell, N (0 when slack or untethered)",
    "rcs_cmd_x": "RCS torque command about x, -1..1 of capability",
    "rcs_cmd_y": "RCS torque command about y",
    "rcs_cmd_z": "RCS torque command about z",
    "height_m": "footpad height above the pad (laser / baro), m",
}

SENSOR_NOISE = {
    "gimbal_deg": 0.02,
    "gyro_dps": 0.05,
    "accel_mps2": 0.05,
    "prop_kg": 0.1,
    "tension_n": 5.0,
    "height_m": 0.02,
}


def fly_sysid(
    vehicle: VehicleSpec,
    world: WorldSpec | None = None,
    seed: int | None = 0,
    noise: bool = True,
    log_hz: float = 100.0,
    hover_agl: float = 1.5,
    tether_length: float = 4.0,
    on_frame: Callable[[dict], None] | None = None,
):
    """Tethered system-identification hover. Returns ``(recorder, log DataFrame)``.

    Sequence after the rig settles in the hover: throttle doublets (+/-0.10, 1 s), a
    3-2-1-1 throttle multistep, a gimbal chirp on x then on y (+/-1 deg, 0.3-3 Hz, 8 s,
    RCS pitch/yaw off), then a landing. Sensor noise (``SENSOR_NOISE``) is added when
    ``noise`` is set.
    """
    from plume.hoprig.scenarios import CONTROL_DT, _setup, default_tether

    world = world or WorldSpec()
    tether = default_tether(vehicle, tether_length)
    f = _setup(
        vehicle,
        world,
        seed,
        f"System identification hover: {vehicle.name}",
        [("Rig pad", np.zeros(3))],
        None,
        tether,
        on_frame,
    )
    sim = f.sim
    rng = np.random.default_rng(seed)
    log_every = max(1, round(1.0 / (log_hz * sim.dt)))
    ctrl_every = round(CONTROL_DT / sim.dt)
    gmax = math.radians(vehicle.engine.gimbal_max_deg)
    rows: list[dict] = []
    phase = "ascent"
    t_h = t_land = None
    seq = [
        ("throttle_doublet", 3.0, 5.0),
        ("throttle_3211", 8.0, 14.0),
        ("gimbal_chirp_x", 15.0, 23.0),
        ("gimbal_chirp_y", 25.0, 33.0),
    ]
    end_hover = 35.0
    label = "pad"
    v_prev = sim.state.vel_com.copy()
    g_vec = sim.gravity.accel(sim.state.com)
    n = 0
    throttle = 0.0
    f.rec.event(0.0, "phase", "Lift-off on tether")
    while sim.t < 80.0:
        if n % ctrl_every == 0:
            st = sim.state
            if phase == "ascent" and st.agl > 0.9 * hover_agl and abs(st.vertical_speed) < 0.2:
                phase, t_h = "hover", sim.t
                f.rec.event(sim.t, "phase", "Hover: identification sequence")
            if phase == "hover" and sim.t - t_h > end_hover:
                phase = "descent"
                f.rec.event(sim.t, "phase", "Descent")
            if phase == "descent" and f.touched_down():
                phase, t_land = "landed", sim.t
                f.rec.event(sim.t, "touchdown", "Touchdown")
            label = {"ascent": "ascent", "descent": "descent", "landed": "landed"}.get(
                phase, "hover"
            )
            du, dg = 0.0, np.zeros(2)
            if phase == "hover":
                tau = sim.t - t_h
                for name, a, b in seq:
                    if a <= tau < b:
                        label = name
                if label == "throttle_doublet":
                    du = doublet(tau, 3.0, 1.0, 0.10)
                elif label == "throttle_3211":
                    du = multistep_3211(tau, 8.0, 0.5, 0.08)
                elif label == "gimbal_chirp_x":
                    dg[0] = chirp(tau, 15.0, 8.0, 0.3, 3.0, math.radians(1.0))
                elif label == "gimbal_chirp_y":
                    dg[1] = chirp(tau, 25.0, 8.0, 0.3, 3.0, math.radians(1.0))
            if phase == "landed":
                sim.set_controls(0.0)
                throttle = 0.0
            else:
                if phase == "descent":
                    target, tv = f.descend(0.0, 0.0, max_rate=1.0)
                else:
                    target, tv = f.com_target(0.0, 0.0, hover_agl), None
                throttle = f.control(target, tv)
                if du or dg.any():
                    eng = sim.engine
                    gim = np.clip(eng.gimbal_cmd / gmax + dg / gmax, -1, 1)
                    rcs = sim.rcs.cmd.copy()
                    if dg.any():
                        rcs[:2] = 0.0  # identification axis: gimbal only
                    throttle = float(np.clip(throttle + du, 0.0, vehicle.engine.throttle_max))
                    sim.set_controls(throttle, gim, rcs)
            if phase == "landed" and sim.t - t_land > 1.0:
                break
        if n % log_every == 0:
            st = sim.state
            dt_log = log_every * sim.dt
            a_w = (st.vel_com - v_prev) / dt_log - g_vec
            v_prev = st.vel_com.copy()
            a_b = st.rot.T @ a_w
            eng = sim.engine
            rows.append(
                {
                    "time_s": sim.t,
                    "manoeuvre": label,
                    "throttle_cmd": eng.throttle_cmd,
                    "gimbal_cmd_x_deg": math.degrees(eng.gimbal_cmd[0]),
                    "gimbal_cmd_y_deg": math.degrees(eng.gimbal_cmd[1]),
                    "gimbal_x_deg": math.degrees(eng.gimbal[0]),
                    "gimbal_y_deg": math.degrees(eng.gimbal[1]),
                    "gyro_x_dps": math.degrees(st.omega[0]),
                    "gyro_y_dps": math.degrees(st.omega[1]),
                    "gyro_z_dps": math.degrees(st.omega[2]),
                    "accel_x_mps2": a_b[0],
                    "accel_y_mps2": a_b[1],
                    "accel_z_mps2": a_b[2],
                    "prop_mass_kg": st.prop_mass,
                    "tether_tension_n": f.tether.tension,
                    "rcs_cmd_x": sim.rcs.cmd[0],
                    "rcs_cmd_y": sim.rcs.cmd[1],
                    "rcs_cmd_z": sim.rcs.cmd[2],
                    "height_m": max(st.agl, 0.0),
                }
            )
        sim.step(1)
        n += 1
        if n % ctrl_every == 0:
            frame = sim.frame(phase, {"tether_tension": f.tether.tension})
            f.rec.record(frame)
            if on_frame:
                on_frame(frame)
    df = pd.DataFrame(rows, columns=list(RIG_LOG_COLUMNS))
    if noise:
        s = SENSOR_NOISE
        k = len(df)
        for c in ("gimbal_x_deg", "gimbal_y_deg"):
            df[c] += rng.normal(0, s["gimbal_deg"], k)
        for c in ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps"):
            df[c] += rng.normal(0, s["gyro_dps"], k)
        for c in ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2"):
            df[c] += rng.normal(0, s["accel_mps2"], k)
        df["prop_mass_kg"] += rng.normal(0, s["prop_kg"], k)
        df["tether_tension_n"] = np.maximum(
            df["tether_tension_n"] + rng.normal(0, s["tension_n"], k), 0.0
        )
        df["height_m"] += rng.normal(0, s["height_m"], k)
    ok = phase == "landed" and not sim.ever_body_contact
    f.rec.set_outcome(
        ok,
        "landed" if ok else "failed",
        {"flight_time_s": sim.t, "fuel_used_kg": sim.prop_used, "log_rows": len(df)},
    )
    return f.rec, df


def read_rig_log(path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    missing = [
        c for c in ("time_s", "throttle_cmd", "accel_z_mps2", "prop_mass_kg") if c not in df.columns
    ]
    if missing:
        raise KeyError(f"rig log is missing required columns {missing}; see RIG_LOG_COLUMNS")
    return df


# ----------------------------------------------------------------------------- estimators
def lagged_throttle(
    t: np.ndarray, cmd: np.ndarray, tau: float, t_min: float, t_max: float
) -> np.ndarray:
    """Delivered throttle: the Engine's on/off logic and first-order lag on the log grid."""
    u = np.zeros(len(t))
    x = 0.0
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        c = cmd[k - 1]
        on = c >= 0.5 * t_min and c > 1e-6
        target = min(max(c, t_min), t_max) if on else 0.0
        a = math.exp(-dt / tau) if tau > 0 else 0.0
        x = target + (x - target) * a
        if not on and x < 0.02:
            x = 0.0
        u[k] = x
    return u


@dataclass
class EngineFit:
    thrust_vac: float
    exit_area: float
    mdot_max: float
    throttle_tau: float
    isp_vac: float
    isp_sl: float
    rms_thrust_n: float
    samples: int


def fit_engine(
    df: pd.DataFrame,
    vehicle: VehicleSpec,
    p_amb: float = P0,
    min_height: float = 0.3,
) -> EngineFit:
    """Thrust law F = u F_vac - p A_e, throttle lag and mass flow from a hover log.

    Measured thrust: (m a_z + tether tension) / cos(nozzle angle), with m the weighed dry
    mass (+ RCS gas) plus the logged propellant. Only airborne samples are used (the
    accelerometer reading on the pad includes the leg reactions)."""
    t = df["time_s"].to_numpy(float)
    cmd = df["throttle_cmd"].to_numpy(float)
    prop = df["prop_mass_kg"].to_numpy(float)
    m = vehicle.mass.dry + vehicle.cargo.mass + vehicle.rcs.gas + prop
    gx = np.radians(df.get("gimbal_x_deg", pd.Series(0.0, index=df.index)).to_numpy(float))
    gy = np.radians(df.get("gimbal_y_deg", pd.Series(0.0, index=df.index)).to_numpy(float))
    tension = df.get("tether_tension_n", pd.Series(0.0, index=df.index)).to_numpy(float)
    f_meas = (m * df["accel_z_mps2"].to_numpy(float) + tension) / (np.cos(gx) * np.cos(gy))
    height = df.get("height_m", pd.Series(np.inf, index=df.index)).to_numpy(float)
    e = vehicle.engine
    sel = (height > min_height) & (cmd >= e.throttle_min * 0.5)
    if sel.sum() < 50:
        raise ValueError("not enough airborne powered samples to fit the engine")

    def solve(tau: float):
        u = lagged_throttle(t, cmd, tau, e.throttle_min, e.throttle_max)
        A = np.column_stack([u[sel], -np.ones(sel.sum())])
        coef, *_ = np.linalg.lstsq(A, f_meas[sel], rcond=None)
        r = A @ coef - f_meas[sel]
        return float(r @ r), coef, u

    res = minimize_scalar(lambda x: solve(x)[0], bounds=(0.005, 1.5), method="bounded")
    tau = float(res.x)
    sse, (f_vac, pa), u = solve(tau)
    # mass flow: prop(t) = P0 - mdot_max * integral(u dt)  (engine running)
    iu = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(t))])
    A = np.column_stack([np.ones(len(t)), -iu])
    (_p0, mdot_max), *_ = np.linalg.lstsq(A, prop, rcond=None)
    exit_area = max(pa, 0.0) / p_amb
    isp_vac = f_vac / (mdot_max * G0)
    isp_sl = (f_vac - P0 * exit_area) / (mdot_max * G0)
    return EngineFit(
        thrust_vac=float(f_vac),
        exit_area=float(exit_area),
        mdot_max=float(mdot_max),
        throttle_tau=tau,
        isp_vac=float(isp_vac),
        isp_sl=float(isp_sl),
        rms_thrust_n=math.sqrt(sse / sel.sum()),
        samples=int(sel.sum()),
    )


def _first_order(t, cmd, tau, rate, h=0.005):
    g = np.zeros(len(t))
    x = cmd[0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        n = max(1, math.ceil(dt / h - 1e-9))
        for _ in range(n):
            hh = dt / n
            r = np.clip((cmd[k - 1] - x) / max(tau, hh), -rate, rate)
            x += r * hh
        g[k] = x
    return g


def _second_order(t, cmd, wn, zeta, delay):
    """Second-order actuator driven by the command delayed by ``delay``.

    Exact zero-order-hold discretisation on the (uniform) log grid, run as an IIR filter;
    a fractional delay is applied by linear interpolation of the command sequence, which
    keeps the fit smooth in the delay."""
    from scipy.signal import cont2discrete, lfilter

    dt = float(np.median(np.diff(t)))
    num, den, _ = cont2discrete(([wn * wn], [1.0, 2 * zeta * wn, wn * wn]), dt, method="zoh")
    k = np.arange(len(cmd), dtype=float)
    u = np.interp(k - delay / dt, k, cmd)
    y0 = cmd[0]
    return lfilter(np.ravel(num), den, u - y0) + y0


@dataclass
class GimbalFit:
    tau: float
    rate_deg_s: float
    rms_first_deg: float
    wn_hz: float
    zeta: float
    delay_s: float
    rms_second_deg: float

    @property
    def best(self) -> str:
        return "first" if self.rms_first_deg <= self.rms_second_deg else "second"


def fit_gimbal(df: pd.DataFrame, axis: str = "x", max_rows: int = 3000) -> GimbalFit:
    """Fit first-order (+ rate limit) and second-order (+ delay) actuator models to the
    commanded vs measured nozzle angle. Uses the chirp window when labelled."""
    sub = df
    if "manoeuvre" in df.columns:
        w = df["manoeuvre"].astype(str).str.startswith(f"gimbal_chirp_{axis}")
        if w.sum() > 100:
            sub = df[w]
    sub = sub.iloc[:max_rows]
    t = sub["time_s"].to_numpy(float)
    c = sub[f"gimbal_cmd_{axis}_deg"].to_numpy(float)
    y = sub[f"gimbal_{axis}_deg"].to_numpy(float)
    if np.ptp(c) < 0.1:
        raise ValueError(f"gimbal axis {axis}: commands do not vary enough to identify")

    def r1(p):
        return _first_order(t, c, p[0], p[1]) - y

    s1 = least_squares(r1, [0.05, 60.0], bounds=([0.002, 1.0], [1.0, 500.0]), diff_step=1e-3)

    def r2(p):
        return _second_order(t, c, 2 * math.pi * p[0], p[1], p[2]) - y

    s2 = least_squares(
        r2, [5.0, 0.7, 0.01], bounds=([0.3, 0.1, 0.0], [40.0, 3.0, 0.2]), x_scale=[1.0, 0.1, 0.01]
    )
    delay_fit = float(s2.x[2])
    rms = lambda r: float(np.sqrt(np.mean(r**2)))  # noqa: E731
    return GimbalFit(
        tau=float(s1.x[0]),
        rate_deg_s=float(s1.x[1]),
        rms_first_deg=rms(s1.fun),
        wn_hz=float(s2.x[0]),
        zeta=float(s2.x[1]),
        delay_s=delay_fit,
        rms_second_deg=rms(s2.fun),
    )


@dataclass
class InertiaFit:
    scale: dict[str, float]  # measured / model pitch-yaw inertia
    sigma: dict[str, float]
    windows: dict[str, int]


def fit_inertia(
    df: pd.DataFrame, vehicle: VehicleSpec, engine: EngineFit, window_s: float = 0.3
) -> InertiaFit:
    """Scale on the model's pitch/yaw inertia from the gimbal torque and gyro rates.

    Over each short window, the rate change equals the integral of torque / inertia:
    delta(omega) = s * integral(tau_model / I_model) dt, solved for s by least squares.
    The torque is the gimbal's (arm x thrust x nozzle angle) plus the RCS command times
    its capability; windows with the tether taut are skipped."""
    from plume.physics.propulsion import RCS

    t = df["time_s"].to_numpy(float)
    prop = df["prop_mass_kg"].to_numpy(float)
    mm = MassModel(vehicle)
    cap = vehicle.prop_capacity
    rcs_gas = vehicle.rcs.gas
    loads = np.linspace(0.0, cap, 12)
    caps = [tk.capacity for tk in vehicle.tanks]
    props = [mm.evaluate([c * x / cap for c in caps], rcs_gas) for x in loads]
    cg_tab = np.array([p.cg_z for p in props])
    ixx_tab = np.array([p.inertia[0] for p in props])
    iyy_tab = np.array([p.inertia[1] for p in props])
    pc = np.clip(prop, 0.0, cap)
    cg = np.interp(pc, loads, cg_tab)
    inertia = {"x": np.interp(pc, loads, ixx_tab), "y": np.interp(pc, loads, iyy_tab)}
    e = vehicle.engine
    u = lagged_throttle(
        t, df["throttle_cmd"].to_numpy(float), engine.throttle_tau, e.throttle_min, e.throttle_max
    )
    thrust = np.maximum(u * engine.thrust_vac - P0 * engine.exit_area, 0.0)
    thrust[u <= 0] = 0.0
    a = np.radians(df["gimbal_x_deg"].to_numpy(float))
    b = np.radians(df["gimbal_y_deg"].to_numpy(float))
    rz = e.gimbal_z - cg
    tau_g = {"x": rz * thrust * np.sin(a), "y": rz * thrust * np.cos(a) * np.sin(b)}
    rcs_cap = (
        RCS(vehicle.rcs, vehicle, float(cg_tab[-1])).torque_cap
        if vehicle.rcs.enabled
        else np.zeros(3)
    )
    tension = df.get("tether_tension_n", pd.Series(0.0, index=df.index)).to_numpy(float)
    height = df.get("height_m", pd.Series(np.inf, index=df.index)).to_numpy(float)
    scale, sigma, nwin = {}, {}, {}
    dt = float(np.median(np.diff(t)))
    step = max(2, round(window_s / dt))
    for k, ax in enumerate(("x", "y")):
        rcs_t = df.get(f"rcs_cmd_{ax}", pd.Series(0.0, index=df.index)).to_numpy(float) * rcs_cap[k]
        accel = (tau_g[ax] + rcs_t) / inertia[ax]
        w = np.radians(df[f"gyro_{ax}_dps"].to_numpy(float))
        use = (height > 0.3) & (tension < 50.0) & (thrust > 0)
        if "manoeuvre" in df.columns:
            lab = df["manoeuvre"].astype(str).to_numpy()
            chirp_w = lab == f"gimbal_chirp_{ax}"
            if chirp_w.sum() > 3 * step:
                use &= chirp_w
        X, Y = [], []
        idx = np.nonzero(use)[0]
        for i0 in range(0, len(idx) - step, step // 2):
            seg = idx[i0 : i0 + step]
            if seg[-1] - seg[0] != step - 1:
                continue  # not contiguous
            integral = float(np.trapezoid(accel[seg], t[seg]))
            X.append(integral)
            Y.append(float(w[seg[-1]] - w[seg[0]]))
        X, Y = np.array(X), np.array(Y)
        if len(X) < 5 or float(X @ X) <= 0:
            scale[ax], sigma[ax], nwin[ax] = float("nan"), float("nan"), len(X)
            continue
        s = float(X @ Y / (X @ X))  # = I_model / I_true
        resid = Y - s * X
        var_s = float(resid @ resid) / max(len(X) - 1, 1) / float(X @ X)
        scale[ax] = 1.0 / s
        sigma[ax] = math.sqrt(var_s) / (s * s)
        nwin[ax] = len(X)
    return InertiaFit(scale=scale, sigma=sigma, windows=nwin)


@dataclass
class RigCalibration:
    engine: EngineFit
    gimbal: dict[str, GimbalFit]
    inertia: InertiaFit
    vehicle: VehicleSpec
    rows: list[tuple[str, str, str]] = field(default_factory=list)  # (parameter, model, fit)


def calibrate_rig(df: pd.DataFrame, vehicle: VehicleSpec, p_amb: float = P0) -> RigCalibration:
    """Run all estimators and return a calibrated copy of the vehicle."""
    ef = fit_engine(df, vehicle, p_amb)
    gim = {}
    for ax in ("x", "y"):
        try:
            gim[ax] = fit_gimbal(df, ax)
        except (ValueError, KeyError):
            pass
    inf = fit_inertia(df, vehicle, ef)
    e = vehicle.engine
    rows = [
        ("engine.thrust_vac (N)", f"{e.thrust_vac:.0f}", f"{ef.thrust_vac:.0f}"),
        ("engine.isp_vac (s)", f"{e.isp_vac:.1f}", f"{ef.isp_vac:.1f}"),
        ("engine.isp_sl (s)", f"{e.isp_sea_level:.1f}", f"{ef.isp_sl:.1f}"),
        ("engine.throttle_tau (s)", f"{e.throttle_tau:.3f}", f"{ef.throttle_tau:.3f}"),
        ("thrust residual rms (N)", "", f"{ef.rms_thrust_n:.1f}"),
    ]
    cal = vehicle.model_copy(deep=True)
    cal.engine.thrust_vac = round(ef.thrust_vac, 1)
    cal.engine.isp_vac = round(ef.isp_vac, 2)
    cal.engine.isp_sl = round(min(ef.isp_sl, ef.isp_vac), 2)
    cal.engine.throttle_tau = round(ef.throttle_tau, 4)
    if gim:
        gf = list(gim.values())
        tau = float(np.mean([g.tau for g in gf]))
        wn = float(np.mean([g.wn_hz for g in gf]))
        zeta = float(np.mean([g.zeta for g in gf]))
        delay = float(np.mean([g.delay_s for g in gf]))
        rows += [
            ("engine.gimbal_tau (s)", f"{e.gimbal_tau:.3f}", f"{tau:.3f}"),
            ("engine.gimbal_wn_hz", f"{e.gimbal_wn_hz:.2f}", f"{wn:.2f}"),
            ("engine.gimbal_zeta", f"{e.gimbal_zeta:.2f}", f"{zeta:.2f}"),
            ("engine.gimbal_delay_s", f"{e.gimbal_delay_s:.3f}", f"{delay:.3f}"),
            (
                "gimbal fit rms 1st / 2nd order (deg)",
                "",
                " / ".join(f"{g.rms_first_deg:.3f} / {g.rms_second_deg:.3f}" for g in gf),
            ),
        ]
        cal.engine.gimbal_tau = round(tau, 4)
        # the second-order model (high fidelity) is only updated when the data show
        # second-order behaviour: a clearly better fit, not pinned at a bound
        second = all(
            g.rms_second_deg < 0.9 * g.rms_first_deg and g.zeta < 2.9 and g.wn_hz < 39.0 for g in gf
        )
        if second:
            cal.engine.gimbal_wn_hz = round(wn, 3)
            cal.engine.gimbal_zeta = round(zeta, 3)
            cal.engine.gimbal_delay_s = round(delay, 4)
        rows.append(
            (
                "actuator model updated",
                "",
                "first + second order" if second else "first order only (no 2nd-order signature)",
            )
        )
    if vehicle.mass.dry_inertia is not None and all(
        math.isfinite(inf.scale.get(a, float("nan"))) for a in ("x", "y")
    ):
        ix, iy, iz = vehicle.mass.dry_inertia
        cal.mass.dry_inertia = [
            round(ix * inf.scale["x"], 2),
            round(iy * inf.scale["y"], 2),
            iz,
        ]
    for ax in ("x", "y"):
        s = inf.scale.get(ax, float("nan"))
        rows.append(
            (
                f"pitch/yaw inertia scale ({ax})",
                "1.000",
                f"{s:.3f} +/- {inf.sigma.get(ax, float('nan')):.3f}"
                f" ({inf.windows.get(ax, 0)} windows)",
            )
        )
    cal = VehicleSpec.model_validate(cal.model_dump())
    cal.name = f"{vehicle.name}_calibrated"
    return RigCalibration(engine=ef, gimbal=gim, inertia=inf, vehicle=cal, rows=rows)
