"""Simulate the flight a log came from, compare it with the log, plot and export replays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume.config import VehicleSpec, WorldSpec
from plume.constants import G0
from plume.flightdata.importer import FlightLog
from plume.physics.pointmass import PointMassSim
from plume.recording import Recorder


@dataclass
class SimTrace:
    label: str
    t: np.ndarray
    altitude: np.ndarray  # m above the launch point
    velocity: np.ndarray  # vertical, m/s
    accel: np.ndarray  # axial specific force, m/s^2 (comparable to the log)
    horizontal: np.ndarray | None = None  # (n, 2) east/north, m

    @property
    def apogee(self) -> float:
        return float(self.altitude.max())

    @property
    def t_apogee(self) -> float:
        return float(self.t[int(np.argmax(self.altitude))])


def simulate(
    vehicle: VehicleSpec,
    rail_length: float = 1.5,
    rail_tilt_deg: float = 0.0,
    t_end: float = 400.0,
    dt: float = 0.01,
    stop_after_apogee: float | None = None,
    thrust_curve: np.ndarray | None = None,
    cd_scale: float | None = None,
    chute_scale: float = 1.0,
    t_offset: float = 0.0,
    world: WorldSpec | None = None,
    label: str = "sim",
) -> SimTrace:
    """3-DOF flight from a launch rail (t = 0 at ignition, shifted by ``t_offset``)."""
    pm = PointMassSim(vehicle, world or WorldSpec(), thrust_curve=thrust_curve, cd_scale=cd_scale)
    pm.chute_scale = chute_scale
    tilt = math.radians(rail_tilt_deg)
    rail = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    r0 = np.array([0.0, 0.0, vehicle.mass.dry_cg_z])
    state = {"apogee_t": None}

    def stop(t, r, v, prop):
        if stop_after_apogee is None:
            return False
        if state["apogee_t"] is None and t > 1.0 and v[2] < 0:
            state["apogee_t"] = t
        return state["apogee_t"] is not None and t > state["apogee_t"] + stop_after_apogee

    def dt_fn(t, r, v):
        # fine steps for the boost, coarser under the parachute
        return dt if (state["apogee_t"] is None or t < state["apogee_t"] + 3) else max(dt, 0.05)

    traj = pm.run(
        r0,
        np.zeros(3),
        t_end=t_end,
        dt=dt,
        rail_length=rail_length,
        rail_dir=rail,
        stop_fn=stop,
        dt_fn=dt_fn,
    )
    v = traj.vel
    speed = np.linalg.norm(v, axis=1)
    vhat = np.where(speed[:, None] > 1e-3, v / np.maximum(speed[:, None], 1e-9), rail[None, :])
    axial = np.einsum("ij,ij->i", traj.accel, vhat)
    # before liftoff and under the parachute the body is not along the velocity
    axial = np.where(speed < 1e-3, G0, axial)
    return SimTrace(
        label=label,
        t=traj.t + t_offset,
        altitude=traj.pos[:, 2] - r0[2],
        velocity=v[:, 2],
        accel=axial,
        horizontal=traj.pos[:, :2],
    )


def metrics(log: FlightLog, trace: SimTrace) -> dict[str, float]:
    t_end = log.t_apogee + 1.0
    m = log.window(0.0, t_end)
    alt_sim = np.interp(log.t[m], trace.t, trace.altitude)
    out = {
        "apogee_real_m": log.apogee,
        "apogee_sim_m": trace.apogee,
        "apogee_error_m": trace.apogee - log.apogee,
        "t_apogee_real_s": log.t_apogee,
        "t_apogee_sim_s": trace.t_apogee,
        "rms_altitude_ascent_m": float(np.sqrt(np.mean((alt_sim - log.altitude[m]) ** 2))),
        "max_velocity_real_mps": float(log.velocity.max()),
        "max_velocity_sim_mps": float(trace.velocity.max()),
    }
    if log.accel is not None:
        mb = log.window(0.05, log.t_apogee)
        acc_sim = np.interp(log.t[mb], trace.t, trace.accel)
        out["rms_accel_ascent_g"] = float(np.sqrt(np.mean((acc_sim - log.accel[mb]) ** 2)) / G0)
    return out


def plot_comparison(
    log: FlightLog, traces: list[SimTrace], path: str | Path, title: str = ""
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#e8743b", "#19a979", "#945ecf", "#13a4b4"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    axes[1].sharex(axes[0])
    t_max = min(log.t[-1], max(tr.t[-1] for tr in traces))
    real_kw = {"color": "#1f3a5f", "lw": 1.4, "label": "real (log)"}
    axes[0].plot(log.t, log.altitude, **real_kw)
    axes[1].plot(log.t, log.velocity, **real_kw)
    if log.accel is not None:
        axes[2].plot(log.t, log.accel / G0, **{**real_kw, "lw": 0.8})
    for tr, c in zip(traces, colors, strict=False):
        axes[0].plot(tr.t, tr.altitude, color=c, lw=1.6, ls="--", label=tr.label)
        axes[1].plot(tr.t, tr.velocity, color=c, lw=1.6, ls="--", label=tr.label)
        asc = tr.t <= tr.t_apogee
        axes[2].plot(tr.t[asc], tr.accel[asc] / G0, color=c, lw=1.6, ls="--", label=tr.label)
    axes[0].set_ylabel("altitude (m)")
    axes[1].set_ylabel("vertical velocity (m/s)")
    axes[2].set_ylabel("axial accel (g)")
    axes[1].set_xlabel("time since liftoff (s)")
    axes[2].set_xlabel("time since liftoff (s), boost and coast")
    axes[2].set_xlim(-0.5, min(log.t_apogee + 4, t_max))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[0].set_xlim(-1, t_max)
    axes[1].set_xlim(-1, t_max)
    fig.suptitle(title or "Real vs simulated flight", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- replays
def _quat_from_axis(axis: np.ndarray) -> list[float]:
    from plume.physics.sim import quat_from_z_axis

    return quat_from_z_axis(axis).tolist()


def _vehicle_meta(vehicle: VehicleSpec) -> dict:
    return {
        "name": vehicle.name,
        "length": vehicle.geometry.length,
        "diameter": vehicle.geometry.diameter,
        "nose_length": vehicle.geometry.nose_length,
        "legs": {"count": 0, "span": 0.0, "height": 0.0},
        "engine": {"nozzle_radius": vehicle.engine.nozzle_radius, "gimbal_z": 0.0},
        "dry_mass": vehicle.mass.dry,
        "prop_mass_initial": vehicle.prop_initial,
    }


def _scene() -> dict:
    return {
        "frame": "flat",
        "ground": {"type": "plane"},
        "pads": [{"name": "Rail", "pos": [0.0, 0.0, 0.0], "radius": 1.5}],
        "target": None,
    }


def log_to_replay(log: FlightLog, vehicle: VehicleSpec, title: str = "Real flight") -> dict:
    """Replay of a logged flight. Attitude is integrated from the gyro when present."""
    rec = Recorder(
        {"title": title, "source": "real", "vehicle": _vehicle_meta(vehicle), "scene": _scene()},
        every=2,
    )
    q = np.array([1.0, 0.0, 0.0, 0.0])
    east = log.east if log.east is not None else np.zeros_like(log.t)
    north = log.north if log.north is not None else np.zeros_like(log.t)
    burnout = log.burnout_time or 0.0
    for k, t in enumerate(log.t):
        if log.gyro is not None and k and t > 0:
            w = log.gyro[k] * (log.t[k] - log.t[k - 1])
            ang = float(np.linalg.norm(w))
            if ang > 1e-12:
                dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * w / ang)])
                w0, x0, y0, z0 = q
                w1, x1, y1, z1 = dq
                q = np.array(
                    [
                        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
                    ]
                )
                q /= np.linalg.norm(q)
        acc = log.accel[k] if log.accel is not None else G0
        rec.record(
            {
                "t": t,
                "pos": [east[k], north[k], max(log.altitude[k], 0.0)],
                "quat": q.tolist(),
                "vel": [0.0, 0.0, log.velocity[k]],
                "alt": max(log.altitude[k], 0.0),
                "throttle": 1.0 if 0 <= t <= burnout else 0.0,
                "g_load": abs(acc) / G0,
                "phase": "boost"
                if 0 <= t <= burnout
                else ("coast" if t <= log.t_apogee else "descent"),
            }
        )
    rec.event(0.0, "ignition", "Liftoff")
    if burnout:
        rec.event(burnout, "cutoff", "Burnout")
    rec.event(log.t_apogee, "phase", "Apogee")
    return rec.to_dict()


def trace_to_replay(trace: SimTrace, vehicle: VehicleSpec, title: str) -> dict:
    rec = Recorder(
        {"title": title, "source": "sim", "vehicle": _vehicle_meta(vehicle), "scene": _scene()},
        every=1,
    )
    t_apo = trace.t_apogee
    step = max(1, len(trace.t) // 6000)
    hz = trace.horizontal if trace.horizontal is not None else np.zeros((len(trace.t), 2))
    for k in range(0, len(trace.t), step):
        t = trace.t[k]
        if k + 1 < len(trace.t):
            d = np.array(
                [
                    hz[k + 1, 0] - hz[k, 0],
                    hz[k + 1, 1] - hz[k, 1],
                    trace.altitude[k + 1] - trace.altitude[k],
                ]
            )
        else:
            d = np.array([0.0, 0.0, 1.0])
        axis = d if t <= t_apo and np.linalg.norm(d) > 1e-6 else np.array([0.0, 0.0, 1.0])
        rec.record(
            {
                "t": t,
                "pos": [hz[k, 0], hz[k, 1], max(trace.altitude[k], 0.0)],
                "quat": _quat_from_axis(axis),
                "vel": [0.0, 0.0, trace.velocity[k]],
                "alt": max(trace.altitude[k], 0.0),
                "throttle": 1.0 if trace.accel[k] > 0 and t <= t_apo else 0.0,
                "g_load": abs(trace.accel[k]) / G0,
                "phase": "boost"
                if trace.accel[k] > 0 and t <= t_apo
                else ("coast" if t <= t_apo else "descent"),
            }
        )
    rec.event(t_apo, "phase", "Apogee")
    return rec.to_dict()
