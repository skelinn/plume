"""Sim-to-real calibration: fit drag and thrust curve (and parachute) to a flight log.

Parameters (``mode="scale"``)
    cd_scale        multiplier on all aerodynamic coefficients
    impulse_scale   multiplier on the motor's total impulse
    time_scale      stretch of the thrust curve in time (burn time)
    t_offset        log/sim time alignment (liftoff detection vs ignition)
``mode="knots"`` adds piecewise-linear thrust multipliers at ``n_knots`` points of the
burn (shape changes), regularised toward smoothness.

The ascent (liftoff to just past apogee) is fitted with ``scipy.optimize.least_squares``
on altitude and axial-acceleration residuals using the 3-DOF model. The parachute
drag area follows in closed form from the steady descent rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from plume.config import VehicleSpec
from plume.constants import G0
from plume.flightdata.compare import SimTrace, metrics, simulate
from plume.flightdata.importer import FlightLog
from plume.physics.atmosphere import Atmosphere
from plume.physics.propulsion import read_eng


def nominal_curve(vehicle: VehicleSpec) -> np.ndarray:
    e = vehicle.engine
    if e.thrust_curve:
        curve = np.asarray(e.thrust_curve, dtype=float)
    elif e.motor_file:
        curve, _ = read_eng(e.motor_file)
    else:
        raise ValueError("calibration needs a solid motor (thrust_curve or motor_file)")
    if curve[0, 0] > 0:
        curve = np.vstack([[0.0, 0.0], curve])
    return curve


def scaled_curve(
    curve: np.ndarray, impulse_scale: float, time_scale: float, knots=None
) -> np.ndarray:
    out = curve.copy()
    tb = curve[-1, 0]
    if knots is not None and len(knots):
        xk = np.linspace(0.0, tb, len(knots))
        out[:, 1] = out[:, 1] * np.interp(out[:, 0], xk, knots)
        # renormalise so impulse_scale keeps meaning "total impulse ratio"
        out[:, 1] *= np.trapezoid(curve[:, 1], curve[:, 0]) / max(
            np.trapezoid(out[:, 1], out[:, 0]), 1e-12
        )
    out[:, 0] *= time_scale
    out[:, 1] *= impulse_scale / time_scale
    return out


@dataclass
class CalibrationResult:
    params: dict[str, float]
    sigma: dict[str, float]
    before: dict[str, float]
    after: dict[str, float]
    vehicle: VehicleSpec
    nominal_trace: SimTrace
    calibrated_trace: SimTrace
    curve_nominal: np.ndarray
    curve_fit: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.curve_fit[:, 1], self.curve_fit[:, 0]))


def descent_cd_area(log: FlightLog, mass: float, skip: float = 4.0) -> float | None:
    """Parachute Cd*A from the steady descent rate: m g = 0.5 rho v^2 CdA."""
    m = (log.t > log.t_apogee + skip) & (log.altitude > 20.0)
    if m.sum() < 20:
        return None
    v = -log.velocity[m]
    if np.median(v) < 1.0:
        return None
    rho = np.array([Atmosphere().density(a) for a in log.altitude[m]])
    cda = 2.0 * mass * G0 / (rho * v * v)
    return float(np.median(cda))


def calibrate(
    log: FlightLog,
    vehicle: VehicleSpec,
    rail_length: float = 1.5,
    mode: str = "scale",
    n_knots: int = 5,
    fit_chute: bool = True,
    sigma_alt: float = 1.5,
    sigma_acc_g: float = 0.25,
) -> CalibrationResult:
    curve0 = nominal_curve(vehicle)
    cd0 = vehicle.aero.cd_scale
    t_stop = 1.0

    def trace(p, label="fit"):
        cd, imp, ts, toff = p[:4]
        knots = p[4:] if mode == "knots" else None
        return simulate(
            vehicle,
            rail_length=rail_length,
            thrust_curve=scaled_curve(curve0, imp, ts, knots),
            cd_scale=cd0 * cd,
            t_offset=toff,
            t_end=log.t_apogee * 2 + 20,
            stop_after_apogee=t_stop,
            label=label,
        )

    win = log.window(0.0, log.t_apogee + t_stop * 0.8)
    t_fit = log.t[win][::2]
    alt_fit = log.altitude[win][::2]
    acc_mask = log.window(0.02, log.t_apogee)
    t_acc = log.t[acc_mask][::2]
    acc_fit = log.accel[acc_mask][::2] if log.accel is not None else None

    def residuals(p):
        tr = trace(p)
        r = [(np.interp(t_fit, tr.t, tr.altitude) - alt_fit) / sigma_alt]
        if acc_fit is not None:
            r.append((np.interp(t_acc, tr.t, tr.accel) - acc_fit) / (sigma_acc_g * G0))
        if mode == "knots":
            k = np.asarray(p[4:])
            r.append(np.diff(k, 2) / 0.05)  # smoothness prior
        return np.concatenate(r)

    p0 = [1.0, 1.0, 1.0, 0.0]
    lo = [0.3, 0.5, 0.5, -0.3]
    hi = [3.0, 1.5, 2.0, 0.3]
    names = ["cd_scale", "impulse_scale", "time_scale", "t_offset"]
    if mode == "knots":
        p0 += [1.0] * n_knots
        lo += [0.3] * n_knots
        hi += [2.0] * n_knots
        names += [f"knot_{k}" for k in range(n_knots)]
    sol = least_squares(residuals, p0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3, max_nfev=200)
    dof = max(len(sol.fun) - len(sol.x), 1)
    s2 = 2.0 * sol.cost / dof
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sig = np.full(len(sol.x), np.nan)
    params = {n: float(x) for n, x in zip(names, sol.x, strict=True)}
    sigma = {n: float(s) for n, s in zip(names, sig, strict=True)}
    knots = sol.x[4:] if mode == "knots" else None
    curve_fit = scaled_curve(curve0, params["impulse_scale"], params["time_scale"], knots)

    notes = []
    chute_scale = 1.0
    calibrated = vehicle.model_copy(deep=True)
    calibrated.aero.cd_scale = cd0 * params["cd_scale"]
    calibrated.engine.thrust_curve = np.round(curve_fit, 4).tolist()
    calibrated.engine.motor_file = None
    if fit_chute and vehicle.recovery.chutes:
        dry = vehicle.mass.dry + vehicle.cargo.mass + vehicle.rcs.gas
        cda = descent_cd_area(log, dry)
        if cda is not None:
            total0 = sum(c.cd_area for c in vehicle.recovery.chutes)
            chute_scale = cda / total0
            params["chute_cd_area"] = cda
            for c in calibrated.recovery.chutes:
                c.cd_area = round(c.cd_area * chute_scale, 4)
        else:
            notes.append("no steady parachute descent found; parachute left unchanged")
    calibrated = VehicleSpec.model_validate(calibrated.model_dump())
    calibrated.name = f"{vehicle.name}_calibrated"
    calibrated.description = (
        f"{vehicle.description} Calibrated to {log.meta.get('source', 'a flight log')}."
    ).strip()

    nominal = simulate(vehicle, rail_length=rail_length, label="nominal sim")
    fitted = simulate(
        vehicle,
        rail_length=rail_length,
        thrust_curve=curve_fit,
        cd_scale=calibrated.aero.cd_scale,
        chute_scale=chute_scale,
        t_offset=params["t_offset"],
        label="calibrated sim",
    )
    return CalibrationResult(
        params=params,
        sigma=sigma,
        before=metrics(log, nominal),
        after=metrics(log, fitted),
        vehicle=calibrated,
        nominal_trace=nominal,
        calibrated_trace=fitted,
        curve_nominal=curve0,
        curve_fit=curve_fit,
        notes=notes,
    )


def impulse_of(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve[:, 1], curve[:, 0]))


def describe(res: CalibrationResult) -> list[tuple[str, str]]:
    rows = []
    for k, v in res.params.items():
        s = res.sigma.get(k)
        rows.append(
            (k, f"{v:.4f}" + (f" +/- {s:.4f}" if s is not None and math.isfinite(s) else ""))
        )
    rows.append(
        ("total impulse", f"{impulse_of(res.curve_nominal):.1f} -> {res.total_impulse:.1f} N s")
    )
    rows.append(("burn time", f"{res.curve_nominal[-1, 0]:.2f} -> {res.curve_fit[-1, 0]:.2f} s"))
    rows.append(
        (
            "apogee error",
            f"{res.before['apogee_error_m']:+.1f} m -> {res.after['apogee_error_m']:+.1f} m",
        )
    )
    rows.append(
        (
            "RMS altitude (ascent)",
            f"{res.before['rms_altitude_ascent_m']:.1f} m -> {res.after['rms_altitude_ascent_m']:.1f} m",
        )
    )
    if "rms_accel_ascent_g" in res.before:
        rows.append(
            (
                "RMS accel (ascent)",
                f"{res.before['rms_accel_ascent_g']:.2f} g -> {res.after['rms_accel_ascent_g']:.2f} g",
            )
        )
    return rows
