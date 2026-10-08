"""NASA NESC 6-DOF atmospheric check-cases 1-10 (NASA/TM-2015-218675).

Each case flies a simple vehicle (a sphere or a "tumbling brick") for 30 s through
Plume's high-fidelity models (WGS-84 or round Earth, J2 or inverse-square gravity,
rotating or fixed Earth, US Standard Atmosphere 1976, optional winds) and compares the
trajectory against the reference trajectories produced by NASA's simulation tools
(stored in ``tests/data/nasa_checkcases``; public domain, from
https://nescacademy.nasa.gov/flightsim/2015).

Units in the reference files are imperial (ft, slug, deg); Plume works in SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from plume.config import EarthSpec, VehicleSpec, WorldSpec
from plume.physics.sim import RocketSim
from plume.physics.wind import TableWind

FT = 0.3048
SLUG = 14.593_902_94
SLUG_FT2 = SLUG * FT * FT
DATA = Path(__file__).resolve().parents[3] / "tests" / "data" / "nasa_checkcases"

# NASA scenario constants (Initial_Conditions.xlsx)
NASA_J2 = 0.001_082_629_82
NASA_GM = 14_076_443_110_000_000 * FT**3
NASA_OMEGA = 7.292_115e-5
NASA_ROUND_RADIUS = 20_902_255.199 * FT


@dataclass
class CheckCase:
    number: int
    name: str
    vehicle: str  # "sphere" | "brick"
    shape: str = "wgs84"  # "wgs84" | "sphere"
    rotating: bool = True
    zonal: bool = True  # J2 (else inverse-square)
    cd: float = 0.0
    damping: bool = False
    alt_ft: float = 30_000.0
    v_ned_fps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll (body wrt NED)
    rates_inertial_dps: tuple[float, float, float] = (0.0, 0.0, 0.0)  # body axes, wrt inertial
    wind: str | None = None  # "steady" | "shear"
    duration: float = 30.0


CASES: dict[int, CheckCase] = {
    1: CheckCase(1, "Dropped sphere, no drag", "sphere"),
    # (the spreadsheet lists 0 ft for the bricks; the reference trajectories start at 30,000 ft)
    2: CheckCase(2, "Tumbling brick, no damping", "brick", rates_inertial_dps=(10.0, 20.0, 30.0)),
    3: CheckCase(
        3,
        "Tumbling brick with damping",
        "brick",
        damping=True,
        rates_inertial_dps=(10.0, 20.0, 30.0),
    ),
    4: CheckCase(
        4,
        "Sphere, round non-rotating Earth",
        "sphere",
        shape="sphere",
        rotating=False,
        zonal=False,
        cd=0.1,
        rates_inertial_dps=(10.0, 20.0, 30.0),
    ),
    5: CheckCase(
        5,
        "Sphere, round rotating Earth",
        "sphere",
        shape="sphere",
        rotating=True,
        zonal=False,
        cd=0.1,
        rates_inertial_dps=(10.0, 20.0, 30.0),
    ),
    6: CheckCase(6, "Sphere, WGS-84 rotating Earth", "sphere", cd=0.1),
    7: CheckCase(7, "Sphere, steady wind", "sphere", cd=0.1, wind="steady"),
    8: CheckCase(8, "Sphere, wind shear", "sphere", cd=0.1, wind="shear"),
    9: CheckCase(
        9,
        "Cannonball eastward along the equator",
        "sphere",
        cd=0.1,
        alt_ft=0.0,
        v_ned_fps=(0.0, 1000.0, -1000.0),
        euler_deg=(90.0, 0.0, 0.0),
        rates_inertial_dps=(0.0, -0.004178073, 0.0),
    ),
    10: CheckCase(
        10,
        "Cannonball northward along the prime meridian",
        "sphere",
        cd=0.1,
        alt_ft=0.0,
        v_ned_fps=(1000.0, 0.0, -1000.0),
        rates_inertial_dps=(0.004178073, 0.0, 0.0),
    ),
}

SPHERE = {"mass": 1.0 * SLUG, "inertia": (3.6, 3.6, 3.6), "S": 0.196_349_5 * FT**2}
BRICK = {
    "mass": 0.155_404_754 * SLUG,
    "inertia": (0.001_894_22, 0.006_211_019, 0.007_194_665),
    "S": 0.222_22 * FT**2,
    "b": 0.333_33 * FT,
    "c": 0.666_67 * FT,
}


def _vehicle(case: CheckCase) -> VehicleSpec:
    v = SPHERE if case.vehicle == "sphere" else BRICK
    inertia = [i * SLUG_FT2 for i in v["inertia"]]
    return VehicleSpec.model_validate(
        {
            "name": f"nesc_{case.vehicle}",
            "geometry": {"length": 0.2, "diameter": 0.15},
            "legs": {"count": 0},
            "mass": {"dry": v["mass"], "dry_cg_z": 0.0, "dry_inertia": inertia},
            "engine": {"type": "liquid", "thrust_vac": 1.0, "nozzle_radius": 0.01, "gimbal_z": 0.0},
            "rcs": {"enabled": False},
            "aero": {"enabled": False},
        }
    )


def _world(case: CheckCase) -> WorldSpec:
    earth = EarthSpec(
        shape=case.shape,
        radius=NASA_ROUND_RADIUS,
        rotating=case.rotating,
        zonal_degree=2 if case.zonal else 0,
        gm=NASA_GM,
        omega=NASA_OMEGA,
        j2=NASA_J2,
    )
    return WorldSpec(fidelity="high", gravity="wgs84", earth=earth, ground="none", dt=0.002)


# NED <-> ENU (both at the same point)
_NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _dcm_body_to_ned(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ]
    )


def _quat_from_dcm(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(float))
    return q


def _forces(case: CheckCase):
    """Sphere drag (constant CD) or brick rate damping, as an extra-force callback."""
    if case.vehicle == "sphere":
        S, cd = SPHERE["S"], case.cd

        def sphere(sim, v_air_w, atm, R, omega_b, mp):
            if cd == 0.0:
                return np.zeros(3), np.zeros(3)
            v = float(np.linalg.norm(v_air_w))
            return -0.5 * atm.density * cd * S * v * v_air_w, np.zeros(3)

        return sphere
    S, b, c = BRICK["S"], BRICK["b"], BRICK["c"]

    def brick(sim, v_air_w, atm, R, omega_b, mp):
        if not case.damping:
            return np.zeros(3), np.zeros(3)
        v = max(float(np.linalg.norm(v_air_w)), 0.5 * FT)
        q = 0.5 * atm.density * v * v
        p, qq, r = omega_b
        tau = np.array(
            [
                q * S * b * (-1.0) * p * b / (2 * v),
                q * S * c * (-1.0) * qq * c / (2 * v),
                q * S * b * (-1.0) * r * b / (2 * v),
            ]
        )
        return np.zeros(3), tau

    return brick


@dataclass
class CaseResult:
    case: CheckCase
    ours: pd.DataFrame
    reference: pd.DataFrame  # per-tool rows
    errors: dict[str, float] = field(default_factory=dict)  # max |ours - median|
    spread: dict[str, float] = field(default_factory=dict)  # max |tool - median|


VARS = [
    "alt_ft",
    "lat_deg",
    "lon_deg",
    "vn_fps",
    "ve_fps",
    "vd_fps",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "p_dps",
    "q_dps",
    "r_dps",
]


def run_case(number: int, output_dt: float = 0.5) -> pd.DataFrame:
    case = CASES[number]
    world = _world(case)
    sim = RocketSim(_vehicle(case), world)
    g = sim.gravity
    if case.wind == "steady":
        sim.wind = TableWind([0.0, 1.0], [20.0 * FT, 20.0 * FT], [0.0, 0.0])
    elif case.wind == "shear":
        h = np.array([0.0, 30_000.0]) * FT
        sim.wind = TableWind(h, np.array([-20.0, 0.003 * 30_000 - 20.0]) * FT, [0.0, 0.0])
    sim.extra_forces.append(_forces(case))
    yaw, pitch, roll = (math.radians(a) for a in case.euler_deg)
    R_wb = _NED_TO_ENU @ _dcm_body_to_ned(yaw, pitch, roll)
    pos = g.world_point(0.0, 0.0, case.alt_ft * FT)
    v_w = _NED_TO_ENU @ (np.array(case.v_ned_fps) * FT)
    w_in = np.radians(case.rates_inertial_dps)
    w_rel = w_in - R_wb.T @ g.omega_w
    sim.reset(pos=pos, vel=v_w, quat=_quat_from_dcm(R_wb), omega=w_rel)
    steps = round(output_dt / world.dt)
    rows = []
    n_out = round(case.duration / output_dt) + 1
    for k in range(n_out):
        if k:
            sim.step(steps)
        st = sim.state
        lat, lon, h = g.geodetic(st.com)
        enu = g.enu_at(st.com)
        ned = np.column_stack([enu[:, 1], enu[:, 0], -enu[:, 2]])  # world coords of N, E, D
        v_ned = ned.T @ st.vel_com
        R_nb = ned.T @ st.rot
        yaw_o = math.degrees(math.atan2(R_nb[1, 0], R_nb[0, 0]))
        pitch_o = math.degrees(-math.asin(max(-1.0, min(1.0, R_nb[2, 0]))))
        roll_o = math.degrees(math.atan2(R_nb[2, 1], R_nb[2, 2]))
        w = np.degrees(g.inertial_rates(st.rot, st.omega))
        atm = sim.atmosphere.at(h)
        rows.append(
            {
                "t": round(k * output_dt, 6),
                "alt_ft": h / FT,
                "lat_deg": math.degrees(lat),
                "lon_deg": math.degrees(lon),
                "vn_fps": v_ned[0] / FT,
                "ve_fps": v_ned[1] / FT,
                "vd_fps": v_ned[2] / FT,
                "yaw_deg": yaw_o,
                "pitch_deg": pitch_o,
                "roll_deg": roll_o,
                "p_dps": w[0],
                "q_dps": w[1],
                "r_dps": w[2],
                "rho_slugft3": atm.density / (SLUG / FT**3),
                "a_fps": atm.speed_of_sound / FT,
            }
        )
    return pd.DataFrame(rows)


def _wrap(d: np.ndarray, var: str) -> np.ndarray:
    if var in {"yaw_deg", "roll_deg", "lon_deg"}:
        return (d + 180.0) % 360.0 - 180.0
    return d


def compare(number: int) -> CaseResult:
    ours = run_case(number)
    ref = pd.read_csv(DATA / f"atmos_{number:02d}.csv")
    res = CaseResult(CASES[number], ours, ref)
    med = ref.groupby("t").median(numeric_only=True)
    for var in VARS:
        if var not in med.columns:
            continue
        m = med[var].reindex(ours["t"].to_numpy()).to_numpy()
        res.errors[var] = float(np.nanmax(np.abs(_wrap(ours[var].to_numpy() - m, var))))
        spread = 0.0
        for _, grp in ref.groupby("tool"):
            g = grp.set_index("t")[var].reindex(ours["t"].to_numpy()).to_numpy()
            if np.isfinite(g).any():
                spread = max(spread, float(np.nanmax(np.abs(_wrap(g - m, var)))))
        res.spread[var] = spread
    return res


# Agreement thresholds (absolute floors; a case passes a variable when our error is within
# max(floor, 2 x the inter-tool spread)). Floors are ~0.1 ft / 0.01 ft/s / 0.01 deg scale,
# i.e. far below anything that matters for flight.
FLOORS = {
    "alt_ft": 0.1,
    "lat_deg": 1e-6,
    "lon_deg": 1e-6,
    "vn_fps": 0.01,
    "ve_fps": 0.01,
    "vd_fps": 0.01,
    "yaw_deg": 0.01,
    "pitch_deg": 0.01,
    "roll_deg": 0.01,
    "p_dps": 0.01,
    "q_dps": 0.01,
    "r_dps": 0.01,
}


def passed(res: CaseResult, var: str) -> bool:
    return res.errors[var] <= max(FLOORS[var], 2.0 * res.spread[var])


def summary_table(results: list[CaseResult]) -> str:
    lines = [
        "| Case | Variable | Plume error | NASA tool spread | Pass |",
        "|---|---|---:|---:|:---:|",
    ]
    for r in results:
        for var, err in r.errors.items():
            lines.append(
                f"| {r.case.number}. {r.case.name} | {var} | {err:.3g} | {r.spread[var]:.3g} | "
                f"{'yes' if passed(r, var) else '**no**'} |"
            )
    return "\n".join(lines)
