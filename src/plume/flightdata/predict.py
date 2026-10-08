"""Pre-flight prediction of a hobby-rocket flight with Monte Carlo uncertainty bands.

``plume predict <vehicle> --motor <motor>`` flies the vehicle many times on the 3-DOF
model (the same one ``plume calibrate`` fits), each time with drag, motor impulse and
burn time, dry mass, parachute size, rail tilt, wind and temperature drawn from the
stated uncertainties, and records:

* the nominal (best-estimate) flight,
* percentile bands (2.5/5/50/95/97.5 %) for apogee, time to apogee, maximum speed and
  Mach, maximum acceleration, rail-exit speed, descent rate and flight time,
* the landing dispersion: mean point, 95 % ellipse and 95th-percentile distance.

The point of a prediction is to write it down *before* the flight: the record carries a
UTC timestamp, the git commit, and SHA-256 hashes of the vehicle definition and the
motor file. ``plume validate`` later compares the real flight with these bands; a
prediction made after the flight is reported as such (it is not a blind test).

Model limits (stated in the report): point mass at zero angle of attack (the rocket
weathercocks instantly into the relative wind after leaving the rail), no
weathercocking dynamics or coning, parachute opens as configured, flat Earth,
power-law wind profile with one mean speed and direction.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from plume import __version__
from plume.config import REPO_ROOT, VehicleSpec, WorldSpec, data_root
from plume.constants import G0
from plume.physics.atmosphere import atmosphere_from_world
from plume.physics.pointmass import PointMassSim
from plume.physics.propulsion import read_eng

FORMAT = "plume-prediction"
VERSION = 1

# metric key -> (label, unit, format)
METRICS: dict[str, tuple[str, str, str]] = {
    "apogee_m": ("apogee (above the pad)", "m", ".0f"),
    "t_apogee_s": ("time to apogee", "s", ".1f"),
    "max_speed_mps": ("maximum speed", "m/s", ".0f"),
    "max_mach": ("maximum Mach", "", ".2f"),
    "max_accel_g": ("maximum acceleration", "g", ".1f"),
    "rail_exit_mps": ("rail-exit speed", "m/s", ".1f"),
    "descent_rate_mps": ("descent rate under the parachute", "m/s", ".1f"),
    "flight_time_s": ("flight time", "s", ".0f"),
    "landing_distance_m": ("landing distance from the pad", "m", ".0f"),
}
PERCENTILES = (2.5, 5.0, 50.0, 95.0, 97.5)


@dataclass
class Dispersions:
    """1-sigma uncertainties (fractions unless the name says otherwise)."""

    cd: float = 0.10  # drag coefficient (hobby-rocket drag estimates are this uncertain)
    impulse: float = 0.03  # motor total impulse (batch-to-batch)
    burn_time: float = 0.03
    dry_mass: float = 0.02  # weigh the rocket: then this is the scale's error
    chute_cd_area: float = 0.10
    rail_tilt_deg: float = 1.0  # rail set-up error
    wind_speed_mps: float | None = None  # default max(1, 0.3 x mean wind)
    wind_dir_deg: float = 30.0
    temperature_k: float = 5.0


@dataclass
class LaunchConditions:
    rail_length: float = 1.5
    rail_tilt_deg: float = 2.0  # nominal tilt away from vertical
    rail_azimuth_deg: float | None = None  # tilt direction (from north, clockwise);
    # None = into the wind (common practice: the rocket weathercocks into it anyway)
    wind_speed_mps: float = 3.0  # mean wind at 10 m
    wind_from_deg: float = 270.0  # meteorological: direction the wind comes FROM
    temperature_offset_k: float = 0.0  # vs. the standard atmosphere at the site


@dataclass
class Prediction:
    meta: dict
    settings: dict
    nominal: dict
    summary: dict
    landing: dict
    runs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, **asdict(self)}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Prediction:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != FORMAT:
            raise ValueError(f"{path} is not a Plume prediction record")
        return cls(d["meta"], d["settings"], d["nominal"], d["summary"], d["landing"], d["runs"])


# ----------------------------------------------------------------------------- inputs
def resolve_motor(motor: str) -> Path:
    p = Path(motor)
    if p.suffix.lower() == ".eng" and p.exists():
        return p.resolve()
    cand = data_root() / "motors" / (p.name if p.suffix else f"{p.name}.eng")
    if cand.exists():
        return cand
    # case-insensitive match on the stem (e.g. "h180" -> Plume_H180.eng)
    for f in sorted((data_root() / "motors").glob("*.eng")):
        if f.stem.lower() == p.stem.lower() or f.stem.lower().endswith("_" + p.stem.lower()):
            return f
    raise FileNotFoundError(f"motor {motor!r} not found (path or name in data/motors/)")


def with_motor(vehicle: VehicleSpec, motor: str | Path) -> tuple[VehicleSpec, list[str]]:
    """The vehicle flying another motor: propellant from the .eng header; the dry mass
    changes by the difference in empty-casing mass when the old motor file is known."""
    path = resolve_motor(str(motor))
    _, info = read_eng(path)
    notes = []
    v = vehicle.model_copy(deep=True)
    old = vehicle.engine.motor_file
    if old and Path(old).exists():
        _, oi = read_eng(old)
        d_case = (info["total_mass_kg"] - info["propellant_kg"]) - (
            oi["total_mass_kg"] - oi["propellant_kg"]
        )
        if abs(d_case) > 1e-9:
            v.mass.dry = round(v.mass.dry + d_case, 4)
            notes.append(f"dry mass {d_case:+.3f} kg for the motor casing")
    elif old != str(path):
        notes.append(
            "previous motor unknown: dry mass left unchanged (check it includes the casing)"
        )
    if v.tanks:
        prop = info["propellant_kg"]
        v.tanks[0].capacity = prop
        v.tanks[0].initial = None
        for t in v.tanks[1:]:
            t.capacity = max(t.capacity, 1e-6)
            t.initial = 0.0
    v.engine.motor_file = str(path)
    v.engine.thrust_curve = None
    return VehicleSpec.model_validate(v.model_dump()), notes


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vehicle_hash(vehicle: VehicleSpec) -> str:
    d = vehicle.model_dump(mode="json")
    d["engine"].pop("motor_file", None)  # absolute paths differ between machines
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def git_state(root: Path = REPO_ROOT) -> dict:
    def run(*args):
        try:
            r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(status)}


# ----------------------------------------------------------------------------- one flight
def wind_profile(speed: float, from_deg: float, alpha: float = 0.143) -> Callable:
    """Power-law mean wind (to the direction it blows), constant above 2 km."""
    to = math.radians(from_deg + 180.0)
    d = np.array([math.sin(to), math.cos(to), 0.0])  # east, north

    def fn(r, alt):
        h = min(max(r[2], 1.0), 2000.0)
        return speed * (h / 10.0) ** alpha * d

    return fn


def fly(
    vehicle: VehicleSpec,
    launch: LaunchConditions,
    cd_scale: float = 1.0,
    impulse_scale: float = 1.0,
    time_scale: float = 1.0,
    chute_scale: float = 1.0,
    rail_tilt_deg: float | None = None,
    wind_speed: float | None = None,
    wind_from_deg: float | None = None,
    temperature_offset: float | None = None,
    dt: float = 0.01,
    t_end: float = 600.0,
) -> dict:
    """One 3-DOF flight from the rail to landing; returns the metrics in ``METRICS``
    plus the landing point."""
    from plume.flightdata.calibrate import nominal_curve, scaled_curve

    ws = launch.wind_speed_mps if wind_speed is None else wind_speed
    wf = launch.wind_from_deg if wind_from_deg is None else wind_from_deg
    tilt = math.radians(launch.rail_tilt_deg if rail_tilt_deg is None else rail_tilt_deg)
    az = launch.rail_azimuth_deg
    az = math.radians(wf if az is None else az)  # tilt toward where the wind comes from
    rail = np.array([math.sin(tilt) * math.sin(az), math.sin(tilt) * math.cos(az), math.cos(tilt)])
    world = WorldSpec(
        temperature_offset=launch.temperature_offset_k
        if temperature_offset is None
        else temperature_offset
    )
    wind = wind_profile(ws, wf)
    curve = scaled_curve(nominal_curve(vehicle), impulse_scale, time_scale)
    pm = PointMassSim(
        vehicle,
        world,
        thrust_curve=curve,
        cd_scale=vehicle.aero.cd_scale * cd_scale,
        wind_fn=wind,
    )
    pm.chute_scale = chute_scale
    r0 = np.array([0.0, 0.0, vehicle.mass.dry_cg_z])
    state = {"apogee_t": None, "rail_exit": None}

    def direction(t, r, v):
        along = float((r - r0) @ rail)
        if along < launch.rail_length:
            return rail
        if state["rail_exit"] is None:
            state["rail_exit"] = float(np.linalg.norm(v))
        va = v - wind(r, r[2])
        sp = float(np.linalg.norm(va))
        return va / sp if sp > 5.0 else rail  # weathercocked into the relative wind

    def dt_fn(t, r, v):
        if state["apogee_t"] is None and t > 0.5 and v[2] < 0:
            state["apogee_t"] = t
        a = state["apogee_t"]
        return dt if (a is None or t < a + 3.0) else max(dt, 0.1)  # under the parachute

    traj = pm.run(
        r0,
        np.zeros(3),
        t_end=t_end,
        dt=dt,
        direction=direction,
        rail_length=launch.rail_length,
        rail_dir=rail,
        dt_fn=dt_fn,
        ground_altitude=r0[2],
    )
    alt = traj.pos[:, 2] - r0[2]
    k_apo = int(np.argmax(alt))
    speed = np.linalg.norm(traj.vel, axis=1)
    atm = atmosphere_from_world(world)
    a_sound = np.array([atm.at(max(h, 0.0)).speed_of_sound for h in traj.pos[:, 2]])
    after = (traj.t > traj.t[k_apo] + 5.0) & (alt > 20.0)
    descent = float(np.median(-traj.vel[after, 2])) if after.sum() > 5 else float("nan")
    land = traj.pos[-1, :2]
    return {
        "apogee_m": float(alt[k_apo]),
        "t_apogee_s": float(traj.t[k_apo]),
        "max_speed_mps": float(speed[: k_apo + 1].max()),
        "max_mach": float((speed / np.maximum(a_sound, 1.0))[: k_apo + 1].max()),
        "max_accel_g": float(np.linalg.norm(traj.accel, axis=1).max() / G0),
        "rail_exit_mps": state["rail_exit"] if state["rail_exit"] is not None else float("nan"),
        "descent_rate_mps": descent,
        "flight_time_s": float(traj.t[-1]),
        "landing_east_m": float(land[0]),
        "landing_north_m": float(land[1]),
        "landing_distance_m": float(np.linalg.norm(land)),
    }


def _draw(rng: np.random.Generator, disp: Dispersions, launch: LaunchConditions) -> dict:
    ws_sigma = (
        disp.wind_speed_mps
        if disp.wind_speed_mps is not None
        else max(1.0, 0.3 * launch.wind_speed_mps)
    )
    return {
        "cd_scale": float(max(rng.normal(1.0, disp.cd), 0.3)),
        "impulse_scale": float(max(rng.normal(1.0, disp.impulse), 0.5)),
        "time_scale": float(max(rng.normal(1.0, disp.burn_time), 0.5)),
        "dry_mass_scale": float(max(rng.normal(1.0, disp.dry_mass), 0.5)),
        "chute_scale": float(max(rng.normal(1.0, disp.chute_cd_area), 0.2)),
        "rail_tilt_deg": float(abs(rng.normal(launch.rail_tilt_deg, disp.rail_tilt_deg))),
        "wind_speed": float(max(rng.normal(launch.wind_speed_mps, ws_sigma), 0.0)),
        "wind_from_deg": float(launch.wind_from_deg + rng.normal(0.0, disp.wind_dir_deg)),
        "temperature_offset": float(
            launch.temperature_offset_k + rng.normal(0.0, disp.temperature_k)
        ),
    }


def _run_one(args) -> dict:
    vehicle, launch, draw = args
    v = vehicle
    if draw.get("dry_mass_scale", 1.0) != 1.0:
        v = vehicle.model_copy(deep=True)
        v.mass.dry = vehicle.mass.dry * draw["dry_mass_scale"]
    kw = {k: x for k, x in draw.items() if k != "dry_mass_scale"}
    return {**draw, **fly(v, launch, **kw)}


def ellipse95(east: np.ndarray, north: np.ndarray) -> dict:
    pts = np.column_stack([east, north])
    c = pts.mean(axis=0)
    if len(pts) < 3:
        return {"center": c.tolist(), "semi_axes_m": [0.0, 0.0], "angle_deg": 0.0}
    cov = np.cov(pts.T)
    w, vec = np.linalg.eigh(cov)
    k = math.sqrt(5.991)  # chi-square, 2 dof, 95 %
    major = vec[:, 1]
    return {
        "center": [float(c[0]), float(c[1])],
        "semi_axes_m": [float(k * math.sqrt(max(w[1], 0))), float(k * math.sqrt(max(w[0], 0)))],
        "angle_deg": float(math.degrees(math.atan2(major[1], major[0]))),  # from east
    }


def predict(
    vehicle: VehicleSpec,
    launch: LaunchConditions | None = None,
    dispersions: Dispersions | None = None,
    runs: int = 200,
    seed: int = 0,
    workers: int = 1,
    flight_id: str | None = None,
    motor_notes: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> Prediction:
    launch = launch or LaunchConditions()
    disp = dispersions or Dispersions()
    rng = np.random.default_rng(seed)
    draws = [_draw(rng, disp, launch) for _ in range(runs)]
    nominal = fly(vehicle, launch)
    jobs = [(vehicle, launch, d) for d in draws]
    workers = max(1, min(int(workers), 4))  # shared machine: never more than 4 processes
    if workers > 1 and runs > 8:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, jobs, chunksize=max(1, runs // (4 * workers))))
    else:
        results = []
        for k, j in enumerate(jobs):
            results.append(_run_one(j))
            if log and (k + 1) % 50 == 0:
                log(f"  {k + 1}/{runs} flights")
    summary = {}
    for key in METRICS:
        x = np.array([r[key] for r in results], dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        summary[key] = {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            **{f"p{p:g}": float(np.percentile(x, p)) for p in PERCENTILES},
        }
    east = np.array([r["landing_east_m"] for r in results])
    north = np.array([r["landing_north_m"] for r in results])
    landing = ellipse95(east, north)
    landing["p95_distance_m"] = float(np.percentile(np.hypot(east, north), 95))
    e = vehicle.engine
    motor = {}
    if e.motor_file:
        curve, info = read_eng(e.motor_file)
        motor = {
            "file": Path(e.motor_file).name,
            "sha256": _sha256_file(e.motor_file),
            "name": info["name"],
            "total_impulse_ns": float(np.trapezoid(curve[:, 1], curve[:, 0])),
            "burn_time_s": float(curve[-1, 0]),
        }
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    meta = {
        "flight_id": flight_id or f"{vehicle.name}_{now[:10]}",
        "created_utc": now,
        "plume_version": __version__,
        "git": git_state(),
        "vehicle": {"name": vehicle.name, "sha256": vehicle_hash(vehicle)},
        "vehicle_spec": vehicle.model_dump(mode="json"),
        "motor": motor,
        "notes": list(motor_notes or []),
        "model": "3-DOF point mass, zero angle of attack, power-law wind (flightdata.predict)",
    }
    settings = {
        "runs": runs,
        "seed": seed,
        "launch": asdict(launch),
        "dispersions": asdict(disp),
    }
    return Prediction(meta, settings, nominal, summary, landing, results)


# ----------------------------------------------------------------------------- report
def report_markdown(p: Prediction, figure: str | None = None) -> str:
    m, s = p.meta, p.settings
    la, d = s["launch"], s["dispersions"]
    lines = [
        f"# Pre-flight prediction: {m['flight_id']}",
        "",
        f"**Recorded:** {m['created_utc']} (UTC) · Plume {m['plume_version']} · commit "
        f"`{m['git'].get('commit') or 'unknown'}`"
        + (" + uncommitted changes" if m["git"].get("dirty") else ""),
        f"**Vehicle:** {m['vehicle']['name']} (sha256 `{m['vehicle']['sha256'][:12]}`)"
        + (
            f" · **motor:** {m['motor']['name']} ({m['motor']['total_impulse_ns']:.0f} N s, "
            f"{m['motor']['burn_time_s']:.2f} s, sha256 `{m['motor']['sha256'][:12]}`)"
            if m.get("motor")
            else ""
        ),
        f"**Model:** {m['model']}.",
        "",
        "Commit this file (and the `.json` next to it) **before the flight**. After the "
        "flight, `plume validate <log.csv> --prediction <this .json>` compares the real "
        "flight with these bands.",
        "",
        f"## Launch conditions ({s['runs']} Monte Carlo flights, seed {s['seed']})",
        "",
        f"- rail {la['rail_length']:g} m, tilted {la['rail_tilt_deg']:g} deg "
        + (
            "into the wind"
            if la["rail_azimuth_deg"] is None
            else f"toward {la['rail_azimuth_deg']:g} deg"
        ),
        f"- wind {la['wind_speed_mps']:g} m/s at 10 m from {la['wind_from_deg']:g} deg, "
        f"temperature {la['temperature_offset_k']:+g} K vs standard",
        "",
        "Uncertainties (1 sigma): "
        f"drag {100 * d['cd']:g} %, impulse {100 * d['impulse']:g} %, burn time "
        f"{100 * d['burn_time']:g} %, dry mass {100 * d['dry_mass']:g} %, parachute "
        f"{100 * d['chute_cd_area']:g} %, rail tilt {d['rail_tilt_deg']:g} deg, wind "
        + (
            f"{d['wind_speed_mps']:g} m/s"
            if d["wind_speed_mps"] is not None
            else f"max(1, 30 %) = {max(1.0, 0.3 * la['wind_speed_mps']):g} m/s"
        )
        + f" and {d['wind_dir_deg']:g} deg, temperature {d['temperature_k']:g} K.",
        "",
        "## Predictions",
        "",
        "| quantity | nominal | 95 % band | 90 % band | median |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, (label, unit, fmt) in METRICS.items():
        if key not in p.summary:
            continue
        b = p.summary[key]
        u = f" {unit}" if unit else ""
        nom = p.nominal.get(key, float("nan"))
        lines.append(
            f"| {label} | {nom:{fmt}}{u} | {b['p2.5']:{fmt}} – {b['p97.5']:{fmt}}{u} | "
            f"{b['p5']:{fmt}} – {b['p95']:{fmt}}{u} | {b['p50']:{fmt}}{u} |"
        )
    ld = p.landing
    a, b = ld["semi_axes_m"]
    lines += [
        "",
        f"**Landing:** mean point {ld['center'][0]:+.0f} m east, {ld['center'][1]:+.0f} m "
        f"north of the pad; 95 % ellipse {a:.0f} x {b:.0f} m (major axis "
        f"{ld['angle_deg']:.0f} deg from east); 95 % of flights land within "
        f"{ld['p95_distance_m']:.0f} m. Keep the recovery area and spectators clear of it.",
        "",
    ]
    if figure:
        lines += [f"![prediction]({figure})", ""]
    if m.get("notes"):
        lines += ["Notes: " + "; ".join(m["notes"]), ""]
    lines += [
        "## Limits of this prediction",
        "",
        "- Point mass: the rocket is assumed to point into the relative wind as soon as it "
        "leaves the rail. Real weathercocking is slower, so with wind the real apogee is "
        "usually a little lower and the landing a little further upwind than predicted.",
        "- Drag comes from the vehicle file's coefficient tables; for a new airframe the "
        "10 % drag uncertainty may be optimistic. The first flight calibrates it.",
        "- The parachute opens exactly as configured (no tangling, no late deployment).",
        "- Wind is one mean speed and direction with a power-law profile; gusts are not "
        "modelled. Use the forecast or a measurement on the day.",
        "",
    ]
    return "\n".join(lines)


def plot_prediction(p: Prediction, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    apo = np.array([r["apogee_m"] for r in p.runs])
    ax[0].hist(apo, bins=30, color="#8a8f98")
    s = p.summary["apogee_m"]
    for q in ("p2.5", "p97.5"):
        ax[0].axvline(s[q], color="#222", ls="--", lw=1)
    ax[0].axvline(p.nominal["apogee_m"], color="#e8743b", lw=2, label="nominal")
    ax[0].set_xlabel("apogee above the pad (m)")
    ax[0].set_ylabel("flights")
    ax[0].legend(frameon=False)
    e = np.array([r["landing_east_m"] for r in p.runs])
    n = np.array([r["landing_north_m"] for r in p.runs])
    ax[1].scatter(e, n, s=6, color="#8a8f98")
    ld = p.landing
    a, b = ld["semi_axes_m"]
    ax[1].add_patch(
        Ellipse(ld["center"], 2 * a, 2 * b, angle=ld["angle_deg"], fill=False, color="#222")
    )
    ax[1].plot(0, 0, "^", color="#e8743b", ms=9, label="pad")
    ax[1].plot(p.nominal["landing_east_m"], p.nominal["landing_north_m"], "x", color="#e8743b")
    ax[1].set_aspect("equal", adjustable="datalim")
    ax[1].set_xlabel("east (m)")
    ax[1].set_ylabel("north (m)")
    ax[1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(alpha=0.25)
    fig.suptitle(f"Prediction {p.meta['flight_id']} ({len(p.runs)} flights)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_prediction(p: Prediction, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    fid = p.meta["flight_id"]
    png = plot_prediction(p, out_dir / f"{fid}.png")
    js = p.save(out_dir / f"{fid}.json")
    md = out_dir / f"{fid}.md"
    md.write_text(report_markdown(p, png.name), encoding="utf-8")
    return {"json": js, "markdown": md, "figure": png}
