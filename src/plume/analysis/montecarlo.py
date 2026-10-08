"""Monte Carlo dependability analysis for missions.

A dispersion file (``configs/dispersions/<name>.yaml``) names a mission and a set of
uncertain parameters. Each run samples the *truth* (vehicle, environment) while the
flight software keeps flying with the *nominal* vehicle and the forecast environment, as
it would in reality. Results are written incrementally (``runs.jsonl``, resumable) and
summarised as:

* success probability with a 95 % Wilson score interval,
* landing dispersion at the target: mean offset, CEP50/CEP90 and the 99 % ellipse,
* percentiles of fuel margin, cargo g-load, touchdown speed and flight time,
* failure modes,
* sensitivity: Spearman rank correlation of every dispersed parameter with the miss
  distance and with failure.

See docs/models/montecarlo.md.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from plume.config import config_root


class ParamDispersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dist: Literal["normal", "uniform", "choice"] = "normal"
    sigma: float = Field(
        0.0, ge=0, description="normal: 1-sigma (absolute, or relative if relative)"
    )
    mean: float | None = Field(None, description="normal: mean override (default: nominal value)")
    low: float | None = None
    high: float | None = None
    clip: tuple[float, float] | None = Field(None, description="truncate samples to [lo, hi]")
    relative: bool = Field(False, description="sigma/low/high are fractions of the nominal value")
    values: list | None = Field(None, description="choice: candidate values")


class DispersionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    mission: str
    runs: int = Field(200, ge=1)
    seed: int = 1
    fidelity: Literal["fast", "high"] = "fast"
    params: dict[str, ParamDispersion] = Field(default_factory=dict)


def load_dispersion(name_or_path: str) -> DispersionSpec:
    p = Path(name_or_path)
    if not p.exists():
        p = config_root() / "dispersions" / f"{name_or_path}.yaml"
    data = yaml.safe_load(p.read_text())
    return DispersionSpec(**data)


# ----------------------------------------------------------------------------- sampling
def _get(d: dict, path: list[str]):
    for k in path:
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def _set(d: dict, path: list[str], value) -> None:
    for k in path[:-1]:
        d = d[int(k)] if isinstance(d, list) else d[k]
    last = path[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


def _draw(p: ParamDispersion, nominal, rng: np.random.Generator):
    if p.dist == "choice":
        return p.values[int(rng.integers(len(p.values)))]
    vec = isinstance(nominal, (list, tuple))
    nom = np.atleast_1d(np.asarray(nominal if nominal is not None else 0.0, dtype=float))
    if p.dist == "normal":
        mean = nom if p.mean is None else np.full_like(nom, p.mean)
        sig = p.sigma * (np.abs(nom) if p.relative else 1.0)
        x = mean + sig * rng.standard_normal(nom.shape)
    else:  # uniform
        lo = p.low if p.low is not None else 0.0
        hi = p.high if p.high is not None else 1.0
        if p.relative:
            x = nom * (1.0 + rng.uniform(lo, hi, nom.shape))
        else:
            x = rng.uniform(lo, hi, nom.shape)
    if p.clip is not None:
        x = np.clip(x, *p.clip)
    return [float(v) for v in x] if vec else float(x[0])


def sample_run(ds: DispersionSpec, i: int, mission: dict, vehicle: dict) -> dict:
    """Draw run ``i``'s parameter values (independent stream per run, reproducible)."""
    rng = np.random.default_rng([ds.seed, i])
    out = {}
    for path, p in ds.params.items():
        root, *rest = path.split(".")
        src = vehicle if root == "vehicle" else mission
        keys = rest if root == "vehicle" else ([root, *rest] if root != "mission" else rest)
        try:
            nominal = _get(src, keys)
        except (KeyError, IndexError, TypeError):
            nominal = None
        out[path] = _draw(p, nominal, rng)
    return out


def apply_sample(mission: dict, vehicle: dict, values: dict) -> tuple[dict, dict]:
    import copy

    m, v = copy.deepcopy(mission), copy.deepcopy(vehicle)
    for path, val in values.items():
        root, *rest = path.split(".")
        if root == "vehicle":
            _set(v, rest, val)
        elif root == "mission":
            _set(m, rest, val)
        else:
            _set(m, [root, *rest], val)
    return m, v


# ----------------------------------------------------------------------------- one run
def _lower_priority() -> None:
    """Be polite to interactive use: below-normal CPU priority for worker processes."""
    try:
        import psutil

        proc = psutil.Process()
        if sys.platform == "win32":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except Exception:  # pragma: no cover - best effort
        pass


def _nominal(ds: DispersionSpec):
    from plume.config import load_mission, load_vehicle

    spec = load_mission(ds.mission)
    vehicle = load_vehicle(spec.vehicle)
    if spec.cargo_mass is not None:
        vehicle = vehicle.with_cargo(spec.cargo_mass)
    return spec, vehicle


def plan_nominal(ds: DispersionSpec) -> dict:
    """Pre-flight ascent plan from the nominal vehicle and forecast (shared by all runs)."""
    from plume.missions.hop import MissionWorld, mission_frame, plan_ascent

    spec, vehicle = _nominal(ds)
    spec2, world, gravity = mission_frame(spec, ds.fidelity)
    mw = MissionWorld(spec2, gravity)
    if spec.guidance.kick_angle_deg is not None:
        return {"rise_time": spec.guidance.rise_time, "kick_deg": spec.guidance.kick_angle_deg}
    rise, kick = plan_ascent(spec2, mw, vehicle, vehicle.cargo.mass, world)
    return {"rise_time": float(rise), "kick_deg": float(kick)}


def run_one(ds_json: str, i: int, plan: dict) -> dict:
    """Fly run ``i`` (top level so it pickles for worker processes)."""
    from plume.config import MissionSpec, VehicleSpec
    from plume.missions.hop import mission_frame, run_mission

    ds = DispersionSpec(**json.loads(ds_json))
    spec, vehicle = _nominal(ds)
    m_dict, v_dict = spec.model_dump(), vehicle.model_dump()
    values = sample_run(ds, i, m_dict, v_dict)
    m2, v2 = apply_sample(m_dict, v_dict, values)
    truth_spec = MissionSpec(**m2)
    truth_vehicle = VehicleSpec(**v2)
    _, nominal_world, _ = mission_frame(spec, ds.fidelity)
    t0 = time.perf_counter()
    try:
        run = run_mission(
            truth_spec,
            seed=ds.seed * 100_003 + i,
            fidelity=ds.fidelity,
            vehicle=truth_vehicle,
            nominal_vehicle=vehicle,
            nominal_world=nominal_world,
            kick_deg=plan["kick_deg"],
            rise_time=plan["rise_time"],
        )
        res = asdict(run.result)
        err = None
    except Exception as e:  # a crash of the simulation itself is recorded, not hidden
        res, err = {"success": False, "reason": "sim_error"}, f"{type(e).__name__}: {e}"
    return {
        "run": i,
        "values": values,
        "result": res,
        "error": err,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


# ----------------------------------------------------------------------------- runner
def _busy_reasons(detector) -> list[str]:
    if detector is None:
        return []
    try:
        return detector.reasons({os.getpid()})
    except Exception:  # pragma: no cover
        return []


def run_mc(
    ds: DispersionSpec,
    out_dir: str | Path,
    workers: int | None = None,
    idle_aware: bool = True,
    runs: int | None = None,
    log=print,
) -> list[dict]:
    """Run (or resume) a Monte Carlo campaign; returns all run records."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = runs or ds.runs
    (out / "dispersion.json").write_text(ds.model_dump_json(indent=2))
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        log("planning the nominal ascent...")
        plan = plan_nominal(ds)
        plan_path.write_text(json.dumps(plan))
    jl = out / "runs.jsonl"
    done: dict[int, dict] = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["run"]] = r
    todo = [i for i in range(n) if i not in done]
    if not todo:
        return [done[i] for i in sorted(done)]
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    detector = None
    if idle_aware:
        from plume.rl.idle import BusyDetector, IdleConfig

        detector = BusyDetector(IdleConfig.load())
    ds_json = ds.model_dump_json()
    log(f"{len(todo)} runs to fly ({len(done)} already done), {workers} workers")
    t_start = time.time()
    with (
        ProcessPoolExecutor(max_workers=workers, initializer=_lower_priority) as pool,
        jl.open("a") as f,
    ):
        pending = set()
        it = iter(todo)
        finished = 0
        while True:
            while len(pending) < workers:
                reasons = _busy_reasons(detector)
                if reasons:
                    if not pending:
                        log(f"paused: {', '.join(reasons)}")
                        time.sleep(30)
                        continue
                    break  # let the running ones finish, start nothing new
                try:
                    i = next(it)
                except StopIteration:
                    break
                pending.add(pool.submit(run_one, ds_json, i, plan))
            if not pending:
                break
            fin, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for fut in fin:
                r = fut.result()
                done[r["run"]] = r
                f.write(json.dumps(r) + "\n")
                f.flush()
                finished += 1
                res = r["result"]
                rate = (time.time() - t_start) / finished
                log(
                    f"run {r['run']:4d}: {res.get('reason', '?'):14s} "
                    f"miss {res.get('landing_error_m', float('nan')):9.1f} m   "
                    f"[{len(done)}/{n}, eta {rate * (len(todo) - finished) / 60:.0f} min]"
                )
    return [done[i] for i in sorted(done)]


# ----------------------------------------------------------------------------- statistics
def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def error_ellipse(points: np.ndarray, prob: float = 0.99) -> dict:
    """Bivariate-normal ellipse containing ``prob`` of the population: centre, semi-axes
    (major first) and the major-axis angle from east toward north."""
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    if len(pts) < 3:
        return {
            "center": c.tolist(),
            "semi_axes_m": [0.0, 0.0],
            "angle_rad": 0.0,
            "probability": prob,
        }
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    k = math.sqrt(-2.0 * math.log(1.0 - prob))  # chi-square, 2 dof
    major = v[:, 1]
    return {
        "center": c.tolist(),
        "semi_axes_m": [float(k * math.sqrt(max(w[1], 0))), float(k * math.sqrt(max(w[0], 0)))],
        "angle_rad": float(math.atan2(major[1], major[0])),
        "probability": prob,
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _pct(a, qs=(1, 5, 50, 95, 99)):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean())}


def summarize(records: list[dict], ds: DispersionSpec, target_radius: float | None = None) -> dict:
    n = len(records)
    res = [r["result"] for r in records]
    ok = [bool(x.get("success")) for x in res]
    k = sum(ok)
    lo, hi = wilson_interval(k, n)
    reasons = Counter(x.get("reason", "?") for x in res)
    # touchdowns (anything that came to rest on legs, on or off target)
    td = [
        r
        for r in records
        if np.isfinite(r["result"].get("touchdown_vz_mps", math.nan) or math.nan)
        and r["result"].get("reason") not in ("sim_error",)
    ]
    pts = np.array([[r["result"]["landing_east_m"], r["result"]["landing_north_m"]] for r in td])
    radial = np.hypot(pts[:, 0], pts[:, 1]) if len(pts) else np.array([])
    land = {}
    if len(pts):
        land = {
            "count": len(pts),
            "mean_offset_m": pts.mean(axis=0).tolist(),
            "cep50_m": float(np.median(radial)),
            "cep90_m": float(np.percentile(radial, 90)),
            "max_m": float(radial.max()),
            "ellipse99": error_ellipse(pts, 0.99),
            "ellipse50": error_ellipse(pts, 0.5),
        }
    # flatten parameters (vector params -> one column per component)
    cols: dict[str, list] = {}
    for r in records:
        for path, val in r["values"].items():
            vals = val if isinstance(val, list) else [val]
            for j, x in enumerate(vals):
                name = path if not isinstance(val, list) else f"{path}[{j}]"
                cols.setdefault(name, []).append(x if isinstance(x, (int, float)) else math.nan)
    miss = [x.get("landing_error_m", math.nan) for x in res]
    fail = [0.0 if s else 1.0 for s in ok]
    sens = []
    for name, xs in cols.items():
        sens.append(
            {
                "param": name,
                "rho_miss": spearman(xs, miss),
                "rho_failure": spearman(xs, fail) if 0 < k < n else 0.0,
            }
        )
    sens.sort(key=lambda s: -max(abs(s["rho_miss"]), abs(s["rho_failure"])))
    return {
        "name": ds.name,
        "mission": ds.mission,
        "fidelity": ds.fidelity,
        "runs": n,
        "successes": k,
        "success_probability": k / n if n else 0.0,
        "success_ci95": [lo, hi],
        "failure_modes": dict(reasons.most_common()),
        "target_radius_m": target_radius,
        "landing": land,
        "points": pts.tolist(),
        "metrics": {
            "fuel_remaining_kg": _pct([x.get("fuel_remaining_kg") for x in res]),
            "max_cargo_g": _pct([x.get("max_cargo_g") for x in res]),
            "touchdown_vz_mps": _pct([x.get("touchdown_vz_mps") for x in res]),
            "touchdown_vh_mps": _pct([x.get("touchdown_vh_mps") for x in res]),
            "flight_time_s": _pct([x.get("flight_time_s") for x in res]),
            "landing_error_m": _pct(miss),
        },
        "sensitivity": sens,
        "significance_threshold": 2.0 / math.sqrt(n) if n else 1.0,
        "sim_errors": [r["error"] for r in records if r.get("error")][:10],
    }


def dispersion_meta(summary: dict) -> dict | None:
    """``meta.dispersion`` block for a replay (schema: local east/north at the target)."""
    land = summary.get("landing") or {}
    e = land.get("ellipse99")
    if not e:
        return None
    return {
        "center": [*e["center"], 0.0],
        "semi_axes_m": e["semi_axes_m"],
        "angle_rad": e["angle_rad"],
        "probability": e["probability"],
        "runs": summary["runs"],
        "points": [[float(x), float(y), 0.0] for x, y in summary["points"][:2000]],
    }


# ----------------------------------------------------------------------------- report
def _fmt(x, d=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:,.{d}f}"


def _scatter_svg(summary: dict, size: int = 440) -> str:
    pts = np.asarray(summary["points"], dtype=float).reshape(-1, 2)
    land = summary.get("landing") or {}
    r_t = summary.get("target_radius_m") or 0.0
    ext = max([r_t * 1.2, 10.0] + ([float(np.abs(pts).max()) * 1.1] if len(pts) else []))
    s = size / (2 * ext)
    cx = cy = size / 2

    def X(e):
        return cx + e * s

    def Y(n):
        return cy - n * s

    out = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
        f'aria-label="landing dispersion">',
        f'<rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="var(--rule)"/>',
        f'<line x1="{cx}" y1="0" x2="{cx}" y2="{size}" stroke="var(--rule)"/>',
        f'<line x1="0" y1="{cy}" x2="{size}" y2="{cy}" stroke="var(--rule)"/>',
    ]
    if r_t:
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r_t * s:.2f}" fill="none" stroke="var(--fg)" '
            f'stroke-dasharray="4 3"/>'
        )
    for key, dash in (("ellipse99", ""), ("ellipse50", ' stroke-dasharray="1 2"')):
        e = land.get(key)
        if e and e["semi_axes_m"][0] > 0:
            a, b = e["semi_axes_m"]
            ex, ey = e["center"]
            deg = -math.degrees(e["angle_rad"])
            out.append(
                f'<ellipse cx="{X(ex):.2f}" cy="{Y(ey):.2f}" rx="{a * s:.2f}" ry="{b * s:.2f}" '
                f'transform="rotate({deg:.2f} {X(ex):.2f} {Y(ey):.2f})" fill="none" '
                f'stroke="var(--fg)"{dash}/>'
            )
    for e, n in pts:
        out.append(f'<circle cx="{X(e):.2f}" cy="{Y(n):.2f}" r="1.8" fill="var(--fg)"/>')
    out.append(
        f'<text x="6" y="14" class="ax">N</text><text x="{size - 14}" y="{cy - 6}" class="ax">E</text>'
        f'<text x="6" y="{size - 8}" class="ax">half-width {ext:,.0f} m</text>'
    )
    out.append("</svg>")
    return "".join(out)


def _hist_svg(values, label: str, w: int = 300, h: int = 90) -> str:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(a) < 2 or np.ptp(a) == 0:
        return ""
    counts, edges = np.histogram(a, bins=24)
    m = counts.max()
    bw = w / len(counts)
    bars = "".join(
        f'<rect x="{k * bw:.2f}" y="{h - c / m * (h - 14):.2f}" width="{bw - 1:.2f}" '
        f'height="{c / m * (h - 14):.2f}" fill="var(--fg)"/>'
        for k, c in enumerate(counts)
    )
    return (
        f'<figure><svg viewBox="0 0 {w} {h + 14}" width="{w}" height="{h + 14}">{bars}'
        f'<text x="0" y="{h + 12}" class="ax">{edges[0]:,.1f}</text>'
        f'<text x="{w}" y="{h + 12}" class="ax" text-anchor="end">{edges[-1]:,.1f}</text></svg>'
        f"<figcaption>{label}</figcaption></figure>"
    )


def write_report(summary: dict, records: list[dict], path: str | Path) -> Path:
    """Self-contained monochrome HTML report."""
    s = summary
    lo, hi = s["success_ci95"]
    land = s.get("landing") or {}
    e99 = land.get("ellipse99") or {}
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{_fmt(v.get('p1'))}</td><td class=n>{_fmt(v.get('p5'))}</td>"
        f"<td class=n>{_fmt(v.get('p50'))}</td><td class=n>{_fmt(v.get('p95'))}</td>"
        f"<td class=n>{_fmt(v.get('p99'))}</td></tr>"
        for k, v in s["metrics"].items()
        if v
    )
    fm = "".join(
        f"<tr><td>{k}</td><td class=n>{v}</td><td class=n>{100 * v / s['runs']:.1f} %</td></tr>"
        for k, v in s["failure_modes"].items()
    )
    thr = s["significance_threshold"]
    sens = "".join(
        f"<tr><td><code>{x['param']}</code></td><td class=n>{x['rho_miss']:+.2f}</td>"
        f"<td class=n>{x['rho_failure']:+.2f}</td>"
        f"<td>{'significant' if max(abs(x['rho_miss']), abs(x['rho_failure'])) > thr else ''}</td></tr>"
        for x in s["sensitivity"]
    )
    res = [r["result"] for r in records]
    hists = "".join(
        [
            _hist_svg([x.get("landing_error_m") for x in res], "miss distance, m"),
            _hist_svg([x.get("fuel_remaining_kg") for x in res], "propellant remaining, kg"),
            _hist_svg([x.get("max_cargo_g") for x in res], "peak cargo load, g"),
            _hist_svg([x.get("touchdown_vz_mps") for x in res], "touchdown sink rate, m/s"),
        ]
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monte Carlo: {s["name"]}</title>
<style>
:root{{--bg:#fff;--fg:#0a0a0a;--mute:#666;--rule:#d0d0d0}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0a0a0a;--fg:#f2f2f2;--mute:#8a8a8a;--rule:#2a2a2a}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:32px 16px}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}} h2{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:32px 0 8px;font-weight:500}}
p.sub{{color:var(--mute);margin:0 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));border-top:1px solid var(--rule);border-left:1px solid var(--rule)}}
.kpi{{padding:12px 16px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.kpi b{{display:block;font:500 22px "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.kpi span{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:left}}
th{{font-weight:500;color:var(--mute);font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
td.n{{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.row{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}} svg{{max-width:100%;height:auto}}
.ax{{font:10px "IBM Plex Mono",monospace;fill:var(--mute)}} figure{{margin:0}} figcaption{{font-size:11px;color:var(--mute)}}
code{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px}}
.note{{color:var(--mute);font-size:12px}}
</style></head><body><main>
<h1>Monte Carlo: {s["name"]}</h1>
<p class="sub">mission <code>{s["mission"]}</code> · fidelity {s["fidelity"]} · {s["runs"]} runs</p>
<div class="kpis">
<div class="kpi"><b>{100 * s["success_probability"]:.1f} %</b><span>success</span></div>
<div class="kpi"><b>{100 * lo:.1f}–{100 * hi:.1f} %</b><span>95 % interval</span></div>
<div class="kpi"><b>{_fmt(land.get("cep50_m"))} m</b><span>CEP50</span></div>
<div class="kpi"><b>{_fmt(land.get("cep90_m"))} m</b><span>CEP90</span></div>
<div class="kpi"><b>{_fmt((e99.get("semi_axes_m") or [None])[0])} m</b><span>99 % ellipse, major</span></div>
</div>
<h2>Landing dispersion at the target</h2>
<div class="row">{_scatter_svg(s)}
<div class="note" style="max-width:420px">Local east/north offsets of every touchdown from the
target. Dashed circle: target radius ({_fmt(s.get("target_radius_m"), 0)} m). Solid ellipse: 99 %
bivariate-normal fit; dotted: 50 %. Mean offset {_fmt(land.get("mean_offset_m", [0, 0])[0])} m E,
{_fmt(land.get("mean_offset_m", [0, 0])[1])} m N.</div></div>
<h2>Distributions</h2><div class="row">{hists}</div>
<h2>Metrics (percentiles)</h2>
<table><tr><th>metric</th><th>p1</th><th>p5</th><th>p50</th><th>p95</th><th>p99</th></tr>{rows}</table>
<h2>Outcomes</h2>
<table><tr><th>outcome</th><th>runs</th><th>share</th></tr>{fm}</table>
<h2>Sensitivity</h2>
<p class="note">Spearman rank correlation of each dispersed parameter with miss distance and with
failure. |&rho;| above {thr:.2f} (2/&radic;n) is marked significant.</p>
<table><tr><th>parameter</th><th>&rho; miss</th><th>&rho; failure</th><th></th></tr>{sens}</table>
<h2>Method</h2>
<p class="note">Each run samples the true vehicle and environment from the dispersion file; the flight
software flies with the nominal vehicle, the forecast environment and the pre-flight ascent plan.
Success probability uses the Wilson score interval. Ellipses assume a bivariate normal landing
distribution. These are simulation results: model-form error is not included beyond the
dispersed parameters (see docs/models).</p>
</main></body></html>"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
