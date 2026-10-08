"""Mission-planner endpoints for ``plume viz`` (see docs/planner.md).

``GET  /planner``                         planner page (static/planner.html)
``GET  /api/planner/vehicles``            hop-capable vehicle presets
``GET  /api/planner/curve?vehicle=``      max range vs cargo (bundled table, else computed)
``POST /api/planner/feasibility``         quick 3-DOF feasibility of a route (a few seconds)
``POST /api/planner/jobs``                queue a full 6-DOF flight or a reliability estimate
``GET  /api/planner/jobs``                all jobs; ``GET /api/planner/jobs/{id}`` one job
``GET  /api/planner/jobs/{id}/files/{f}`` safety export / report of a finished job

Jobs run one at a time in the background (the machine is shared): a flight uses one worker
process, a reliability estimate at most four.
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MAX_WORKERS = 4
CAPABILITY_FILE = Path(__file__).with_name("static") / "planner" / "capability.json"
JOB_FILES = {
    "safety.geojson": "application/geo+json",
    "safety.kml": "application/vnd.google-earth.kml+xml",
    "safety.html": "text/html",
    "safety.json": "application/json",
    "report.html": "text/html",
}


class Site(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=360)


class FeasibilityRequest(BaseModel):
    launch: Site
    landing: Site
    vehicle: str = "cargo_hopper"
    cargo_kg: float = Field(250.0, ge=0, le=20_000)


class JobRequest(FeasibilityRequest):
    kind: Literal["flight", "reliability"] = "flight"
    fidelity: Literal["fast", "high"] = "fast"
    runs: int = Field(16, ge=4, le=64)


# ----------------------------------------------------------------------------- capability
_curve_lock = threading.Lock()
_curve_cache: dict[str, list[dict]] = {}


def bundled_tables() -> dict:
    try:
        return json.loads(CAPABILITY_FILE.read_text(encoding="utf-8")).get("vehicles", {})
    except (OSError, ValueError):
        return {}


def capability_curve(vehicle: str) -> tuple[list[dict], str]:
    """(max range vs cargo curve, source): the bundled table when it matches the preset,
    otherwise computed now (about a minute) and cached."""
    from plume.config import load_vehicle
    from plume.missions.planner import range_curve, vehicle_hash

    v = load_vehicle(vehicle)
    table = bundled_tables().get(v.name)
    if table and table.get("hash") == vehicle_hash(v):
        return table["curve"], "bundled table"
    with _curve_lock:
        if v.name not in _curve_cache:
            _curve_cache[v.name] = range_curve(v)
    return _curve_cache[v.name], "computed"


def _clean(obj):
    """JSON-safe: numpy scalars -> float, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, req: JobRequest, workdir: Path):
        self.id = uuid.uuid4().hex[:10]
        self.req = req
        self.dir = workdir / "jobs" / self.id
        self.state = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.message = ""
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.expected_s = 600.0  # nominal flight time; refined from the quick planner
        self.files: list[str] = []

    def view(self) -> dict:
        self._poll()
        r = self.req
        return _clean(
            {
                "id": self.id,
                "kind": r.kind,
                "fidelity": r.fidelity,
                "runs": r.runs if r.kind == "reliability" else None,
                "vehicle": r.vehicle,
                "cargo_kg": r.cargo_kg,
                "launch": r.launch.model_dump(),
                "landing": r.landing.model_dump(),
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "message": self.message,
                "result": self.result,
                "error": self.error,
                "files": self.files,
                "created": self.created,
            }
        )

    def _poll(self) -> None:
        """Progress from the files the workers write."""
        if self.state != "running":
            return
        prog = self.dir / "progress.json"
        try:
            p = json.loads(prog.read_text(encoding="utf-8"))
            frac = min(float(p.get("t", 0.0)) / self.expected_s, 0.99)
            if self.stage == "nominal flight":
                span = 1.0 if self.req.kind == "flight" else 0.25
                self.progress = span * frac
                self.message = f"{p.get('stage', '')} t = {p.get('t', 0):.0f} s" + (
                    f" ({p['phase']})" if p.get("phase") else ""
                )
        except (OSError, ValueError):
            pass
        if self.stage == "monte carlo":
            jl = self.dir / "mc" / "runs.jsonl"
            try:
                n = sum(1 for line in jl.read_text().splitlines() if line.strip())
            except OSError:
                n = 0
            self.progress = 0.25 + 0.7 * n / self.req.runs
            self.message = f"{n} / {self.req.runs} runs flown"


class JobManager:
    def __init__(self, workdir: Path, replay_dir: Path, terrain_dir: Path, replay_id_of=None):
        self.workdir = workdir
        self.replay_id_of = replay_id_of or (lambda p: p.name)
        self.replay_dir = replay_dir
        self.terrain_dir = terrain_dir
        self.jobs: dict[str, Job] = {}
        self._queue: list[Job] = []
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def submit(self, req: JobRequest) -> Job:
        job = Job(req, self.workdir)
        with self._cv:
            self.jobs[job.id] = job
            self._queue.append(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            self._cv.notify()
        return job

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    if not self._cv.wait(timeout=30):
                        self._thread = None
                        return
                job = self._queue.pop(0)
            try:
                job.state = "running"
                self.run(job)
                job.state = "done"
                job.progress = 1.0
            except Exception as exc:  # reported to the page, not hidden
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                (job.dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    # -- the work ---------------------------------------------------------------
    def run(self, job: Job) -> None:
        from plume.missions.route import build_route_mission, fly_route

        r = job.req
        job.dir.mkdir(parents=True, exist_ok=True)
        name = f"planner_{job.id}"
        job.stage = "building mission"
        mission = build_route_mission(
            name,
            (r.launch.lat, r.launch.lon),
            (r.landing.lat, r.landing.lon),
            r.vehicle,
            r.cargo_kg,
            job.dir,
            self.terrain_dir,
            fidelity=r.fidelity,
        )
        replay = self.replay_dir / f"{name}_{r.fidelity}.plume.json.gz"
        job.stage = "nominal flight"
        job.message = "planning the ascent (about a minute)"
        with ProcessPoolExecutor(max_workers=1) as pool:
            flight = pool.submit(
                fly_route, str(mission), str(replay), r.fidelity, str(job.dir / "progress.json")
            ).result()
        flight["replay_id"] = self.replay_id_of(replay)
        job.result = {"flight": flight}
        if r.kind == "flight":
            job.message = "flight complete"
            return
        job.stage = "monte carlo"
        mc_dir = job.dir / "mc"
        summary = run_reliability(mission, mc_dir, r.runs, r.fidelity, replay)
        job.result["reliability"] = summary
        for f in ("safety.geojson", "safety.kml", "safety.html", "safety.json", "report.html"):
            if (mc_dir / f).exists():
                job.files.append(f)
        job.message = "reliability estimate complete"


def run_reliability(mission: Path, mc_dir: Path, runs: int, fidelity: str, replay: Path) -> dict:
    """Small Monte Carlo campaign on the route (screening estimate), its report and the
    flight-safety export."""
    from plume.analysis.montecarlo import (
        DispersionSpec,
        _get,
        load_dispersion,
        run_mc,
        summarize,
        write_report,
    )
    from plume.analysis.safety import GeoFrame, analyze, load_replay, viewer_overlay, write_outputs
    from plume.config import load_mission, load_vehicle
    from plume.recording import save_replay

    spec = load_mission(mission)
    template = load_dispersion("real_hop" if fidelity == "high" else "demo_hop")
    vdict = load_vehicle(spec.vehicle).model_dump()

    def applies(path: str) -> bool:  # e.g. no grid-fin dispersion for a finless vehicle
        root, *rest = path.split(".")
        if root != "vehicle":
            return True
        try:
            return _get(vdict, rest) is not None
        except (KeyError, IndexError, TypeError):
            return False

    params = {k: v for k, v in template.params.items() if applies(k)}
    ds = DispersionSpec(
        name=f"{mission.stem}_{fidelity}",
        description="planner reliability estimate (screening)",
        mission=str(mission.resolve()),
        runs=runs,
        seed=1,
        fidelity=fidelity,
        params=params,
    )
    records = run_mc(
        ds, mc_dir, workers=min(MAX_WORKERS, runs), idle_aware=True, log=lambda *_: None
    )
    summary = summarize(records, ds, spec.target_radius)
    (mc_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, records, mc_dir / "report.html")
    rep = load_replay(replay)
    analysis = analyze(mc_dir=mc_dir, replay=rep)
    write_outputs(analysis, mc_dir)
    # IIP trace and hazard areas in the replay, for the viewer's engineering view
    rep["meta"]["safety"] = viewer_overlay(analysis, GeoFrame.from_replay_meta(rep["meta"]))
    save_replay(rep, replay)
    lo, hi = summary["success_ci95"]
    land = summary.get("landing") or {}
    return _clean(
        {
            "runs": summary["runs"],
            "success_probability": summary["success_probability"],
            "success_ci95": [lo, hi],
            "recovery_probability": summary["recovery_probability"],
            "failure_modes": summary["failure_modes"],
            "cep50_m": land.get("cep50_m"),
            "cep90_m": land.get("cep90_m"),
            "fuel_remaining_p5_kg": (summary["metrics"].get("fuel_remaining_kg") or {}).get("p5"),
            "label": f"screening estimate: {runs} runs at {fidelity} fidelity",
        }
    )


# ----------------------------------------------------------------------------- routes
def register_planner(
    app: FastAPI,
    static_dir: Path,
    workdir: Path,
    replay_dirs: list[Path],
    terrain_dir: Path,
) -> JobManager:
    from plume.missions.planner import planner_vehicles, route_feasibility

    def replay_id_of(path: Path) -> str:
        for root in replay_dirs:
            try:
                return path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
        return path.name

    jobs = JobManager(workdir, workdir, terrain_dir, replay_id_of)
    app.state.planner_jobs = jobs

    @app.get("/planner", include_in_schema=False)
    def planner_page():
        return FileResponse(static_dir / "planner.html", media_type="text/html")

    @app.get("/api/planner/vehicles")
    def vehicles():
        return planner_vehicles()

    @app.get("/api/planner/curve")
    def curve(vehicle: str = "cargo_hopper"):
        try:
            rows, source = capability_curve(vehicle)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"vehicle": vehicle, "source": source, "curve": _clean(rows)}

    @app.post("/api/planner/feasibility")
    def feasibility(req: FeasibilityRequest):
        names = {v["name"] for v in planner_vehicles()}
        if req.vehicle not in names:
            raise HTTPException(status_code=404, detail=f"unknown vehicle {req.vehicle!r}")
        t0 = time.perf_counter()
        try:
            rows, _ = capability_curve(req.vehicle)
        except Exception:
            rows = None
        out = route_feasibility(
            (req.launch.lat, req.launch.lon),
            (req.landing.lat, req.landing.lon),
            req.vehicle,
            req.cargo_kg,
            curve=rows,
        )
        out["compute_s"] = time.perf_counter() - t0
        return _clean(out)

    @app.post("/api/planner/jobs")
    def submit(req: JobRequest):
        names = {v["name"] for v in planner_vehicles()}
        if req.vehicle not in names:
            raise HTTPException(status_code=404, detail=f"unknown vehicle {req.vehicle!r}")
        return jobs.submit(req).view()

    @app.get("/api/planner/jobs")
    def list_jobs():
        return [j.view() for j in sorted(jobs.jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/planner/jobs/{jid}")
    def get_job(jid: str):
        job = jobs.jobs.get(jid)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.view()

    @app.get("/api/planner/jobs/{jid}/files/{fname}")
    def job_file(jid: str, fname: str):
        job = jobs.jobs.get(jid)
        if job is None or fname not in JOB_FILES or fname not in job.files:
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(job.dir / "mc" / fname, media_type=JOB_FILES[fname], filename=fname)

    return jobs
