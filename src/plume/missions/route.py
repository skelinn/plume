"""Missions between arbitrary (lat, lon) sites, for the browser planner's full-flight and
reliability jobs.

:func:`build_route_mission` writes a mission YAML (from the ``real_hop`` template: guidance,
wind, scoring) plus *flat* terrain at sea level around both sites: the planner has no
elevation data for arbitrary sites. Real terrain for a route comes from
``plume terrain fetch`` (docs/models/terrain.md) and a hand-written mission file.

Map coordinates: fast fidelity flies a non-rotating sphere whose map coordinates are the
spherical azimuthal-equidistant projection about the launch site (the ``sphere`` backend of
:func:`plume.terrain.dem.site_projection`, the same sphere as the simulator), so lat/lon
round-trip exactly. High fidelity recomputes the target from lat/lon on WGS-84
(:func:`plume.missions.hop.mission_frame`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from plume.config import MissionSpec, load_mission
from plume.terrain.heightmap import Heightmap

PAD_HALF_M = 200.0
LZ_HALF_M = 1000.0
REGION_MARGIN_M = 60_000.0


def _flat(name: str, cx: float, cy: float, half: float, n: int) -> Heightmap:
    return Heightmap(
        np.zeros((n, n)), cx - half, cx + half, cy - half, cy + half, name=name, meta={}
    )


def build_route_mission(
    name: str,
    launch: tuple[float, float],
    landing: tuple[float, float],
    vehicle: str,
    cargo_kg: float,
    out_dir: str | Path,
    terrain_dir: str | Path,
    fidelity: str = "fast",
) -> Path:
    """Write ``<out_dir>/<name>.yaml`` and flat terrain ``<terrain_dir>/<name>_*.yaml``."""
    from plume.terrain.dem import site_projection

    out_dir, terrain_dir = Path(out_dir), Path(terrain_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(parents=True, exist_ok=True)
    # the same projection mission_frame uses for this fidelity (tiles must sit on the site)
    backend = "auto" if fidelity == "high" else "sphere"
    u, v = site_projection(launch[0], launch[1], backend=backend).forward(landing[0], landing[1])
    u, v = float(u), float(v)
    x0, x1 = min(0.0, u) - REGION_MARGIN_M, max(0.0, u) + REGION_MARGIN_M
    y0, y1 = min(0.0, v) - REGION_MARGIN_M, max(0.0, v) + REGION_MARGIN_M
    region = Heightmap(np.zeros((33, 33)), x0, x1, y0, y1, name=f"{name}_region", meta={})
    pad = _flat(f"{name}_pad_a", 0.0, 0.0, PAD_HALF_M, 201)
    lz = _flat(f"{name}_lz_b", u, v, LZ_HALF_M, 201)
    paths = {}
    for hm in (region, pad, lz):
        paths[hm.name] = hm.save(terrain_dir / f"{hm.name}.yaml")
    tmpl = load_mission("real_hop").model_dump(mode="json")
    world = tmpl["world"]
    if fidelity != "high":  # the fast model flies US76 (MSIS needs a geodetic origin)
        world["atmosphere_model"] = {"model": "us76"}
    mission = {
        **tmpl,
        "name": name,
        "description": (
            f"Planner route: {cargo_kg:.0f} kg, ({launch[0]:.4f}, {launch[1]:.4f}) -> "
            f"({landing[0]:.4f}, {landing[1]:.4f}); flat terrain at sea level"
        ),
        "vehicle": vehicle,
        "cargo_mass": float(cargo_kg),
        "terrain": {
            "base": str(paths[region.name].resolve()),
            "launch_tile": str(paths[pad.name].resolve()),
            "landing_tile": str(paths[lz.name].resolve()),
        },
        "launch": {"name": "Launch site", "u": 0.0, "v": 0.0, "lat": launch[0], "lon": launch[1]},
        "target": {"name": "Landing site", "u": u, "v": v, "lat": landing[0], "lon": landing[1]},
        "world": world,
    }
    MissionSpec.model_validate(mission)  # fail early
    path = out_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(mission, sort_keys=False), encoding="utf-8")
    return path


def fix_replay_meta(meta: dict, mission_path: str | Path, fidelity: str) -> dict:
    """Terrain ids by name (the server resolves bare stems) and the geodetic origin, so the
    viewer and the safety export can place a planner flight on the map."""
    spec = load_mission(mission_path)
    ground = (meta.get("scene") or {}).get("ground") or {}
    if ground.get("terrain_id"):
        ground["terrain_id"] = Path(ground["terrain_id"]).stem
    if ground.get("detail_terrain_ids"):
        ground["detail_terrain_ids"] = [Path(t).stem for t in ground["detail_terrain_ids"]]
    meta["geo"] = {
        "origin_lat_deg": spec.launch.lat,
        "origin_lon_deg": spec.launch.lon,
        "map_projection": "aeqd-sphere" if fidelity != "high" else "aeqd-wgs84",
        "launch": {"lat": spec.launch.lat, "lon": spec.launch.lon},
        "landing": {"lat": spec.target.lat, "lon": spec.target.lon},
    }
    meta.setdefault("mission", {})["mission_file"] = Path(mission_path).name
    return meta


def fly_route(
    mission_path: str | Path,
    replay_path: str | Path,
    fidelity: str = "fast",
    progress_path: str | Path | None = None,
    seed: int = 0,
) -> dict:
    """Fly the 6-DOF mission and save its replay; writes ``{"t": sim time, ...}`` progress
    to ``progress_path`` every few seconds of wall time (for the server's job status)."""
    from plume.missions.hop import run_mission

    def write(obj):
        if progress_path is not None:
            Path(progress_path).write_text(json.dumps(obj), encoding="utf-8")

    write({"stage": "planning ascent", "t": 0.0})
    last = [0.0]

    def on_frame(frame):
        now = time.monotonic()
        if now - last[0] > 1.0:
            last[0] = now
            write({"stage": "flying", "t": float(frame.get("t", 0.0)), "phase": frame.get("phase")})

    spec = load_mission(mission_path)
    run = run_mission(spec, seed=seed, fidelity=fidelity, on_frame=on_frame)
    rec = run.recorder
    fix_replay_meta(rec.meta, mission_path, fidelity)
    out = rec.save(replay_path)
    res = run.result
    write({"stage": "done", "t": res.flight_time_s})
    return {
        "replay": str(out),
        "success": res.success,
        "reason": res.reason,
        "landing_error_m": res.landing_error_m,
        "fuel_remaining_kg": res.fuel_remaining_kg,
        "max_cargo_g": res.max_cargo_g,
        "flight_time_s": res.flight_time_s,
        "apogee_km": res.apogee_km,
    }
