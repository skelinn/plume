"""Fly the demo hop with and without grid fins over several seeds and wind levels.

    uv run python scripts/compare_grid_fins.py [--seeds 4] [--workers 4]

Writes docs/grid_fins.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

WINDS = {"light (5 m/s)": (5.0, 0.6, 3.0), "strong (10 m/s, gusts)": (10.0, 1.5, 6.0)}
VEHICLES = {"no grid fins": "cargo_hopper_finless", "grid fins": "cargo_hopper"}


def fly(args):
    vehicle, wind, seed = args
    from plume.config import load_mission
    from plume.missions.hop import run_mission

    spec = load_mission("demo_hop")
    spec.vehicle = vehicle
    speed, turb, gust = WINDS[wind]
    spec.world.wind = spec.world.wind.model_copy(
        update={"speed": speed, "turbulence": turb, "gust_max": gust}
    )
    r = run_mission(spec, seed=seed).result
    return (
        vehicle,
        wind,
        seed,
        r.success,
        r.landing_error_m,
        r.fuel_remaining_kg,
        r.max_cargo_g,
        r.rcs_used_kg,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(v, w, s) for v in VEHICLES.values() for w in WINDS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(fly, jobs))
    out = []
    lines = [
        "| Wind | Vehicle | Landed on target | Median miss (m) | Propellant left (kg) | RCS gas used (kg) | Peak cargo load (g) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for w in WINDS:
        for label, v in VEHICLES.items():
            rs = [r for r in rows if r[0] == v and r[1] == w]
            ok = sum(r[3] for r in rs)
            rec = {
                "wind": w,
                "vehicle": label,
                "runs": len(rs),
                "landed": ok,
                "median_miss_m": float(np.median([r[4] for r in rs])),
                "fuel_left_kg": float(np.mean([r[5] for r in rs])),
                "rcs_used_kg": float(np.mean([r[7] for r in rs])),
                "max_cargo_g": float(np.max([r[6] for r in rs])),
            }
            out.append(rec)
            lines.append(
                f"| {w} | {label} | {ok}/{len(rs)} | {rec['median_miss_m']:.0f} | "
                f"{rec['fuel_left_kg']:.0f} | {rec['rcs_used_kg']:.0f} | {rec['max_cargo_g']:.2f} |"
            )
    Path("docs").mkdir(exist_ok=True)
    Path("docs/grid_fins.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
