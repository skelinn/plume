"""Precompute the planner capability tables (static demo + cargo suggestions).

    uv run python scripts/build_planner_table.py [--workers 4]

Writes ``src/plume/viz/static/planner/capability.json``: for every hop-capable vehicle
preset, the max range vs cargo curve and the 3-DOF feasibility over a (range, cargo) grid.
Each table carries a hash of its vehicle preset; ``plume viz`` recomputes when it is stale.
Takes a few minutes per vehicle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from plume.missions.planner import capability_table, planner_vehicles

OUT = Path(__file__).resolve().parents[1] / "src/plume/viz/static/planner/capability.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vehicle", action="append", help="only these presets")
    args = ap.parse_args()
    existing = json.loads(OUT.read_text()) if OUT.exists() else {"vehicles": {}}
    for v in planner_vehicles():
        if args.vehicle and v["name"] not in args.vehicle:
            continue
        t0 = time.time()
        existing["vehicles"][v["name"]] = capability_table(v["name"], workers=args.workers)
        print(f"{v['name']}: {time.time() - t0:.0f} s")
    existing["generated_by"] = "scripts/build_planner_table.py"
    existing["model"] = "3-DOF point mass, calm air, non-rotating spherical Earth (fast)"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, separators=(",", ":")), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
