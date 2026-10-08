"""Design trade study: compare Monte Carlo campaigns of design variants against the baseline.

All campaigns use the same seeded draws (see DispersionSpec.overrides), so run i is the same
weather/engine/mass sample in every campaign and differences are paired.

    uv run python scripts/trade_study.py            # writes docs/design/trade_study.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from plume.analysis.montecarlo import INTACT_REASONS, wilson_interval

CATASTROPHIC = {"terrain_impact", "crash_hull", "crash_legs", "tipped_over"}

VARIANTS = [
    (
        "baseline",
        "runs/mc/real_hop_high",
        "Current design: 7 deg gimbal, 6800 kg propellant, 1400 m/s entry",
    ),
    ("gimbal9", "runs/mc/real_hop_gimbal9_high", "Engine gimbal range 7 -> 9 deg"),
    (
        "tanks105",
        "runs/mc/real_hop_tanks105_high",
        "Propellant tanks +5 % (+340 kg propellant, +15 kg structure)",
    ),
    ("entry1300", "runs/mc/real_hop_entry1300_high", "Entry-burn cutoff speed 1400 -> 1300 m/s"),
    ("combined", "runs/mc/real_hop_combined_high", "All three changes"),
]

COST = {
    "baseline": "-",
    "gimbal9": "Longer actuator stroke, larger engine-bay clearance and heat-shield cut-out; "
    "roughly +2-4 kg; may need a stiffer thrust structure. No performance cost.",
    "tanks105": "Longer tanks (~0.3 m) or higher fill; +15 kg dry, +355 kg at liftoff "
    "(liftoff mass 8340 -> 8695 kg, sea-level thrust-to-weight 2.10 -> 2.02); longer ascent burn.",
    "entry1300": "No hardware change (flight software parameter); costs entry-burn propellant.",
    "combined": "Sum of the above.",
}


def load(d: Path) -> dict[int, dict]:
    f = d / "runs.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["run"]] = r["result"]
    return out


def pct(vals, q):
    a = np.array([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    return float(np.percentile(a, q)) if len(a) else math.nan


def main(out: str = "docs/design/trade_study.md") -> None:
    data = {name: load(Path(d)) for name, d, _ in VARIANTS}
    base = data["baseline"]
    rows = []
    for name, _d, desc in VARIANTS:
        R = data[name]
        if not R:
            rows.append(f"| {name} | {desc} | _not run yet_ | | | | | | |")
            continue
        n = len(R)
        k = sum(r["success"] for r in R.values())
        lo, hi = wilson_interval(k, n)
        intact = sum(r["reason"] in INTACT_REASONS for r in R.values())
        cat = sum(r["reason"] in CATASTROPHIC for r in R.values())
        miss = [r["landing_error_m"] for r in R.values() if r["success"]]
        fuel = [r.get("fuel_remaining_kg") for r in R.values()]
        g = [r.get("max_cargo_g") for r in R.values() if r["reason"] in INTACT_REASONS]
        common = sorted(set(R) & set(base))
        won = (
            sum(R[i]["success"] and not base[i]["success"] for i in common)
            if name != "baseline"
            else 0
        )
        lost = (
            sum(base[i]["success"] and not R[i]["success"] for i in common)
            if name != "baseline"
            else 0
        )
        paired = f"+{won} / -{lost}" if name != "baseline" else "-"
        rows.append(
            f"| {name} | {desc} | **{100 * k / n:.1f} %** ({100 * lo:.0f}-{100 * hi:.0f} %) | "
            f"{100 * intact / n:.1f} % | {cat} | {paired} | {pct(miss, 50):.0f} m | "
            f"{pct(fuel, 5):.0f} / {pct(fuel, 50):.0f} kg | {pct(g, 50):.2f} / {pct(g, 95):.2f} g |"
        )
    cost_rows = "\n".join(f"| {n} | {COST[n]} |" for n, _, _ in VARIANTS[1:])
    text = f"""# Design trade study: cargo hopper on the real route

Monte Carlo at **high fidelity** on `real_hop` (Spaceport America to Burns Flat, 761 km, 250 kg
cargo, 64 runs per variant). Every variant uses the **same seeded draws** (engine, mass, aero,
wind, turbulence, temperature), so the "paired" column counts the draws a change rescued (+) or
lost (-) against the baseline. The design changes are known to the flight software
(`overrides` in `configs/dispersions/real_hop_<variant>.yaml`).

| variant | change | mission success (95 % CI) | vehicle recovered | catastrophic | paired vs baseline | median miss (landed) | propellant left p5 / p50 | peak cargo g p50 / p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

_Catastrophic = vehicle lost (terrain impact, hull or leg crash, tip-over). Peak cargo g is over
recovered vehicles (a crash impact is not a flight load). Limit 6 g._

## Cost of each change

| variant | engineering cost |
|---|---|
{cost_rows}

## Reading the results

64 runs give a 95 % interval of roughly +/-10 points on a success rate near 85 %, so small
differences between variants are not significant on their own; the paired column is the sharper
test (a change that rescues draws without losing others is a real improvement). Regenerate with
`uv run python scripts/trade_study.py` after re-running the campaigns
(`uv run plume mc real_hop_<variant> --workers 6`).
"""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
