"""Generate the bundled demo terrain for cargo-hop missions.

    uv run python scripts/generate_terrain.py

Writes to ``data/terrain/``:

* ``demo_region``  -- 880 x 400 km regional map at 1 km resolution, launch site A at
                      map (0, 0) flattened, landing site B ~750 km away smoothed.
* ``demo_pad_a``   -- 400 m launch-pad tile at 2 m (prepared, flat).
* ``demo_lz_b``    -- 2 km landing-zone tile at 2.5 m: *unprepared* ground with
                      undulation, bumps and scattered rocks.

Map coordinates are east/north arc lengths (m) from site A (see
``SphericalGravity.surface_point``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plume.terrain import Heightmap, flatten_sites, generate_detail_tile, generate_regional_map

OUT = Path("data/terrain")
SITE_A = (0.0, 0.0)
SITE_B = (730_000.0, 170_000.0)


def main() -> None:
    base = generate_regional_map(
        (-60_000.0, 820_000.0), (-120_000.0, 280_000.0), 1000.0, seed=7, relief=1600.0, base=250.0
    )
    base = flatten_sites(base, [SITE_A], radius=3000.0, blend=8000.0)
    base = flatten_sites(base, [SITE_B], radius=1500.0, blend=12_000.0)
    base.name = "demo_region"
    base.meta = {"site_a": list(SITE_A), "site_b": list(SITE_B), "resolution_m": 1000.0}
    base.save(OUT / "demo_region.yaml")

    pad = generate_detail_tile(base, SITE_A, 200.0, 2.0, seed=11, roughness=0.0, rock_density=0.0)
    pad.heights[:] = base.height(*SITE_A)
    pad.name = "demo_pad_a"
    pad.save(OUT / "demo_pad_a.yaml")

    lz = generate_detail_tile(
        base,
        SITE_B,
        1000.0,
        2.5,
        seed=12,
        roughness=1.2,
        rock_density=1.2e-4,
        rock_height=(0.1, 0.5),
    )
    lz.name = "demo_lz_b"
    lz.save(OUT / "demo_lz_b.yaml")

    for hm in (base, pad, lz):
        h = hm.heights
        print(f"{hm.name:12s} {h.shape[1]}x{h.shape[0]}  z {h.min():8.1f} .. {h.max():8.1f} m")
    slopes = [
        lz.slope_deg(SITE_B[0] + dx, SITE_B[1] + dy) for dx in (-50, 0, 50) for dy in (-50, 0, 50)
    ]
    print(f"landing zone slope near B: max {max(slopes):.1f} deg")
    print("site B height", round(Heightmap.load(OUT / "demo_lz_b.yaml").height(*SITE_B), 2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
