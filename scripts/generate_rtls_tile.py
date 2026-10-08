"""Generate the flat booster landing-zone tile for the orbital demo (``demo_orbit``).

    uv run python scripts/generate_rtls_tile.py

Writes ``data/terrain/demo_lz_rtls``: a prepared, flat 300 m tile 1.5 km south-west
(uprange) of Pad A, at the height of the flattened launch-site plateau.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plume.terrain import Heightmap

OUT = Path("data/terrain")
CENTER = (-1340.0, -670.0)
HALF = 150.0


def main() -> None:
    base = Heightmap.load(OUT / "demo_region.yaml")
    h = float(base.height(0.0, 0.0))  # the plateau is flat within 3 km of Pad A
    n = round(2 * HALF / 2.0) + 1
    tile = Heightmap(
        np.full((n, n), h),
        CENTER[0] - HALF,
        CENTER[0] + HALF,
        CENTER[1] - HALF,
        CENTER[1] + HALF,
        "demo_lz_rtls",
    )
    tile.save(OUT / "demo_lz_rtls.yaml")
    print("demo_lz_rtls at", CENTER, "height", round(h, 2), "base there", base.height(*CENTER))


if __name__ == "__main__":
    main()
