"""Build the planner's vendored world map from Natural Earth GeoJSON (public domain).

Download (https://github.com/nvkelso/natural-earth-vector, ``geojson/``):

* ``ne_50m_coastline.geojson``
* ``ne_50m_admin_0_boundary_lines_land.geojson``
* ``ne_50m_admin_1_states_provinces_lines.geojson``

then run::

    uv run python scripts/build_planner_map.py <download dir>

It writes ``src/plume/viz/static/planner/world.json``: polylines as flat
``[lon0, lat0, lon1, lat1, ...]`` arrays rounded to 0.01 deg (about 1 km), with
consecutive duplicates removed. Made with Natural Earth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/plume/viz/static/planner/world.json"
LAYERS = {
    "coast": "ne_50m_coastline.geojson",
    "borders": "ne_50m_admin_0_boundary_lines_land.geojson",
    "states": "ne_50m_admin_1_states_provinces_lines.geojson",
}


def lines(geom: dict):
    if geom["type"] == "LineString":
        yield geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        yield from geom["coordinates"]
    elif geom["type"] == "Polygon":
        yield from geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield from poly


def compact(coords, digits: int = 2) -> list[float]:
    out: list[float] = []
    last = None
    for lon, lat, *_ in coords:
        p = (round(float(lon), digits), round(float(lat), digits))
        if p != last:
            out.extend(p)
            last = p
    return out


def main(src: str) -> None:
    src_dir = Path(src)
    data = {
        "source": "Made with Natural Earth (public domain), 1:50m, naturalearthdata.com",
        "format": "polylines as flat [lon, lat, lon, lat, ...] arrays, degrees",
    }
    for key, name in LAYERS.items():
        gj = json.loads((src_dir / name).read_text(encoding="utf-8"))
        polys = []
        for feat in gj["features"]:
            for line in lines(feat["geometry"]):
                c = compact(line)
                if len(c) >= 4:
                    polys.append(c)
        data[key] = polys
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
