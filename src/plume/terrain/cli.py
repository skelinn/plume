"""``plume terrain ...``: fetch real-world terrain, inspect heightmaps, map landing hazards.

Mount in the main CLI with ``app.add_typer(terrain_app, name="terrain")``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

terrain_app = typer.Typer(
    help="Real-world terrain: Copernicus DEM download, projection, landing hazards.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _parse_pair(text: str, what: str) -> tuple[float, float]:
    try:
        a, b = (float(v) for v in text.replace(" ", "").split(","))
    except ValueError as e:
        raise typer.BadParameter(f"{what} must look like 'LAT,LON' (got {text!r})") from e
    return a, b


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
        force=True,
    )


def _summary(hm, title: str) -> Table:
    h = hm.heights
    t = Table(title=title, show_header=False, title_style="bold")
    t.add_row("name", hm.name)
    t.add_row("grid", f"{h.shape[1]} x {h.shape[0]}  ({hm.dx:.2f} x {hm.dy:.2f} m)")
    t.add_row(
        "extent x / y",
        f"{hm.x_min / 1e3:.3f} .. {hm.x_max / 1e3:.3f} km / "
        f"{hm.y_min / 1e3:.3f} .. {hm.y_max / 1e3:.3f} km",
    )
    t.add_row("height", f"{h.min():.1f} .. {h.max():.1f} m (mean {h.mean():.1f})")
    m = hm.meta
    if "source" in m:
        t.add_row("source", str(m["source"]))
    if "origin_lat" in m:
        t.add_row("origin (A)", f"{m['origin_lat']:.6f}, {m['origin_lon']:.6f}")
    if "tiles" in m:
        t.add_row("tiles", f"{len(m['tiles'])} DEM + {len(m.get('sea_level_tiles', []))} sea")
    if "source_resolution_m" in m:
        t.add_row("source posting", f"~{m['source_resolution_m']} m")
    if "vertical_datum" in m:
        t.add_row("vertical datum", str(m["vertical_datum"]))
    if "route_length_m" in m:
        t.add_row(
            "route A->B",
            f"{m['route_length_m'] / 1e3:.2f} km @ {m['route_azimuth_deg']:.2f} deg, "
            f"B at map ({m['site_b'][0] / 1e3:.3f}, {m['site_b'][1] / 1e3:.3f}) km",
        )
    if "site_xy" in m:
        t.add_row("site map xy", f"({m['site_xy'][0]:.2f}, {m['site_xy'][1]:.2f}) m")
    synth = m.get("synthetic_detail")
    if synth is not None:
        t.add_row(
            "synthetic detail",
            "[yellow]YES (procedural, not measured)[/]" if synth else "[green]no[/]",
        )
    if m.get("modifications"):
        t.add_row("modifications", "; ".join(str(x.get("type")) for x in m["modifications"]))
    return t


@terrain_app.command()
def fetch(
    site: Annotated[str, typer.Option(help="centre / launch site 'LAT,LON' (degrees)")],
    out: Annotated[Path, typer.Option(help="output heightmap YAML, e.g. data/terrain/x.yaml")],
    site2: Annotated[
        str | None, typer.Option(help="second site 'LAT,LON': fetch the A->B corridor")
    ] = None,
    origin: Annotated[
        str | None,
        typer.Option(help="projection centre 'LAT,LON' (default: --site); use site A for B tiles"),
    ] = None,
    radius_km: Annotated[
        float, typer.Option(help="half-width of the square (or corridor padding), km")
    ] = 20.0,
    resolution: Annotated[float, typer.Option(help="grid spacing, m")] = 100.0,
    product: Annotated[str, typer.Option(help="glo30 | glo90")] = "glo30",
    fmt: Annotated[str, typer.Option(help="png (16-bit) | npy (float32)")] = "png",
    name: Annotated[str | None, typer.Option(help="heightmap name (default: file stem)")] = None,
    synthetic_detail: Annotated[
        bool, typer.Option(help="add flagged procedural roughness + rocks (landing tiles)")
    ] = False,
    roughness: Annotated[float, typer.Option(help="synthetic roughness RMS, m")] = 0.15,
    rock_density: Annotated[float, typer.Option(help="synthetic rocks per m^2")] = 1.0e-4,
    seed: int = 0,
    flatten_radius: Annotated[
        float, typer.Option(help="flatten a prepared pad of this radius (m) at the site")
    ] = 0.0,
    geotiff: Annotated[
        Path | None, typer.Option(help="use this survey GeoTIFF (any CRS) instead of the DEM")
    ] = None,
    max_cells: Annotated[int, typer.Option(help="corridor: max grid cells per side")] = 4096,
    offline: Annotated[bool, typer.Option(help="use cached tiles only")] = False,
    verbose: Annotated[bool, typer.Option("--verbose/--quiet")] = True,
):
    """Download DEM tiles and write a projected Heightmap (YAML + PNG/NPY)."""
    from plume.terrain import dem
    from plume.terrain.heightmap import flatten_sites

    _logging(verbose)
    a = _parse_pair(site, "--site")
    o = _parse_pair(origin, "--origin") if origin else a
    nm = name or out.stem
    try:
        if site2:
            if origin:
                raise typer.BadParameter("--origin is not used with --site2 (A is the origin)")
            hm = dem.corridor_heightmap(
                a,
                _parse_pair(site2, "--site2"),
                radius_km * 1e3,
                resolution,
                max_cells=max_cells,
                name=nm,
                product=product,
                offline=offline,
            )
        else:
            hm = dem.landing_tile(
                a,
                radius_km * 1e3,
                resolution,
                origin=o,
                synthetic_detail=synthetic_detail,
                seed=seed,
                roughness=roughness,
                rock_density=rock_density,
                geotiff=geotiff,
                name=nm,
                product=product,
                offline=offline,
            )
            if flatten_radius > 0:
                cx, cy = hm.meta["site_xy"]
                meta = dict(hm.meta)
                hm = flatten_sites(hm, [(cx, cy)], flatten_radius, 2.0 * flatten_radius)
                meta.setdefault("modifications", []).append(
                    {
                        "type": "flattened_pad",
                        "centre_xy": [cx, cy],
                        "radius_m": flatten_radius,
                        "blend_m": 2.0 * flatten_radius,
                        "height_m": round(float(hm.height(cx, cy)), 3),
                    }
                )
                hm.meta = meta
    except dem.DemDownloadError as e:
        console.print(f"[red]download failed:[/] {e}")
        raise typer.Exit(1) from e
    path = hm.save(out, fmt=fmt)
    console.print(_summary(hm, f"wrote {path}"))
    if "attribution" in hm.meta:
        console.print(f"[dim]{hm.meta['attribution']}[/]")


@terrain_app.command()
def info(yaml_path: Annotated[Path, typer.Argument(help="heightmap YAML")]):
    """Summarise a heightmap (extent, heights, source, datum, synthetic-detail flag)."""
    from plume.terrain.heightmap import Heightmap

    hm = Heightmap.load(yaml_path)
    console.print(_summary(hm, str(yaml_path)))
    m = hm.meta
    if "origin_lat" in m:
        from plume.terrain.dem import site_projection

        proj = site_projection(m["origin_lat"], m["origin_lon"])
        lat, lon = proj.inverse(
            np.array([hm.x_min, hm.x_max, hm.x_max, hm.x_min]),
            np.array([hm.y_min, hm.y_min, hm.y_max, hm.y_max]),
        )
        corners = ", ".join(f"({la:.4f}, {lo:.4f})" for la, lo in zip(lat, lon, strict=True))
        console.print(f"corners SW, SE, NE, NW (lat, lon): {corners}")
    if m.get("synthetic_detail_params"):
        p = m["synthetic_detail_params"]
        console.print(
            f"[yellow]synthetic detail:[/] roughness {p['roughness_rms_m']} m RMS "
            f"(< {p['cutoff_wavelength_m']:.0f} m wavelengths), {p['rock_count']} rocks, "
            f"seed {p['seed']}"
        )
    if "attribution" in m:
        console.print(f"[dim]{m['attribution']}[/]")


@terrain_app.command()
def hazard(
    yaml_path: Annotated[Path, typer.Argument(help="heightmap YAML")],
    footprint: Annotated[float, typer.Option(help="landing footprint radius, m")] = 3.0,
    max_slope: Annotated[float, typer.Option(help="max slope over the footprint, deg")] = 8.0,
    max_roughness: Annotated[float, typer.Option(help="max RMS roughness, m")] = 0.1,
    target: Annotated[
        str | None, typer.Option(help="target map 'X,Y' in m (default: site / tile centre)")
    ] = None,
    clearance: Annotated[float, typer.Option(help="extra safe margin around the point, m")] = 0.0,
    png: Annotated[Path | None, typer.Option(help="write a safe/unsafe mask PNG")] = None,
):
    """Slope / roughness hazard map and the nearest safe landing point to a target."""
    from plume.terrain.dem import hazard_map
    from plume.terrain.heightmap import Heightmap

    hm = Heightmap.load(yaml_path)
    hz = hazard_map(hm, footprint, max_slope, max_roughness)
    if target:
        tx, ty = _parse_pair(target, "--target")
    elif "site_xy" in hm.meta:
        tx, ty = hm.meta["site_xy"]
    else:
        tx, ty = 0.5 * (hm.x_min + hm.x_max), 0.5 * (hm.y_min + hm.y_max)
    v = hz.valid
    t = Table(title=f"hazards: {hm.name}", show_header=False, title_style="bold")
    t.add_row("footprint", f"r = {footprint} m on a {hm.dx:.2f} m grid")
    t.add_row("limits", f"slope <= {max_slope} deg, roughness <= {max_roughness} m RMS")
    t.add_row("safe area", f"{100 * hz.safe_fraction:.1f} %")
    t.add_row(
        "slope",
        f"median {np.median(hz.slope_deg[v]):.2f}, p99 {np.percentile(hz.slope_deg[v], 99):.2f}, "
        f"max {hz.slope_deg[v].max():.2f} deg",
    )
    t.add_row(
        "roughness",
        f"median {np.median(hz.roughness[v]):.3f}, "
        f"p99 {np.percentile(hz.roughness[v], 99):.3f}, max {hz.roughness[v].max():.3f} m",
    )
    t.add_row(
        "target",
        f"({tx:.1f}, {ty:.1f}) m: {'[green]safe' if hz.is_safe(tx, ty) else '[red]UNSAFE'}[/]",
    )
    best = hz.nearest_safe(tx, ty, clearance=clearance)
    if best is None:
        t.add_row("nearest safe", "[red]none in this map[/]")
    else:
        t.add_row("nearest safe", f"({best[0]:.1f}, {best[1]:.1f}) m, {best[2]:.1f} m away")
    if hm.meta.get("synthetic_detail"):
        t.add_row("note", "[yellow]map contains flagged synthetic detail[/]")
    console.print(t)
    if png:
        from PIL import Image

        img = np.zeros((*hz.safe.shape, 3), np.uint8)
        img[hz.safe] = (40, 170, 70)
        img[~hz.safe & v] = (200, 50, 40)
        img[~v] = (90, 90, 90)
        png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img[::-1]).save(png)
        console.print(f"mask written to {png}")
