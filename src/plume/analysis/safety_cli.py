"""``plume safety``: flight-safety analysis export (see plume.analysis.safety)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _origin(text: str | None):
    if not text:
        return None
    lat, lon = (float(x) for x in text.split(","))
    return lat, lon


def safety(
    source: Annotated[
        Path, typer.Argument(help="Monte Carlo directory (runs.jsonl) or a replay file")
    ],
    replay: Annotated[
        Path | None, typer.Option(help="nominal replay for the ground track and IIP trace")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="output folder [runs/safety/<name>]")] = None,
    iip_step: Annotated[float, typer.Option(help="IIP sample spacing, s")] = 2.0,
    drag: Annotated[bool, typer.Option("--drag/--vacuum-only", help="drag-aware IIP")] = True,
    corridor_km: Annotated[float, typer.Option(help="IIP corridor half-width, km")] = 5.0,
    impact_buffer_km: Annotated[float, typer.Option(help="buffer around failure impacts")] = 5.0,
    landing_buffer_m: Annotated[float, typer.Option(help="buffer around touchdowns")] = 500.0,
    origin: Annotated[
        str | None, typer.Option(help="lat,lon of the launch site if the input has none")
    ] = None,
    embed: Annotated[
        Path | None,
        typer.Option(help="also write a copy of the replay with the overlay for the viewer"),
    ] = None,
):
    """Flight-safety analysis: ground track, IIP trace, landing dispersion, failure impact
    points, failure probability by phase and hazard areas as GeoJSON, KML and HTML.
    An engineering input to a licence application, not a certified analysis."""
    from plume.analysis.safety import (
        GeoFrame,
        analyze,
        load_replay,
        viewer_overlay,
        write_outputs,
    )

    mc_dir = source if source.is_dir() else None
    rep = replay if mc_dir is not None else source
    if mc_dir is not None and not (mc_dir / "runs.jsonl").exists():
        console.print(f"[red]{mc_dir}: no runs.jsonl (not a Monte Carlo directory)[/]")
        raise typer.Exit(2)
    rep_data = load_replay(rep) if rep is not None else None
    sa = analyze(
        mc_dir=mc_dir,
        replay=rep_data,
        origin=_origin(origin),
        iip_step_s=iip_step,
        drag=drag,
        corridor_half_width_m=corridor_km * 1000.0,
        impact_buffer_m=impact_buffer_km * 1000.0,
        landing_buffer_m=landing_buffer_m,
    )
    stem = (mc_dir.name if mc_dir else source.name.split(".")[0]).replace(" ", "_")
    out = out or Path("runs") / "safety" / stem
    paths = write_outputs(sa, out)
    table = Table(title=f"Flight safety: {sa.name}", show_header=True)
    for col in ("phase", "failed", "P(failure)", "95 % interval", "vehicle lost"):
        table.add_column(col)
    for p in sa.phases:
        table.add_row(
            p["label"],
            str(p["failures"]),
            f"{100 * p['probability']:.1f} %",
            f"{100 * p['ci95'][0]:.1f}-{100 * p['ci95'][1]:.1f} %",
            str(p["vehicle_lost"]),
        )
    if sa.phases:
        console.print(table)
    for h in sa.hazard_areas:
        console.print(f"{h['label']}: {h['area_km2']:,.1f} km2  ({h['basis']})")
    if embed is not None and rep_data is not None:
        geo = GeoFrame.from_replay_meta(rep_data["meta"], _origin(origin))
        rep_data["meta"]["safety"] = viewer_overlay(sa, geo)
        from plume.recording import save_replay

        save_replay(rep_data, embed)
        console.print(f"replay with overlay: [cyan]{embed}[/]")
    console.print("[yellow]engineering input, not a certified flight safety analysis[/]")
    console.print("  ".join(f"{k}: [cyan]{v}[/]" for k, v in paths.items()))
