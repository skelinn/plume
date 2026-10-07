"""Plume command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from plume.config import WindSpec, WorldSpec, load_vehicle

app = typer.Typer(
    help="Plume: 6-DOF reusable cargo rocket simulator.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _outcome_table(outcome: dict | None, title: str) -> Table:
    table = Table(title=title, show_header=False, title_style="bold")
    if outcome:
        table.add_row(
            "result", "[green]success[/]" if outcome["success"] else f"[red]{outcome['reason']}[/]"
        )
        for k, v in outcome.get("metrics", {}).items():
            table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    return table


def _live(enabled: bool):
    if not enabled:
        return None, None
    try:
        from plume.viz.live import LiveStreamer
    except ImportError:  # viz extra missing
        console.print("[yellow]live streaming unavailable (install the viz extra)[/]")
        return None, None
    streamer = LiveStreamer()
    return streamer, streamer.push


@app.command()
def sim(
    vehicle: Annotated[
        str, typer.Argument(help="vehicle preset name or YAML path")
    ] = "lander_small",
    script: Annotated[str, typer.Option(help="scenario: hop_test | drop")] = "hop_test",
    out: Annotated[Path | None, typer.Option(help="replay output path")] = None,
    seed: int = 0,
    wind: Annotated[float, typer.Option(help="mean wind speed, m/s")] = 0.0,
    gusts: Annotated[float, typer.Option(help="max gust speed, m/s")] = 0.0,
    live: Annotated[bool, typer.Option(help="stream to a running `plume viz`")] = False,
):
    """Run a scripted 6-DOF flight and save a replay."""
    from plume.scenarios import SCENARIOS

    v = load_vehicle(vehicle)
    world = WorldSpec(
        wind=WindSpec(
            speed=wind, turbulence=0.15 * wind, gust_rate=0.1 if gusts else 0.0, gust_max=gusts
        )
    )
    out = out or Path("runs") / f"{script}_{v.name}.plume.json.gz"
    streamer, on_frame = _live(live)
    rec = SCENARIOS[script](v, world, seed=seed, on_frame=on_frame)
    if streamer:
        streamer.end(rec.meta.get("outcome"))
    path = rec.save(out)
    console.print(_outcome_table(rec.meta.get("outcome"), f"{script} - {v.name}"))
    console.print(f"replay: [cyan]{path}[/]")


@app.command()
def viz(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
):
    """Serve the 3D viewer (replays from runs/ and data/replays/)."""
    from plume.viz.server import serve

    serve(host=host, port=port, open_browser=open_browser)


@app.command()
def info(vehicle: Annotated[str, typer.Argument()] = "lander_small"):
    """Print derived properties of a vehicle preset."""
    from plume.constants import G0
    from plume.physics.massprops import MassModel
    from plume.physics.propulsion import Engine

    v = load_vehicle(vehicle)
    mm = MassModel(v)
    full = mm.evaluate([t.initial_mass for t in v.tanks], v.rcs.propellant)
    empty = mm.evaluate([0.0 for _ in v.tanks], v.rcs.propellant)
    e = Engine(v.engine, v.prop_capacity)
    import math

    table = Table(title=v.name, show_header=False)
    table.add_row("description", v.description)
    table.add_row("wet / dry mass", f"{full.mass:,.0f} / {empty.mass:,.0f} kg")
    table.add_row("cargo", f"{v.cargo.mass:,.0f} kg")
    table.add_row(
        "max thrust (SL / vac)", f"{e.max_thrust(101325):,.0f} / {e.max_thrust(0):,.0f} N"
    )
    table.add_row("T/W at liftoff", f"{e.max_thrust(101325) / (full.mass * G0):.2f}")
    if not e.is_solid:
        dv = v.engine.isp_vac * G0 * math.log(full.mass / empty.mass)
        table.add_row("ideal delta-v (vac)", f"{dv:,.0f} m/s")
        table.add_row("burn time at 100%", f"{v.prop_initial / e.mdot_max:,.1f} s")
    else:
        table.add_row("total impulse", f"{e.total_impulse:,.1f} N s")
    console.print(table)


if __name__ == "__main__":
    app()
