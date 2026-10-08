"""CLI for the test-flight programme (docs/test_flight_programme.md, docs/hop_rig.md).

* ``plume flightlog inspect <csv>``  columns, guessed roles and units, draft mapping YAML
* ``plume predict <vehicle>``        pre-flight Monte Carlo prediction (record it before flying)
* ``plume validate <csv>``           compare a real flight with its prediction, calibrate and
                                     write a validation record (docs/validation/)
* ``plume rig plan | sysid | calibrate``  hop-rig test campaign and system identification
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

flightlog_app = typer.Typer(help="Flight-computer CSV logs: inspect columns, draft a mapping.")
rig_app = typer.Typer(help="Hop-test rig: test plan, system-identification flight, calibration.")


# ----------------------------------------------------------------------------- flight logs
@flightlog_app.command("inspect")
def flightlog_inspect(
    csv: Annotated[Path, typer.Argument(help="flight-computer CSV export")],
    out: Annotated[
        Path | None, typer.Option(help="draft mapping YAML [runs/flightlogs/<csv>_draft.yaml]")
    ] = None,
):
    """Print the columns of a CSV, guess roles and units, write a draft mapping YAML."""
    from plume.config import LogMappingSpec
    from plume.flightdata.importer import load_log
    from plume.flightdata.inspect import inspect_csv, write_draft

    ins = inspect_csv(csv)
    sep = {"	": "tab", ",": "comma", ";": "semicolon"}.get(ins.delimiter, repr(ins.delimiter))
    table = Table(title=f"{csv.name}: {ins.rows:,} rows, {sep}-separated")
    for col in ("column", "valid", "min", "max", "pad value", "guess", "unit", "decided by"):
        table.add_column(
            col, justify="right" if col in ("valid", "min", "max", "pad value") else "left"
        )
    for c in ins.columns:
        table.add_row(
            c.name,
            f"{c.n_valid:,}",
            f"{c.vmin:.6g}",
            f"{c.vmax:.6g}",
            f"{c.first:.4g}",
            c.role or "[dim]-[/]",
            c.unit or "",
            c.why,
        )
    console.print(table)
    for w in ins.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    path = write_draft(ins, out or Path("runs") / "flightlogs" / f"{csv.stem}_draft.yaml")
    console.print(
        f"draft mapping: [cyan]{path}[/]  (confirm every unit, then copy it to configs/flightlogs/)"
    )
    try:
        log = load_log(csv, LogMappingSpec.model_validate(ins.mapping))
    except Exception as exc:  # the draft may be incomplete; say why
        console.print(f"[yellow]the draft does not import yet:[/] {exc}")
        return
    console.print(
        f"trial import with the draft: liftoff at raw t = {log.meta['launch_time_raw']:.2f} s, "
        f"apogee {log.apogee:,.1f} m at T+{log.t_apogee:.1f} s, max vertical speed "
        f"{log.velocity.max():.1f} m/s. [dim]Implausible numbers usually mean a wrong unit.[/]"
    )


# ----------------------------------------------------------------------------- hop rig
@rig_app.command("plan")
def rig_plan():
    """Which test excites which model parameter (system-identification plan)."""
    from plume.hoprig.sysid import MANOEUVRES

    table = Table(title="Hop-rig test campaign", show_lines=True)
    for col in ("test", "where", "inputs", "identifies (vehicle YAML)", "measure"):
        table.add_column(col)
    for m in MANOEUVRES:
        table.add_row(m.name, m.where, m.inputs, m.excites, m.measure)
    console.print(table)
    console.print("details and calibration path: [cyan]docs/hop_rig.md[/]")


@rig_app.command("sysid")
def rig_sysid(
    vehicle: Annotated[str, typer.Argument(help="vehicle preset or YAML")] = "hop_rig",
    out: Annotated[Path, typer.Option(help="rig log CSV to write")] = Path(
        "runs/rig/sysid_log.csv"
    ),
    fidelity: Annotated[str, typer.Option(help="fast | high")] = "fast",
    seed: int = 0,
    noise: Annotated[bool, typer.Option("--noise/--no-noise", help="add sensor noise")] = True,
):
    """Fly the tethered identification hover in the simulator and write a rig log.

    Use it to rehearse the pipeline (and to check `plume rig calibrate` recovers a known
    vehicle) before real rig data exists. The real rig logs the same columns."""
    from plume.config import WorldSpec, load_vehicle
    from plume.hoprig.sysid import fly_sysid

    v = load_vehicle(vehicle)
    rec, df = fly_sysid(v, WorldSpec(fidelity=fidelity), seed=seed, noise=noise)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    replay = rec.save(out.with_suffix("").with_suffix(".plume.json.gz"))
    o = rec.meta["outcome"]
    console.print(
        f"{'[green]landed[/]' if o['success'] else '[red]' + o['reason'] + '[/]'}: "
        f"{len(df):,} log rows, {o['metrics']['flight_time_s']:.1f} s"
    )
    console.print(f"rig log: [cyan]{out}[/]   replay: [cyan]{replay}[/]")


@rig_app.command("calibrate")
def rig_calibrate(
    log: Annotated[Path, typer.Argument(help="rig log CSV (columns: docs/hop_rig.md)")],
    vehicle: Annotated[str, typer.Option(help="nominal vehicle preset or YAML")] = "hop_rig",
    out_dir: Annotated[Path, typer.Option(help="where to write results")] = Path(
        "runs/rig/calibration"
    ),
    p_amb: Annotated[float, typer.Option(help="ambient pressure at the rig, Pa")] = 101325.0,
):
    """Fit engine (thrust, Isp, throttle lag), gimbal actuator and inertia to a rig log."""
    from plume.config import dump_yaml, load_vehicle
    from plume.hoprig.sysid import calibrate_rig, read_rig_log

    v = load_vehicle(vehicle)
    df = read_rig_log(log)
    with console.status("fitting engine, gimbal and inertia..."):
        cal = calibrate_rig(df, v, p_amb=p_amb)
    table = Table(title=f"rig calibration: {log.name}")
    for col in ("parameter", "model", "fit"):
        table.add_column(col, justify="left" if col == "parameter" else "right")
    for row in cal.rows:
        table.add_row(*row)
    console.print(table)
    path = dump_yaml(cal.vehicle, out_dir / f"{v.name}_{log.stem}.yaml")
    console.print(f"calibrated vehicle: [cyan]{path}[/]")
    console.print(
        "[dim]Check every fitted value is physically plausible before using it; refly the "
        "scenario with the calibrated vehicle and compare against the log.[/]"
    )


def register(app: typer.Typer) -> None:
    """Attach the test-flight commands to the main ``plume`` app."""
    app.add_typer(flightlog_app, name="flightlog")
    app.add_typer(rig_app, name="rig")
