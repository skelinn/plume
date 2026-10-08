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
    fidelity: Annotated[str, typer.Option(help="fast | high")] = "fast",
):
    """Run a scripted 6-DOF flight and save a replay."""
    from plume.scenarios import SCENARIOS

    v = load_vehicle(vehicle)
    world = WorldSpec(
        fidelity=fidelity,
        wind=WindSpec(
            speed=wind, turbulence=0.15 * wind, gust_rate=0.1 if gusts else 0.0, gust_max=gusts
        ),
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
def land(
    controller: Annotated[str, typer.Option(help="pid | ppo")] = "pid",
    stage: Annotated[str, typer.Option(help="curriculum stage name or index")] = "full_descent",
    seed: int = 1,
    episodes: int = 1,
    run_dir: Annotated[Path, typer.Option(help="PPO run directory")] = Path("runs/ppo_landing"),
    save: Annotated[Path | None, typer.Option(help="replay path (first episode)")] = None,
    fidelity: Annotated[
        str, typer.Option(help="fast | high (aero database, actuator dynamics)")
    ] = "fast",
):
    """Fly the landing task with the PID/guidance autopilot or a trained PPO agent."""
    from plume.envs.landing_env import AutopilotPolicy, LandingEnv
    from plume.recording import save_replay
    from plume.rl.evaluate import run_episodes, sb3_policy_factory

    env = LandingEnv(record=True, fixed_stage=True, fidelity=fidelity)
    names = [s.name for s in env.stages]
    idx = int(stage) if stage.isdigit() else names.index(stage)
    if controller == "pid":
        make = AutopilotPolicy
    else:
        from plume.rl.train import load_trained

        model, vecnorm = load_trained(run_dir)
        make = sb3_policy_factory(model, vecnorm)
    res = run_episodes(env, make, idx, episodes, seed=seed, controller=controller)
    out = save or Path("runs") / f"land_{controller}_{names[idx]}_{seed}.plume.json.gz"
    if env.last_replay is not None and episodes == 1:
        save_replay(env.last_replay, out)
        console.print(
            _outcome_table(env.last_replay["meta"].get("outcome"), f"{controller} - {names[idx]}")
        )
        console.print(f"replay: [cyan]{out}[/]")
    else:
        console.print(res.row(), res.reasons)


@app.command()
def train(
    config: Annotated[Path | None, typer.Option(help="training config YAML")] = None,
    when_idle: Annotated[
        bool, typer.Option("--when-idle", help="only train while the PC is idle")
    ] = False,
    status: Annotated[bool, typer.Option("--status", help="show progress and exit")] = False,
    timesteps: Annotated[int | None, typer.Option(help="cap steps for this session")] = None,
    device: Annotated[str | None, typer.Option(help="cuda | cpu")] = None,
    fresh: Annotated[bool, typer.Option("--fresh", help="ignore existing checkpoints")] = False,
):
    """Train the PPO landing agent (curriculum, resumable, optionally idle-aware)."""
    from plume.rl.train import TrainConfig, eta_seconds, read_state

    cfg = TrainConfig.load(config)
    if device:
        cfg.device = device
    run_dir = Path(cfg.run_dir)
    if status:
        st = read_state(run_dir)
        table = Table(title=f"training: {run_dir}", show_header=False)
        pct = 100 * st["timesteps"] / cfg.total_timesteps
        table.add_row(
            "progress", f"{st['timesteps']:,} / {cfg.total_timesteps:,} steps ({pct:.1f}%)"
        )
        table.add_row("curriculum stage", str(st["stage"]))
        table.add_row("sessions", str(st.get("sessions", 0)))
        last = st.get("last_session") or {}
        if last:
            table.add_row(
                "last session",
                f"{last.get('steps', 0):,} steps at {last.get('steps_per_second', 0):,.0f}/s on {last.get('device')}",
            )
        eta = eta_seconds(st, cfg.total_timesteps)
        if eta == eta:  # not NaN
            table.add_row("remaining (active time)", f"{eta / 3600:.1f} h")
        if st.get("waiting_reason"):
            table.add_row("now", st["waiting_reason"])
        if st.get("last_pause_reason"):
            table.add_row("last pause", f"{st['last_pause_reason']} at {st.get('last_pause_time')}")
        table.add_row("updated", str(st.get("updated", "-")))
        console.print(table)
        return
    if fresh and run_dir.exists():
        for f in ("model.zip", "vecnormalize.pkl", "state.json"):
            (run_dir / f).unlink(missing_ok=True)
    if when_idle:
        from plume.rl.idle import IdleConfig, Supervisor

        Supervisor(cfg, IdleConfig.load(), config_path=str(config) if config else None).run()
    else:
        from plume.rl.train import train as run_train

        st = run_train(cfg, timesteps=timesteps)
        console.print(f"trained to {st['timesteps']:,} steps, stage {st['stage']}")


@app.command()
def hop(
    mission: Annotated[str, typer.Argument(help="mission preset name or YAML path")] = "demo_hop",
    cargo: Annotated[float | None, typer.Option(help="override cargo mass, kg")] = None,
    seed: int = 0,
    out: Annotated[Path | None, typer.Option(help="replay output path")] = None,
    live: Annotated[bool, typer.Option(help="stream to a running `plume viz`")] = False,
    fidelity: Annotated[
        str | None, typer.Option(help="fast | high (WGS-84, rotating Earth, RK4-stage forces)")
    ] = None,
):
    """Fly a point-to-point cargo hop over terrain and score it."""
    from plume.config import load_mission
    from plume.missions.hop import run_mission

    spec = load_mission(mission)
    streamer, on_frame = _live(live)
    if streamer:
        console.print("[dim]streaming to the viewer...[/]")
    with console.status(f"flying {spec.name} (planning ascent, then ~10 min of flight)..."):
        run = run_mission(spec, seed=seed, cargo_mass=cargo, on_frame=on_frame, fidelity=fidelity)
    if streamer:
        streamer.end(run.recorder.meta.get("outcome"))
    path = run.recorder.save(out or Path("runs") / f"hop_{spec.name}.plume.json.gz")
    res = run.result
    table = Table(title=f"{spec.name}: {spec.launch.name} -> {spec.target.name}", show_header=False)
    table.add_row(
        "result", "[green]landed on target[/]" if res.success else f"[red]{res.reason}[/]"
    )
    table.add_row(
        "landing error", f"{res.landing_error_m:,.1f} m (radius {spec.target_radius:g} m)"
    )
    table.add_row("fuel used / left", f"{res.fuel_used_kg:,.0f} / {res.fuel_remaining_kg:,.0f} kg")
    table.add_row(
        "max cargo load", f"{res.max_cargo_g:.2f} g (limit {spec.guidance.cargo_g_limit:g} g)"
    )
    table.add_row("flight time", f"{res.flight_time_s / 60:.1f} min")
    table.add_row("apogee", f"{res.apogee_km:,.0f} km")
    table.add_row(
        "touchdown",
        f"{res.touchdown_vz_mps:.2f} m/s down, {res.touchdown_vh_mps:.2f} m/s across, {res.ground_slope_deg:.1f} deg slope",
    )
    table.add_row("score", f"[bold]{res.score:.1f}[/]")
    console.print(table)
    console.print(f"replay: [cyan]{path}[/]")


def _load_flight(csv: Path, mapping: str):
    from plume.flightdata.importer import load_log

    log = load_log(csv, mapping)
    table = Table(title=f"flight log: {csv.name}", show_header=False)
    table.add_row("mapping", log.meta["mapping"])
    table.add_row("samples", f"{len(log.t):,} at {log.meta['rate_hz']:g} Hz")
    table.add_row("liftoff (raw clock)", f"{log.meta['launch_time_raw']:.3f} s")
    table.add_row("apogee", f"{log.apogee:,.1f} m at T+{log.t_apogee:.2f} s")
    table.add_row("max velocity (fused)", f"{log.velocity.max():.1f} m/s")
    if log.burnout_time:
        table.add_row("burnout", f"T+{log.burnout_time:.2f} s")
    table.add_row(
        "sensors",
        ", ".join(k for k in ("accel", "gyro", "east") if getattr(log, k) is not None).replace(
            "east", "gps"
        ),
    )
    return log, table


@app.command("import-log")
def import_log(
    csv: Annotated[Path, typer.Argument(help="flight computer CSV export")],
    mapping: Annotated[str, typer.Option(help="column mapping name or YAML")] = "generic_altimeter",
    vehicle: Annotated[
        str, typer.Option(help="vehicle used for the 3-D model in the replay")
    ] = "hobby_rocket",
    out: Annotated[Path | None, typer.Option(help="replay output")] = None,
):
    """Import a CSV log (unit conversion, liftoff detection, Kalman velocity) to a replay."""
    from plume.flightdata.compare import log_to_replay
    from plume.recording import save_replay

    log, table = _load_flight(csv, mapping)
    console.print(table)
    path = save_replay(
        log_to_replay(log, load_vehicle(vehicle), f"Real flight: {csv.stem}"),
        out or Path("runs") / f"real_{csv.stem}.plume.json.gz",
    )
    console.print(f"replay: [cyan]{path}[/]")


@app.command()
def calibrate(
    csv: Annotated[Path, typer.Argument(help="flight computer CSV export")],
    mapping: Annotated[str, typer.Option(help="column mapping name or YAML")] = "generic_altimeter",
    vehicle: Annotated[
        str, typer.Option(help="vehicle preset/YAML with the nominal motor")
    ] = "hobby_rocket",
    rail: Annotated[float, typer.Option(help="launch rail length, m")] = 1.5,
    mode: Annotated[str, typer.Option(help="scale | knots (thrust-curve shape)")] = "scale",
    out_dir: Annotated[Path, typer.Option(help="where to write results")] = Path(
        "runs/calibration"
    ),
):
    """Fit drag, thrust curve and parachute to a real flight; overlay real vs sim."""
    from plume.config import dump_yaml
    from plume.flightdata.calibrate import calibrate as run_calibration
    from plume.flightdata.calibrate import describe
    from plume.flightdata.compare import log_to_replay, plot_comparison, trace_to_replay
    from plume.recording import save_replay

    v = load_vehicle(vehicle)
    log, table = _load_flight(csv, mapping)
    console.print(table)
    with console.status("fitting drag + thrust curve (least squares on the 3-DOF model)..."):
        res = run_calibration(log, v, rail_length=rail, mode=mode)
    fit = Table(title="calibration", show_header=False)
    for k, val in describe(res):
        fit.add_row(k, val)
    console.print(fit)
    for note in res.notes:
        console.print(f"[yellow]{note}[/]")
    stem = csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dump_yaml(res.vehicle, out_dir / f"{v.name}_{stem}.yaml")
    png = plot_comparison(
        log,
        [res.nominal_trace, res.calibrated_trace],
        out_dir / f"{stem}_overlay.png",
        f"{stem}: real vs simulated",
    )
    real = save_replay(
        log_to_replay(log, v, f"Real flight: {stem}"), Path("runs") / f"real_{stem}.plume.json.gz"
    )
    sim = save_replay(
        trace_to_replay(res.calibrated_trace, res.vehicle, f"Calibrated sim: {stem}"),
        Path("runs") / f"sim_{stem}.plume.json.gz",
    )
    console.print(f"calibrated vehicle: [cyan]{yaml_path}[/]")
    console.print(f"overlay plot:       [cyan]{png}[/]")
    console.print(f"replays:            [cyan]{real}[/], [cyan]{sim}[/]")
    console.print(
        f"compare in the viewer: [cyan]plume viz[/] -> ?compare=real_{stem}.plume.json.gz,sim_{stem}.plume.json.gz"
    )


@app.command()
def bench(
    episodes: Annotated[int, typer.Option(help="episodes per stage and controller")] = 200,
    run_dir: Path = Path("runs/ppo_landing"),
    readme: Annotated[bool, typer.Option(help="write the table into README.md")] = True,
    workers: Annotated[
        int | None, typer.Option(help="parallel processes (default: cores - 2)")
    ] = None,
    pid_only: Annotated[bool, typer.Option("--pid-only", help="skip the PPO agent")] = False,
):
    """PID vs PPO on every curriculum stage -> results table (and README)."""
    from plume.rl.benchmark import run_benchmark

    run_benchmark(
        episodes=episodes,
        run_dir=run_dir,
        update_readme=readme,
        console=console,
        workers=workers,
        include_ppo=not pid_only,
    )


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
    full = mm.evaluate([t.initial_mass for t in v.tanks], v.rcs.gas)
    empty = mm.evaluate([0.0 for _ in v.tanks], v.rcs.gas)
    e = Engine(v.engine, v.prop_capacity)
    import math

    table = Table(title=v.name, show_header=False)
    table.add_row("description", v.description)
    fmt = ",.0f" if full.mass >= 100 else ".3f"
    table.add_row("wet / dry mass", f"{full.mass:{fmt}} / {empty.mass:{fmt}} kg")
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


@app.command()
def mc(
    dispersion: Annotated[
        str, typer.Argument(help="dispersion preset (configs/dispersions) or YAML")
    ],
    runs: Annotated[int | None, typer.Option(help="override the number of runs")] = None,
    workers: Annotated[
        int | None, typer.Option(help="parallel processes (default: half the cores)")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="output directory")] = None,
    idle_aware: Annotated[
        bool,
        typer.Option(
            "--idle-aware/--always", help="pause while a Steam game or heavy GPU job runs"
        ),
    ] = True,
    fidelity: Annotated[str | None, typer.Option(help="override: fast | high")] = None,
):
    """Monte Carlo dependability analysis: success probability, landing dispersion, sensitivity."""
    import json

    from plume.analysis.montecarlo import (
        dispersion_meta,
        load_dispersion,
        run_mc,
        summarize,
        write_report,
    )
    from plume.config import load_mission

    ds = load_dispersion(dispersion)
    if fidelity:
        ds = ds.model_copy(update={"fidelity": fidelity})
    out_dir = out or Path("runs") / "mc" / f"{ds.name}_{ds.fidelity}"
    records = run_mc(
        ds, out_dir, workers=workers, idle_aware=idle_aware, runs=runs, log=console.print
    )
    spec = load_mission(ds.mission)
    summary = summarize(records, ds, spec.target_radius)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = write_report(summary, records, Path("docs") / "mc" / f"{ds.name}_{ds.fidelity}.html")
    meta = dispersion_meta(summary)
    if meta:
        (out_dir / "dispersion_meta.json").write_text(json.dumps(meta))
    lo, hi = summary["success_ci95"]
    land = summary.get("landing") or {}
    table = Table(
        title=f"Monte Carlo: {ds.name} ({ds.fidelity}, {summary['runs']} runs)", show_header=False
    )
    table.add_row(
        "success",
        f"{100 * summary['success_probability']:.1f} % (95 % CI {100 * lo:.1f}-{100 * hi:.1f} %)",
    )
    rlo, rhi = summary["recovery_ci95"]
    table.add_row(
        "vehicle recovered",
        f"{100 * summary['recovery_probability']:.1f} % (95 % CI {100 * rlo:.1f}-{100 * rhi:.1f} %)",
    )
    if land:
        table.add_row("CEP50 / CEP90", f"{land['cep50_m']:,.1f} / {land['cep90_m']:,.1f} m")
        a, b = land["ellipse99"]["semi_axes_m"]
        table.add_row("99 % ellipse", f"{a:,.1f} x {b:,.1f} m")
    fuel = summary["metrics"].get("fuel_remaining_kg") or {}
    if fuel:
        table.add_row("fuel remaining p1 / p50", f"{fuel['p1']:,.0f} / {fuel['p50']:,.0f} kg")
    table.add_row(
        "failure modes", ", ".join(f"{k} {v}" for k, v in summary["failure_modes"].items())
    )
    top = [s for s in summary["sensitivity"][:3]]
    table.add_row("top drivers", ", ".join(f"{s['param']} ({s['rho_miss']:+.2f})" for s in top))
    console.print(table)
    console.print(f"report: [cyan]{report}[/]   data: [cyan]{out_dir}[/]")


@app.command("vv-report")
def vv_report(
    run: Annotated[
        bool, typer.Option("--run/--no-run", help="run the test suite (else read --junit)")
    ] = True,
    junit: Annotated[
        Path | None,
        typer.Option(help="JUnit XML written by --run / read by --no-run [runs/vv/junit.xml]"),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="HTML report path [docs/vv/report.html]")] = None,
    nasa: Annotated[
        bool, typer.Option("--nasa/--no-nasa", help="run the NASA check cases (~1-2 min)")
    ] = True,
):
    """Verification & validation report: tests by model area, NASA check cases, model
    documentation status and Monte Carlo headline numbers, as one HTML page."""
    from plume.analysis.vv import AREAS, build_report

    try:
        res = build_report(run=run, junit=junit, out=out, nasa=nasa, log=console.print)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from None
    if res["exit_code"] not in (None, 0, 1):
        console.print(f"[yellow]pytest exited with code {res['exit_code']}[/]")
    table = Table(title="Verification", show_header=True)
    for col in ("area", "passed", "failed", "skipped", "total"):
        table.add_column(col, justify="left" if col == "area" else "right")
    for area, c in res["areas"].items():
        failed = f"[red]{c['failed']}[/]" if c["failed"] else "0"
        table.add_row(AREAS[area], str(c["passed"]), failed, str(c["skipped"]), str(c["total"]))
    table.add_row(
        "[bold]all[/]",
        str(res["passed"]),
        str(res["failed"]),
        str(res["skipped"]),
        str(res["total"]),
    )
    console.print(table)
    if res["nasa"] is not None:
        ok = sum(
            1
            for c in res["nasa"]
            if not c["error"] and c["vars"] and all(v["passed"] for v in c["vars"])
        )
        console.print(f"NASA check cases passed: {ok} / {len(res['nasa'])}")
    console.print("validation against real data: [yellow]pending[/] for every model")
    console.print(f"report: [cyan]{res['out']}[/]   junit: [cyan]{res['junit']}[/]")
    if res["failed"]:
        raise typer.Exit(1)


@app.command("export-site")
def export_site_cmd(
    out: Annotated[Path, typer.Option(help="output folder")] = Path("site"),
    replays: Annotated[Path, typer.Option(help="replay folder to publish")] = Path("data/replays"),
):
    """Export the viewer, bundled replays, terrain and reports as a static site."""
    from plume.viz.export import export_site

    export_site(out, replay_dirs=(replays,), log=console.print)


from plume.analysis.safety_cli import safety as _safety_cmd  # noqa: E402

app.command("safety")(_safety_cmd)


try:  # real-terrain tools (plume terrain fetch | info | hazard)
    from plume.terrain.cli import terrain_app

    app.add_typer(terrain_app, name="terrain", help="Real-world terrain (Copernicus DEM).")
except ImportError:  # pragma: no cover
    pass


if __name__ == "__main__":
    app()
