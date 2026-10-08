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
def land(
    controller: Annotated[str, typer.Option(help="pid | ppo")] = "pid",
    stage: Annotated[str, typer.Option(help="curriculum stage name or index")] = "full_descent",
    seed: int = 1,
    episodes: int = 1,
    run_dir: Annotated[Path, typer.Option(help="PPO run directory")] = Path("runs/ppo_landing"),
    save: Annotated[Path | None, typer.Option(help="replay path (first episode)")] = None,
):
    """Fly the landing task with the PID/guidance autopilot or a trained PPO agent."""
    from plume.envs.landing_env import AutopilotPolicy, LandingEnv
    from plume.recording import save_replay
    from plume.rl.evaluate import run_episodes, sb3_policy_factory

    env = LandingEnv(record=True, fixed_stage=True)
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
):
    """Fly a point-to-point cargo hop over terrain and score it."""
    from plume.config import load_mission
    from plume.missions.hop import run_mission

    spec = load_mission(mission)
    streamer, on_frame = _live(live)
    if streamer:
        console.print("[dim]streaming to the viewer...[/]")
    with console.status(f"flying {spec.name} (planning ascent, then ~10 min of flight)..."):
        run = run_mission(spec, seed=seed, cargo_mass=cargo, on_frame=on_frame)
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
    workers: Annotated[int | None, typer.Option(help="parallel processes (default: cores - 2)")] = None,
):
    """PID vs PPO on every curriculum stage -> results table (and README)."""
    from plume.rl.benchmark import run_benchmark

    run_benchmark(
        episodes=episodes, run_dir=run_dir, update_readme=readme, console=console, workers=workers
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


if __name__ == "__main__":
    app()
