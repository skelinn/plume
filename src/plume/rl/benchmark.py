"""PID vs PPO benchmark on every curriculum stage, written to a Markdown table."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _chunk(args) -> list[dict]:
    controller, stage, seeds, run_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from plume.envs.landing_env import AutopilotPolicy, LandingEnv
    from plume.rl.evaluate import sb3_policy_factory

    env = LandingEnv(fixed_stage=True)
    if controller == "pid":
        make = AutopilotPolicy
    else:
        import torch

        torch.set_num_threads(1)
        from plume.rl.train import load_trained

        model, vecnorm = load_trained(run_dir)
        make = sb3_policy_factory(model, vecnorm)
    policy = make(env)
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed, options={"stage": stage})
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        out.append(
            {
                "success": bool(info["success"]),
                "fuel": info["fuel_used"],
                "error": info["landing_error"],
                "vz": info["touchdown_vz"],
                "reason": info["reason"],
            }
        )
    return out


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["success"]]

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return float(np.mean(xs)) if xs else math.nan

    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    n = len(rows)
    p = len(ok) / max(n, 1)
    return {
        "episodes": n,
        "success_rate": p,
        "success_ci95": 1.96 * math.sqrt(p * (1 - p) / max(n, 1)),
        "fuel_kg": mean([r["fuel"] for r in ok]),
        "landing_error_m": mean([r["error"] for r in ok]),
        "touchdown_vz_mps": mean([r["vz"] for r in ok]),
        "reasons": reasons,
    }


def markdown_table(results: dict, stages: list[str]) -> str:
    lines = [
        "| Stage | Controller | Success rate | Fuel used (kg) | Landing error (m) | Touchdown speed (m/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def fmt(x, d=1):
        return "–" if x != x else f"{x:.{d}f}"

    for st in stages:
        for ctrl, label in (("pid", "PID / guidance"), ("ppo", "PPO")):
            r = results.get(ctrl, {}).get(st)
            if not r:
                continue
            sr = f"**{100 * r['success_rate']:.0f}%** ± {100 * r['success_ci95']:.0f}"
            lines.append(
                f"| {st} | {label} | {sr} | {fmt(r['fuel_kg'])} | {fmt(r['landing_error_m'], 2)} | {fmt(r['touchdown_vz_mps'], 2)} |"
            )
    return "\n".join(lines)


def run_benchmark(
    episodes: int = 200,
    run_dir: Path = Path("runs/ppo_landing"),
    update_readme: bool = True,
    console=None,
    out_json: Path = Path("docs/results.json"),
    seed0: int = 50_000,
    workers: int | None = None,
) -> dict:
    from plume.config import load_landing_env

    stages = [s.name for s in load_landing_env().curriculum.stages]
    controllers = ["pid"]
    if (Path(run_dir) / "model.zip").exists():
        controllers.append("ppo")
    jobs = []
    chunk = 25
    for ctrl in controllers:
        for si in range(len(stages)):
            seeds = list(range(seed0, seed0 + episodes))
            for k in range(0, episodes, chunk):
                jobs.append((ctrl, si, seeds[k : k + chunk], str(run_dir)))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(_chunk, jobs))
    grouped: dict[tuple[str, int], list[dict]] = {}
    for (ctrl, si, _, _), rows in zip(jobs, outputs, strict=True):
        grouped.setdefault((ctrl, si), []).extend(rows)
    results: dict = {c: {} for c in controllers}
    for (ctrl, si), rows in grouped.items():
        results[ctrl][stages[si]] = _summarise(rows)
    table = markdown_table(results, stages)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"episodes": episodes, "seed0": seed0, "results": results}, indent=2)
    )
    if console is not None:
        console.print(table)
    if update_readme:
        readme = Path("README.md")
        text = readme.read_text(encoding="utf-8") if readme.exists() else ""
        if START in text and END in text:
            pre, rest = text.split(START, 1)
            _, post = rest.split(END, 1)
            note = f"\n_{episodes} seeded episodes per stage and controller; ± is the 95% interval. Fuel, error and touchdown speed are averaged over successful landings._\n\n"
            readme.write_text(pre + START + note + table + "\n" + END + post, encoding="utf-8")
    return results
