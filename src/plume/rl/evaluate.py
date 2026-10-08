"""Policy evaluation on the landing task (works for the autopilot and SB3 agents)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plume.envs.landing_env import AutopilotPolicy, LandingEnv


@dataclass
class EvalResult:
    stage: str
    controller: str
    episodes: int
    success_rate: float
    fuel_mean: float  # kg, successful landings
    fuel_all_mean: float  # kg, all episodes
    landing_error_mean: float  # m, successful landings
    touchdown_vz_mean: float  # m/s, episodes with a touchdown
    reasons: dict[str, int] = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "stage": self.stage,
            "controller": self.controller,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "fuel_kg": self.fuel_mean,
            "landing_error_m": self.landing_error_mean,
            "touchdown_vz_mps": self.touchdown_vz_mean,
        }


def _nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def run_episodes(
    env: LandingEnv,
    make_policy: Callable[[LandingEnv], Callable[[np.ndarray], np.ndarray]],
    stage: int,
    episodes: int,
    seed: int = 10_000,
    controller: str = "policy",
    save_replays: Path | None = None,
    save_first: int = 3,
) -> EvalResult:
    infos = []
    policy = make_policy(env)
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k, options={"stage": stage, "controller": controller})
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        info = {}
        while not done:
            action = policy(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        infos.append(info)
        if save_replays is not None and k < save_first and env.last_replay is not None:
            from plume.recording import save_replay

            name = f"{controller}_{env.stages[stage].name}_{k}.plume.json.gz"
            save_replay(env.last_replay, save_replays / name)
    ok = [i for i in infos if i["success"]]
    reasons: dict[str, int] = {}
    for i in infos:
        reasons[i["reason"]] = reasons.get(i["reason"], 0) + 1
    return EvalResult(
        stage=env.stages[stage].name,
        controller=controller,
        episodes=episodes,
        success_rate=len(ok) / max(episodes, 1),
        fuel_mean=_nanmean([i["fuel_used"] for i in ok]),
        fuel_all_mean=_nanmean([i["fuel_used"] for i in infos]),
        landing_error_mean=_nanmean([i["landing_error"] for i in ok]),
        touchdown_vz_mean=_nanmean([i["touchdown_vz"] for i in infos]),
        reasons=reasons,
    )


def evaluate_autopilot(
    stage: int, episodes: int = 50, seed: int = 10_000, env_spec=None, **kw
) -> EvalResult:
    env = LandingEnv(env_spec, record=kw.get("save_replays") is not None, fixed_stage=True)
    return run_episodes(env, AutopilotPolicy, stage, episodes, seed, controller="pid", **kw)


def sb3_policy_factory(model, vecnorm=None, deterministic: bool = True):
    """Build a policy callable for ``run_episodes`` from an SB3 model (+ VecNormalize stats)."""

    def make(env):
        def act(obs):
            o = obs[None, :]
            if vecnorm is not None:
                o = vecnorm.normalize_obs(o)
            action, _ = model.predict(o, deterministic=deterministic)
            return action[0]

        return act

    return make
