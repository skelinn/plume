"""PPO training with curriculum learning, resumable checkpoints and a stop file.

A run directory holds::

    model.zip, vecnormalize.pkl   latest checkpoint
    best/                         best evaluation on the hardest stage reached
    state.json                    timesteps, curriculum stage, history
    STOP                          if present, the worker checkpoints and exits
    tb/                           TensorBoard logs
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from plume.config import config_root, load_landing_env

STOP_FILE = "STOP"


@dataclass
class TrainConfig:
    env: str = "landing"
    run_dir: str = "runs/ppo_landing"
    total_timesteps: int = 50_000_000
    n_envs: int = 32
    n_workers: int = 12
    device: str = "cuda"
    seed: int = 0
    checkpoint_every: int = 1_000_000
    ppo: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> TrainConfig:
        p = Path(path) if path else config_root() / "rl" / "ppo_landing.yaml"
        data = yaml.safe_load(p.read_text()) or {}
        return cls(**data)


# ----------------------------------------------------------------------------- state
def read_state(run_dir: Path) -> dict:
    f = run_dir / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"timesteps": 0, "stage": 0, "history": [], "sessions": 0}


def write_state(run_dir: Path, state: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(run_dir / "state.json")


# ----------------------------------------------------------------------------- envs
def _make_env(env_name: str, rank: int, seed: int, stage: int):
    def thunk():
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from plume.envs.landing_env import LandingEnv

        env = LandingEnv(env_name, stage=stage)
        env.reset(seed=seed + rank)
        return env

    return thunk


def make_vec_env(
    cfg: TrainConfig,
    stage: int,
    n_envs: int | None = None,
    subprocess: bool = True,
    n_workers: int | None = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from plume.rl.multivec import MultiSubprocVecEnv

    n = n_envs or cfg.n_envs
    w = n_workers or cfg.n_workers
    seed = cfg.seed * 1000 + int(time.time()) % 100_000
    fns = [_make_env(cfg.env, i, seed, stage) for i in range(n)]
    venv = MultiSubprocVecEnv(fns, n_workers=w) if subprocess and n > 1 else DummyVecEnv(fns)
    return VecMonitor(venv, info_keywords=("success", "fuel_used", "stage"))


# ----------------------------------------------------------------------------- callbacks
def _callbacks(cfg: TrainConfig, run_dir: Path, state: dict, vecnorm):
    from stable_baselines3.common.callbacks import BaseCallback

    cur = load_landing_env(cfg.env).curriculum
    n_stages = len(cur.stages)

    class Curriculum(BaseCallback):
        """Promote the curriculum stage on rolling success; checkpoint; honour the stop file."""

        def __init__(self):
            super().__init__()
            self.window: deque[float] = deque(maxlen=cur.window)
            self.seen = 0
            self.all_success: deque[float] = deque(maxlen=500)
            self.last_ckpt = state["timesteps"]
            self.stopped = False

        def _on_step(self) -> bool:
            stage = state["stage"]
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if not done:
                    continue
                s = float(info.get("success", False))
                self.all_success.append(s)
                if info.get("stage") == stage:
                    self.window.append(s)
                    self.seen += 1
            ts = state["timesteps_at_start"] + self.num_timesteps
            if (
                stage < n_stages - 1
                and self.seen >= cur.min_episodes
                and len(self.window) == self.window.maxlen
                and np.mean(self.window) >= cur.promote_success
            ):
                state["stage"] = stage + 1
                state["history"].append(
                    {
                        "timesteps": ts,
                        "stage": stage + 1,
                        "success_prev": float(np.mean(self.window)),
                    }
                )
                self.training_env.env_method("set_stage", stage + 1)
                self.window.clear()
                self.seen = 0
                if self.verbose:
                    print(f"[curriculum] promoted to stage {stage + 1} at {ts:,} steps")
            if self.n_calls % 50 == 0:
                self.logger.record("curriculum/stage", state["stage"])
                if self.window:
                    self.logger.record("curriculum/stage_success", float(np.mean(self.window)))
                if self.all_success:
                    self.logger.record("curriculum/success_all", float(np.mean(self.all_success)))
            if ts - self.last_ckpt >= cfg.checkpoint_every:
                self.last_ckpt = ts
                save_checkpoint(self.model, vecnorm, run_dir, state, ts)
            if self.n_calls % 64 == 0 and (run_dir / STOP_FILE).exists():
                self.stopped = True
                return False
            return True

    return Curriculum()


def save_checkpoint(model, vecnorm, run_dir: Path, state: dict, timesteps: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.zip")
    vecnorm.save(str(run_dir / "vecnormalize.pkl"))
    state["timesteps"] = int(timesteps)
    write_state(run_dir, state)


# ----------------------------------------------------------------------------- train
def train(
    cfg: TrainConfig,
    timesteps: int | None = None,
    resume: bool = True,
    verbose: int = 1,
    subprocess: bool = True,
) -> dict:
    """Train (or continue training) until ``cfg.total_timesteps`` or the stop file.

    ``timesteps`` caps the number of steps in *this* session.
    """
    import torch
    from stable_baselines3.common.vec_env import VecNormalize

    import plume.envs  # noqa: F401  (registers the env)
    from plume.rl.hybrid_ppo import HybridPPO as PPO

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / STOP_FILE).unlink(missing_ok=True)
    state = (
        read_state(run_dir)
        if resume
        else {"timesteps": 0, "stage": 0, "history": [], "sessions": 0}
    )
    remaining = cfg.total_timesteps - state["timesteps"]
    if timesteps is not None:
        remaining = min(remaining, timesteps)
    if remaining <= 0:
        return state
    state["sessions"] = state.get("sessions", 0) + 1
    state["timesteps_at_start"] = state["timesteps"]
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    venv = make_vec_env(cfg, state["stage"], subprocess=subprocess)
    model_path = run_dir / "model.zip"
    if resume and model_path.exists():
        vecnorm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), venv)
        vecnorm.training = True
        model = PPO.load(
            model_path, env=vecnorm, device=device, tensorboard_log=str(run_dir / "tb")
        )
    else:
        vecnorm = VecNormalize(
            venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.ppo.get("gamma", 0.995)
        )
        act = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[cfg.policy.get("activation", "tanh")]
        policy_kwargs = {
            "net_arch": {
                "pi": cfg.policy.get("net_arch", [256, 256]),
                "vf": cfg.policy.get("net_arch", [256, 256]),
            },
            "log_std_init": cfg.policy.get("log_std_init", -0.5),
            "activation_fn": act,
        }
        model = PPO(
            "MlpPolicy",
            vecnorm,
            device=device,
            seed=cfg.seed,
            verbose=0,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=policy_kwargs,
            **cfg.ppo,
        )
    cb = _callbacks(cfg, run_dir, state, vecnorm)
    cb.verbose = verbose
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=cb,
            reset_num_timesteps=False,
            tb_log_name="ppo",
            progress_bar=False,
        )
    finally:
        done_steps = state["timesteps_at_start"] + cb.num_timesteps
        save_checkpoint(model, vecnorm, run_dir, state, done_steps)
        elapsed = time.time() - t0
        state["last_session"] = {
            "steps": int(cb.num_timesteps),
            "seconds": round(elapsed, 1),
            "steps_per_second": round(cb.num_timesteps / max(elapsed, 1e-9), 1),
            "stopped_by_supervisor": cb.stopped,
            "device": str(model.device),
        }
        write_state(run_dir, state)
        venv.close()
    return state


def load_trained(run_dir: str | Path, which: str = "model"):
    """Load (model, vecnormalize) for evaluation."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from plume.envs.landing_env import LandingEnv

    run_dir = Path(run_dir)
    model_file = run_dir / ("best/model.zip" if which == "best" else "model.zip")
    norm_file = model_file.parent / "vecnormalize.pkl"
    model = PPO.load(model_file, device="cpu")
    dummy = DummyVecEnv([lambda: LandingEnv()])
    vecnorm = VecNormalize.load(str(norm_file), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return model, vecnorm


def eta_seconds(state: dict, total: int) -> float:
    sps = (state.get("last_session") or {}).get("steps_per_second") or 0
    if sps <= 0:
        return math.nan
    return (total - state["timesteps"]) / sps


if __name__ == "__main__":  # worker entry point used by the idle supervisor
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    c = TrainConfig.load(args.config)
    if args.device:
        c.device = args.device
    train(c, timesteps=args.timesteps)
