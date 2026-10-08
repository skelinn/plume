"""Landing environment, autopilot baseline, vectorised envs and PPO plumbing."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from gymnasium.utils.env_checker import check_env

from plume.envs.landing_env import OBS_DIM, AutopilotPolicy, LandingEnv
from plume.rl.evaluate import run_episodes


def test_env_checker_passes():
    env = LandingEnv()
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (OBS_DIM,)


def test_registered_with_gymnasium():
    import gymnasium as gym

    import plume.envs  # noqa: F401

    env = gym.make("Plume/Landing-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)


def test_seeded_determinism():
    def rollout():
        env = LandingEnv(stage=1)
        obs, _ = env.reset(seed=123)
        rng = np.random.default_rng(0)
        traj = [obs]
        for _ in range(40):
            obs, _r, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            traj.append(obs)
            if term or trunc:
                break
        return np.array(traj)

    np.testing.assert_array_equal(rollout(), rollout())


def test_autopilot_lands_easy_stage():
    env = LandingEnv(fixed_stage=True)
    res = run_episodes(env, AutopilotPolicy, stage=0, episodes=20, seed=777)
    assert res.success_rate >= 0.9, res.reasons
    assert res.landing_error_mean < 5.0


def test_engine_off_crashes_with_penalty():
    env = LandingEnv(fixed_stage=True)
    env.reset(seed=5, options={"stage": 1})
    total, info = 0.0, {}
    for _ in range(4000):
        _, r, term, trunc, info = env.step(np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32))
        total += r
        if term or trunc:
            break
    assert info["reason"] in {"crash_legs", "crash_hull"}
    assert not info["success"]
    assert total < -50


def test_success_scores_higher_than_crash():
    env = LandingEnv(fixed_stage=True)
    pol = AutopilotPolicy(env)
    obs, _ = env.reset(seed=11, options={"stage": 0})
    pol.reset()
    total_ok, done, info = 0.0, False, {}
    while not done:
        obs, r, term, trunc, info = env.step(pol(obs))
        total_ok += r
        done = term or trunc
    assert info["success"]
    assert total_ok > 100


def test_curriculum_stage_controls():
    env = LandingEnv()
    env.set_stage(99)
    assert env.get_stage() == len(env.stages) - 1
    env.set_stage(-3)
    assert env.get_stage() == 0
    env.set_stage(2)
    stages = set()
    for k in range(60):
        env.reset(seed=k)
        stages.add(env.episode_stage)
    assert 2 in stages and stages <= {0, 1, 2}  # earlier stages replayed sometimes


def test_recording_produces_replay():
    env = LandingEnv(record=True, fixed_stage=True)
    pol = AutopilotPolicy(env)
    obs, _ = env.reset(seed=3, options={"stage": 0})
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(pol(obs))
        done = term or trunc
    rep = env.last_replay
    assert rep is not None and rep["meta"]["outcome"] is not None
    assert len(rep["frames"]["t"]) > 10


def test_multivec_matches_dummy():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from plume.rl.multivec import MultiSubprocVecEnv

    def fn(i):
        def make():
            e = LandingEnv(stage=1)
            return e

        return make

    fns = [fn(i) for i in range(4)]
    multi = MultiSubprocVecEnv(fns, n_workers=2)
    dummy = DummyVecEnv(fns)
    try:
        multi.seed(42)
        dummy.seed(42)
        o1, o2 = multi.reset(), dummy.reset()
        np.testing.assert_allclose(o1, o2)
        acts = np.tile(np.array([0.4, 0.1, -0.1], dtype=np.float32), (4, 1))
        for _ in range(30):
            a1 = multi.step(acts)
            a2 = dummy.step(acts)
            np.testing.assert_allclose(a1[0], a2[0], rtol=1e-6)
            np.testing.assert_allclose(a1[1], a2[1], rtol=1e-6)
        assert multi.env_method("get_stage") == [1, 1, 1, 1]
        multi.env_method("set_stage", 2, indices=[1, 3])
        assert multi.get_attr("stage") == [1, 2, 1, 2]
    finally:
        multi.close()
        dummy.close()


@pytest.mark.slow
def test_ppo_smoke_train_resume_and_stop(tmp_path):
    from plume.rl.train import STOP_FILE, TrainConfig, load_trained, read_state, train

    cfg = TrainConfig.load()
    cfg.run_dir = str(tmp_path / "run")
    cfg.device = "cpu"
    cfg.n_envs = 4
    cfg.n_workers = 2
    cfg.total_timesteps = 10**9
    cfg.checkpoint_every = 10**9
    cfg.ppo = {**cfg.ppo, "n_steps": 128, "batch_size": 256, "n_epochs": 1}
    st = train(cfg, timesteps=1024, verbose=0)
    assert st["timesteps"] >= 1024
    first = st["timesteps"]
    st = train(cfg, timesteps=1024, verbose=0)  # resumes from the checkpoint
    assert st["timesteps"] >= first + 1024
    assert read_state(tmp_path / "run")["sessions"] == 2
    model, vecnorm = load_trained(tmp_path / "run")
    obs = vecnorm.normalize_obs(np.zeros((1, OBS_DIM), dtype=np.float32))
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 3)  # guidance mode: throttle + axis tilt

    # the STOP file ends a session early with a checkpoint
    def stopper():
        time.sleep(8)
        (tmp_path / "run" / STOP_FILE).touch()

    threading.Thread(target=stopper, daemon=True).start()
    st = train(cfg, timesteps=10_000_000, verbose=0)
    assert st["last_session"]["stopped_by_supervisor"]
    assert st["timesteps"] < first + 2048 + 10_000_000
