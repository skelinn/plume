"""A subprocess VecEnv where each worker process runs several environments.

SB3's ``SubprocVecEnv`` uses one process per environment; with cheap physics steps
the per-step IPC and policy overhead dominate. Grouping ``k`` environments per
worker amortises that overhead over a larger batch while still using every core.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv


def _worker(remote, parent_remote, fns_wrapper) -> None:
    parent_remote.close()
    envs = [fn() for fn in fns_wrapper.var]
    reset_infos: list[dict] = [{} for _ in envs]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for k, (env, action) in enumerate(zip(envs, data, strict=True)):
                    obs, rew, term, trunc, info = env.step(action)
                    done = term or trunc
                    info["TimeLimit.truncated"] = trunc and not term
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_infos[k] = env.reset()
                    out.append((obs, rew, done, info))
                remote.send(out)
            elif cmd == "reset":
                seeds, options = data
                obs = []
                for k, env in enumerate(envs):
                    o, reset_infos[k] = env.reset(seed=seeds[k], options=options[k])
                    obs.append(o)
                remote.send((obs, list(reset_infos)))
            elif cmd == "env_method":
                name, args, kwargs, idx = data
                remote.send([getattr(envs[i], name)(*args, **kwargs) for i in idx])
            elif cmd == "get_attr":
                name, idx = data
                remote.send([getattr(envs[i], name) for i in idx])
            elif cmd == "set_attr":
                name, value, idx = data
                for i in idx:
                    setattr(envs[i], name, value)
                remote.send(None)
            elif cmd == "spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class MultiSubprocVecEnv(VecEnv):
    def __init__(
        self, env_fns: Sequence[Callable[[], Any]], n_workers: int, start_method: str = "spawn"
    ):
        n = len(env_fns)
        n_workers = max(1, min(n_workers, n))
        self.groups = [list(range(n))[w::n_workers] for w in range(n_workers)]
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_workers)], strict=True)
        self.processes = []
        for wr, r, grp in zip(self.work_remotes, self.remotes, self.groups, strict=True):
            fns = CloudpickleWrapper([env_fns[i] for i in grp])
            p = ctx.Process(target=_worker, args=(wr, r, fns), daemon=True)
            p.start()
            self.processes.append(p)
            wr.close()
        self.remotes[0].send(("spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(n, obs_space, act_space)
        self.closed = False
        self._actions = None

    # -- helpers
    def _where(self, indices) -> dict[int, list[tuple[int, int]]]:
        idx = self._get_indices(indices)
        out: dict[int, list[tuple[int, int]]] = {}
        for w, grp in enumerate(self.groups):
            for j, i in enumerate(grp):
                if i in idx:
                    out.setdefault(w, []).append((j, i))
        return out

    def _gather(self, per_worker: list[list[Any]]) -> list[Any]:
        flat: list[Any] = [None] * self.num_envs
        for grp, items in zip(self.groups, per_worker, strict=True):
            for i, item in zip(grp, items, strict=True):
                flat[i] = item
        return flat

    # -- VecEnv API
    def step_async(self, actions: np.ndarray) -> None:
        for remote, grp in zip(self.remotes, self.groups, strict=True):
            remote.send(("step", [actions[i] for i in grp]))

    def step_wait(self):
        results = self._gather([remote.recv() for remote in self.remotes])
        obs, rews, dones, infos = zip(*results, strict=True)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones), list(infos)

    def reset(self):
        for remote, grp in zip(self.remotes, self.groups, strict=True):
            seeds = [self._seeds[i] for i in grp]
            options = [self._options[i] for i in grp]
            remote.send(("reset", (seeds, options)))
        parts = [remote.recv() for remote in self.remotes]
        obs = self._gather([p[0] for p in parts])
        self.reset_infos = self._gather([p[1] for p in parts])
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs)

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)
        self.closed = True

    def _call(self, cmd, make_payload, indices):
        where = self._where(indices)
        for w, pairs in where.items():
            self.remotes[w].send((cmd, make_payload([j for j, _ in pairs])))
        out: dict[int, Any] = {}
        for w, pairs in where.items():
            res = self.remotes[w].recv()
            if res is None:
                continue
            for (_, i), r in zip(pairs, res, strict=True):
                out[i] = r
        return [out[i] for i in sorted(out)]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return self._call("get_attr", lambda js: (attr_name, js), indices)

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        self._call("set_attr", lambda js: (attr_name, value, js), indices)

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> list[Any]:
        return self._call(
            "env_method", lambda js: (method_name, method_args, method_kwargs, js), indices
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
