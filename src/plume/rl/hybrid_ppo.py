"""PPO that collects rollouts with a CPU copy of the policy and trains on the GPU.

With small MLP policies, per-step GPU inference on a batch of ~16 observations is
dominated by transfer latency, which roughly halves throughput. Here the rollout
phase runs on a CPU clone (weights synced after every update) while gradient
updates still run on CUDA.
"""

from __future__ import annotations

import copy

import torch
from stable_baselines3 import PPO


class HybridPPO(PPO):
    _cpu_policy = None

    def _sync_cpu_policy(self):
        if self._cpu_policy is None:
            self._cpu_policy = copy.deepcopy(self.policy).to("cpu")
        else:
            state = {k: v.detach().to("cpu") for k, v in self.policy.state_dict().items()}
            self._cpu_policy.load_state_dict(state)
        return self._cpu_policy

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self.device.type != "cuda":
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        gpu_policy, gpu_device = self.policy, self.device
        self.policy = self._sync_cpu_policy()
        self.device = torch.device("cpu")
        try:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        finally:
            self.policy, self.device = gpu_policy, gpu_device
