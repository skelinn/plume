"""Idle-aware training supervisor: train only while the PC is not busy.

The supervisor polls a few cheap signals every ``poll_s`` seconds:

* **Steam game running** -- ``HKCU\\Software\\Valve\\Steam\\RunningAppID`` != 0, or any
  process whose executable lives under a Steam library's ``steamapps\\common``.
* **Heavy GPU use by other processes** -- NVML per-process SM utilisation summed over
  processes that are not part of the training run, above ``gpu_threshold`` % for
  ``gpu_busy_s`` seconds.
* **Blocklisted processes** -- e.g. video editors or 3D tools listed in the config.

While idle for ``resume_after_s`` it runs the training worker (``python -m
plume.rl.train``) in a child process. When busy, it creates the ``STOP`` file;
the worker finishes its current rollout, checkpoints and exits, freeing the GPU
and CPU. Progress accumulates in ``state.json`` across sessions.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from plume.config import config_root
from plume.rl.train import STOP_FILE, TrainConfig, read_state, write_state


@dataclass
class IdleConfig:
    poll_s: float = 5.0
    resume_after_s: float = 300.0
    gpu_threshold: float = 50.0
    gpu_busy_s: float = 20.0
    stop_timeout_s: float = 120.0
    steam: bool = True
    blocklist: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> IdleConfig:
        p = Path(path) if path else config_root() / "rl" / "idle_trainer.yaml"
        if not p.exists():
            return cls()
        return cls(**(yaml.safe_load(p.read_text()) or {}))


# ----------------------------------------------------------------------------- probes
def steam_running_app_id() -> int:
    """Steam's RunningAppID (0 when no game is running, or not on Windows)."""
    if sys.platform != "win32":
        return 0
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _ = winreg.QueryValueEx(key, "RunningAppID")
            return int(value)
    except OSError:
        return 0


def steam_library_dirs() -> list[Path]:
    """``steamapps/common`` directories of every Steam library on this machine."""
    if sys.platform != "win32":
        return []
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam = Path(winreg.QueryValueEx(key, "SteamPath")[0])
    except OSError:
        return []
    libs = {steam}
    vdf = steam / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore")):
            libs.add(Path(m.group(1).replace("\\\\", "\\")))
    return [lib / "steamapps" / "common" for lib in libs if (lib / "steamapps" / "common").exists()]


def _process_snapshot():
    import psutil

    out = []
    for p in psutil.process_iter(["pid", "name", "exe"]):
        info = p.info
        out.append((info["pid"], (info.get("name") or "").lower(), info.get("exe") or ""))
    return out


def own_pids(root_pid: int | None) -> set[int]:
    """PIDs of the training worker and all its children (env workers)."""
    if root_pid is None:
        return set()
    import psutil

    try:
        root = psutil.Process(root_pid)
        return {root_pid, *(c.pid for c in root.children(recursive=True))}
    except psutil.Error:
        return set()


class GpuMonitor:
    def __init__(self):
        self.ok = False
        self._last_ts = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:  # no NVIDIA GPU / driver
            self.ok = False

    def other_utilisation(self, exclude: set[int]) -> float:
        """Summed SM utilisation (%) of GPU processes not in ``exclude``."""
        if not self.ok:
            return 0.0
        n = self.nvml
        try:
            samples = n.nvmlDeviceGetProcessUtilization(self.handle, self._last_ts)
        except n.NVMLError:
            # per-process stats unavailable: fall back to whole-GPU utilisation
            if exclude:
                return 0.0
            return float(n.nvmlDeviceGetUtilizationRates(self.handle).gpu)
        if samples:
            self._last_ts = max(s.timeStamp for s in samples)
        latest: dict[int, float] = {}
        for s in samples:
            latest[s.pid] = max(latest.get(s.pid, 0.0), float(s.smUtil))
        return sum(u for pid, u in latest.items() if pid not in exclude and pid != os.getpid())


# ----------------------------------------------------------------------------- detector
class BusyDetector:
    """Combines the probes into a busy/idle decision with debouncing."""

    def __init__(self, cfg: IdleConfig, probes: dict | None = None, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        probes = probes or {}
        self.steam_app = probes.get("steam_app", steam_running_app_id)
        self.processes = probes.get("processes", _process_snapshot)
        gpu = probes.get("gpu")
        if gpu is None:
            mon = GpuMonitor()
            gpu = mon.other_utilisation
        self.gpu = gpu
        self.steam_dirs = [str(d).lower() for d in probes.get("steam_dirs", steam_library_dirs())]
        self._gpu_high_since: float | None = None
        self._idle_since: float | None = None

    def reasons(self, exclude: set[int]) -> list[str]:
        out = []
        if self.cfg.steam:
            app = self.steam_app()
            if app:
                out.append(f"steam game running (app {app})")
        procs = None
        if self.cfg.steam and self.steam_dirs and not out:
            procs = self.processes()
            for pid, name, exe in procs:
                if pid in exclude:
                    continue
                exe_l = exe.lower()
                if any(exe_l.startswith(d) for d in self.steam_dirs):
                    out.append(f"steam game process {name}")
                    break
        if self.cfg.blocklist:
            procs = procs if procs is not None else self.processes()
            block = {b.lower() for b in self.cfg.blocklist}
            for pid, name, _ in procs:
                if pid not in exclude and name in block:
                    out.append(f"blocklisted process {name}")
                    break
        util = self.gpu(exclude)
        now = self.clock()
        if util > self.cfg.gpu_threshold:
            if self._gpu_high_since is None:
                self._gpu_high_since = now
            if now - self._gpu_high_since >= self.cfg.gpu_busy_s:
                out.append(f"GPU busy ({util:.0f}% by other processes)")
        else:
            self._gpu_high_since = None
        return out

    def poll(self, exclude: set[int]) -> tuple[bool, bool, list[str]]:
        """Returns (busy, idle_long_enough_to_start, reasons)."""
        reasons = self.reasons(exclude)
        now = self.clock()
        if reasons:
            self._idle_since = None
            return True, False, reasons
        if self._idle_since is None:
            self._idle_since = now
        return False, now - self._idle_since >= self.cfg.resume_after_s, []


# ----------------------------------------------------------------------------- supervisor
class Supervisor:
    def __init__(
        self,
        train_cfg: TrainConfig,
        idle_cfg: IdleConfig,
        config_path: str | None = None,
        detector: BusyDetector | None = None,
        log=print,
    ):
        self.train_cfg = train_cfg
        self.idle_cfg = idle_cfg
        self.config_path = config_path
        self.detector = detector or BusyDetector(idle_cfg)
        self.run_dir = Path(train_cfg.run_dir)
        self.proc: subprocess.Popen | None = None
        self._print = log
        self._last_wait: str | None = None

    def log(self, msg: str) -> None:
        """Timestamped line to stdout (flushed) and ``<run_dir>/supervisor.log``."""
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        try:
            self._print(line, flush=True)
        except TypeError:
            self._print(line)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "supervisor.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _start_worker(self) -> None:
        (self.run_dir / STOP_FILE).unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "plume.rl.train"]
        if self.config_path:
            cmd += ["--config", str(self.config_path)]
        cmd += ["--device", self.train_cfg.device]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.run_dir / "worker.log", "a")
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        self.log(f"[idle] started training worker (pid {self.proc.pid})")

    def _stop_worker(self, reason: str) -> None:
        if self.proc is None:
            return
        (self.run_dir / STOP_FILE).touch()
        self.log(f"[idle] pausing: {reason}")
        try:
            self.proc.wait(timeout=self.idle_cfg.stop_timeout_s)
        except subprocess.TimeoutExpired:
            self.log("[idle] worker did not stop in time; terminating")
            self.proc.terminate()
            self.proc.wait(timeout=30)
        self.proc = None
        state = read_state(self.run_dir)
        state["last_pause_reason"] = reason
        state["last_pause_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_state(self.run_dir, state)

    def done(self) -> bool:
        return read_state(self.run_dir)["timesteps"] >= self.train_cfg.total_timesteps

    def run(self, max_seconds: float | None = None) -> None:
        t_end = None if max_seconds is None else time.monotonic() + max_seconds
        self.log(
            f"[idle] supervising {self.run_dir} (target {self.train_cfg.total_timesteps:,} steps)"
        )
        try:
            while not self.done():
                if t_end is not None and time.monotonic() > t_end:
                    break
                exclude = own_pids(self.proc.pid if self.proc else None)
                busy, can_start, reasons = self.detector.poll(exclude)
                if self.proc is not None and self.proc.poll() is not None:
                    self.log(f"[idle] worker exited with code {self.proc.returncode}")
                    self.proc = None
                if busy and self.proc is not None:
                    self._stop_worker("; ".join(reasons))
                elif not busy and can_start and self.proc is None:
                    self._start_worker()
                    self._last_wait = None
                elif self.proc is None:
                    why = "; ".join(reasons) if busy else "waiting for the PC to stay idle"
                    if why != self._last_wait:
                        self.log(f"[idle] {why}")
                        self._last_wait = why
                    state = read_state(self.run_dir)
                    state["waiting_reason"] = why
                    write_state(self.run_dir, state)
                time.sleep(self.idle_cfg.poll_s)
        finally:
            if self.proc is not None:
                self._stop_worker("supervisor exiting")
        self.log("[idle] finished" if self.done() else "[idle] stopped")
