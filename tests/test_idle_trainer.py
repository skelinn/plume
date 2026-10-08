"""Idle-aware training: busy detection logic with mocked probes."""

from __future__ import annotations

import pytest

pytest.importorskip("psutil")

from plume.rl.idle import BusyDetector, IdleConfig


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(cfg=None, app=0, procs=(), gpu=0.0, steam_dirs=()):
    clock = Clock()
    state = {"app": app, "procs": list(procs), "gpu": gpu}
    det = BusyDetector(
        cfg or IdleConfig(resume_after_s=300, gpu_threshold=50, gpu_busy_s=20),
        probes={
            "steam_app": lambda: state["app"],
            "processes": lambda: state["procs"],
            "gpu": lambda exclude: (
                state["gpu"](exclude) if callable(state["gpu"]) else state["gpu"]
            ),
            "steam_dirs": list(steam_dirs),
        },
        clock=clock,
    )
    return det, state, clock


def test_idle_requires_hysteresis():
    det, _, clock = make()
    busy, can_start, _ = det.poll(set())
    assert not busy and not can_start
    clock.t = 299
    assert det.poll(set())[1] is False
    clock.t = 301
    assert det.poll(set())[1] is True


def test_steam_registry_pauses_immediately():
    det, state, clock = make()
    clock.t = 1000
    det.poll(set())
    state["app"] = 570
    busy, can_start, reasons = det.poll(set())
    assert busy and not can_start
    assert "steam" in reasons[0]
    # after the game exits the idle timer starts over
    state["app"] = 0
    clock.t = 1100
    assert det.poll(set())[1] is False
    clock.t = 1401
    assert det.poll(set())[1] is True


def test_steam_library_process_detected():
    det, state, _ = make(steam_dirs=[r"d:\steamlibrary\steamapps\common"])
    state["procs"] = [(42, "game.exe", r"D:\SteamLibrary\steamapps\common\Game\game.exe")]
    busy, _, reasons = det.poll(set())
    assert busy and "game.exe" in reasons[0]


def test_gpu_busy_needs_sustained_load_and_excludes_own_pids():
    own = {111, 112}

    def gpu(exclude):
        # our training uses 90%, someone else 60%
        return 60.0 if exclude == own else 150.0

    det, state, clock = make(gpu=gpu)
    busy, _, _ = det.poll(own)
    assert not busy  # load just started
    clock.t = 25
    busy, _, reasons = det.poll(own)
    assert busy and "GPU" in reasons[0]
    state["gpu"] = 10.0
    clock.t = 30
    assert det.poll(own)[0] is False


def test_blocklist():
    cfg = IdleConfig(blocklist=["Blender.exe"])
    det, state, _ = make(cfg=cfg)
    state["procs"] = [(7, "blender.exe", "C:/apps/blender.exe")]
    busy, _, reasons = det.poll(set())
    assert busy and "blender" in reasons[0]
    assert det.poll({7})[0] is False  # excluded pids are ignored
