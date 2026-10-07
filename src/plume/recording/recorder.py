"""Episode recording in the Plume replay format (see ``schema.json``)."""

from __future__ import annotations

import datetime as _dt
import gzip
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from plume import __version__

FORMAT = "plume-replay"
VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("schema.json")

# Decimal places kept per column; keeps replays small without visible loss.
_PRECISION = {
    "t": 4,
    "pos": 3,
    "quat": 6,
    "vel": 3,
    "omega": 4,
    "alt": 2,
    "throttle": 3,
    "thrust": 1,
    "gimbal": 5,
    "rcs": 3,
    "prop_mass": 2,
    "mass": 2,
    "g_load": 3,
    "mach": 3,
    "q_dyn": 1,
    "wind": 2,
}


def _round(value: Any, ndigits: int) -> Any:
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), ndigits).tolist()
    if isinstance(value, (list, tuple)):
        return [_round(v, ndigits) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), ndigits)
    if isinstance(value, np.integer):
        return int(value)
    return value


class Recorder:
    """Collects frames and events, then writes a gzipped JSON replay."""

    def __init__(self, meta: Mapping[str, Any], every: int = 1):
        self.meta: dict[str, Any] = {
            "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "plume_version": __version__,
            **meta,
        }
        self.every = max(1, int(every))
        self.frames: dict[str, list[Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._calls = 0

    def __len__(self) -> int:
        return len(self.frames.get("t", []))

    def record(self, frame: Mapping[str, Any], force: bool = False) -> None:
        """Append one frame. Only every ``every``-th call is kept unless ``force``."""
        keep = force or self._calls % self.every == 0
        self._calls += 1
        if not keep:
            return
        n = len(self)
        for key, value in frame.items():
            col = self.frames.setdefault(key, [])
            if len(col) != n:
                raise ValueError(f"column {key!r} appeared mid-recording")
            col.append(_round(value, _PRECISION.get(key, 4)))
        for key, col in self.frames.items():
            if len(col) != n + 1:
                raise ValueError(f"frame is missing column {key!r}")

    def event(self, t: float, type_: str, label: str = "") -> None:
        self.events.append({"t": round(float(t), 4), "type": type_, "label": label})

    def set_outcome(self, success: bool, reason: str, metrics: Mapping[str, Any] | None = None):
        clean = {k: _round(v, 4) for k, v in (metrics or {}).items()}
        self.meta["outcome"] = {"success": bool(success), "reason": reason, "metrics": clean}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "meta": self.meta,
            "frames": self.frames,
            "events": self.events,
        }

    def save(self, path: str | Path) -> Path:
        return save_replay(self.to_dict(), path)


def save_replay(replay: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(replay, separators=(",", ":")).encode()
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as f:
            f.write(payload)
    else:
        path.write_bytes(payload)
    return path


def load_replay(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    replay = json.loads(raw)
    if replay.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Plume replay")
    return replay


def frames_as_arrays(replay: Mapping[str, Any], keys: Iterable[str] | None = None):
    """Return the replay's columns as numpy arrays (strings stay lists)."""
    out = {}
    for key, col in replay["frames"].items():
        if keys is not None and key not in keys:
            continue
        out[key] = col if col and isinstance(col[0], str) else np.asarray(col, dtype=float)
    return out
