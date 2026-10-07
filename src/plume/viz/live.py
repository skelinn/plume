"""Tiny client for streaming live frames from a simulation to the Plume viewer.

    live = LiveStreamer()          # http://127.0.0.1:8765, channel "default"
    live.start(meta)               # meta as in the replay format (vehicle, scene, ...)
    live.push({"t": 0.0, "pos": [...], "quat": [...], ...})   # one frame per step
    live.end({"success": True, "reason": "landed"})

Uses only the standard library. Frames are buffered and flushed from a background thread
at ~10 Hz; every network error is swallowed so a simulation never fails (or blocks for
long) just because no viewer is running.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from typing import Any

try:  # numpy is optional here; convert arrays/scalars if it is around
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


def _jsonable(obj: Any) -> Any:
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


class LiveStreamer:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8765",
        channel: str = "default",
        flush_hz: float = 10.0,
        timeout: float = 0.5,
    ):
        self.url = f"{url.rstrip('/')}/api/live/{channel}"
        self.period = 1.0 / flush_hz
        self.timeout = timeout
        self._buf: dict[str, list] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._failures = 0

    # ------------------------------------------------------------------ public
    def start(self, meta: Mapping[str, Any]) -> None:
        """Begin a new live episode (viewers reset)."""
        with self._lock:
            self._buf = {}
        self._post({"meta": dict(meta)})
        self._ensure_thread()

    def push(self, frame: Mapping[str, Any]) -> None:
        """Queue one frame (a dict of the same keys the Recorder takes)."""
        with self._lock:
            for key, value in frame.items():
                self._buf.setdefault(key, []).append(value)

    def end(self, outcome: Mapping[str, Any] | None = None) -> None:
        """Flush remaining frames and mark the episode finished."""
        self.flush()
        self._post({"end": True, "outcome": dict(outcome) if outcome else None})
        self._stop = True
        self._wake.set()

    def flush(self) -> None:
        with self._lock:
            frames, self._buf = self._buf, {}
        if frames.get("t"):
            self._post({"frames": frames})

    # ------------------------------------------------------------------ internals
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="plume-live", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            time.sleep(self.period)
            self.flush()

    def _post(self, payload: dict) -> bool:
        if self._failures >= 3:  # viewer clearly absent: stop trying until start() again
            if payload.get("meta") is None:
                return False
            self._failures = 0
        try:
            data = json.dumps(payload, separators=(",", ":"), default=_jsonable).encode()
            req = urllib.request.Request(
                self.url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
            self._failures = 0
            return True
        except Exception:
            self._failures += 1
            return False
