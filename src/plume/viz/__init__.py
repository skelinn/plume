"""Browser-based 3-D viewer for Plume replays and live runs.

Names are imported lazily so ``from plume.viz.live import LiveStreamer`` in a simulation
does not need the viewer's web dependencies.
"""

from __future__ import annotations

__all__ = ["LiveStreamer", "create_app", "serve"]


def __getattr__(name: str):
    if name == "LiveStreamer":
        from plume.viz.live import LiveStreamer

        return LiveStreamer
    if name in ("create_app", "serve"):
        from plume.viz import server

        return getattr(server, name)
    raise AttributeError(name)
