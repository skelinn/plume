"""FastAPI server for the Plume 3-D viewer.

Endpoints
---------
``GET /``                              viewer (static/index.html)
``GET /static/...``                    viewer assets (three.js and uPlot are vendored)
``GET /api/replays``                   list of ``*.plume.json[.gz]`` found in the replay dirs
``GET /api/replays/{id:path}``         one replay (gzip passthrough, ``Content-Encoding: gzip``)
``GET /api/terrain``                   list of heightmaps found in the terrain dirs
``GET /api/terrain/{id:path}/heights`` raw little-endian float32, row-major, row 0 = south
``GET /api/terrain/{id:path}``         heightmap metadata
``POST /api/live/{channel}``           sims push ``{"meta"}`` / ``{"frames"}`` / ``{"end"}`` messages
``GET /api/live``                      state of the live channels
``WS /ws/live?channel=default``        fan-out of the above to viewers

Replay and terrain ids are POSIX paths relative to the directory they were found in
(terrain ids also resolve by bare file stem or by the ``name:`` key of the YAML, which is
what replays reference in ``scene.ground.terrain_id``).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import mimetypes
import threading
import webbrowser
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from plume.terrain.heightmap import Heightmap

# Windows maps .js to text/plain through the registry, which breaks ES modules.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

STATIC_DIR = Path(__file__).with_name("static")

DEFAULT_REPLAY_DIRS = ("runs", "data/replays")
DEFAULT_TERRAIN_DIRS = ("data/terrain", "runs/fixtures/terrain", "runs/terrain")
REPLAY_SUFFIXES = (".plume.json.gz", ".plume.json")
MAX_LIVE_FRAMES = 400_000  # per-channel history kept for late-joining viewers


# --------------------------------------------------------------------------- helpers
def _as_paths(dirs: Iterable[str | Path] | None, default: Sequence[str]) -> list[Path]:
    return [Path(d) for d in (default if dirs is None else dirs)]


def _safe_id(rid: str) -> str:
    """Reject ids that could escape their root (checked again after resolving)."""
    if (
        not rid
        or "\\" in rid
        or "\x00" in rid
        or rid.startswith("/")
        or ":" in rid
        or any(part in ("..", "") for part in rid.split("/"))
    ):
        raise HTTPException(status_code=400, detail="invalid id")
    return rid


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _peek_meta(path: Path) -> dict[str, Any]:
    """Read ``meta`` from the start of a (gzipped) replay without decoding all the frames."""
    try:
        with open(path, "rb") as f:
            raw = f.read(1 << 20 if path.suffix == ".gz" else 1 << 18)
        if path.suffix == ".gz":
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw, 1 << 20)
        text = raw.decode("utf-8", errors="ignore")
        k = text.find('"meta"')
        if k >= 0:
            colon = text.index(":", k)
            meta, _ = json.JSONDecoder().raw_decode(text[colon + 1 :].lstrip())
            if isinstance(meta, dict):
                return meta
    except Exception:
        pass
    try:
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        return json.loads(data).get("meta", {})
    except Exception:
        return {}


class _TerrainEntry:
    __slots__ = ("heightmap", "mtime", "path")

    def __init__(self, path: Path):
        self.path = path
        self.mtime = path.stat().st_mtime
        self.heightmap: Heightmap | None = None


# --------------------------------------------------------------------------- live
class _Channel:
    """One live episode: remembers what was sent so late viewers can catch up."""

    def __init__(self):
        self.meta: dict | None = None
        self.frames: dict[str, list] = {}
        self.n_frames = 0
        self.outcome: dict | None = None
        self.ended = False
        self.subscribers: set[_Subscriber] = set()

    def reset(self, meta: dict | None):
        self.meta = meta
        self.frames = {}
        self.n_frames = 0
        self.outcome = None
        self.ended = False

    def append(self, frames: dict[str, list]):
        n = len(frames.get("t", []))
        if self.n_frames + n > MAX_LIVE_FRAMES:  # keep memory bounded: drop history
            self.frames = {}
            self.n_frames = 0
        for key, col in frames.items():
            self.frames.setdefault(key, []).extend(col)
        self.n_frames += n


class _Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue[dict] = asyncio.Queue()

    def send(self, msg: dict):
        # POSTs and websockets may run on different event loops/threads (e.g. in tests).
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)


# --------------------------------------------------------------------------- app
def create_app(
    replay_dirs: Iterable[str | Path] | None = None,
    terrain_dirs: Iterable[str | Path] | None = None,
) -> FastAPI:
    rdirs = _as_paths(replay_dirs, DEFAULT_REPLAY_DIRS)
    tdirs = _as_paths(terrain_dirs, DEFAULT_TERRAIN_DIRS)
    app = FastAPI(title="Plume viewer", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.replay_dirs = rdirs
    app.state.terrain_dirs = tdirs

    meta_cache: dict[tuple[str, float], dict] = {}
    terrain_cache: dict[str, _TerrainEntry] = {}
    channels: dict[str, _Channel] = {}
    lock = threading.Lock()

    # ---- replay index -------------------------------------------------------
    def scan_replays() -> dict[str, tuple[Path, Path]]:
        """id -> (root, file). The first root wins on id collisions."""
        out: dict[str, tuple[Path, Path]] = {}
        for root in rdirs:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name.endswith(REPLAY_SUFFIXES):
                    out.setdefault(p.relative_to(root).as_posix(), (root, p))
        return out

    @app.get("/api/replays")
    def list_replays():
        rows = []
        for rid, (_root, p) in scan_replays().items():
            st = p.stat()
            key = (str(p), st.st_mtime)
            if key not in meta_cache:
                meta_cache[key] = _peek_meta(p)
            meta = meta_cache[key]
            rows.append(
                {
                    "id": rid,
                    "title": meta.get("title") or rid,
                    "source": meta.get("source", "sim"),
                    "path": str(p),
                    "size": st.st_size,
                    "modified": st.st_mtime,
                }
            )
        rows.sort(key=lambda r: r["modified"], reverse=True)
        return rows

    @app.get("/api/replays/{rid:path}")
    def get_replay(rid: str):
        _safe_id(rid)
        hit = scan_replays().get(rid)
        if hit is None or not _inside(*hit):
            raise HTTPException(status_code=404, detail="replay not found")
        p = hit[1]
        headers = {"Cache-Control": "no-cache"}
        if p.suffix == ".gz":
            headers["Content-Encoding"] = "gzip"
        return Response(p.read_bytes(), media_type="application/json", headers=headers)

    # ---- terrain ------------------------------------------------------------
    def scan_terrain() -> dict[str, _TerrainEntry]:
        out: dict[str, _TerrainEntry] = {}
        aliases: dict[str, _TerrainEntry] = {}
        for root in tdirs:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*.yaml")):
                key = str(p)
                ent = terrain_cache.get(key)
                try:
                    if ent is None or ent.mtime != p.stat().st_mtime:
                        ent = terrain_cache[key] = _TerrainEntry(p)
                except OSError:
                    continue
                rid = p.relative_to(root).with_suffix("").as_posix()
                out.setdefault(rid, ent)
                aliases.setdefault(p.stem, ent)
        for k, v in aliases.items():
            out.setdefault(k, v)
        return out

    def find_terrain(tid: str) -> _TerrainEntry:
        _safe_id(tid)
        ents = scan_terrain()
        ent = ents.get(tid)
        if ent is None:  # fall back to the YAML ``name:`` key
            import yaml

            for cand in {id(e): e for e in ents.values()}.values():
                try:
                    if yaml.safe_load(cand.path.read_text()).get("name") == tid:
                        ent = cand
                        break
                except Exception:
                    continue
        if ent is None or not any(_inside(r, ent.path) for r in tdirs if r.is_dir()):
            raise HTTPException(status_code=404, detail="terrain not found")
        return ent

    def load_heightmap(ent: _TerrainEntry) -> Heightmap:
        if ent.heightmap is None:
            with lock:
                if ent.heightmap is None:
                    ent.heightmap = Heightmap.load(ent.path)
        return ent.heightmap

    def view_of(hm: Heightmap, max_dim: int) -> tuple[np.ndarray, int, int]:
        ny, nx = hm.shape
        if max(nx, ny) <= max_dim:
            return hm.heights, nx, ny
        scale = max_dim / max(nx, ny)
        new_nx, new_ny = max(2, round(nx * scale)), max(2, round(ny * scale))
        grid = hm.sample_grid(hm.x_min, hm.x_max, hm.y_min, hm.y_max, new_nx, new_ny)
        return grid, new_nx, new_ny

    @app.get("/api/terrain")
    def list_terrain():
        rows = []
        for tid, ent in scan_terrain().items():
            rows.append({"id": tid, "path": str(ent.path)})
        return rows

    @app.get("/api/terrain/{tid:path}/heights")
    def terrain_heights(tid: str, max_dim: int = Query(1024, ge=2, le=8192)):
        hm = load_heightmap(find_terrain(tid))
        grid, nx, ny = view_of(hm, max_dim)
        data = np.ascontiguousarray(grid, dtype="<f4").tobytes()
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Nx": str(nx), "X-Ny": str(ny), "Cache-Control": "no-cache"},
        )

    @app.get("/api/terrain/{tid:path}")
    def terrain_meta(tid: str, max_dim: int = Query(1024, ge=2, le=8192)):
        hm = load_heightmap(find_terrain(tid))
        grid, nx, ny = view_of(hm, max_dim)
        return {
            "id": tid,
            "x_min": hm.x_min,
            "x_max": hm.x_max,
            "y_min": hm.y_min,
            "y_max": hm.y_max,
            "nx": nx,
            "ny": ny,
            "native_nx": hm.shape[1],
            "native_ny": hm.shape[0],
            "z_min": float(np.min(grid)),
            "z_max": float(np.max(grid)),
        }

    # ---- live ---------------------------------------------------------------
    def channel(name: str) -> _Channel:
        return channels.setdefault(name, _Channel())

    @app.get("/api/live")
    def live_state():
        return {
            name: {
                "title": (ch.meta or {}).get("title"),
                "frames": ch.n_frames,
                "ended": ch.ended,
                "viewers": len(ch.subscribers),
                "active": ch.meta is not None,
            }
            for name, ch in channels.items()
        }

    @app.post("/api/live/{name}")
    async def live_post(name: str, request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        ch = channel(name)
        msgs: list[dict] = []
        if "meta" in body:
            ch.reset(body["meta"])
            msgs.append({"type": "meta", "meta": body["meta"]})
        if "frames" in body:
            frames = body["frames"]
            if not isinstance(frames, dict) or not isinstance(frames.get("t"), list):
                raise HTTPException(status_code=400, detail="frames must be columnar with 't'")
            if ch.meta is None:
                raise HTTPException(status_code=409, detail="send {'meta': ...} first")
            ch.append(frames)
            msgs.append({"type": "frames", "frames": frames})
        if body.get("end"):
            ch.ended = True
            ch.outcome = body.get("outcome")
            msgs.append({"type": "end", "outcome": ch.outcome})
        for msg in msgs:
            for sub in list(ch.subscribers):
                sub.send(msg)
        return {"ok": True, "viewers": len(ch.subscribers), "frames": ch.n_frames}

    @app.websocket("/ws/live")
    async def live_ws(ws: WebSocket, channel: str = "default"):
        await ws.accept()
        ch = channels.setdefault(channel, _Channel())
        sub = _Subscriber(asyncio.get_running_loop())
        # Catch the viewer up on the current episode before streaming new messages.
        if ch.meta is not None:
            await ws.send_json({"type": "meta", "meta": ch.meta})
            if ch.n_frames:
                await ws.send_json({"type": "frames", "frames": ch.frames})
            if ch.ended:
                await ws.send_json({"type": "end", "outcome": ch.outcome})
        else:
            await ws.send_json({"type": "idle"})
        ch.subscribers.add(sub)

        async def pump():
            while True:
                await ws.send_json(await sub.queue.get())

        task = asyncio.create_task(pump())
        try:
            while True:  # also notices client disconnects
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            ch.subscribers.discard(sub)

    # ---- static -------------------------------------------------------------
    @app.middleware("http")
    async def no_stale_assets(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    replay_dirs: Iterable[str | Path] | None = None,
    terrain_dirs: Iterable[str | Path] | None = None,
    open_browser: bool = True,
    url_path: str = "",
) -> None:
    """Run the viewer server (blocking)."""
    import uvicorn

    app = create_app(replay_dirs, terrain_dirs)
    url = f"http://{host}:{port}/{url_path}"
    print(f"Plume viewer: {url}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
