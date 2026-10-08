"""Tests for the viewer server (replay/terrain API, live fan-out, static files)."""

from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from plume.recording import Recorder
from plume.terrain import Heightmap
from plume.viz.live import LiveStreamer
from plume.viz.server import STATIC_DIR, create_app

VEHICLE = {"name": "t", "length": 5.0, "diameter": 1.0}


def _replay(title: str, source: str = "sim", n: int = 5) -> Recorder:
    rec = Recorder(
        {
            "title": title,
            "source": source,
            "vehicle": VEHICLE,
            "scene": {"frame": "flat", "ground": {"type": "plane"}},
        }
    )
    for i in range(n):
        rec.record({"t": 0.1 * i, "pos": [0, 0, 10.0 - i], "quat": [1, 0, 0, 0]})
    return rec


@pytest.fixture()
def env(tmp_path):
    rdir = tmp_path / "runs"
    tdir = tmp_path / "terrain"
    _replay("Alpha").save(rdir / "alpha.plume.json.gz")
    _replay("Beta (real)", source="real").save(rdir / "sub" / "beta.plume.json.gz")
    (rdir / "plain.plume.json").write_text(json.dumps(_replay("Plain").to_dict()))
    (rdir / "notes.txt").write_text("ignored")
    (tmp_path / "secret.plume.json.gz").write_bytes(gzip.compress(b"{}"))

    # 200 x 100 samples over 2 km x 1 km; height = x + 10*y (non-trivial, easy to check).
    xs = np.linspace(-1000, 1000, 200)
    ys = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(xs, ys)
    Heightmap(X * 0.1 + Y * 0.05, -1000, 1000, -500, 500, name="ramp").save(
        tdir / "ramp.yaml", fmt="npy"
    )
    app = create_app([rdir], [tdir])
    with TestClient(app) as client:
        yield client, rdir, tdir


def test_index_and_static(env):
    client, *_ = env
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "importmap" in r.text
    for path in (
        "/static/vendor/three/three.module.min.js",
        "/static/vendor/three/examples/jsm/controls/OrbitControls.js",
        "/static/vendor/uplot/uPlot.esm.js",
        "/static/vendor/uplot/uPlot.min.css",
        "/static/js/main.js",
        "/static/js/atmosphere.js",
        "/static/js/post.js",
        "/static/js/engineering.js",
        "/static/js/models/gridfin.js",
        "/static/js/models/legs.js",
        "/static/js/models/engine.js",
        "/static/js/models/hull.js",
        "/static/css/style.css",
    ):
        r = client.get(path)
        assert r.status_code == 200, path
        if path.endswith(".js"):
            # ES modules are rejected by browsers unless served as a JavaScript MIME type.
            assert "javascript" in r.headers["content-type"], path
    assert client.get("/static/nope.js").status_code == 404


def test_viewer_assets_are_self_contained(env):
    """Fonts are vendored (no CDN at runtime) and served with a font MIME type."""
    client, *_ = env
    for name in ("ibm-plex-sans-latin-400-normal.woff2", "ibm-plex-mono-latin-500-normal.woff2"):
        r = client.get(f"/static/vendor/fonts/{name}")
        assert r.status_code == 200, name
        assert r.headers["content-type"].startswith("font/woff2"), name
        assert r.content[:4] == b"wOF2"
    css = client.get("/static/css/style.css").text
    html = client.get("/").text
    for text in (css, html):
        assert "https://" not in text and "http://" not in text.replace("http://www.w3.org", "")
    # every relative ES-module import in the viewer resolves
    for js in (STATIC_DIR / "js").rglob("*.js"):
        for spec in re.findall(r"from '(\.{1,2}/[^']+)'", js.read_text(encoding="utf-8")):
            target = (js.parent / spec).resolve()
            assert target.is_file(), f"{js.name}: {spec}"


def test_list_replays(env):
    client, *_ = env
    rows = client.get("/api/replays").json()
    ids = {r["id"] for r in rows}
    assert ids == {"alpha.plume.json.gz", "sub/beta.plume.json.gz", "plain.plume.json"}
    by_id = {r["id"]: r for r in rows}
    assert by_id["sub/beta.plume.json.gz"]["title"] == "Beta (real)"
    assert by_id["sub/beta.plume.json.gz"]["source"] == "real"
    assert by_id["alpha.plume.json.gz"]["size"] > 0
    mods = [r["modified"] for r in rows]
    assert mods == sorted(mods, reverse=True)
    assert set(rows[0]) >= {"id", "title", "source", "path", "size", "modified"}


def test_get_replay(env):
    client, *_ = env
    for rid in ("alpha.plume.json.gz", "sub/beta.plume.json.gz", "plain.plume.json"):
        r = client.get(f"/api/replays/{rid}")
        assert r.status_code == 200
        data = r.json()  # httpx transparently handles Content-Encoding: gzip
        assert data["format"] == "plume-replay"
        assert len(data["frames"]["t"]) == 5
    assert client.get("/api/replays/missing.plume.json.gz").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/replays/..%2Fsecret.plume.json.gz",
        "/api/replays/%2e%2e/secret.plume.json.gz",
        "/api/replays/sub/..%2F..%2Fsecret.plume.json.gz",
        "/api/replays/..%5Csecret.plume.json.gz",
        "/api/replays/%2Fetc%2Fpasswd",
        "/api/terrain/..%2F..%2Fx",
        "/api/terrain/%2e%2e%2Framp/heights",
    ],
)
def test_traversal_rejected(env, path):
    client, *_ = env
    r = client.get(path)
    assert r.status_code in (400, 404)
    assert "format" not in r.text


def test_terrain_meta_and_heights(env):
    client, *_ = env
    meta = client.get("/api/terrain/ramp").json()
    assert meta["id"] == "ramp"
    assert (meta["nx"], meta["ny"]) == (200, 100)
    assert (meta["x_min"], meta["x_max"], meta["y_min"], meta["y_max"]) == (-1000, 1000, -500, 500)
    assert meta["z_max"] > meta["z_min"]

    r = client.get("/api/terrain/ramp/heights")
    assert r.status_code == 200
    assert len(r.content) == 200 * 100 * 4
    h = np.frombuffer(r.content, dtype="<f4").reshape(100, 200)
    # row 0 = south (y_min), column 0 = west (x_min); height = 0.1 x + 0.05 y
    assert h[0, 0] == pytest.approx(-100 - 25, abs=1e-3)
    assert h[99, 199] == pytest.approx(100 + 25, abs=1e-3)
    assert h[0, 199] == pytest.approx(100 - 25, abs=1e-3)
    assert float(h.min()) == pytest.approx(meta["z_min"], abs=1e-3)
    assert client.get("/api/terrain/nope").status_code == 404
    assert client.get("/api/terrain/nope/heights").status_code == 404


def test_terrain_downsample(env):
    client, *_ = env
    meta = client.get("/api/terrain/ramp?max_dim=50").json()
    assert meta["nx"] == 50 and meta["ny"] == 25
    assert meta["native_nx"] == 200
    r = client.get("/api/terrain/ramp/heights?max_dim=50")
    assert r.headers["x-nx"] == "50" and r.headers["x-ny"] == "25"
    h = np.frombuffer(r.content, dtype="<f4").reshape(25, 50)
    assert h[0, 0] == pytest.approx(-125, abs=1e-3)  # corners are preserved
    assert h[-1, -1] == pytest.approx(125, abs=1e-3)
    # not downsampled when under the limit
    assert len(client.get("/api/terrain/ramp/heights?max_dim=512").content) == 200 * 100 * 4


def test_live_fanout(env):
    client, *_ = env
    meta = {"title": "live test", "source": "sim", "vehicle": VEHICLE, "scene": {"frame": "flat"}}
    with (
        client.websocket_connect("/ws/live?channel=c1") as a,
        client.websocket_connect("/ws/live?channel=c1") as b,
        client.websocket_connect("/ws/live?channel=other") as other,
    ):
        assert a.receive_json()["type"] == "idle"
        assert b.receive_json()["type"] == "idle"
        assert other.receive_json()["type"] == "idle"

        assert client.post("/api/live/c1", json={"meta": meta}).json()["ok"]
        frames = {"t": [0.0, 0.1], "pos": [[0, 0, 1], [0, 0, 2]], "quat": [[1, 0, 0, 0]] * 2}
        assert client.post("/api/live/c1", json={"frames": frames}).status_code == 200
        outcome = {"success": True, "reason": "landed"}
        client.post("/api/live/c1", json={"end": True, "outcome": outcome})

        for ws in (a, b):
            assert ws.receive_json() == {"type": "meta", "meta": meta}
            got = ws.receive_json()
            assert got["type"] == "frames" and got["frames"] == frames
            assert ws.receive_json() == {"type": "end", "outcome": outcome}

        # a late joiner is caught up with meta + all frames + end
        with client.websocket_connect("/ws/live?channel=c1") as late:
            assert late.receive_json()["type"] == "meta"
            assert late.receive_json()["frames"]["t"] == [0.0, 0.1]
            assert late.receive_json()["type"] == "end"

    # frames before any meta are refused; garbage is a 400
    assert client.post("/api/live/fresh", json={"frames": {"t": [0]}}).status_code == 409
    assert client.post("/api/live/fresh", json=[1, 2]).status_code == 400
    state = client.get("/api/live").json()
    assert state["c1"]["frames"] == 2 and state["c1"]["ended"]


def test_live_streamer_swallows_errors():
    # Nothing is listening here: must not raise or hang.
    live = LiveStreamer(url="http://127.0.0.1:9", timeout=0.2)
    live.start({"title": "x"})
    for i in range(3):
        live.push({"t": 0.1 * i, "pos": np.zeros(3)})
    live.end({"success": False})
