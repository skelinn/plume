"""Static export of the viewer (e.g. for GitHub Pages).

``plume export-site --out site`` writes a self-contained folder:

* ``index.html`` + ``static/`` - the viewer with relative asset paths and
  ``window.PLUME_STATIC = true`` (see ``static/js/api.js``),
* ``api/replays.json`` and ``api/replays/<id>.json`` - the replay list and the replays
  (decompressed: static hosts do not send ``Content-Encoding`` for ``.gz`` files),
* ``api/terrain/<id>.<max_dim>.json`` / ``.f32`` - every terrain the replays use,
* ``reports/`` - the V&V report and Monte Carlo reports from ``docs/``.

The files are produced by calling the real server endpoints through FastAPI's test
client, so the static data is byte-for-byte what ``plume viz`` serves.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from plume.viz.server import STATIC_DIR, create_app

TERRAIN_DIM = 1024  # the resolution the viewer requests (scene.js)
FEATURED = ("real_hop_high.plume.json.gz", "cargo_hop_demo.plume.json.gz")


def _replay_bytes(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def export_site(
    out: str | Path,
    replay_dirs=("data/replays",),
    terrain_dirs=None,
    docs_dir: str | Path = "docs",
    log=print,
) -> Path:
    from fastapi.testclient import TestClient

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "api" / "replays").mkdir(parents=True)
    (out / "api" / "terrain").mkdir(parents=True)

    # viewer: relative paths, static flag before the import map
    shutil.copytree(STATIC_DIR, out / "static")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('"/static/', '"./static/')  # import maps need ./ for relative URLs
    html = html.replace("<head>", "<head>\n  <script>window.PLUME_STATIC = true;</script>", 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "static" / "index.html").unlink(missing_ok=True)

    app = create_app(replay_dirs=replay_dirs, terrain_dirs=terrain_dirs)
    client = TestClient(app)
    rows = client.get("/api/replays").json()
    order = {rid: k for k, rid in enumerate(FEATURED)}
    rows.sort(key=lambda r: (order.get(r["id"], len(order)), -r["modified"]))
    terrains: set[str] = set()
    for r in rows:
        r.pop("path", None)  # local filesystem path: not for the web
        rid = r["id"]
        resp = client.get(f"/api/replays/{rid}")
        resp.raise_for_status()
        body = resp.content  # the test client already undoes Content-Encoding: gzip
        data = json.loads(_replay_bytes(body))
        target = out / "api" / "replays" / f"{rid}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(json.dumps(data, separators=(",", ":")).encode())
        r["size"] = target.stat().st_size
        ground = (data.get("meta", {}).get("scene") or {}).get("ground") or {}
        if ground.get("terrain_id"):
            terrains.add(ground["terrain_id"])
        terrains.update(ground.get("detail_terrain_ids") or [])
        log(f"replay  {rid}  ({r['size'] / 1e6:.1f} MB)")
    (out / "api" / "replays.json").write_text(json.dumps(rows), encoding="utf-8")

    for tid in sorted(terrains):
        meta = client.get(f"/api/terrain/{tid}", params={"max_dim": TERRAIN_DIM})
        heights = client.get(f"/api/terrain/{tid}/heights", params={"max_dim": TERRAIN_DIM})
        if meta.status_code != 200 or heights.status_code != 200:
            log(f"terrain {tid}: not found, skipped")
            continue
        base = out / "api" / "terrain" / tid
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{base}.{TERRAIN_DIM}.json").write_text(meta.text, encoding="utf-8")
        Path(f"{base}.{TERRAIN_DIM}.f32").write_bytes(heights.content)
        log(f"terrain {tid}  ({len(heights.content) / 1e6:.1f} MB)")

    docs = Path(docs_dir)
    rep = out / "reports"
    rep.mkdir()
    links = []
    if (docs / "vv" / "report.html").exists():
        shutil.copy(docs / "vv" / "report.html", rep / "vv.html")
        links.append(("vv.html", "Verification & validation report"))
    for p in sorted((docs / "mc").glob("*.html")):
        (rep / "mc").mkdir(exist_ok=True)
        shutil.copy(p, rep / "mc" / p.name)
        links.append((f"mc/{p.name}", f"Monte Carlo: {p.stem}"))
    items = "".join(f'<li><a href="{h}">{t}</a></li>' for h, t in links)
    (rep / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Plume reports</title>
<style>:root{{--bg:#fff;--fg:#0a0a0a;--mute:#666}}@media (prefers-color-scheme:dark){{:root{{--bg:#0a0a0a;--fg:#f2f2f2;--mute:#8a8a8a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 "IBM Plex Sans",system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:40px 16px}}a{{color:inherit}}h1{{font-size:20px;font-weight:600}}
p{{color:var(--mute)}}</style></head><body><main><h1>Plume reports</h1>
<p>Verification results and Monte Carlo dependability campaigns. <a href="../">Open the 3-D viewer</a> ·
<a href="https://github.com/skelinn/plume">Source on GitHub</a></p><ul>{items}</ul></main></body></html>""",
        encoding="utf-8",
    )
    (out / ".nojekyll").write_text("", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"site: {out}  ({total / 1e6:.1f} MB)")
    return out
