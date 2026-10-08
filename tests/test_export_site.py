"""Static site export (GitHub Pages)."""

from __future__ import annotations

import json

from plume.viz.export import export_site


def test_export_site_is_self_contained(tmp_path):
    out = export_site(tmp_path / "site", log=lambda *_: None)
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "window.PLUME_STATIC = true" in html
    assert '"/static/' not in html and '"./static/js/main.js"' in html
    rows = json.loads((out / "api" / "replays.json").read_text())
    assert rows and all("path" not in r for r in rows)  # no local filesystem paths
    first = rows[0]["id"]
    data = json.loads((out / "api" / "replays" / f"{first}.json").read_text())
    assert "frames" in data and "meta" in data
    ground = data["meta"]["scene"]["ground"]
    for tid in [ground["terrain_id"], *ground.get("detail_terrain_ids", [])]:
        assert (out / "api" / "terrain" / f"{tid}.1024.json").exists()
        assert (out / "api" / "terrain" / f"{tid}.1024.f32").stat().st_size > 0
    assert (out / "static" / "js" / "api.js").exists()
    assert (out / "reports" / "index.html").exists()
    assert (out / ".nojekyll").exists()
