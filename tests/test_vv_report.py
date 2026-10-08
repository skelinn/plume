"""V&V report generator (``plume vv-report``): fast checks on a synthetic JUnit file."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from plume.analysis.vv import REPO, area_counts, classify, parse_junit, read_model_docs
from plume.cli import app

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="1" tests="7"
 timestamp="2026-10-08T12:00:00">
<testcase classname="tests.vv.test_nasa_checkcases" name="test_nasa_atmospheric_checkcase[1]" time="3.5"/>
<testcase classname="tests.vv.test_nasa_checkcases" name="test_nasa_atmospheric_checkcase[2]" time="4.0"/>
<testcase classname="tests.vv.test_analytic" name="test_convergence_with_timestep" time="1.2"/>
<testcase classname="tests.vv.test_analytic" name="test_geodetic_roundtrip" time="0.01"/>
<testcase classname="tests.test_aerodb" name="test_slender_body_limit" time="0.3">
  <failure message="assert 2.1 &lt; 2.0">long traceback</failure>
</testcase>
<testcase classname="tests.test_dem" name="test_network_fetch_real_tiles" time="0.0">
  <skipped type="pytest.skip" message="offline">skipped</skipped>
</testcase>
<testcase classname="tests.test_actuators" name="test_ignition_delay" time="0.1"/>
</testsuite></testsuites>
"""


def _junit(tmp_path: Path) -> Path:
    p = tmp_path / "junit.xml"
    p.write_text(JUNIT, encoding="utf-8")
    return p


def test_parse_and_group(tmp_path):
    tests, meta = parse_junit(_junit(tmp_path))
    assert len(tests) == 7 and meta["timestamp"] == "2026-10-08T12:00:00"
    counts = area_counts(tests)
    assert counts["nasa"] == {"passed": 2, "failed": 0, "skipped": 0, "total": 2, "time": 7.5}
    assert counts["integration"]["passed"] == 1 and counts["earth"]["passed"] == 1
    assert counts["aero"]["failed"] == 1 and counts["terrain"]["skipped"] == 1
    assert counts["propulsion"]["total"] == 1
    assert classify("tests.test_physics_conservation", "test_tsiolkovsky_delta_v") == "propulsion"
    assert classify("tests.test_physics_models", "test_cg_drops_as_tanks_drain") == "mass"


def test_model_docs_have_status_lines():
    docs = {d.path.stem: d for d in read_model_docs(REPO / "docs" / "models")}
    for stem in ("earth", "atmosphere", "aero", "navigation", "terrain", "propulsion", "mass"):
        assert stem in docs, stem
        assert docs[stem].status and docs[stem].code, stem
        assert docs[stem].areas, stem


def test_cli_no_run_no_nasa(tmp_path):
    out = tmp_path / "vv" / "report.html"
    res = CliRunner().invoke(
        app,
        ["vv-report", "--no-run", "--no-nasa", "--junit", str(_junit(tmp_path)), "--out", str(out)],
    )
    assert res.exit_code == 1, res.output  # the synthetic suite has one failure
    page = out.read_text(encoding="utf-8")
    assert "Plume - verification &amp; validation report" in page
    assert "<th>verification status</th>" in page  # model table
    assert "Earth model, frames and integration" in page
    assert page.count("Validation: pending real data") == len(
        read_model_docs(REPO / "docs" / "models")
    )
    assert "<b>5 / 7</b><span>tests passed</span>" in page
    assert "2 / 2" in page  # NASA area
    assert "test_slender_body_limit" in page
    assert "assert 2.1 &lt; 2.0" in page and "2.1 < 2.0" not in page  # escaped
    assert "--no-nasa" in page  # NASA table not run
    assert "http://" not in page and "https://" not in page and "<script" not in page


def test_mc_headline_from_report_page():
    from plume.analysis.vv import _mc_from_html

    page = (
        '<p class="sub">mission <code>demo_hop</code> · fidelity fast · 200 runs</p>'
        '<div class="kpis"><div class="kpi"><b>19.0 %</b><span>success</span></div>'
        '<div class="kpi"><b>14.2–25.0 %</b><span>95 % interval</span></div>'
        '<div class="kpi"><b>234.0 m</b><span>CEP50</span></div></div>'
    )
    h = _mc_from_html(page)
    assert h["runs"] == "200" and h["fidelity"] == "fast" and h["mission"] == "demo_hop"
    assert h["success"] == "19.0 %" and h["ci"] == "14.2–25.0 %" and h["cep50"] == "234 m"
