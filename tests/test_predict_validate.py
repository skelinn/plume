"""Pre-flight prediction (`plume predict`) and validation records (`plume validate`)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from plume.analysis.vv import REPO, environment, read_model_docs, render_html
from plume.cli import app
from plume.config import load_vehicle
from plume.flightdata.predict import (
    LaunchConditions,
    Prediction,
    fly,
    predict,
    report_markdown,
    resolve_motor,
    vehicle_hash,
    with_motor,
    write_prediction,
)
from plume.flightdata.validation import (
    Check,
    build_record,
    read_records,
    status_by_doc,
    verdicts,
    write_record,
)

SAMPLE_A = "data/flights/sample_flight.csv"


@pytest.fixture(scope="module")
def hobby():
    return load_vehicle("hobby_rocket")


@pytest.fixture(scope="module")
def prediction(hobby, tmp_path_factory):
    launch = LaunchConditions(wind_speed_mps=0.0, rail_tilt_deg=4.0)
    p = predict(hobby, launch, runs=16, seed=1, flight_id="test_pred")
    paths = write_prediction(p, tmp_path_factory.mktemp("pred"))
    return p, paths


def test_fly_physics_trends(hobby):
    calm = LaunchConditions(wind_speed_mps=0.0, rail_tilt_deg=0.0)
    base = fly(hobby, calm)
    assert 600 < base["apogee_m"] < 1100
    assert abs(base["landing_distance_m"]) < 5.0  # vertical, no wind: lands near the pad
    assert fly(hobby, calm, cd_scale=1.3)["apogee_m"] < base["apogee_m"]
    assert fly(hobby, calm, impulse_scale=1.05)["apogee_m"] > base["apogee_m"]
    windy = fly(hobby, LaunchConditions(wind_speed_mps=5.0, wind_from_deg=270.0, rail_tilt_deg=0))
    assert windy["landing_east_m"] > 100.0  # drifts downwind (east) under the parachute
    assert 5.0 < base["descent_rate_mps"] < 7.5  # ~36 in canopy
    assert base["rail_exit_mps"] > 15.0


def test_motor_swap(hobby):
    assert resolve_motor("H180").name == "Plume_H180.eng"
    with pytest.raises(FileNotFoundError):
        resolve_motor("Z9999")
    g, notes = with_motor(hobby, "G75")
    assert g.tanks[0].capacity == pytest.approx(0.062)
    assert g.mass.dry == pytest.approx(hobby.mass.dry - 0.077, abs=1e-6)  # lighter casing
    assert notes and vehicle_hash(g) != vehicle_hash(hobby)
    calm = LaunchConditions(wind_speed_mps=0.0)
    assert fly(g, calm)["apogee_m"] < 0.6 * fly(hobby, calm)["apogee_m"]


def test_prediction_record(prediction):
    p, paths = prediction
    s = p.summary["apogee_m"]
    assert s["p2.5"] <= s["p5"] <= s["p50"] <= s["p95"] <= s["p97.5"]
    assert s["p2.5"] < p.nominal["apogee_m"] < s["p97.5"]
    assert p.landing["p95_distance_m"] > 0 and len(p.landing["semi_axes_m"]) == 2
    assert len(p.runs) == 16
    m = p.meta
    assert m["created_utc"] and m["motor"]["name"] == "H180" and len(m["motor"]["sha256"]) == 64
    assert m["vehicle"]["sha256"] == vehicle_hash(load_vehicle("hobby_rocket"))
    q = Prediction.load(paths["json"])
    assert q.summary == p.summary and q.meta["flight_id"] == "test_pred"
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "before the flight" in md and "95 % band" in md and paths["figure"].exists()
    assert "Limits of this prediction" in report_markdown(p)


def test_same_seed_same_prediction(hobby):
    a = predict(hobby, runs=4, seed=7)
    b = predict(hobby, runs=4, seed=7)
    assert a.summary == b.summary


def test_validation_record_on_synthetic_log(prediction, tmp_path):
    p, paths = prediction
    rec, cal, log = build_record(
        SAMPLE_A,
        "generic_altimeter",
        p,
        "rehearsal_a",
        flight_date="2099-01-01T00:00Z",
        synthetic=True,
    )
    assert rec["synthetic"] and rec["prediction"]["blind"] is True
    # the synthetic "truth" has 22 % more drag than the model: outside the 10 % sigma
    assert rec["calibration"]["params"]["cd_scale"] == pytest.approx(1.22, abs=0.05)
    assert rec["verdicts"]["drag"] == "discrepancy"
    apo = next(r for r in rec["comparison"] if r["metric"] == "apogee_m")
    assert apo["observed"] == pytest.approx(log.apogee) and not apo["inside_95"]
    out = write_record(rec, cal, log, paths["json"], tmp_path)
    assert out["markdown"].exists() and out["overlay"].exists() and out["vehicle"].exists()
    assert "Synthetic data" in out["markdown"].read_text(encoding="utf-8")
    recs = read_records(tmp_path)
    assert [r["flight_id"] for r in recs] == ["rehearsal_a"]
    assert status_by_doc(recs) == {}  # rehearsals never count


def test_verdict_rules():
    ok = [Check("drag", "q", 1.0, "c", True), Check("motor", "q", 1.0, "c", True)]
    ok += [Check("parachute", "q", 1.0, "c", True)]
    assert verdicts(ok, blind=True, synthetic=False)["drag"] == "validated"
    assert verdicts(ok, blind=None, synthetic=False)["drag"] == "consistent"
    assert verdicts(ok, blind=False, synthetic=False)["drag"] == "consistent"
    assert verdicts(ok, blind=True, synthetic=True)["drag"] == "consistent"
    bad = [*ok, Check("drag", "q", 2.0, "c", False), Check("parachute", "q", None, "c", None)]
    v = verdicts(bad, blind=True, synthetic=False)
    assert v["drag"] == "discrepancy" and v["parachute"] == "inconclusive"
    assert v["motor"] == "validated"


def test_vv_report_shows_real_validation(tmp_path):
    models = read_model_docs(REPO / "docs" / "models")
    real = {
        "flight_id": "flight_001",
        "flight_date": "2027-03-01",
        "synthetic": False,
        "prediction": {"blind": True},
        "envelope": "hobby_rocket on H180, Mach < 0.5",
        "verdicts": {"drag": "validated", "motor": "discrepancy", "parachute": "validated"},
        "_md": str(tmp_path / "flight_001.md"),
    }
    fake = {**real, "flight_id": "rehearsal", "synthetic": True, "_md": str(tmp_path / "r.md")}
    page = render_html(
        env=environment(REPO),
        tests=[],
        junit_meta={},
        junit_path=None,
        models=models,
        nasa=None,
        mc=[],
        out_dir=tmp_path,
        root=REPO,
        ran=False,
        validation=[real, fake],
    )
    assert f"<b>1 / {len(models)}</b><span>models validated</span>" in page
    assert "flight_001</a>: validated (hobby_rocket on H180" in page
    assert "flight_001</a>: discrepancy" in page  # propulsion row
    assert page.count("Validation: pending real data") == len(models) - 2
    assert "synthetic rehearsal" in page


def test_cli_predict_and_validate(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "predict",
            "hobby_rocket",
            "--motor",
            "H180",
            "--runs",
            "6",
            "--workers",
            "1",
            "--wind",
            "0",
            "--flight-id",
            "cli_pred",
            "--out-dir",
            str(tmp_path / "pred"),
        ],
    )
    assert res.exit_code == 0, res.output
    pj = tmp_path / "pred" / "cli_pred.json"
    assert json.loads(pj.read_text())["format"] == "plume-prediction"
    res = runner.invoke(
        app,
        [
            "validate",
            SAMPLE_A,
            "--prediction",
            str(pj),
            "--flight-id",
            "cli_rehearsal",
            "--synthetic",
            "--out-root",
            str(tmp_path / "val"),
        ],
    )
    assert res.exit_code == 0, res.output
    rec = json.loads((tmp_path / "val" / "cli_rehearsal" / "record.json").read_text())
    assert rec["prediction"]["blind"] is None and rec["synthetic"]
    assert np.isfinite(rec["observed"]["apogee_m"])
