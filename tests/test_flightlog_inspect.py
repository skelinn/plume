"""`plume flightlog inspect`: column/unit guessing on the bundled logs and odd formats."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from plume.cli import app
from plume.config import LogMappingSpec, load_log_mapping
from plume.flightdata.importer import load_log
from plume.flightdata.inspect import inspect_csv, write_draft

SAMPLE_A = "data/flights/sample_flight.csv"
SAMPLE_B = "data/flights/sample_flight_b.csv"


@pytest.mark.parametrize(
    ("csv", "reference"), [(SAMPLE_A, "generic_altimeter"), (SAMPLE_B, "simple_altimeter")]
)
def test_draft_matches_hand_written_mapping(csv, reference):
    ins = inspect_csv(csv)
    ref = load_log_mapping(reference)
    draft = LogMappingSpec.model_validate(ins.mapping)
    assert draft.delimiter == ref.delimiter
    for part in ("time", "altitude", "acceleration", "gyro", "gps"):
        a, b = getattr(draft, part), getattr(ref, part)
        assert (a is None) == (b is None), part
        if a is not None:
            assert a.model_dump() == b.model_dump(), part
    assert not ins.warnings
    # and it imports the same flight
    assert load_log(csv, draft).apogee == pytest.approx(load_log(csv, ref).apogee)


def test_units_from_values_when_names_do_not_say(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(0, 20000, 10)  # ms clock, but the column is just "Clock"
    alt = np.where(t > 5000, (t - 5000) / 1000.0 * 40.0, 0.0) + rng.normal(0, 0.3, len(t))
    acc = np.where((t > 5000) & (t < 6500), 8.0, 1.0) + rng.normal(0, 0.02, len(t))
    p = tmp_path / "odd.csv"
    with open(p, "w") as f:
        f.write("Clock\tAltitude\tAccZ\tTemp\n")
        for row in zip(t, alt, acc, np.full(len(t), 21.0), strict=True):
            f.write("\t".join(f"{v:.3f}" for v in row) + "\n")
    ins = inspect_csv(p)
    m = ins.mapping
    assert m["delimiter"] == "\t"
    assert m["time"] == {"column": "Clock", "unit": "ms"}
    assert m["altitude"]["unit"] == "m"
    assert m["acceleration"] == {"column": "AccZ", "unit": "g", "includes_gravity": True}
    assert any("altitude unit" in w for w in ins.warnings)  # metres is only a guess
    assert "gyro" not in m and "gps" not in m
    out = write_draft(ins, tmp_path / "draft.yaml")
    text = out.read_text()
    assert text.startswith("# DRAFT") and "CONFIRM" in text and "WARNING" in text
    assert yaml.safe_load(text)["time"]["column"] == "Clock"


def test_inverted_accelerometer_and_missing_altitude(tmp_path):
    p = tmp_path / "inv.csv"
    t = np.arange(0, 10, 0.01)
    with open(p, "w") as f:
        f.write("time_s,pressure_pa,acc_x_g\n")
        for tt in t:
            f.write(f"{tt:.2f},{101325 - 10 * tt:.1f},{-1.0 if tt < 5 else -6.0}\n")
    ins = inspect_csv(p)
    assert ins.mapping["acceleration"]["scale"] == -1.0
    assert any("no altitude column" in w and "pressure_pa" in w for w in ins.warnings)


def test_cli_inspect(tmp_path):
    out = tmp_path / "a.yaml"
    res = CliRunner().invoke(app, ["flightlog", "inspect", SAMPLE_A, "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and "trial import" in res.output
    assert LogMappingSpec.model_validate(yaml.safe_load(out.read_text())).gyro is not None


def test_template_mapping_is_valid():
    m = load_log_mapping("template")
    assert m.time.unit == "ms" and m.acceleration.includes_gravity
