"""Config loading/validation, replay recording, and terrain heightmaps."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from plume.config import VehicleSpec, dump_yaml, load_vehicle
from plume.recording import Recorder, frames_as_arrays, load_replay
from plume.recording.recorder import SCHEMA_PATH
from plume.terrain import Heightmap, flatten_sites, generate_detail_tile, generate_regional_map
from tests.conftest import make_test_vehicle

# --------------------------------------------------------------------------- config


def test_presets_load():
    v = load_vehicle("lander_small")
    assert v.prop_initial == pytest.approx(1000.0)
    assert v.engine.isp_sea_level == 275.0


def test_yaml_roundtrip(tmp_path, lander):
    path = dump_yaml(lander, tmp_path / "v.yaml")
    again = load_vehicle(path)
    assert again == lander


def test_validation_rejects_typos_and_bad_values():
    good = make_test_vehicle().model_dump()
    bad = json.loads(json.dumps(good))
    bad["engine"]["thrust_vacc"] = 1.0  # typo
    with pytest.raises(ValidationError, match="thrust_vacc"):
        VehicleSpec.model_validate(bad)
    bad = json.loads(json.dumps(good))
    bad["tanks"][0]["z_top"] = 0.5
    with pytest.raises(ValidationError, match="z_top"):
        VehicleSpec.model_validate(bad)
    bad = json.loads(json.dumps(good))
    bad["cargo"] = {"mass": 500, "max_mass": 250}
    with pytest.raises(ValidationError, match="exceeds max"):
        VehicleSpec.model_validate(bad)
    bad = json.loads(json.dumps(good))
    bad["engine"] = {"type": "solid"}
    with pytest.raises(ValidationError, match="thrust_curve"):
        VehicleSpec.model_validate(bad)


def test_unknown_preset_message():
    with pytest.raises(FileNotFoundError, match="no vehicle config"):
        load_vehicle("does_not_exist")


def test_with_cargo_validates(lander):
    v = make_test_vehicle(cargo={"mass": 0, "max_mass": 100, "cg_z": 6})
    assert v.with_cargo(80).cargo.mass == 80
    with pytest.raises(ValidationError):
        v.with_cargo(150)


# --------------------------------------------------------------------------- recording


def test_recorder_roundtrip_and_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    rec = Recorder(
        {
            "title": "t",
            "source": "sim",
            "vehicle": {"name": "x", "length": 1.0, "diameter": 0.1},
            "scene": {"frame": "flat", "ground": {"type": "plane"}},
        },
        every=2,
    )
    for i in range(10):
        rec.record(
            {"t": i * 0.1, "pos": np.array([i, 0.0, 1.0]), "quat": [1, 0, 0, 0], "phase": "a"}
        )
    rec.event(0.5, "ignition", "go")
    rec.set_outcome(True, "landed", {"err": np.float64(1.23456)})
    assert len(rec) == 5
    path = rec.save(tmp_path / "r.plume.json.gz")
    data = load_replay(path)
    jsonschema.validate(data, json.loads(SCHEMA_PATH.read_text()))
    arrays = frames_as_arrays(data)
    assert arrays["pos"].shape == (5, 3)
    assert arrays["phase"] == ["a"] * 5
    assert data["meta"]["outcome"]["metrics"]["err"] == pytest.approx(1.2346)


def test_recorder_rejects_ragged_frames():
    rec = Recorder({"title": "t"})
    rec.record({"t": 0.0, "pos": [0, 0, 0]})
    with pytest.raises(ValueError):
        rec.record({"t": 0.1})


def test_sim_replay_validates_against_schema(tmp_path, lander):
    jsonschema = pytest.importorskip("jsonschema")
    from plume.config import WorldSpec
    from plume.physics.sim import RocketSim

    sim = RocketSim(lander, WorldSpec())
    sim.reset(pos=(0, 0, 50))
    rec = Recorder(sim.replay_meta("unit"))
    for _ in range(20):
        sim.step(10)
        rec.record(sim.frame("fall"))
    data = rec.to_dict()
    jsonschema.validate(json.loads(json.dumps(data)), json.loads(SCHEMA_PATH.read_text()))


# --------------------------------------------------------------------------- terrain


def test_heightmap_bilinear_exact_on_planes():
    xs = np.linspace(0, 100, 11)
    ys = np.linspace(-50, 50, 21)
    X, Y = np.meshgrid(xs, ys)
    hm = Heightmap(2.0 + 0.1 * X - 0.05 * Y, 0, 100, -50, 50)
    for x, y in [(0, -50), (37.3, 12.1), (99.9, 49.9), (55, 0)]:
        assert hm.height(x, y) == pytest.approx(2.0 + 0.1 * x - 0.05 * y)
    n = hm.normal(40.0, 10.0)
    expected = np.array([-0.1, 0.05, 1.0])
    np.testing.assert_allclose(n, expected / np.linalg.norm(expected), atol=1e-9)
    assert hm.slope_deg(40, 10) == pytest.approx(math.degrees(math.atan(math.hypot(0.1, 0.05))))


@pytest.mark.parametrize("fmt", ["png", "npy"])
def test_heightmap_save_load(tmp_path, fmt):
    hm = generate_regional_map((0, 50_000), (0, 30_000), 1000.0, seed=1)
    path = hm.save(tmp_path / "region.yaml", fmt=fmt)
    back = Heightmap.load(path)
    assert back.shape == hm.shape
    tol = (hm.heights.max() - hm.heights.min()) / 65535 * 1.01 if fmt == "png" else 1e-3
    np.testing.assert_allclose(back.heights, hm.heights, atol=tol)
    assert back.height(12_345, 6_789) == pytest.approx(hm.height(12_345, 6_789), abs=tol * 2)


def test_flatten_sites_and_detail_tile():
    hm = generate_regional_map((0, 100_000), (0, 100_000), 1000.0, seed=2)
    flat = flatten_sites(hm, [(50_000, 50_000)], radius=4000, blend=4000)
    assert flat.slope_deg(50_000, 50_000) < 0.5
    tile = generate_detail_tile(flat, (50_000, 50_000), 500.0, 5.0, seed=3)
    assert tile.shape == (201, 201)
    assert abs(tile.height(50_000, 50_000) - flat.height(50_000, 50_000)) < 6.0
