"""End-to-end scripted flights through the CLI and scenario runners."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from plume.cli import app
from plume.config import WindSpec, WorldSpec
from plume.recording import load_replay
from plume.scenarios import drop, hop_test


def test_hop_test_lands_on_pad_b(lander):
    rec = hop_test(lander, WorldSpec(), seed=0)
    out = rec.meta["outcome"]
    assert out["success"], out
    assert out["metrics"]["landing_error_m"] < 1.0
    assert out["metrics"]["touchdown_speed_mps"] < 1.5


def test_hop_test_in_wind(lander):
    world = WorldSpec(wind=WindSpec(speed=8.0, turbulence=1.2, gust_rate=0.1, gust_max=5.0))
    rec = hop_test(lander, world, seed=1)
    assert rec.meta["outcome"]["success"], rec.meta["outcome"]
    assert rec.meta["outcome"]["metrics"]["landing_error_m"] < 5.0


def test_drop_hits_ground(lander):
    rec = drop(lander, WorldSpec(), altitude=300.0)
    assert rec.meta["outcome"]["reason"] == "impact"
    assert rec.frames["alt"][-1] < 1.0


def test_cli_sim_and_info(tmp_path):
    runner = CliRunner()
    out = tmp_path / "hop.plume.json.gz"
    res = runner.invoke(app, ["sim", "lander_small", "--script", "hop_test", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert load_replay(out)["meta"]["outcome"]["success"]
    res = runner.invoke(app, ["info", "lander_small"])
    assert res.exit_code == 0 and "delta-v" in res.output


@pytest.mark.parametrize("bad", ["nope"])
def test_cli_unknown_vehicle(bad):
    res = CliRunner().invoke(app, ["info", bad])
    assert res.exit_code != 0
