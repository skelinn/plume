"""Hop-test rig: vehicle preset, tethered and free-flight scenarios, CLI."""

from __future__ import annotations

import math

import numpy as np
import pytest
from typer.testing import CliRunner

from plume.cli import app
from plume.config import WindSpec, WorldSpec, load_vehicle
from plume.hoprig.scenarios import (
    free_hop,
    step_metrics,
    tether_catch,
    tethered_hover,
    translation_step,
)
from plume.recording import load_replay


@pytest.fixture(scope="module")
def rig():
    return load_vehicle("hop_rig")


def test_rig_preset_is_hover_capable(rig):
    from plume.physics.massprops import MassModel
    from plume.physics.propulsion import Engine

    mm = MassModel(rig)
    wet = mm.evaluate([t.initial_mass for t in rig.tanks], rig.rcs.gas).mass
    dry = mm.evaluate([0.0], rig.rcs.gas).mass
    e = Engine(rig.engine, rig.prop_capacity)
    assert 100 <= wet <= 300  # small representative rig
    t_sl = e.max_thrust(101325.0)
    assert t_sl / (wet * 9.80665) > 1.5  # climbs when full
    # can still throttle down to hover when nearly empty: F(u_min) < weight
    f_min = rig.engine.throttle_min * rig.engine.thrust_vac - (rig.engine.thrust_vac - t_sl)
    assert f_min < dry * 9.80665
    assert "Placeholder" in rig.description or "representative" in rig.description.lower()


def test_tethered_hover(rig):
    rec = tethered_hover(rig, WorldSpec(), seed=0)
    out = rec.meta["outcome"]
    assert out["success"], out
    m = out["metrics"]
    assert m["max_footpad_agl_m"] < m["tether_limit_agl_m"]  # rope stayed slack
    assert m["max_tether_tension_n"] == 0.0
    assert m["touchdown_speed_mps"] < 1.0
    assert "tether_tension" in rec.frames and rec.meta["tether"]["length"] == 3.0


def test_tether_catches_stuck_throttle(rig):
    rec = tether_catch(rig, WorldSpec(), seed=0)
    out = rec.meta["outcome"]
    assert out["success"], out
    m = out["metrics"]
    assert m["caught_by_tether"]
    assert m["max_tether_tension_n"] > rig.mass.dry * 9.80665  # it really took the load
    # the rope stretches a little but stops the climb near its length
    assert m["max_footpad_agl_m"] < m["tether_limit_agl_m"] + 0.5
    labels = [e["label"] for e in rec.events]
    assert "Throttle stuck at 100 %" in labels and "Fault cleared" in labels


def test_translation_step_response(rig):
    rec = translation_step(rig, WorldSpec(), seed=0)
    out = rec.meta["outcome"]
    assert out["success"], out
    m = out["metrics"]
    assert m["landing_error_m"] < 0.5
    assert 0.5 < m["rise_time_s"] < 10.0 and m["overshoot_pct"] < 25.0
    assert math.isfinite(m["settling_time_s"])


def test_free_hop_50m(rig):
    rec = free_hop(rig, WorldSpec(wind=WindSpec(speed=3.0, turbulence=0.4)), seed=1)
    out = rec.meta["outcome"]
    assert out["success"], out
    m = out["metrics"]
    assert m["max_agl_m"] == pytest.approx(50.0, abs=3.0)
    assert m["landing_error_m"] < 1.5 and m["propellant_left_kg"] > 5.0


def test_step_metrics_on_known_response():
    t = np.linspace(0, 20, 2001)
    y = 1 - np.exp(-t)  # first order, tau = 1 s
    m = step_metrics(t, y, 0.0, 0.0, 1.0)
    assert m["rise_time_s"] == pytest.approx(math.log(9), abs=0.02)
    assert m["overshoot_pct"] == 0.0
    assert m["settling_time_s"] == pytest.approx(-math.log(0.05), abs=0.02)


def test_cli_sim_hop_rig(tmp_path):
    out = tmp_path / "hover.plume.json.gz"
    res = CliRunner().invoke(
        app, ["sim", "hop_rig", "--script", "tethered_hover", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert load_replay(out)["meta"]["outcome"]["success"]
    res = CliRunner().invoke(app, ["sim", "hop_rig", "--script", "nope"])
    assert res.exit_code == 2
