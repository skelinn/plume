"""Hop-rig system identification: test inputs, synthetic rig log, parameter recovery."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import WorldSpec, load_vehicle
from plume.hoprig.sysid import (
    MANOEUVRES,
    RIG_LOG_COLUMNS,
    _first_order,
    _second_order,
    calibrate_rig,
    chirp,
    doublet,
    fly_sysid,
    lagged_throttle,
    multistep_3211,
    read_rig_log,
)


def test_input_signals():
    assert doublet(0.5, 0.0, 1.0, 0.1) == 0.1 and doublet(1.5, 0.0, 1.0, 0.1) == -0.1
    assert doublet(2.5, 0.0, 1.0, 0.1) == 0.0
    vals = [multistep_3211(t, 0.0, 1.0, 1.0) for t in (0.5, 3.5, 5.5, 6.5, 7.5)]
    assert vals == [1.0, -1.0, 1.0, -1.0, 0.0]
    ts = np.linspace(0, 10, 10001)
    c = np.array([chirp(t, 0.0, 10.0, 0.5, 4.0, 2.0) for t in ts])
    assert abs(c).max() <= 2.0 + 1e-9 and c[0] == 0.0 and abs(c[-1]) < 1e-6
    # instantaneous frequency rises: more zero crossings in the second half
    z = np.nonzero(np.diff(np.sign(c)))[0]
    assert (z > 5000).sum() > 2 * (z < 5000).sum()


def test_actuator_models_match_their_definitions():
    t = np.arange(0, 2, 0.01)
    cmd = np.where(t >= 0.5, 1.0, 0.0)
    g = _first_order(t, cmd, 0.1, 1e3, h=1e-4)
    k = 60  # t = 0.6 s: one time constant after cmd[50] (t = 0.5 s) took effect
    assert g[k] == pytest.approx(1 - math.exp(-1), abs=0.02)
    g2 = _second_order(t, cmd, 2 * math.pi * 3.0, 0.7, 0.05)
    assert g2[np.searchsorted(t, 0.55)] == pytest.approx(0.0, abs=1e-6)  # still delayed
    assert g2[-1] == pytest.approx(1.0, abs=1e-3)
    u = lagged_throttle(t, np.full_like(t, 0.8), 0.2, 0.3, 1.0)
    assert u[np.searchsorted(t, 0.2)] == pytest.approx(0.8 * (1 - math.exp(-1)), abs=0.01)


def test_plan_covers_the_rig_model():
    text = " ".join(m.excites for m in MANOEUVRES)
    for key in ("thrust_vac", "throttle_tau", "gimbal_wn_hz", "dry_inertia", "rcs.thrust"):
        assert key in text, key


@pytest.fixture(scope="module")
def truth_and_log():
    nominal = load_vehicle("hop_rig")
    truth = nominal.model_copy(deep=True)
    truth.engine.thrust_vac = 3800.0
    truth.engine.isp_vac = 220.0
    truth.engine.isp_sl = 203.0
    truth.engine.throttle_tau = 0.22
    truth.engine.gimbal_tau = 0.06
    truth.mass.dry_inertia = [110.0, 104.0, 9.0]
    rec, df = fly_sysid(truth, WorldSpec(), seed=3)
    return nominal, truth, rec, df


def test_sysid_flight_and_log(truth_and_log, tmp_path):
    _, _, rec, df = truth_and_log
    assert rec.meta["outcome"]["success"], rec.meta["outcome"]
    assert list(df.columns) == list(RIG_LOG_COLUMNS)
    labels = set(df["manoeuvre"])
    assert {"throttle_doublet", "throttle_3211", "gimbal_chirp_x", "gimbal_chirp_y"} <= labels
    assert df["tether_tension_n"].max() < 100.0  # the rope stayed (nearly) slack
    p = tmp_path / "rig.csv"
    df.to_csv(p, index=False)
    assert len(read_rig_log(p)) == len(df)
    df.drop(columns=["prop_mass_kg"]).to_csv(p, index=False)
    with pytest.raises(KeyError, match="prop_mass_kg"):
        read_rig_log(p)


def test_calibration_recovers_truth(truth_and_log):
    nominal, truth, _, df = truth_and_log
    cal = calibrate_rig(df, nominal)
    e, te = cal.engine, truth.engine
    assert e.thrust_vac == pytest.approx(te.thrust_vac, rel=0.015)
    assert e.isp_vac == pytest.approx(te.isp_vac, rel=0.015)
    assert e.isp_sl == pytest.approx(te.isp_sl, rel=0.015)
    assert e.throttle_tau == pytest.approx(te.throttle_tau, rel=0.1)
    for ax in ("x", "y"):
        assert cal.gimbal[ax].tau == pytest.approx(te.gimbal_tau, rel=0.15)
    ix = cal.vehicle.mass.dry_inertia
    assert ix[0] == pytest.approx(110.0, rel=0.05) and ix[1] == pytest.approx(104.0, rel=0.05)
    assert cal.vehicle.engine.thrust_vac == pytest.approx(te.thrust_vac, rel=0.015)
    assert cal.vehicle.name == "hop_rig_calibrated"
    assert any("thrust_vac" in r[0] for r in cal.rows)


def test_cli_rig_plan_sysid_calibrate(tmp_path):
    from typer.testing import CliRunner

    from plume.cli import app
    from plume.config import load_vehicle as lv

    runner = CliRunner()
    res = runner.invoke(app, ["rig", "plan"])
    assert res.exit_code == 0 and "docs/hop_rig.md" in res.output
    log = tmp_path / "rig.csv"
    res = runner.invoke(app, ["rig", "sysid", "--out", str(log), "--seed", "1"])
    assert res.exit_code == 0, res.output
    assert log.exists() and (tmp_path / "rig.plume.json.gz").exists()
    res = runner.invoke(app, ["rig", "calibrate", str(log), "--out-dir", str(tmp_path / "cal")])
    assert res.exit_code == 0, res.output
    cal = lv(tmp_path / "cal" / "hop_rig_rig.yaml")
    assert cal.engine.thrust_vac == pytest.approx(4000.0, rel=0.015)
    # fast-fidelity truth is first order: the second-order parameters stay as they were
    assert cal.engine.gimbal_wn_hz == 6.0
