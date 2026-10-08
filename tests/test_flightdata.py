"""Flight-log import, comparison and sim-to-real calibration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from plume.config import LogMappingSpec, WorldSpec, load_log_mapping, load_vehicle
from plume.constants import G0
from plume.flightdata.calibrate import calibrate, descent_cd_area, impulse_of, nominal_curve
from plume.flightdata.compare import (
    log_to_replay,
    metrics,
    plot_comparison,
    simulate,
    trace_to_replay,
)
from plume.flightdata.importer import kalman_vertical, load_log, to_si
from plume.physics.atmosphere import Atmosphere
from plume.physics.sim import RocketSim, quat_from_axis_angle
from plume.recording.recorder import SCHEMA_PATH

SAMPLE_A = "data/flights/sample_flight.csv"
SAMPLE_B = "data/flights/sample_flight_b.csv"


@pytest.fixture(scope="module")
def truth():
    return json.loads(open("data/flights/sample_truth.json").read())


@pytest.fixture(scope="module")
def log_a():
    return load_log(SAMPLE_A, "generic_altimeter")


@pytest.fixture(scope="module")
def hobby():
    return load_vehicle("hobby_rocket")


def test_unit_conversion():
    assert to_si([1000.0], "ms")[0] == pytest.approx(1.0)
    assert to_si([1.0], "ft")[0] == pytest.approx(0.3048)
    assert to_si([2.0], "g")[0] == pytest.approx(2 * G0)
    assert to_si([180.0], "deg/s")[0] == pytest.approx(math.pi)
    assert to_si([10.0], "m", scale=2.0, offset=1.0)[0] == pytest.approx(21.0)
    with pytest.raises(ValueError, match="unknown unit"):
        to_si([1.0], "furlongs")


def test_import_generic_log(log_a):
    assert log_a.meta["launch_time_raw"] == pytest.approx(12.355, abs=0.03)
    assert 650 < log_a.apogee < 760
    assert abs(log_a.altitude[log_a.t < -0.1]).max() < 3.0  # baro zeroed on the pad
    assert np.allclose(np.diff(log_a.t), 1 / 50, atol=1e-9)
    assert log_a.gyro is not None and log_a.east is not None
    assert 1.2 < log_a.burnout_time < 1.6
    assert 140 < log_a.velocity.max() < 158


def test_both_formats_agree(log_a):
    log_b = load_log(SAMPLE_B, "simple_altimeter")
    assert log_b.gyro is None and log_b.east is None
    # two flights of the same rocket in different wind: similar apogee
    assert log_b.apogee == pytest.approx(log_a.apogee, rel=0.05)
    assert log_b.accel[(log_b.t > -1) & (log_b.t < -0.2)].mean() == pytest.approx(G0, abs=0.5)


def test_altitude_only_launch_detection(tmp_path):
    mapping = LogMappingSpec(
        name="baro_only",
        delimiter=";",
        time={"column": "Time (s)", "unit": "s"},
        altitude={"column": "Altitude (m)", "unit": "m"},
    )
    log = load_log(SAMPLE_B, mapping)
    assert log.accel is None
    assert abs(log.meta["launch_time_raw"]) < 1.0  # within ~0.5 s of the true liftoff
    assert log.velocity.max() > 100


def test_missing_column_is_explained(tmp_path):
    m = load_log_mapping("generic_altimeter").model_copy(deep=True)
    m.altitude.column = "nope"
    with pytest.raises(KeyError, match="nope"):
        load_log(SAMPLE_A, m)


def test_kalman_recovers_velocity():
    t = np.arange(0, 10, 0.02)
    a_true = np.where(t < 2, 30.0, -G0)
    v_true = np.cumsum(a_true) * 0.02
    h_true = np.cumsum(v_true) * 0.02
    rng = np.random.default_rng(0)
    _, v = kalman_vertical(
        t, h_true + rng.normal(0, 1.0, t.size), a_true + G0 + rng.normal(0, 1, t.size)
    )
    assert np.sqrt(np.mean((v - v_true) ** 2)) < 1.0


def test_parachute_descent_rate(hobby):
    tr = simulate(hobby, rail_length=1.5)
    late = (tr.t > tr.t_apogee + 20) & (tr.altitude > 50)
    m = hobby.mass.dry
    rho = Atmosphere().density(float(np.mean(tr.altitude[late])))
    v_expected = math.sqrt(2 * m * G0 / (rho * hobby.recovery.chutes[0].cd_area))
    assert -np.mean(tr.velocity[late]) == pytest.approx(v_expected, rel=0.03)


def test_fins_make_the_6dof_rocket_stable_and_agree_with_3dof(hobby):
    world = WorldSpec(ground="none", dt=0.002)
    sim = RocketSim(hobby, world)
    sim.reset(pos=(0, 0, 0), quat=quat_from_axis_angle([1, 0, 0], math.radians(2)))
    sim.set_rail(1.5)
    sim.step(500)  # 1 s: off the rail, still boosting
    assert sim.rail is None
    apogee = 0.0
    while sim.t < 20:
        sim.step(50)
        apogee = max(apogee, sim.state.altitude)
        if sim.t < 8:
            assert sim.state.tilt < math.radians(10)  # weathercock-stable, no tumbling
    tr = simulate(hobby, rail_length=1.5, rail_tilt_deg=2.0, stop_after_apogee=1.0)
    assert apogee == pytest.approx(tr.apogee, rel=0.03)


def test_calibration_recovers_truth(log_a, hobby, truth):
    res = calibrate(log_a, hobby)
    p = res.params
    assert p["cd_scale"] == pytest.approx(truth["cd_scale"], rel=0.05)
    nominal_impulse = impulse_of(nominal_curve(hobby))
    assert res.total_impulse == pytest.approx(nominal_impulse * truth["impulse_scale"], rel=0.03)
    assert p["time_scale"] == pytest.approx(truth["time_scale"], rel=0.04)
    assert p["chute_cd_area"] == pytest.approx(truth["chute_cd_area"], rel=0.08)
    assert abs(res.after["apogee_error_m"]) < 5.0
    assert res.after["rms_altitude_ascent_m"] < 0.1 * res.before["rms_altitude_ascent_m"]
    # the calibrated vehicle is a valid, loadable spec
    assert res.vehicle.engine.thrust_curve and res.vehicle.engine.motor_file is None


def test_knots_mode_fits_shape(log_a, hobby):
    res = calibrate(log_a, hobby, mode="knots", n_knots=4, fit_chute=False)
    assert res.after["rms_altitude_ascent_m"] < 2.0
    assert any(k.startswith("knot_") for k in res.params)


def test_descent_cd_area_closed_form(log_a, hobby, truth):
    cda = descent_cd_area(log_a, hobby.mass.dry)
    assert cda == pytest.approx(truth["chute_cd_area"], rel=0.08)


def test_replays_and_plot(tmp_path, log_a, hobby):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    real = log_to_replay(log_a, hobby)
    tr = simulate(hobby)
    sim = trace_to_replay(tr, hobby, "sim")
    for rep in (real, sim):
        jsonschema.validate(json.loads(json.dumps(rep)), schema)
    assert real["meta"]["source"] == "real" and sim["meta"]["source"] == "sim"
    png = plot_comparison(log_a, [tr], tmp_path / "overlay.png")
    assert png.stat().st_size > 10_000
    m = metrics(log_a, tr)
    assert m["apogee_sim_m"] > m["apogee_real_m"]  # nominal model over-predicts
