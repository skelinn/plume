"""Sensor error models and the INS/GNSS navigation filter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import EarthSpec, GnssSpec, ImuSpec, WorldSpec
from plume.control.navigation import Navigator, navigation_mode
from plume.physics.atmosphere import Atmosphere
from plume.physics.sensors import Gnss, Imu, pressure_altitude
from plume.physics.sim import RocketSim, quat_from_z_axis
from tests.conftest import make_test_vehicle


def test_imu_noise_density_and_bias():
    spec = ImuSpec(
        gyro_bias_instability_deg_h=0.0,
        accel_bias_instability_mg=0.0,
        scale_factor_ppm=0.0,
        misalignment_mrad=0.0,
        gyro_quantum_rad=0.0,
        accel_quantum_m_s=0.0,
    )
    imu = Imu(spec, np.random.default_rng(3))
    dt = 0.01
    g = np.array([imu.gyro.measure(np.zeros(3), dt) for _ in range(20000)])
    arw = spec.gyro_arw_deg_rt_h * math.pi / 180 / 60  # rad/sqrt(s)
    np.testing.assert_allclose(g.std(axis=0), arw / math.sqrt(dt), rtol=0.05)
    np.testing.assert_allclose(
        g.mean(axis=0), imu.gyro.bias0, atol=4 * arw / math.sqrt(dt * len(g))
    )


def test_gnss_rate_and_latency():
    gn = Gnss(GnssSpec(rate_hz=10.0, latency_s=0.1), np.random.default_rng(0))
    fixes = []
    for k in range(200):
        t = k * 0.01
        f = gn.step(t, np.zeros(3), np.zeros(3), 0.0, 0.01)
        if f is not None:
            fixes.append((t, f.t))
    assert 18 <= len(fixes) <= 20  # 10 Hz over 2 s, minus the latency
    assert all(abs((t - tv) - 0.1) < 0.011 for t, tv in fixes)


@pytest.mark.parametrize("z", [0.0, 1500.0, 8000.0, 18000.0])
def test_pressure_altitude_inverts_standard_atmosphere(z):
    p = Atmosphere().pressure(z)
    h_geopot = 6_356_766.0 * z / (6_356_766.0 + z)
    assert pressure_altitude(p) == pytest.approx(h_geopot, abs=1.0)


def test_navigation_mode_defaults():
    assert navigation_mode(WorldSpec()) == "truth"
    assert navigation_mode(WorldSpec(fidelity="high")) == "ekf"
    assert navigation_mode(WorldSpec(fidelity="high", navigation="truth")) == "truth"


def test_navigator_on_pad_and_in_powered_flight():
    """Rotating WGS-84 Earth: on the pad and through a powered climb the estimate stays
    within GNSS-class accuracy and the attitude error at the alignment level."""
    v = make_test_vehicle()
    w = WorldSpec(
        fidelity="high",
        gravity="wgs84",
        ground="plane",
        dt=0.005,
        earth=EarthSpec(origin_lat_deg=33.0, origin_lon_deg=-107.0),
    )
    sim = RocketSim(v, w, seed=2)
    up = sim.gravity.up(np.zeros(3))
    sim.reset(pos=up * (v.legs.height - 0.005), quat=quat_from_z_axis(up), seed=2)
    nav = Navigator(sim, seed=2)
    for _ in range(400):  # 20 s on the pad
        sim.set_controls(0.0, np.zeros(2), np.zeros(3))
        sim.step(10)
        nav.update(0.05)
    for _ in range(200):  # 10 s of thrust
        sim.set_controls(0.9, np.zeros(2), np.zeros(3))
        sim.step(10)
        nav.update(0.05)
    s = nav.summary()
    assert s["pos_err_final_m"] < 3.0
    assert s["vel_err_max_m_s"] < 0.5
    assert s["att_err_max_deg"] < 0.6
    est = nav.estimate()
    assert np.linalg.norm(est.com - sim.state.com) < 3.0
    assert est.mass == sim.state.mass  # sensed directly
