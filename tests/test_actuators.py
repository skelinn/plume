"""High-fidelity actuator models: gimbal actuator, ignition delay, RCS PWM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import load_vehicle
from plume.physics.propulsion import RCS, Engine


@pytest.fixture(scope="module")
def hopper():
    return load_vehicle("cargo_hopper")


def test_gimbal_actuator_delay_rate_limit_and_settling(hopper):
    e = Engine(hopper.engine, hopper.prop_capacity, high_fidelity=True)
    dt = 0.005
    e.command(0.0, (0.5, 0.0))
    out = []
    for _ in range(200):
        e.update(dt, 101325.0, 1000.0)
        out.append(math.degrees(e.gimbal[0]))
    target = 0.5 * hopper.engine.gimbal_max_deg
    delay_steps = math.ceil(hopper.engine.gimbal_delay_s / dt)
    assert all(abs(x) < 1e-12 for x in out[: delay_steps - 1])  # transport delay
    rate = np.diff(out) / dt
    assert rate.max() <= hopper.engine.gimbal_rate_deg_s + 1e-6  # rate limit
    half_bl = 0.5 * hopper.engine.gimbal_backlash_deg
    assert out[-1] == pytest.approx(target, abs=half_bl + 1e-6)  # settles within free play
    assert max(out) < target * 1.05  # well damped


def test_ignition_delay(hopper):
    e = Engine(hopper.engine, hopper.prop_capacity, high_fidelity=True)
    e.command(1.0, (0, 0))
    thrust = [e.update(0.005, 101325.0, 1000.0)[0] for _ in range(200)]
    t_first = next(i for i, x in enumerate(thrust) if x > 0) * 0.005
    assert t_first == pytest.approx(hopper.engine.ignition_delay_s, abs=0.011)
    fast = Engine(hopper.engine, hopper.prop_capacity)
    fast.command(1.0, (0, 0))
    fast_thrust = [fast.update(0.005, 101325.0, 1000.0)[0] for _ in range(20)]
    assert next(i for i, x in enumerate(fast_thrust) if x > 0) * 0.005 < 0.05  # no delay


def test_rcs_pwm_impulse_and_minimum_impulse_bit(hopper):
    dt = 0.005
    hi = RCS(hopper.rcs, hopper, 4.0, high_fidelity=True)
    lo = RCS(hopper.rcs, hopper, 4.0)
    for r in (hi, lo):
        r.command((0.3, 0.0, 0.0), 4.0)
    imp_hi = sum(hi.update(4.0, 10.0, dt)[1][0] * dt for _ in range(10))
    imp_lo = sum(lo.update(4.0, 10.0, dt)[1][0] * dt for _ in range(10))
    assert imp_hi == pytest.approx(imp_lo, rel=0.15)  # same impulse per PWM frame
    hi.command((0.02, 0.0, 0.0), 4.0)  # below the minimum on-time
    assert not hi.duty.any()
