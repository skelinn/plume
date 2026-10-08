"""Flight sensors: IMU, GNSS receiver, barometric and radar altimeters.

Each sensor turns the simulator's true state into a measurement with the error sources
of real hardware (see ``SensorsSpec`` and docs/models/navigation.md):

* IMU - gyro and accelerometer: constant turn-on bias, bias instability (first-order
  Gauss-Markov), white noise (angle / velocity random walk), scale-factor error, axis
  misalignment, quantisation and range saturation. Outputs are the averages over the
  sample interval (delta-angle / delta-velocity divided by dt), as real IMUs deliver.
  The gyro senses the *inertial* rate (Earth rate included); the accelerometer senses
  specific force (everything but gravity).
* GNSS - CG position and velocity at a fixed rate, with latency, white noise, a slowly
  wandering position bias (atmospheric / multipath errors) and optional outages.
* Barometric altimeter - static pressure converted to altitude with the *standard*
  atmosphere, so off-standard days and model differences show up as bias, plus noise.
* Radar altimeter - height of the CG above the terrain, range-limited, noise
  proportional to range.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from plume.constants import G0, P0, R_AIR

DEG = math.pi / 180.0
HOUR = 3600.0


def _skew_rot(v: np.ndarray) -> np.ndarray:
    """Small-angle rotation matrix I + [v x]."""
    x, y, z = v
    return np.array([[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]])


@dataclass
class ImuSample:
    gyro: np.ndarray  # rad/s, body frame (inertial rate)
    accel: np.ndarray  # m/s^2, body frame (specific force)


class _Channel:
    """Three-axis inertial channel with the standard error model."""

    def __init__(self, rng, bias0, instability, tau, white, scale, misalign, quantum, rng_max):
        self.rng = rng
        self.white = white  # density: units/sqrt(s)
        self.instability = instability  # 1-sigma of the Gauss-Markov bias
        self.tau = tau
        self.quantum = quantum
        self.rng_max = rng_max
        self.bias0 = bias0 * rng.standard_normal(3)
        self.drift = instability * rng.standard_normal(3)
        self.M = (1.0 + scale * rng.standard_normal(3))[:, None] * _skew_rot(
            misalign * rng.standard_normal(3)
        )

    def measure(self, true: np.ndarray, dt: float) -> np.ndarray:
        a = math.exp(-dt / self.tau)
        self.drift = a * self.drift + self.instability * math.sqrt(
            1 - a * a
        ) * self.rng.standard_normal(3)
        m = (
            self.M @ true
            + self.bias0
            + self.drift
            + self.white / math.sqrt(dt) * self.rng.standard_normal(3)
        )
        m = np.clip(m, -self.rng_max, self.rng_max)
        if self.quantum > 0:
            q = self.quantum / dt  # quantised delta-angle / delta-velocity
            m = np.round(m / q) * q
        return m

    @property
    def bias(self) -> np.ndarray:
        return self.bias0 + self.drift


class Imu:
    def __init__(self, spec, rng: np.random.Generator):
        s = spec
        self.gyro = _Channel(
            rng,
            s.gyro_bias_deg_h * DEG / HOUR,
            s.gyro_bias_instability_deg_h * DEG / HOUR,
            s.bias_correlation_s,
            s.gyro_arw_deg_rt_h * DEG / 60.0,
            s.scale_factor_ppm * 1e-6,
            s.misalignment_mrad * 1e-3,
            s.gyro_quantum_rad,
            s.gyro_range_deg_s * DEG,
        )
        self.accel = _Channel(
            rng,
            s.accel_bias_mg * 1e-3 * G0,
            s.accel_bias_instability_mg * 1e-3 * G0,
            s.bias_correlation_s,
            s.accel_vrw_m_s_rt_h / 60.0,
            s.scale_factor_ppm * 1e-6,
            s.misalignment_mrad * 1e-3,
            s.accel_quantum_m_s,
            s.accel_range_g * G0,
        )

    def measure(
        self, omega_inertial_b: np.ndarray, specific_force_b: np.ndarray, dt: float
    ) -> ImuSample:
        return ImuSample(
            self.gyro.measure(omega_inertial_b, dt), self.accel.measure(specific_force_b, dt)
        )


@dataclass
class GnssFix:
    t: float  # time of validity
    pos: np.ndarray
    vel: np.ndarray
    sigma_pos: np.ndarray  # 1-sigma per world axis (x, y horizontal; z vertical, approx.)
    sigma_vel: float


class Gnss:
    def __init__(self, spec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.period = 1.0 / spec.rate_hz
        self._next = 0.0
        self._bias = (
            np.array([spec.sigma_h_m, spec.sigma_h_m, spec.sigma_v_m])
            * rng.standard_normal(3)
            * 0.7
        )
        self._queue: deque[GnssFix] = deque()

    def step(
        self, t: float, pos: np.ndarray, vel: np.ndarray, altitude: float, dt: float
    ) -> GnssFix | None:
        """Sample at the receiver rate; returns a fix once its latency has elapsed."""
        s = self.spec
        a = math.exp(-dt / s.bias_correlation_s)
        sig = np.array([s.sigma_h_m, s.sigma_h_m, s.sigma_v_m]) * 0.7
        self._bias = a * self._bias + sig * math.sqrt(1 - a * a) * self.rng.standard_normal(3)
        if t + 1e-9 >= self._next:
            self._next = t + self.period
            outage = s.max_altitude_m is not None and altitude > s.max_altitude_m
            if not outage:
                white = np.array([s.sigma_h_m, s.sigma_h_m, s.sigma_v_m]) * 0.7
                p = pos + self._bias + white * self.rng.standard_normal(3)
                v = vel + s.sigma_vel_m_s * self.rng.standard_normal(3)
                self._queue.append(
                    GnssFix(
                        t, p, v, np.array([s.sigma_h_m, s.sigma_h_m, s.sigma_v_m]), s.sigma_vel_m_s
                    )
                )
        if self._queue and t - self._queue[0].t >= s.latency_s - 1e-9:
            return self._queue.popleft()
        return None


def pressure_altitude(p: float) -> float:
    """Altitude (m) of static pressure ``p`` in the standard atmosphere (to 20 km,
    isothermal extension above), as a barometric altimeter computes it."""
    t0, lapse = 288.15, 0.0065
    p11 = 22632.06
    if p >= p11:
        return t0 / lapse * (1.0 - (p / P0) ** (R_AIR * lapse / G0))
    return 11_000.0 + R_AIR * 216.65 / G0 * math.log(p11 / max(p, 1e-6))


class Barometer:
    def __init__(self, spec, rng):
        self.spec = spec
        self.rng = rng
        self.bias = spec.bias_m * rng.standard_normal()

    def measure(self, pressure: float) -> float | None:
        s = self.spec
        if pressure < s.min_pressure_pa:
            return None
        return pressure_altitude(pressure) + self.bias + s.sigma_m * self.rng.standard_normal()


class RadarAltimeter:
    def __init__(self, spec, rng):
        self.spec = spec
        self.rng = rng

    def measure(self, height: float, tilt: float) -> float | None:
        s = self.spec
        if height > s.max_range_m or height < 0 or tilt > s.max_tilt_deg * DEG:
            return None
        sig = s.sigma_m + s.sigma_frac * height
        return height + sig * self.rng.standard_normal()
