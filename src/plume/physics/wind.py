"""Wind: mean wind with power-law shear, first-order (Dryden-like) turbulence and
discrete 1-cosine gusts. Fully determined by the seed."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _Gust:
    t_start: float
    duration: float
    vector: np.ndarray

    def value(self, t: float) -> np.ndarray | None:
        s = (t - self.t_start) / self.duration
        if s < 0.0 or s > 1.0:
            return None
        return self.vector * (0.5 * (1.0 - math.cos(2.0 * math.pi * s)))


class WindModel:
    """Wind velocity field (horizontally uniform, varies with altitude and time).

    Parameters mirror :class:`plume.config.WindSpec`. ``from_deg`` uses the
    meteorological convention (direction the wind blows *from*, clockwise from north).
    """

    def __init__(
        self,
        speed: float = 0.0,
        from_deg: float = 270.0,
        shear_exponent: float = 0.143,
        ref_height: float = 10.0,
        shear_top: float = 2000.0,
        fade_top: float = 30_000.0,
        turbulence: float = 0.0,
        turbulence_tau: float = 1.5,
        gust_rate: float = 0.0,
        gust_max: float = 0.0,
        gust_duration: tuple[float, float] = (1.0, 4.0),
        seed: int | None = None,
    ):
        self.speed = speed
        self.from_deg = from_deg
        self.shear_exponent = shear_exponent
        self.ref_height = ref_height
        self.shear_top = shear_top
        self.fade_top = fade_top
        self.turbulence = turbulence
        self.turbulence_tau = turbulence_tau
        self.gust_rate = gust_rate
        self.gust_max = gust_max
        self.gust_duration = gust_duration
        self.reset(seed)

    @classmethod
    def from_spec(cls, spec, seed: int | None = None) -> WindModel:
        return cls(**spec.model_dump(), seed=seed)

    @property
    def enabled(self) -> bool:
        return self.speed > 0 or self.turbulence > 0 or (self.gust_rate > 0 and self.gust_max > 0)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self._turb = np.zeros(3)
        if self.turbulence > 0:
            self._turb = self.turbulence * self.rng.standard_normal(3) * np.array([1, 1, 0.5])
        self._gusts: list[_Gust] = []
        self._next_gust = self._draw_gust_time(0.0)
        th = math.radians(self.from_deg)
        self._mean_dir = -np.array([math.sin(th), math.cos(th), 0.0])

    def _draw_gust_time(self, t: float) -> float:
        if self.gust_rate <= 0 or self.gust_max <= 0:
            return math.inf
        return t + self.rng.exponential(1.0 / self.gust_rate)

    def profile(self, altitude: float) -> float:
        """Mean-wind multiplier versus altitude (1 at the reference height)."""
        h = min(max(altitude, 0.5), self.shear_top)
        f = (h / self.ref_height) ** self.shear_exponent
        if altitude > self.shear_top:
            fade = 1.0 - (altitude - self.shear_top) / max(self.fade_top - self.shear_top, 1.0)
            f *= max(fade, 0.0)
        return f

    def step(self, dt: float) -> None:
        """Advance turbulence and gust processes by ``dt``."""
        self.t += dt
        if self.turbulence > 0:
            a = math.exp(-dt / self.turbulence_tau)
            b = self.turbulence * math.sqrt(1.0 - a * a)
            self._turb = a * self._turb + b * self.rng.standard_normal(3) * np.array([1, 1, 0.5])
        while self.t >= self._next_gust:
            az = self.rng.uniform(0, 2 * math.pi)
            amp = self.rng.uniform(0.3, 1.0) * self.gust_max
            vec = amp * np.array([math.cos(az), math.sin(az), 0.15 * self.rng.standard_normal()])
            dur = self.rng.uniform(*self.gust_duration)
            self._gusts.append(_Gust(self._next_gust, dur, vec))
            self._next_gust = self._draw_gust_time(self._next_gust)
        self._gusts = [g for g in self._gusts if self.t <= g.t_start + g.duration]

    def mean_at(self, altitude: float) -> np.ndarray:
        """Forecast (mean) wind at an altitude: no turbulence or gusts."""
        if self.speed <= 0:
            return np.zeros(3)
        return self.speed * self.profile(altitude) * self._mean_dir

    def at(self, altitude: float) -> np.ndarray:
        """Wind velocity (m/s, world frame) at the current time and given altitude."""
        if not self.enabled:
            return np.zeros(3)
        f = self.profile(altitude)
        w = self.speed * f * self._mean_dir
        if self.turbulence > 0 or self._gusts:
            decay = 1.0 if altitude < self.shear_top else max(f / self.profile(self.shear_top), 0.0)
            w = w + decay * self._turb
            for g in self._gusts:
                v = g.value(self.t)
                if v is not None:
                    w = w + decay * v
        return w
