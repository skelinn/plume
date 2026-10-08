"""Parachute recovery: deployment logic shared by the 6-DOF and 3-DOF simulators."""

from __future__ import annotations

from plume.config import RecoverySpec


class Recovery:
    """Tracks launch/apogee and returns the deployed parachute drag area (Cd*A, m^2)."""

    def __init__(self, spec: RecoverySpec):
        self.chutes = list(spec.chutes)
        self.reset()

    @property
    def enabled(self) -> bool:
        return bool(self.chutes)

    def reset(self) -> None:
        self.launched = False
        self.apogee_t: float | None = None
        self.trigger_t: list[float | None] = [None] * len(self.chutes)

    def update(self, t: float, height: float, vertical_speed: float) -> None:
        """``height`` above the launch point, ``vertical_speed`` positive up."""
        if not self.chutes:
            return
        if not self.launched and height > 5.0 and vertical_speed > 1.0:
            self.launched = True
        if self.launched and self.apogee_t is None and vertical_speed < 0.0:
            self.apogee_t = t
        if self.apogee_t is None:
            return
        for k, c in enumerate(self.chutes):
            if self.trigger_t[k] is not None:
                continue
            if c.deploy == "apogee":
                self.trigger_t[k] = self.apogee_t + c.delay
            elif c.altitude is not None and height <= c.altitude:
                self.trigger_t[k] = t + c.delay

    def cd_area(self, t: float) -> float:
        total = 0.0
        for c, t0 in zip(self.chutes, self.trigger_t, strict=True):
            if t0 is None or t < t0:
                continue
            frac = 1.0 if c.inflation_time <= 0 else min((t - t0) / c.inflation_time, 1.0)
            total += c.cd_area * frac
        return total
