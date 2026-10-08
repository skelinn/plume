"""Landing guidance onto a moving drone-ship deck.

:class:`ShipLandingAutopilot` reuses :class:`LandingAutopilot` unchanged by flying it in a
frame that moves with the deck (a Galilean change of frame):

* the target is the deck landing circle's centre as received over the ship-to-vehicle
  link (latency, noise, extrapolated over the latency);
* the vehicle velocity handed to the autopilot is relative to a *smoothed* deck
  velocity: horizontally the station-keeping drift (low-passed, ``tau_h``; the vehicle
  cannot and should not follow wave-frequency sway), vertically the heave rate
  (``tau_v``), faded in over the last ``heave_track_h`` metres so the final sink rate is
  relative to the deck;
* the wind forecast is shifted by the same velocity, so air-relative velocities - and
  hence the aerodynamic predictions - are unchanged;
* height above ground is already relative to the deck (``RocketSim`` with a ship).

The autopilot cuts the engine at first leg contact as on land; the vehicle then stays on
the pitching deck by footpad friction (and the optional hold-down clamp).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from plume.control.autopilot import LandingAutopilot
from plume.physics.ship import DeckLink


class ShipLandingAutopilot(LandingAutopilot):
    def __init__(
        self,
        sim,
        link: DeckLink,
        control_dt: float = 0.05,
        tau_h: float = 4.0,
        tau_v: float = 0.4,
        heave_track_h: float = 12.0,
        **kw,
    ):
        self.link = link
        self.tau_h = tau_h
        self.tau_v = tau_v
        self.heave_track_h = heave_track_h
        p, _ = link.estimate(sim.t)
        super().__init__(sim, p, control_dt=control_dt, **kw)

    def reset(self) -> None:
        super().reset()
        self.v_deck = None
        self._shift = np.zeros(3)

    def act(self, st):
        p, v = self.link.estimate(st.t)
        k_h = 1.0 - math.exp(-self.dt / self.tau_h)
        k_v = 1.0 - math.exp(-self.dt / self.tau_v)
        if self.v_deck is None:
            self.v_deck = v.copy()
        else:
            self.v_deck[:2] += k_h * (v[:2] - self.v_deck[:2])
            self.v_deck[2] += k_v * (v[2] - self.v_deck[2])
        w = min(max(1.0 - st.agl / self.heave_track_h, 0.0), 1.0)
        self._shift = np.array([self.v_deck[0], self.v_deck[1], w * self.v_deck[2]])
        if self.phase != "landed":
            self.target = p
        rel = dataclasses.replace(st, vel_com=st.vel_com - self._shift)
        return super().act(rel)

    def wind_forecast(self, st):
        return super().wind_forecast(st) - self._shift
