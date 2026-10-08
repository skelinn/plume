"""Continuous atmospheric turbulence per MIL-F-8785C / MIL-HDBK-1797.

The turbulence is a frozen spatial random field (Taylor's hypothesis) sampled along the
air-relative path of the vehicle: the path length ``s`` advances by the airspeed relative
to the mean wind, so a hovering vehicle in a breeze still sees turbulence convected past it
at the wind speed. Each component is synthesised as a sum of ``n_modes`` cosines with
random phases whose amplitudes follow the exact Dryden or von Karman spectrum

    Dryden      Phi_u = s_u^2 (2 L_u / pi) / (1 + (L_u W)^2)
                Phi_v = s_v^2 (L_v / pi) (1 + 3 (L_v W)^2) / (1 + (L_v W)^2)^2
    von Karman  Phi_u = s_u^2 (2 L_u / pi) / (1 + (1.339 L_u W)^2)^(5/6)
                Phi_v = s_v^2 (L_v / pi) (1 + 8/3 (1.339 L_v W)^2) / (1 + (1.339 L_v W)^2)^(11/6)

(W spatial frequency, rad/m; the w component uses the lateral form with L_w, s_w), each
normalised so that its variance is exactly s^2. Intensities and scale lengths depend on
height above ground:

* below 1000 ft (low altitude): L_w = h, L_u = L_v = h / (0.177 + 0.000823 h)^1.2 (ft),
  s_w = 0.1 W20, s_u = s_v = s_w / (0.177 + 0.000823 h)^0.4, with W20 the wind at 20 ft
  (light 15 kt, moderate 30 kt, severe 45 kt);
* above 2000 ft: isotropic, L = 1750 ft (Dryden) / 2500 ft (von Karman), s from the
  probability-of-exceedance table (light 1e-2, moderate 1e-3, severe 1e-5);
* linear interpolation between 1000 and 2000 ft.

Low-altitude axes: u along the mean wind (or the horizontal air path if calm), v
horizontal and perpendicular, w vertical. Above 2000 ft the field is isotropic, so the
axis choice does not matter statistically.
"""

from __future__ import annotations

import math

import numpy as np

FT = 0.3048
KT = 0.514444

# MIL-F-8785C fig. 7: RMS turbulence (ft/s) vs altitude (ft) for probability of exceedance
_HI_ALT_FT = np.array(
    [500, 1750, 3750, 7500, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 80000]
)
_HI_SIGMA_FTPS = {
    "light": np.array([6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0, 0, 0, 0, 0]),  # 1e-2
    "moderate": np.array([8.6, 9.6, 10.6, 10.1, 8.0, 6.6, 5.0, 4.2, 2.7, 0, 0, 0]),  # 1e-3
    "severe": np.array(
        [15.6, 17.6, 23.0, 23.6, 22.1, 20.0, 16.0, 15.1, 12.1, 7.9, 6.2, 5.1]
    ),  # 1e-5
}
_W20_KT = {"light": 15.0, "moderate": 30.0, "severe": 45.0}


def mil_parameters(h_agl: float, severity: str, model: str = "dryden"):
    """((s_u, s_v, s_w) m/s, (L_u, L_v, L_w) m) at height above ground ``h_agl`` (m)."""
    h = max(h_agl / FT, 10.0)  # ft; the low-altitude model is undefined at the ground

    def low(hf):
        k = 0.177 + 0.000823 * hf
        sw = 0.1 * _W20_KT[severity] * KT / FT  # ft/s
        su = sw / k**0.4
        lu = hf / k**1.2
        return (su, su, sw), (lu, lu, hf)

    def high(hf):
        s = float(np.interp(hf, _HI_ALT_FT, _HI_SIGMA_FTPS[severity], right=0.0))
        L = 2500.0 if model == "von_karman" else 1750.0
        return (s, s, s), (L, L, L)

    if h <= 1000.0:
        sig, L = low(h)
    elif h >= 2000.0:
        sig, L = high(h)
    else:
        f = (h - 1000.0) / 1000.0
        s0, l0 = low(1000.0)
        s1, l1 = high(2000.0)
        sig = tuple((1 - f) * a + f * b for a, b in zip(s0, s1, strict=True))
        L = tuple((1 - f) * a + f * b for a, b in zip(l0, l1, strict=True))
    return tuple(x * FT for x in sig), tuple(x * FT for x in L)


def spectrum(omega: np.ndarray, sigma: float, L: float, model: str, longitudinal: bool):
    """One-sided spatial PSD (m^2/s^2 per rad/m); integrates to sigma^2 over [0, inf)."""
    if model == "von_karman":
        x = (1.339 * L * omega) ** 2
        if longitudinal:
            return sigma**2 * (2 * L / math.pi) / (1 + x) ** (5 / 6)
        return sigma**2 * (L / math.pi) * (1 + 8 / 3 * x) / (1 + x) ** (11 / 6)
    x = (L * omega) ** 2
    if longitudinal:
        return sigma**2 * (2 * L / math.pi) / (1 + x)
    return sigma**2 * (L / math.pi) * (1 + 3 * x) / (1 + x) ** 2


class MilTurbulence:
    """Frozen-field MIL-spec turbulence (see the module docstring)."""

    def __init__(
        self,
        model: str = "von_karman",
        severity: str = "light",
        seed: int | None = None,
        n_modes: int = 160,
        omega_min: float = 1e-5,
        omega_max: float = 4.0,
    ):
        if model not in ("dryden", "von_karman"):
            raise ValueError(f"unknown turbulence model {model!r}")
        if severity not in _W20_KT:
            raise ValueError(f"unknown turbulence severity {severity!r}")
        self.model, self.severity = model, severity
        edges = np.geomspace(omega_min, omega_max, n_modes + 1)
        self._om = np.sqrt(edges[:-1] * edges[1:])
        self._dom = np.diff(edges)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._phase = self.rng.uniform(0, 2 * math.pi, (3, len(self._om)))
        self.s = 0.0
        self.value = np.zeros(3)  # m/s, local ENU
        self._axis = np.array([1.0, 0.0])

    def component(self, k: int, sigma: float, L: float) -> float:
        if sigma <= 0.0:
            return 0.0
        phi = spectrum(self._om, sigma, L, self.model, longitudinal=(k == 0))
        a2 = 2.0 * phi * self._dom
        tot = a2.sum()
        if tot <= 0:
            return 0.0
        a = np.sqrt(a2 * (2.0 * sigma**2 / tot))  # exact variance sigma^2 = sum(a^2) / 2
        return float(a @ np.cos(self._om * self.s + self._phase[k]))

    def advance(
        self, dt: float, h_agl: float, v_air_rel: np.ndarray, mean_wind: np.ndarray
    ) -> np.ndarray:
        """Advance the path by ``|v_air_rel| dt`` (airspeed relative to the mean wind, local
        ENU) and return the turbulence velocity (local ENU, m/s) at the new point."""
        self.s += float(np.linalg.norm(v_air_rel)) * dt
        horiz = mean_wind[:2] if np.hypot(*mean_wind[:2]) > 0.5 else v_air_rel[:2]
        n = math.hypot(*horiz)
        if n > 1e-6:
            self._axis = np.asarray(horiz, dtype=float) / n
        sig, L = mil_parameters(h_agl, self.severity, self.model)
        u = self.component(0, sig[0], L[0])
        v = self.component(1, sig[1], L[1])
        w = self.component(2, sig[2], L[2])
        ax = self._axis
        self.value = np.array([u * ax[0] - v * ax[1], u * ax[1] + v * ax[0], w])
        return self.value
