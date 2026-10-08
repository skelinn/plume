"""Mission scoring: landing accuracy, fuel used, cargo g-load and flight time."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MissionResult:
    success: bool
    reason: str
    landing_error_m: float
    fuel_used_kg: float
    fuel_remaining_kg: float
    rcs_used_kg: float
    max_cargo_g: float
    flight_time_s: float
    apogee_km: float
    touchdown_vz_mps: float
    touchdown_vh_mps: float
    final_tilt_deg: float
    ground_slope_deg: float
    score: float

    def summary(self) -> dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "landing error (m)": self.landing_error_m,
            "fuel used (kg)": self.fuel_used_kg,
            "max cargo load (g)": self.max_cargo_g,
            "flight time (s)": self.flight_time_s,
            "apogee (km)": self.apogee_km,
            "score": self.score,
        }


def score_mission(
    sim, mw, spec, failure: str, max_alt: float, prop_initial: float
) -> MissionResult:
    st = sim.state
    td = sim.touchdown if sim.t > 30 else None
    err = mw.miss(st.pos)
    tilt = math.degrees(st.tilt)
    u, v, _ = mw.gravity.map_coords(st.pos)
    slope = 0.0
    for t in mw.tiles:
        if t.contains(u, v):
            slope = t.slope_deg(u, v)
            break
    reason = failure
    if not reason:
        if td is None:
            reason = "no_touchdown"
        elif tilt > 10.0:
            reason = "tipped_over"
        elif st.speed > 0.5:
            reason = "not_at_rest"
        elif err > spec.target_radius:
            reason = "missed_target"
        else:
            reason = "landed"
    success = reason == "landed"
    sc = spec.scoring
    g_lim = spec.guidance.cargo_g_limit
    upright = td is not None and not failure and tilt <= 10.0
    score = 0.0
    if success:
        score += sc.success_points
    if upright:
        score += sc.accuracy_points * max(0.0, 1.0 - err / spec.target_radius)
        score += sc.fuel_points * (sim.prop_mass / max(prop_initial, 1e-9))
    score -= sc.g_penalty_per_g * max(0.0, sim.max_cargo_g - g_lim)
    score -= sc.time_penalty_per_min * sim.t / 60.0
    return MissionResult(
        success=success,
        reason=reason,
        landing_error_m=err,
        fuel_used_kg=sim.prop_used,
        fuel_remaining_kg=sim.prop_mass,
        rcs_used_kg=sim.rcs_used,
        max_cargo_g=sim.max_cargo_g,
        flight_time_s=td.t if td else sim.t,
        apogee_km=max_alt / 1000.0,
        touchdown_vz_mps=td.vertical_speed if td else float("nan"),
        touchdown_vh_mps=td.horizontal_speed if td else float("nan"),
        final_tilt_deg=tilt,
        ground_slope_deg=slope,
        score=round(score, 2),
    )
