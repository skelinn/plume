"""Monte Carlo machinery: sampling, statistics, report."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.analysis.montecarlo import (
    DispersionSpec,
    ParamDispersion,
    apply_sample,
    dispersion_meta,
    error_ellipse,
    load_dispersion,
    sample_run,
    spearman,
    summarize,
    wilson_interval,
    write_report,
)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0370, abs=5e-4)  # "rule of three" ~ 3/n
    lo, hi = wilson_interval(95, 100)
    assert lo == pytest.approx(0.8882, abs=5e-4) and hi == pytest.approx(0.9785, abs=5e-4)


def test_error_ellipse_recovers_covariance():
    rng = np.random.default_rng(0)
    ang = math.radians(30)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pts = (rng.standard_normal((20000, 2)) * [20.0, 5.0]) @ R.T + [3.0, -2.0]
    e = error_ellipse(pts, 0.99)
    k = math.sqrt(-2 * math.log(0.01))
    assert e["semi_axes_m"][0] == pytest.approx(k * 20.0, rel=0.03)
    assert e["semi_axes_m"][1] == pytest.approx(k * 5.0, rel=0.03)
    assert math.cos(e["angle_rad"] - ang) ** 2 == pytest.approx(1.0, abs=1e-3)
    inside = np.mean(
        ((pts - e["center"]) @ R[:, 0] / e["semi_axes_m"][0]) ** 2
        + ((pts - e["center"]) @ R[:, 1] / e["semi_axes_m"][1]) ** 2
        <= 1
    )
    assert inside == pytest.approx(0.99, abs=0.005)


def test_spearman():
    x = np.arange(50.0)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(x, np.zeros(50)) == 0.0


def test_sampling_is_reproducible_and_applies_paths():
    ds = DispersionSpec(
        name="t",
        mission="demo_hop",
        seed=4,
        params={
            "vehicle.engine.thrust_vac": ParamDispersion(sigma=0.1, relative=True),
            "vehicle.engine.misalignment_deg": ParamDispersion(sigma=0.2),
            "world.wind.speed": ParamDispersion(dist="uniform", low=1.0, high=2.0),
            "vehicle.tanks.0.capacity": ParamDispersion(sigma=0.01, relative=True),
            "world.wind.turbulence_severity": ParamDispersion(dist="choice", values=["light"]),
        },
    )
    mission = {"world": {"wind": {"speed": 5.0, "turbulence_severity": "none"}}}
    vehicle = {"engine": {"thrust_vac": 1000.0, "misalignment_deg": [0.0, 0.0]}, "tanks": [{"capacity": 10.0}]}
    a = sample_run(ds, 7, mission, vehicle)
    assert a == sample_run(ds, 7, mission, vehicle)
    assert a != sample_run(ds, 8, mission, vehicle)
    m, v = apply_sample(mission, vehicle, a)
    assert 1.0 <= m["world"]["wind"]["speed"] <= 2.0
    assert m["world"]["wind"]["turbulence_severity"] == "light"
    assert len(v["engine"]["misalignment_deg"]) == 2
    assert v["tanks"][0]["capacity"] != 10.0
    assert vehicle["engine"]["thrust_vac"] == 1000.0  # inputs untouched


def test_presets_load():
    for name in ("demo_hop", "real_hop"):
        ds = load_dispersion(name)
        assert ds.params and ds.runs > 0


def _fake_records(n=60, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        thrust = float(rng.normal(1.0, 0.02))
        e, nth = rng.normal(0, 5) + 1000 * (thrust - 1), rng.normal(0, 5)
        ok = abs(e) < 25
        recs.append(
            {
                "run": i,
                "values": {"vehicle.engine.thrust_vac": thrust, "world.wind.speed": float(rng.uniform(0, 9))},
                "result": {
                    "success": ok,
                    "reason": "landed" if ok else "missed_target",
                    "landing_error_m": math.hypot(e, nth),
                    "landing_east_m": e,
                    "landing_north_m": nth,
                    "touchdown_vz_mps": 1.0,
                    "fuel_remaining_kg": 100.0,
                    "max_cargo_g": 5.0,
                    "flight_time_s": 700.0,
                    "touchdown_vh_mps": 0.1,
                },
                "error": None,
            }
        )
    return recs


def test_summary_sensitivity_and_report(tmp_path):
    ds = DispersionSpec(name="fake", mission="demo_hop")
    recs = _fake_records()
    s = summarize(recs, ds, target_radius=50.0)
    assert s["runs"] == 60 and 0 < s["success_probability"] < 1
    assert s["sensitivity"][0]["param"] == "vehicle.engine.thrust_vac"  # the planted driver
    assert s["landing"]["cep50_m"] > 0
    meta = dispersion_meta(s)
    assert meta["runs"] == 60 and len(meta["center"]) == 3
    html = write_report(s, recs, tmp_path / "r.html").read_text()
    assert "Monte Carlo: fake" in html and "<svg" in html
