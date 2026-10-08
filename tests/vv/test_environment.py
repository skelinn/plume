"""Verification of the environment models: US76 (0-1000 km), NRLMSISE-00, soundings,
and MIL-F-8785C continuous turbulence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import WindSpec, WorldSpec
from plume.physics.atmosphere import (
    _UPPER,
    Atmosphere,
    SoundingAtmosphere,
    read_sounding,
)
from plume.physics.turbulence import FT, KT, MilTurbulence, mil_parameters
from plume.physics.wind import CompositeWind, TableWind, wind_from_world

pytestmark = pytest.mark.vv


# U.S. Standard Atmosphere 1976, Table I (geometric altitude m: T K, p Pa, rho kg/m^3)
US76_REF = {
    0.0: (288.15, 101325.0, 1.2250),
    10_000.0: (223.25, 26500.0, 0.41351),
    20_000.0: (216.65, 5529.3, 0.088910),
    50_000.0: (270.65, 79.779, 1.0269e-3),
    80_000.0: (198.64, 1.0524, 1.8458e-5),
}


@pytest.mark.parametrize("z", sorted(US76_REF))
def test_us76_lower_matches_published_table(z):
    t, p, rho = US76_REF[z]
    s = Atmosphere().at(z)
    assert s.temperature == pytest.approx(t, abs=0.02)
    assert s.pressure == pytest.approx(p, rel=2e-4)
    assert s.density == pytest.approx(rho, rel=3e-4)


def test_us76_upper_table_continuity_and_monotonicity():
    atm = Atmosphere()
    for z_km, t, p, rho in _UPPER:
        s = atm.at(z_km * 1000.0)  # 86 km itself comes from the 7-layer model
        tol = 2e-4 if z_km == 86.0 else 1e-9
        assert s.pressure == pytest.approx(p, rel=tol)
        assert s.density == pytest.approx(rho, rel=tol)
        assert s.temperature == pytest.approx(t, abs=0.1 if z_km == 86.0 else 1e-6)
    # the 7-layer model and the table agree at 86 km
    below, above = atm.at(85_999.0), atm.at(86_001.0)
    assert above.density == pytest.approx(below.density, rel=5e-3)
    assert above.pressure == pytest.approx(below.pressure, rel=5e-3)
    zs = np.linspace(0, 1.2e6, 2000)
    rho = np.array([atm.density(z) for z in zs])
    assert np.all(np.diff(rho) < 0)
    # between table points (published 250 km value, not in the interpolation table)
    assert atm.density(250e3) == pytest.approx(6.073e-11, rel=0.03)


def test_nrlmsise00_reference_case_and_wrapper():
    """NRLMSISE-00 distribution test case 1 (doy 172, 29000 s, 400 km, 60N 70W, lst 16,
    F10.7 = F10.7a = 150, Ap = 4): total mass density 4.074714e-15 g/cm^3."""
    pytest.importorskip("nrlmsise00")
    from datetime import datetime, timedelta

    from nrlmsise00 import msise_model

    from plume.physics.atmosphere import MsisAtmosphere

    t0 = datetime(2000, 1, 1) + timedelta(days=171, seconds=29000)
    d, t = msise_model(t0, 400.0, 60.0, -70.0, 150.0, 150.0, 4.0, lst=16.0)
    assert d[5] == pytest.approx(4.074714e-15, rel=1e-5)
    assert t[0] == pytest.approx(1250.54, abs=0.01)

    atm = MsisAtmosphere(t0, lat_deg=33.0, lon_deg=-107.0)
    s0 = atm.at(0.0)
    assert s0.density == pytest.approx(1.2, rel=0.1)  # near sea-level standard
    ratio = atm.density(300e3) / Atmosphere().density(300e3)
    assert 0.5 < ratio < 5.0  # moderate solar activity vs the US76 mean atmosphere
    # profile interpolation agrees with direct evaluation
    for z in (5e3, 80e3, 150e3, 600e3):
        direct = atm.point(z, 33.0, -107.0).density
        assert atm.density(z) == pytest.approx(direct, rel=0.02)


def test_sounding_reproduces_us76_and_reads_radiosonde(tmp_path):
    std = Atmosphere()
    zs = np.arange(1000.0, 30_001.0, 250.0)
    snd = {"alt": zs, "temperature": np.array([std.at(z).temperature for z in zs])}
    snd["pressure"] = np.full(len(zs), math.nan)
    snd["pressure"][0] = std.at(zs[0]).pressure
    atm = SoundingAtmosphere(snd)
    for z in (5e3, 15e3, 29e3):
        assert atm.at(z).pressure == pytest.approx(std.at(z).pressure, rel=2e-3)
    # continuity at the top of the blended extension
    top = atm._top
    assert atm.density(top + 1) == pytest.approx(atm.density(top - 1), rel=1e-3)

    # University of Wyoming style columns: PRES hPa, HGHT m, TEMP C, DRCT deg, SKNT kt
    csv = tmp_path / "sonde.csv"
    csv.write_text(
        "PRES,HGHT,TEMP,DRCT,SKNT\n"
        "850,1400,20.0,270,10\n"
        "700,3000,8.0,270,20\n"
        "500,5600,-10.0,180,30\n"
    )
    r = read_sounding(csv)
    np.testing.assert_allclose(r["east"][:2], [10 * KT, 20 * KT], rtol=1e-6)  # from W -> blows E
    np.testing.assert_allclose(r["north"][2], 30 * KT, rtol=1e-6)  # from S -> blows N
    assert r["pressure"][0] == pytest.approx(85000.0)
    assert r["temperature"][1] == pytest.approx(281.15)


def test_wind_profile_import_drives_the_sim_wind(tmp_path):
    csv = tmp_path / "wind.csv"
    csv.write_text("altitude_m,speed_mps,from_deg\n0,5,270\n1000,15,270\n10000,40,225\n")
    world = WorldSpec(wind=WindSpec(profile=str(csv)))
    w = wind_from_world(world)
    assert isinstance(w, TableWind)
    np.testing.assert_allclose(w.at(500.0), [10.0, 0.0, 0.0], atol=1e-9)
    s = 40 / math.sqrt(2)
    np.testing.assert_allclose(w.at(10000.0), [s, s, 0.0], atol=1e-9)


def test_mil_low_altitude_parameters():
    (su, _sv, sw), (lu, _lv, lw) = mil_parameters(20 * FT, "light")
    assert sw == pytest.approx(0.1 * 15 * KT, rel=1e-9)  # sigma_w = 0.1 W20
    k = 0.177 + 0.000823 * 20
    assert su == pytest.approx(sw / k**0.4, rel=1e-9)
    assert lw == pytest.approx(20 * FT, rel=1e-9)
    assert lu == pytest.approx(20 * FT / k**1.2, rel=1e-9)
    # continuous through the 1000-2000 ft transition, isotropic above
    a = mil_parameters(999 * FT, "moderate")
    b = mil_parameters(1001 * FT, "moderate")
    assert np.allclose(a[0], b[0], rtol=0.01) and np.allclose(a[1], b[1], rtol=0.01)
    hi = mil_parameters(10_000.0, "severe")
    assert hi[0][0] == hi[0][1] == hi[0][2] and hi[1][0] == pytest.approx(1750 * FT)


@pytest.mark.parametrize("model", ["dryden", "von_karman"])
def test_mil_turbulence_statistics(model):
    """Synthesised field has the specified variance in every component, and the Dryden
    longitudinal autocorrelation at one scale length is exp(-1)."""
    h = 5000.0  # high-altitude regime: isotropic, L = 1750 ft (Dryden) / 2500 ft (vK)
    (sig, _, _), (L, _, _) = mil_parameters(h, "severe", model)
    samples, ac = [], []
    for seed in range(12):
        tb = MilTurbulence(model, "severe", seed=seed)
        dt, v = 0.5, np.array([100.0, 0.0, 0.0])
        n = 4000
        xs = np.array([tb.advance(dt, h, v, np.zeros(3)) for _ in range(n)])
        samples.append(xs)
        lag = round(L / (100.0 * dt))
        u = xs[:, 0]
        ac.append(np.mean(u[:-lag] * u[lag:]))
    allx = np.concatenate(samples)
    np.testing.assert_allclose(allx.std(axis=0), sig, rtol=0.12)
    rho = np.mean(ac) / sig**2
    if model == "dryden":
        assert rho == pytest.approx(math.exp(-1), abs=0.08)
    else:
        assert 0.25 < rho < 0.6  # von Karman decorrelates similarly at one scale length


def test_composite_wind_in_sim_world():
    world = WorldSpec(wind=WindSpec(speed=8.0, from_deg=270, turbulence_severity="moderate"))
    w = wind_from_world(world, seed=3)
    assert isinstance(w, CompositeWind)
    vals = []
    for _ in range(3000):
        w.step(0.01, altitude=30.0, agl=30.0, v_ground=np.zeros(3))
        vals.append(w.at(30.0))
    vals = np.array(vals)
    assert vals[:, 0].mean() == pytest.approx(w.mean_at(30.0)[0], abs=2.5)
    assert vals.std(axis=0).max() > 0.3  # turbulence present while hovering in wind
