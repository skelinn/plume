"""Tests for the table-driven aerodynamic database and its semi-empirical generator."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pytest

from plume.config import VehicleSpec, load_vehicle
from plume.physics.aero import Aero
from plume.physics.aero_gen import base as base_mod
from plume.physics.aero_gen.body import body_cn_cm, crossflow_cd
from plume.physics.aero_gen.friction import cf_karman_schoenherr, skin_friction
from plume.physics.aero_gen.generate import generate_database
from plume.physics.aero_gen.geometry import BodyGeometry
from plume.physics.aero_gen.gridfins import choke_mach, gridfin_cd0, gridfin_cn_alpha, swallow_mach
from plume.physics.aero_gen.heating import HeatingMonitor, heat_load, stagnation_heat_flux
from plume.physics.aero_gen.importers import (
    import_datcom,
    import_generic_csv,
    import_openrocket,
    import_rasaero,
    parse_datcom,
)
from plume.physics.aero_gen.newtonian import cp_max_pitot, newtonian_body
from plume.physics.aero_gen.nose import (
    linnell_bailey,
    supersonic_wave_drag,
    taylor_maccoll_cone,
)
from plume.physics.aero_gen.srp import srp_axial_factor
from plume.physics.aerodb import (
    AeroDatabase,
    AeroModelHiFi,
    GridInterpolator,
    reynolds_per_metre,
    sutherland_viscosity,
)
from plume.physics.massprops import MassModel

REF = Path(__file__).parent / "data" / "aero_ref"
PRESETS = ("hobby_rocket", "lander_small", "cargo_hopper", "cargo_hopper_finless")


def _rows(name: str) -> list[dict]:
    with open(REF / name, encoding="utf-8") as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


@pytest.fixture(scope="module")
def hobby():
    v = load_vehicle("hobby_rocket")
    return v, generate_database(v)


@pytest.fixture(scope="module")
def lander():
    v = load_vehicle("lander_small")
    return v, generate_database(v)


def _cg_range(vehicle) -> list[float]:
    mm = MassModel(vehicle)
    full = np.array([t.initial_mass for t in vehicle.tanks])
    return [mm.evaluate(full * f, vehicle.rcs.gas).cg_z for f in (1.0, 0.5, 0.05)]


# --------------------------------------------------------------------------- atmosphere helpers
def test_sutherland_and_reynolds():
    assert sutherland_viscosity(288.15) == pytest.approx(1.789e-5, rel=2e-3)  # US Std. Atm.
    assert sutherland_viscosity(np.array([216.65]))[0] == pytest.approx(1.422e-5, rel=3e-3)
    assert reynolds_per_metre(1.225, 100.0, 288.15) == pytest.approx(6.85e6, rel=3e-3)


# --------------------------------------------------------------------------- interpolation / IO
def _linear_db() -> AeroDatabase:
    axes = {
        "mach": np.array([0.0, 0.5, 1.0, 2.0, 4.0]),
        "alpha_deg": np.array([0.0, 10.0, 45.0, 90.0, 180.0]),
        "log10_re": np.array([5.0, 6.0, 7.5]),
    }
    M, A, R = np.meshgrid(*axes.values(), indexing="ij")
    coeffs = {
        "CA": 0.3 + 0.2 * M - 0.004 * A + 0.05 * R,
        "CN": 0.05 * A + 0.1 * M,
        "Cm": -0.02 * A + 0.3 * R - 0.1 * M,
    }
    sigma = {"CA": 0.1 * np.abs(coeffs["CA"]), "CN": 0.1 * coeffs["CN"] + 0.01}
    return AeroDatabase(axes, coeffs, sigma, ref_area=0.5, ref_length=0.8, x_ref=1.2,
                        meta={"name": "linear", "body_length": 5.0})  # fmt: skip


def test_interpolation_exact_on_linear_data():
    db = _linear_db()
    rng = np.random.default_rng(1)
    for _ in range(200):
        m, a, r = rng.uniform(0, 4), rng.uniform(0, 180), rng.uniform(5, 7.5)
        c = db.evaluate(m, a, log10_re=r)
        assert c["CA"] == pytest.approx(0.3 + 0.2 * m - 0.004 * a + 0.05 * r, abs=1e-12)
        assert c["CN"] == pytest.approx(0.05 * a + 0.1 * m, abs=1e-12)
        assert c["Cm"] == pytest.approx(-0.02 * a + 0.3 * r - 0.1 * m, abs=1e-12)
    # clamped (no extrapolation) outside the grid
    assert db.evaluate(9.0, 200.0, log10_re=9.0)["CN"] == pytest.approx(0.05 * 180 + 0.1 * 4)


def test_grid_interpolator_collapses_singleton_axes():
    gi = GridInterpolator([np.array([0.0, 1.0]), np.array([3.0])], np.array([[[1.0]], [[3.0]]]))
    assert gi(0.25, 99.0)[0] == pytest.approx(1.5)


def test_save_load_round_trip(tmp_path):
    db = _linear_db()
    path = db.save(tmp_path / "lin")
    assert path.suffix == ".npz" and path.with_suffix(".json").exists()
    back = AeroDatabase.load(path)
    assert back.ref_area == db.ref_area and back.ref_length == db.ref_length
    assert back.x_ref == db.x_ref and back.meta["name"] == "linear"
    for k in db.coeffs:
        np.testing.assert_array_equal(back.coeffs[k], db.coeffs[k])
        np.testing.assert_array_equal(back.sigma[k], db.sigma[k])
    csv_path = db.to_csv(tmp_path / "lin.csv")
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2 + db.coeffs["CA"].size
    # the generic importer reads the export back (standalone, full grid)
    hdr = {"ref_area": db.ref_area, "ref_length": db.ref_length, "x_ref": db.x_ref}
    again = import_generic_csv(csv_path, **hdr)
    assert again.evaluate(1.3, 33.0, log10_re=6.2)["CN"] == pytest.approx(
        db.evaluate(1.3, 33.0, log10_re=6.2)["CN"], rel=1e-6
    )


def test_generated_database_round_trip(tmp_path, hobby):
    _, db = hobby
    back = AeroDatabase.load(db.save(tmp_path / "hobby.npz"))
    assert back.meta["methods"] == db.meta["methods"]
    assert back.evaluate(0.7, 12.0)["Cm"] == pytest.approx(db.evaluate(0.7, 12.0)["Cm"])


# --------------------------------------------------------------------------- runtime physics
def _random_power(model: AeroModelHiFi, rng, n: int, thrust: float = 0.0, cg=(0.0, 1.0)):
    worst = -np.inf
    L = model.length
    for _ in range(n):
        v = rng.normal(size=3) * rng.uniform(0, 1) * rng.choice([5.0, 60.0, 400.0, 1500.0])
        w = rng.normal(size=3) * rng.choice([0.0, 0.3, 3.0])
        if rng.random() < 0.1:
            v[:] = 0.0  # pure rotation
        rho = rng.uniform(1e-3, 1.3)
        a = rng.uniform(220, 340)
        model.thrust = thrust * rng.random()
        cg_z = rng.uniform(*cg) * L
        f, t, _, _ = model.forces(v, w, cg_z, rho, a)
        p = float(f @ v + t @ w)
        scale = rho * model.ref_area * (np.linalg.norm(v) + L * np.linalg.norm(w)) ** 3 + 1e-12
        worst = max(worst, p / scale)
    return worst


@pytest.mark.parametrize("name", ["hobby_rocket", "lander_small"])
def test_dissipative_no_wind(name, hobby, lander):
    """With no wind and no deflection the aerodynamic power F.v + tau.omega <= 0."""
    v, db = hobby if name == "hobby_rocket" else lander
    model = AeroModelHiFi.from_vehicle(v, db)
    rng = np.random.default_rng(7)
    assert _random_power(model, rng, 3000) <= 1e-12
    assert _random_power(model, rng, 1000, thrust=5e4) <= 1e-12  # power-on / SRP too


@pytest.mark.slow
def test_dissipative_monte_carlo_dispersions(hobby):
    v, db = hobby
    rng = np.random.default_rng(3)
    for _ in range(5):
        model = AeroModelHiFi.from_vehicle(v, db.perturbed(rng, scale=3.0))
        assert _random_power(model, rng, 1500) <= 1e-12


def test_axisymmetric_symmetry(hobby):
    v, db = hobby
    m = AeroModelHiFi.from_vehicle(v, db)
    cg = 0.62
    for vel in ([20.0, 0.0, 150.0], [0.0, 35.0, -200.0], [12.0, -7.0, 400.0]):
        vel = np.array(vel)
        f1, t1, _, _ = m.forces(vel, np.zeros(3), cg, 1.0, 330.0)
        mirror = np.array([-vel[0], -vel[1], vel[2]])
        f2, t2, _, _ = m.forces(mirror, np.zeros(3), cg, 1.0, 330.0)
        np.testing.assert_allclose(f2[:2], -f1[:2], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(f2[2], f1[2], rtol=1e-9)
        np.testing.assert_allclose(t2[:2], -t1[:2], rtol=1e-9, atol=1e-9)
        # force lies in the plane of incidence: no side force, no roll moment
        lat = vel[:2] / np.linalg.norm(vel[:2])
        assert abs(f1[0] * lat[1] - f1[1] * lat[0]) < 1e-9 * np.linalg.norm(f1)
        assert abs(t1[2]) < 1e-12
    # CN odd in alpha, zero at 0 and 180 deg
    for mach in (0.3, 1.5, 4.0):
        for a in (3.0, 20.0, 70.0, 130.0):
            assert db.evaluate_signed(mach, -a)["CN"] == pytest.approx(-db.evaluate(mach, a)["CN"])
        assert db.evaluate(mach, 0.0)["CN"] == 0.0
        assert abs(db.evaluate(mach, 180.0)["CN"]) < 1e-12
    assert np.all(db.coeffs["CY"] == 0) and np.all(db.coeffs["Cn"] == 0)


def test_slender_body_limit():
    """CN_alpha -> 2 (Ab/A) per rad as alpha -> 0 for a body alone (Munk)."""
    for shape in ("cone", "tangent_ogive", "von_karman"):
        body = BodyGeometry(10.0, 1.0, 3.0, shape)
        for mach in (0.3, 0.9, 2.0, 3.5):
            cn, _ = body_cn_cm(0.05, mach, 1e6, body, 5.0)
            assert float(cn) / math.radians(0.05) == pytest.approx(2.0, rel=0.01)
    v = load_vehicle("cargo_hopper_finless")
    db = generate_database(v, alpha_grid=(0.0, 0.25, 1.0, 90.0, 180.0))
    assert db.evaluate(0.5, 0.25)["CN"] / math.radians(0.25) == pytest.approx(2.0, rel=0.03)


def test_runtime_matches_tables(hobby):
    """Pure translation: the runtime forces are exactly the tabulated coefficients."""
    v, db = hobby
    m = AeroModelHiFi(db, v.geometry)
    rho, a = 0.9, 320.0
    for alpha in (0.0, 7.0, 35.0, 120.0, 175.0):
        V = 250.0
        ar = math.radians(alpha)
        vel = np.array([V * math.sin(ar), 0.0, V * math.cos(ar)])
        f, t, q, mach = m.forces(vel, np.zeros(3), 0.62, rho, a)
        T = a * a / (1.4 * 287.05287)
        lre = math.log10(rho * V / sutherland_viscosity(T))
        c = db.evaluate(mach, alpha, log10_re=lre)
        qS = q * db.ref_area
        assert f[2] == pytest.approx(-qS * c["CA"], rel=1e-9, abs=1e-9)
        assert f[0] == pytest.approx(-qS * c["CN"], rel=1e-9, abs=1e-9)
        if alpha > 1:
            zcp = db.centre_of_pressure(mach, alpha, log10_re=lre)
            assert t[1] == pytest.approx((zcp - 0.62) * f[0], rel=1e-6)


def test_rate_damping_and_compat_attributes(hobby):
    v, db = hobby
    m = AeroModelHiFi.from_vehicle(v, db)
    fast = Aero(v.aero, v.geometry)
    for attr in ("enabled", "ref_area", "cd_scale", "strip_area", "z"):
        assert hasattr(m, attr)
    assert m.ref_area == pytest.approx(fast.ref_area)
    assert m.axial_coefficient(0.3, True) > 0 and m.axial_coefficient(0.3, False) > 0
    assert 0.3 < m.crossflow_coefficient(0.2) < 1.5
    # pitch-rate damping opposes the rotation
    _, t, _, _ = m.forces(np.array([0.0, 0.0, 200.0]), np.array([0.0, 1.0, 0.0]), 0.62, 1.0, 330.0)
    assert t[1] < 0
    _, t, _, _ = m.forces(np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 5.0]), 0.62, 1.0, 330.0)
    assert t[2] < 0  # fin roll damping
    m.enabled = False
    f, t, q, mach = m.forces(np.array([1.0, 2.0, 50.0]), np.zeros(3), 0.6, 1.0, 330.0)
    assert not f.any() and not t.any() and q > 0 and mach > 0
    # cd_scale scales everything
    m.enabled = True
    f1, _, _, _ = m.forces(np.array([10.0, 0.0, 150.0]), np.zeros(3), 0.6, 1.0, 330.0)
    m.cd_scale = 1.5
    f2, _, _, _ = m.forces(np.array([10.0, 0.0, 150.0]), np.zeros(3), 0.6, 1.0, 330.0)
    np.testing.assert_allclose(f2, 1.5 * f1)


def test_power_on_and_retropropulsion(lander):
    v, db = lander
    m = AeroModelHiFi.from_vehicle(v, db)
    nose_first = np.array([0.0, 0.0, 200.0])
    f_off, _, _, _ = m.forces(nose_first, np.zeros(3), 3.0, 1.0, 330.0)
    m.set_thrust(4e4)
    f_on, _, _, _ = m.forces(nose_first, np.zeros(3), 3.0, 1.0, 330.0)
    assert 0 < -f_on[2] < -f_off[2]  # base drag removed, friction + nose remain
    engine_first = np.array([0.0, 0.0, -300.0])
    m.set_thrust(0.0)
    f_off, _, _, _ = m.forces(engine_first, np.zeros(3), 3.0, 0.5, 330.0)
    m.set_thrust(4e4)
    f_on, _, _, _ = m.forces(engine_first, np.zeros(3), 3.0, 0.5, 330.0)
    ct = 4e4 / (0.5 * 0.5 * 300**2 * m.ref_area)
    assert f_on[2] == pytest.approx(f_off[2] * srp_axial_factor(ct), rel=1e-9)
    assert srp_axial_factor(0.0) == 1.0 and srp_axial_factor(2.0) < 0.01


def test_performance(hobby):
    v, db = hobby
    m = AeroModelHiFi.from_vehicle(v, db)
    vel, w = np.array([20.0, 5.0, 250.0]), np.array([0.1, 0.2, 0.3])
    m.forces(vel, w, 0.6, 1.1, 330.0)
    n = 4000
    t0 = time.perf_counter()
    for _ in range(n):
        m.forces(vel, w, 0.6, 1.1, 330.0)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 50e-6, f"{per_call * 1e6:.1f} us per call"


# --------------------------------------------------------------------------- vehicles
@pytest.mark.slow
@pytest.mark.parametrize("name", PRESETS)
def test_presets_generate(name):
    v = load_vehicle(name)
    db = generate_database(v)
    assert db.check() == []
    assert set(db.axes) == {"mach", "alpha_deg", "log10_re"}
    assert np.all(db.coeffs["CA"][:, db.axes["alpha_deg"] < 90] > 0)
    assert np.all(db.coeffs["CA"][:, db.axes["alpha_deg"] > 90] < 0)
    assert np.all(db.coeffs["CN"] >= -1e-12) and np.all(db.coeffs["Cmq"] <= 0)
    for k, s in db.sigma.items():
        assert np.all(s >= 0), k
    m = AeroModelHiFi.from_vehicle(v, db)
    f, t, _, _ = m.forces(np.array([3.0, -2.0, -60.0]), np.array([0.01, 0.0, 0.0]), 4.0, 1.2, 340.0)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(t))


def test_hobby_rocket_static_margin(hobby):
    """Nose-first: CP well aft of the CG over the flight envelope (stable)."""
    v, db = hobby
    m = AeroModelHiFi.from_vehicle(v, db)
    for cg in _cg_range(v):
        for mach in (0.1, 0.5, 0.9, 1.2, 2.0):
            for a in (1.0, 4.0, 10.0):
                assert m.static_margin(mach, a, cg) > 1.0, (cg, mach, a)
    # CN_alpha of fins + body is consistent with the YAML (fins 9/rad + nose 2/rad)
    cna = db.evaluate(0.1, 1.0)["CN"] / math.radians(1.0)
    assert cna == pytest.approx(9.0 + 2.0, rel=0.12)


@pytest.mark.parametrize("name", ["lander_small", "cargo_hopper_finless"])
def test_engine_first_stability(name):
    """Engine-first (alpha_t -> 180 deg) weathercock stability.

    * ``reverse_potential=0`` (viscous crossflow only, like the fast strip model):
      the CP is at the planform centroid, above the CG -> stable at all angles.
    * Default (Jorgensen TN D-6996 eq. 5, validated on Jernell's base-first data):
      the base-leading potential lift moves the CP toward the base, so the long
      bodies are statically stable only beyond a trim angle (documented finding).
    """
    v = load_vehicle(name)
    cgs = _cg_range(v)
    fast_like = generate_database(v, reverse_potential=0.0, reynolds_effect=False)
    model = AeroModelHiFi.from_vehicle(v, fast_like)
    fast = Aero(v.aero, v.geometry)
    for cg in cgs:
        for mach in (0.1, 0.3, 0.8, 1.5):
            for ap in (2.0, 10.0, 30.0, 60.0):
                assert model.static_margin(mach, 180.0 - ap, cg) > 0.2
        # restoring torque sign agrees with the fast model
        ar = math.radians(10.0)
        vel = np.array([60 * math.sin(ar), 0.0, -60 * math.cos(ar)])
        _, t_db, _, _ = model.forces(vel, np.zeros(3), cg, 1.2, 340.0)
        _, t_fast, _, _ = fast.forces(vel, np.zeros(3), cg, 1.2, 340.0)
        assert np.sign(t_db[1]) == np.sign(t_fast[1]) != 0
    default = generate_database(v)
    m_def = AeroModelHiFi.from_vehicle(v, default)
    cg = max(cgs)
    assert m_def.static_margin(0.3, 175.0, cg) < 0  # unstable near 180 deg (literature)
    assert m_def.static_margin(0.3, 100.0, cg) > 0  # near broadside: turns engine-first


# --------------------------------------------------------------------------- reference data
def _jernell_vehicle(row) -> VehicleSpec:
    l, ln = float(row["l_over_d"]), float(row["lN_over_d"])
    return VehicleSpec.model_validate(
        {
            "name": f"jernell_{row['body']}",
            "geometry": {"length": l, "diameter": 1.0, "nose_length": ln},
            "legs": {"count": 0},
            "mass": {"dry": 1.0, "dry_cg_z": l / 2},
            "engine": {"type": "liquid", "thrust_vac": 1.0},
        }
    )


@pytest.fixture(scope="module")
def jernell():
    """Generated databases for the bodies of TN D-6996 Fig. 9 at M 2.86, Re_d 1.25e5
    (wind-tunnel model: smooth, moments about x_m)."""
    out = {}
    for row in _rows("jernell_m286_bodies.csv"):
        l = float(row["l_over_d"])
        shape = row["nose_shape"] if row["nose_shape"] != "flat" else "tangent_ogive"
        db = generate_database(
            _jernell_vehicle(row),
            mach_grid=(2.5, 2.86, 3.0),
            log10_re_grid=(math.log10(1.25e5),),
            nose_shape=shape,
            roughness=0.0,
            x_ref=l - float(row["xm_over_d"]),
        )
        out[int(row["body"])] = db
    return out


def test_geometry_matches_tn_d6996_fig9():
    table = {  # body: (Ap/d^2, V/d^3, x_c/d, A_s/d^2) from TN D-6996 Fig. 9
        3: (5.500, 3.925, 4.183, 17.34),
        4: (7.500, 5.495, 5.200, 23.62),
        5: (9.500, 7.065, 6.211, 29.91),
        6: (8.011, 5.977, 4.963, 25.24),
        7: (5.340, 3.671, 4.200, 16.82),
        8: (7.340, 5.241, 5.234, 23.10),
        9: (9.340, 6.811, 6.255, 29.38),
    }
    geo = {3: (7, 3, "cone"), 4: (9, 3, "cone"), 5: (11, 3, "cone"), 6: (9, 3, "tangent_ogive"),
           7: (7, 5, "tangent_ogive"), 8: (9, 5, "tangent_ogive"), 9: (11, 5, "tangent_ogive")}  # fmt: skip
    for b, (ap, vol, xc, a_s) in table.items():
        l, ln, shape = geo[b]
        g = BodyGeometry(l, 1.0, ln, shape)
        assert g.planform_area == pytest.approx(ap, rel=2e-3)
        assert g.volume == pytest.approx(vol, rel=2e-3)
        assert g.planform_centroid_x == pytest.approx(xc, rel=2e-3)
        assert g.wetted_area == pytest.approx(a_s, rel=3e-3)


def test_normal_force_vs_jernell(jernell):
    """CN vs NASA TM X-1658 (via TN D-6996 Figs. 10-12), M 2.86, alpha 5-165 deg."""
    fwd, rev = [], []
    for row in _rows("jernell_m286_cn.csv"):
        b, a, cn = int(row["body"]), float(row["alpha_deg"]), float(row["CN"])
        model = jernell[b].evaluate(2.86, a)["CN"]
        err = model - cn
        if a <= 105:
            assert abs(err) <= max(0.20 * cn, 0.7), (b, a, cn, model)
            if a >= 15:
                fwd.append(err / cn)
        else:
            assert abs(err) <= max(0.45 * cn, 0.7), (b, a, cn, model)
            rev.append(err / cn)
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
    assert rms(fwd) < 0.12  # measured: 11.0 % rms, +8 % mean (see docs/models/aero.md)
    assert rms(rev) < 0.22  # measured: 19.9 % rms, +17 % mean (base first)


def test_centre_of_pressure_vs_jernell(jernell):
    cn_data = {
        (int(r["body"]), float(r["alpha_deg"])): float(r["CN"])
        for r in _rows("jernell_m286_cn.csv")
    }
    errs = []
    for row in _rows("jernell_m286_cm.csv"):
        b, a, cm = int(row["body"]), float(row["alpha_deg"]), float(row["Cm"])
        if (b, a) not in cn_data:
            continue
        c = jernell[b].evaluate(2.86, a)
        errs.append(c["Cm"] / c["CN"] - cm / cn_data[(b, a)])  # CP error in calibres
    errs = np.array(errs)
    assert np.sqrt(np.mean(errs**2)) < 0.25  # measured 0.15 d rms
    assert np.max(np.abs(errs)) < 0.5  # measured 0.37 d


def test_axial_force_vs_jernell(jernell):
    for row in _rows("jernell_m286_ca.csv"):
        b, a, ca = int(row["body"]), float(row["alpha_deg"]), float(row["CA"])
        model = jernell[b].evaluate(2.86, a)["CA"]
        # flat faces / base first within 12 %; cone-cylinders nose-first over-predicted
        # by 24-43 % (sting-support interference raises base pressure, TN D-6996 p. 21)
        tol = 0.12 if (b <= 2 or a > 90) else 0.50
        assert abs(model - ca) <= tol * abs(ca), (b, a, ca, model)


def test_taylor_maccoll_and_linnell_bailey():
    tm = taylor_maccoll_cone(2.0, math.radians(10.0))
    assert math.degrees(tm["shock_angle"]) == pytest.approx(31.2, abs=0.15)  # NACA 1135 chart
    for mach in (1.5, 2.0, 3.0, 5.0):
        for deg in (5.0, 10.0, 20.0):
            tm = taylor_maccoll_cone(mach, math.radians(deg))
            lb = float(linnell_bailey(mach, math.radians(deg)))
            assert lb == pytest.approx(tm["cp"], rel=0.06)
    assert taylor_maccoll_cone(1.2, math.radians(25.0)) is None  # detached


def test_ogive_wave_drag_vs_rossow_chart():
    for row in _rows("rossow_wave_drag.csv"):
        k = float(row["K"])
        if k < 0.6:
            continue
        mach = 6.0
        body = BodyGeometry(10.0, 1.0, mach / k, "tangent_ogive")
        param = 0.7 * mach**2 * float(supersonic_wave_drag(mach, body))
        assert param == pytest.approx(float(row["ogive"]), rel=0.06)
        cone = BodyGeometry(10.0, 1.0, mach / k, "cone")
        param = 0.7 * mach**2 * float(supersonic_wave_drag(mach, cone))
        assert param == pytest.approx(float(row["cone"]), rel=0.06)


def test_base_drag_and_face_drag():
    for row in _rows("love_base_pressure.csv"):
        m = float(row["mach"])
        if m >= 1.1:
            assert float(base_mod.base_drag_power_off(m, 0.2)) == pytest.approx(
                -float(row["Cp_base"]), rel=0.01
            )
    m = np.linspace(0.0, 3.0, 301)
    cdb = base_mod.base_drag_power_off(m, 0.2)
    assert np.all(np.abs(np.diff(cdb)) < 0.01)  # continuous through the transonic blend
    assert float(cp_max_pitot(1e-9)) == pytest.approx(1.0)
    assert cp_max_pitot(10.0) == pytest.approx(1.83, abs=0.01)  # TN D-6996 Fig. 7 (1.84 at M 100)
    assert float(base_mod.flat_face_drag(2.86)[0]) == pytest.approx(0.85 * 1.77, rel=0.02)


def test_crossflow_drag_coefficient():
    assert float(crossflow_cd(0.1)) == pytest.approx(1.2)
    assert float(crossflow_cd(1.0)) > 1.9  # transonic peak, TN D-6996 Fig. 1
    assert float(crossflow_cd(6.0)) == pytest.approx(1.29, abs=0.02)  # ~ modified Newtonian
    assert float(crossflow_cd(0.2, 5e5)) < 0.4  # critical crossflow Reynolds number
    assert float(crossflow_cd(0.6, 5e5)) == pytest.approx(float(crossflow_cd(0.6)))


def test_skin_friction():
    assert float(cf_karman_schoenherr(1e7)) == pytest.approx(0.00293, rel=0.01)
    inc = float(skin_friction(1e7, 0.0, 220.0))
    assert inc == pytest.approx(0.00293, rel=0.01)
    vd3 = float(skin_friction(1e7, 3.0, 220.0))
    assert 0.6 < vd3 / inc < 0.7  # classic van Driest II reduction at M 3
    eck = float(skin_friction(1e7, 3.0, 220.0, method="eckert"))
    assert eck == pytest.approx(vd3, rel=0.15)
    rough = float(skin_friction(1e7, 0.3, 288.0, length=1.0, roughness=200e-6))
    assert rough > float(skin_friction(1e7, 0.3, 288.0))


def test_newtonian_limits():
    body = BodyGeometry(10.0, 1.0, 0.0, "flat")
    r = newtonian_body(body, [0.0, 90.0, 180.0], 8.0, 5.0, cp_max=1.0)
    assert r["CA"][0] == pytest.approx(1.0, rel=1e-6)  # flat face, Cp = Cp_max
    assert r["CA"][2] == pytest.approx(-1.0, rel=1e-6)
    assert r["CN"][1] == pytest.approx(2 / 3 * body.planform_area / body.ref_area, rel=0.01)
    assert abs(r["Cm"][1]) < 1e-6  # symmetric cylinder broadside


def test_gridfin_multipliers():
    assert gridfin_cn_alpha(0.0) == pytest.approx(1.0)
    assert gridfin_cd0(0.0) == pytest.approx(1.0)
    mch, msw = choke_mach(), swallow_mach()
    assert 0.6 < mch < 0.8 and 1.4 < msw < 1.8
    mid = 0.5 * (mch + msw)
    assert gridfin_cn_alpha(mid) < 0.75 * gridfin_cn_alpha(mch)  # choked: effectiveness lost
    assert gridfin_cn_alpha(msw + 0.05) > gridfin_cn_alpha(mid)  # recovery after swallowing
    assert gridfin_cd0(1.0) > 1.6  # transonic drag peak
    m = np.linspace(0, 5, 501)
    assert np.all(np.isfinite(gridfin_cn_alpha(m))) and np.all(gridfin_cd0(m) > 0)


def test_heating():
    q = stagnation_heat_flux(1e-4, 7000.0, 1.0)
    assert q == pytest.approx(1.7415e-4 * 0.01 * 7000.0**3)
    t = np.linspace(0, 10, 101)
    assert heat_load(t, np.full_like(t, 1e-4), np.full_like(t, 7000.0), 1.0) == pytest.approx(
        10 * q
    )
    mon = HeatingMonitor(1.0)
    for _ in range(11):
        mon.update(1.0, 1e-4, 7000.0)
    assert mon.load == pytest.approx(10 * q) and mon.q_peak == pytest.approx(q)
    with pytest.raises(ValueError):
        stagnation_heat_flux(1.0, 1.0, 0.0)


def test_perturbed_database(hobby):
    _, db = hobby
    a = db.perturbed(np.random.default_rng(5))
    b = db.perturbed(np.random.default_rng(5))
    np.testing.assert_array_equal(a.coeffs["CN"], b.coeffs["CN"])
    assert not np.array_equal(a.coeffs["CN"], db.coeffs["CN"])
    assert np.all(a.coeffs["Cmq"] <= 0) and "dispersion" in a.meta


# --------------------------------------------------------------------------- importers
def test_import_rasaero(hobby):
    v, db = hobby
    out = import_rasaero(REF / "synthetic_rasaero.csv", db, body_length=v.geometry.length)
    lre = 6.5  # the synthetic file was made at this Re (imports are Re-independent)
    for mach in (0.5, 1.5):
        g, i = db.evaluate(mach, 2.0, log10_re=lre), out.evaluate(mach, 2.0, log10_re=lre)
        assert i["CA"] == pytest.approx(1.10 * g["CA"], rel=1e-3)
        assert i["CN"] == pytest.approx(0.95 * g["CN"], rel=1e-3)
        assert out.centre_of_pressure(mach, 2.0, log10_re=lre) == pytest.approx(
            db.centre_of_pressure(mach, 2.0, log10_re=lre), abs=2e-3
        )
    # outside the imported range (alpha > 4 + margin, M > 2 + margin) the generator stays
    assert out.evaluate(0.5, 30.0)["CN"] == pytest.approx(db.evaluate(0.5, 30.0)["CN"])
    assert out.evaluate(3.0, 2.0)["CA"] == pytest.approx(db.evaluate(3.0, 2.0)["CA"])
    assert "RASAero" in out.meta["provenance"]["CA"]
    assert out.meta["provenance"]["Cmq"] == "generated"


def test_import_openrocket(hobby):
    v, db = hobby
    out = import_openrocket(REF / "synthetic_openrocket.csv", db, body_length=v.geometry.length)
    lre = 6.5
    for mach in (0.4, 0.8, 1.2):
        g, i = db.evaluate(mach, 0.0, log10_re=lre), out.evaluate(mach, 0.0, log10_re=lre)
        assert i["CA"] == pytest.approx(1.05 * g["CA"], rel=0.03)
        g, i = db.evaluate(mach, 3.0, log10_re=lre), out.evaluate(mach, 3.0, log10_re=lre)
        assert i["CN"] == pytest.approx(1.08 * g["CN"], rel=0.05)
    assert "OpenRocket" in out.meta["provenance"]["CN"]


def test_import_datcom(hobby):
    v, db = hobby
    parsed = parse_datcom(REF / "synthetic_datcom_for006.dat")
    assert parsed["mach"] == [0.6, 1.5]
    assert parsed["ref"]["xcg"] == pytest.approx(0.8)
    assert parsed["blocks"][0]["deriv_unit"] == "deg"
    assert "CMQ" in parsed["blocks"][1]["dynamic"] and "CLLP" in parsed["blocks"][1]["dynamic"]
    out = import_datcom(REF / "synthetic_datcom_for006.dat", db, body_length=v.geometry.length)
    for mach, a in ((0.6, 4.0), (1.5, 12.0)):
        g, i = db.evaluate(mach, a, log10_re=6.5), out.evaluate(mach, a, log10_re=6.5)
        assert i["CN"] == pytest.approx(1.1 * g["CN"], rel=1e-3)
        assert i["Cm"] == pytest.approx(1.1 * g["Cm"], rel=1e-3)  # moment transfer from XCG
        assert i["CA"] == pytest.approx(g["CA"], rel=1e-3)
        assert i["Clp"] == pytest.approx(-0.4 * 180 / math.pi, rel=1e-3)
        assert i["Cmq"] < 0
    assert "DATCOM" in out.meta["provenance"]["Cmq"]


def test_import_generic_csv(hobby):
    _, db = hobby
    out = import_generic_csv(REF / "synthetic_generic.csv", db)
    lre = 6.5
    assert out.evaluate(0.8, 10.0, log10_re=lre)["CN"] == pytest.approx(
        0.9 * db.evaluate(0.8, 10.0, log10_re=lre)["CN"], rel=2e-3
    )
    standalone = import_generic_csv(REF / "synthetic_generic.csv")
    assert set(standalone.axes) == {"mach", "alpha_deg"}
    c = standalone.evaluate(0.8, 10.0)
    assert c["Cn"] == 0.0 and c["CN"] > 0  # CN (normal) and Cn (yaw) are distinct columns


def test_roll_dependent_database():
    """A phi axis (e.g. imported cruciform-fin data) is interpolated with periodic wrap."""
    axes = {
        "mach": np.array([0.0, 2.0]),
        "alpha_deg": np.array([0.0, 90.0, 180.0]),
        "phi_deg": np.array([0.0, 22.5, 45.0, 67.5, 90.0]),
    }
    _, A, P = np.meshgrid(*axes.values(), indexing="ij")
    sa = np.sin(np.radians(A))
    coeffs = {
        "CA": 0.4 * np.cos(np.radians(A)),
        "CN": 5 * sa * (1 + 0.1 * np.cos(np.radians(4 * P))),
        "CY": 0.3 * sa * np.sin(np.radians(4 * P)),
    }
    db = AeroDatabase(axes, coeffs, ref_area=0.01, ref_length=0.1, x_ref=0.2,
                      meta={"phi_period_deg": 90.0, "body_length": 1.0})  # fmt: skip
    assert db.roll_dependent
    assert db.evaluate(1.0, 45.0, phi_deg=100.0)["CN"] == pytest.approx(
        db.evaluate(1.0, 45.0, phi_deg=10.0)["CN"]
    )
    m = AeroModelHiFi(db, BodyGeometry(1.0, 0.1, 0.2, "cone"))
    rng = np.random.default_rng(2)
    for _ in range(200):
        v = rng.normal(size=3) * 100
        f, _, _, _ = m.forces(v, np.zeros(3), 0.5, 1.0, 340.0)
        assert np.all(np.isfinite(f)) and float(f @ v) <= 1e-9  # side force is workless
