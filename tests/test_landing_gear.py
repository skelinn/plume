"""Landing gear: crushable-core stroke, soil sinkage, energy accounting, tip-over margin."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plume.config import SoilSpec, WorldSpec, load_vehicle
from plume.constants import G0
from plume.physics.gear import (
    crush_curve,
    crush_energy,
    crush_plateau,
    landing_mass,
    soil_force,
    soil_spec,
    static_sinkage,
)
from plume.physics.sim import RocketSim

VEH = load_vehicle("lander_small")


def _drop(vz: float, soil="rigid", prop=300.0, steps=600, fidelity="high", gap=0.02):
    w = WorldSpec(fidelity=fidelity, dt=0.005, soil=soil, atmosphere=False)
    sim = RocketSim(VEH, w)
    sim.reset(pos=(0, 0, VEH.legs.height + gap), vel=(0, 0, -vz), prop=prop)
    e0 = sim.energy()["total"]
    sim.step(steps)
    return sim, e0 - sim.energy()["total"]


def test_crush_curve_shape_and_energy_integral():
    legs = VEH.legs
    P = crush_plateau(VEH)
    assert crush_curve(legs, P, 0.0) == pytest.approx(legs.crush_onset * P)
    assert crush_curve(legs, P, 0.5 * legs.stroke) == pytest.approx(P)
    assert crush_curve(legs, P, legs.stroke) == pytest.approx(legs.densified_ratio * P)
    # closed form: onset ramp + plateau + densification triangle
    se, sd, s = legs.elastic_stroke, legs.densification * legs.stroke, legs.stroke
    exact = (
        0.5 * (legs.crush_onset + 1.0) * P * se
        + P * (s - se)
        + 0.5 * (legs.densified_ratio - 1.0) * P * (s - sd)
    )
    assert crush_energy(legs, P, s, n=4000) == pytest.approx(exact, rel=1e-3)


def test_default_crush_force_sized_for_design_speed():
    legs = VEH.legs
    m = landing_mass(VEH)
    P = crush_plateau(VEH)
    # all legs on the plateau stop max_touchdown_speed within 75 % of the stroke
    decel = legs.count * P / m - G0
    assert legs.max_touchdown_speed**2 / (2 * decel) == pytest.approx(0.75 * legs.stroke)
    # and a vehicle at rest does not crush the core
    assert crush_curve(legs, P, 0.0) > m * G0 / legs.count


def test_bekker_sinkage_is_inverse_of_pressure_curve():
    soil = soil_spec("dry_sand")
    for load in (1e3, 4e3, 1.5e4):
        z = static_sinkage(soil, load, 0.15)
        assert soil_force(soil, z, 0.15) == pytest.approx(load, rel=1e-9)
    assert static_sinkage(soil_spec("rigid"), 1e5, 0.15) == 0.0
    clay = soil_spec("clay")  # beyond the bearing strength the pad keeps sinking
    assert math.isinf(static_sinkage(clay, 1.01 * clay.bearing_strength * math.pi * 0.15**2, 0.15))


def test_fast_fidelity_keeps_rigid_legs():
    sim = RocketSim(VEH, WorldSpec(fidelity="fast"))
    assert sim.gear.model_name == "rigid" and sim.model.nq == 7
    hi = RocketSim(VEH, WorldSpec(fidelity="high"))
    assert hi.gear.model_name == "crush" and hi.model.nq == 7 + VEH.legs.count
    # hull + stroking leg masses = the dry mass (MuJoCo total = mass model)
    hi.reset(prop=0.0, rcs_prop=0.0)
    total = float(hi.model.body_mass.sum())
    assert total == pytest.approx(hi.mp.mass, rel=1e-6)


@pytest.mark.parametrize("vz", [3.0, 5.0])
def test_crush_energy_accounting_rigid_ground(vz):
    """Mechanical energy lost by the vehicle = energy absorbed by the crushable cores
    and dampers (the rest is contact softness and pad impact: a few per cent)."""
    sim, d_e = _drop(vz)
    r = sim.gear.report()
    absorbed = sum(r.energy_crush_j) + sum(r.energy_damper_j)
    assert sim.gear.failure == ""
    assert absorbed == pytest.approx(d_e, rel=0.05)
    assert max(r.stroke_used_m) < VEH.legs.stroke
    # crush work = the force-stroke curve integrated over the stroke used
    P = sim.gear.plateau
    for s, e in zip(r.stroke_used_m, r.energy_crush_j, strict=True):
        assert e == pytest.approx(crush_energy(VEH.legs, P, s), rel=0.03)
    # at the design speed, about 3/4 of the stroke is used
    if vz == 5.0:
        assert 0.55 < max(r.stroke_fraction) < 0.85
    # at rest each leg carries a quarter of the weight
    w = sim.state.mass * G0
    np.testing.assert_allclose(sim.gear.load, w / 4, rtol=0.03)


def test_hard_landing_bottoms_out():
    sim, _ = _drop(7.0)
    assert sim.leg_failure() in {"gear_bottomed", "gear_overload"}


def test_soil_sinkage_and_energy():
    sim, d_e = _drop(3.0, soil="dry_sand")
    r = sim.gear.report()
    absorbed = sum(r.energy_crush_j) + sum(r.energy_damper_j) + sum(r.energy_soil_j)
    assert absorbed == pytest.approx(d_e, rel=0.1)
    w = sim.state.mass * G0 / 4
    z_static = static_sinkage(soil_spec("dry_sand"), w, VEH.legs.footpad_radius)
    # the touchdown pressure peak drives the pad deeper than the static sinkage
    assert z_static < max(r.sinkage_m) < 6 * z_static
    assert sum(r.energy_soil_j) > 0.1 * absorbed


def test_sudden_placement_sinks_twice_the_static_value():
    """A load set down suddenly on a linear (n = 1) plastic soil sinks until the soil
    work equals the work of the weight: W z = k z^2 / 2, i.e. twice the static sinkage
    (no rebound: the soil does not recover)."""
    soil = SoilSpec(name="soft", k_c=0.0, k_phi=2.0e6, n=1.0)
    sim, _ = _drop(0.0, soil=soil, steps=800, gap=0.001)
    w = sim.state.mass * G0 / 4
    z_static = static_sinkage(soil, w, VEH.legs.footpad_radius)
    np.testing.assert_allclose(sim.gear.sink, 2.0 * z_static, rtol=0.15)
    assert sim.gear.stroke.max() < 0.01  # core not crushed (mm of soft-constraint creep)


def test_tipover_margin_matches_geometry():
    sim, _ = _drop(1.0, fidelity="fast", steps=400)
    st = sim.state
    legs = VEH.legs
    # square support polygon (4 legs): edge distance span cos(45 deg)
    d = legs.span * math.cos(math.pi / legs.count)
    h = st.com[2]  # CG height above the ground (pads at z = 0)
    expected = math.degrees(math.atan2(d, h))
    assert sim.gear.margin == pytest.approx(expected, abs=0.3)
    tilted = sim.gear.tipover_margin(sim, up=np.array([math.sin(0.1), 0.0, math.cos(0.1)]))
    assert tilted < sim.gear.margin
