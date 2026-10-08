"""Convex powered-descent guidance (plume.control.pdg): feasibility, fuel optimality
against an analytic 1-D solution, constraint satisfaction, minimum-miss fallback and
drag handled by successive convexification."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from plume.constants import G0
from plume.control.pdg import PDGConfig, PoweredDescentGuidance

G = np.array([0.0, 0.0, -9.81])
ISP = 300.0
RHO1, RHO2 = 12_000.0, 60_000.0  # N
M0, M_DRY = 2000.0, 1500.0


def _cfg(**kw) -> PDGConfig:
    base = dict(
        thrust_min=RHO1,
        thrust_max=RHO2,
        isp=ISP,
        m_dry=M_DRY,
        accel_max=None,
        tilt_max=math.radians(30.0),
        glide_slope=math.radians(10.0),
        v_final=np.zeros(3),
        n=40,
    )
    base.update(kw)
    return PDGConfig(**base)


# ---------------------------------------------------------------- 1-D analytic optimum
def _burn(h, v, m, thrust, tau):
    """Exact vertical state after burning ``thrust`` for ``tau`` (no drag)."""
    a = 1.0 / (ISP * G0)
    k = a * thrust / m
    w = 1.0 - k * tau
    v1 = v - 9.81 * tau - math.log(w) / a
    h1 = h + v * tau - 4.905 * tau * tau + (w * math.log(w) - w + 1.0) / (a * k)
    return h1, v1, m * w


def _stop_time(h, v, m, thrust):
    lo, hi = 0.0, (m / (thrust / (ISP * G0))) * 0.999
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _burn(h, v, m, thrust, mid)[1] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _analytic_min_fuel(h0, v0, m0):
    """Fuel-optimal vertical landing is bang-bang: minimum thrust until the switch, then
    maximum thrust to touch down at zero speed. Find the switch time by bisection."""

    def touchdown_height(ts):
        h, v, m = _burn(h0, v0, m0, RHO1, ts)
        t2 = _stop_time(h, v, m, RHO2)
        return _burn(h, v, m, RHO2, t2)[0], ts, t2

    lo, hi = 0.0, 30.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if touchdown_height(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    _, ts, t2 = touchdown_height(0.5 * (lo + hi))
    return (RHO1 * ts + RHO2 * t2) / (ISP * G0), ts + t2


def test_vertical_fuel_matches_bang_bang_optimum():
    h0, v0 = 1500.0, -80.0
    fuel_opt, tf_opt = _analytic_min_fuel(h0, v0, M0)
    pdg = PoweredDescentGuidance(_cfg(glide_slope=0.0))
    sol = pdg.solve_free_tf([0, 0, h0], [0, 0, v0], M0, G, (0.5 * tf_opt, 2.0 * tf_opt))
    assert sol is not None
    # discretisation (first-order hold over 40 nodes) costs a little, never less than optimal
    assert sol.fuel == pytest.approx(fuel_opt, rel=0.03)
    assert sol.fuel > 0.995 * fuel_opt
    assert sol.tf == pytest.approx(tf_opt, rel=0.1)
    thrust = sol.thrust
    # bang-bang structure: starts at the lower bound, ends at the upper bound
    assert thrust[1] < 1.1 * RHO1
    assert thrust[-2] > 0.9 * RHO2


# ---------------------------------------------------------------- 3-D feasibility
def _check_constraints(sol, cfg, tol=1e-3):
    T = sol.thrust
    assert np.all(T >= cfg.thrust_min * (1 - 0.02))  # Taylor bound is conservative to ~1 %
    assert np.all(T <= cfg.thrust_max * (1 + tol))
    u = sol.u
    cos_tilt = u[:, 2] / np.linalg.norm(u, axis=1)
    assert np.all(cos_tilt >= math.cos(cfg.tilt_max) - tol)
    rN = sol.r[-1]
    horiz = np.linalg.norm(sol.r[:-1, :2] - rN[:2], axis=1)
    assert np.all(horiz * math.tan(cfg.glide_slope) <= sol.r[:-1, 2] - rN[2] + cfg.glide_apex + 0.1)
    assert np.all(sol.r[:, 2] >= cfg.h_final - 1e-3)
    if cfg.accel_max is not None:
        assert np.all(np.linalg.norm(u + sol.a_aero, axis=1) <= cfg.accel_max * (1 + tol))
    assert sol.m[-1] >= cfg.m_dry * (1 - 1e-6)


def test_3d_divert_is_feasible_and_satisfies_constraints():
    cfg = _cfg(accel_max=3.0 * 9.81, h_final=10.0, v_final=np.array([0.0, 0.0, -3.0]))
    pdg = PoweredDescentGuidance(cfg)
    r0, v0 = np.array([300.0, -150.0, 1200.0]), np.array([-20.0, 10.0, -90.0])
    sol = pdg.solve_free_tf(r0, v0, M0, G, (5.0, 60.0))
    assert sol is not None
    assert sol.miss < 0.1
    np.testing.assert_allclose(sol.r[0], r0, atol=1e-6)
    np.testing.assert_allclose(sol.v[-1], cfg.v_final, atol=1e-3)
    assert sol.r[-1, 2] == pytest.approx(10.0, abs=1e-3)
    _check_constraints(sol, cfg)
    # lossless convexification: the slack is tight, |u| = sigma at the optimum, so the
    # thrust magnitude stays inside the true (non-convex) bounds
    assert np.all(sol.thrust >= 0.98 * RHO1)


def test_out_of_reach_pad_gives_minimum_miss_with_propellant_kept():
    cfg = _cfg(m_dry=1800.0)  # 200 kg of propellant: enough to stop, not to divert 2 km
    pdg = PoweredDescentGuidance(cfg)
    sol = pdg.solve_free_tf([2000.0, 0, 800.0], [0, 0, -60.0], M0, G, (5.0, 40.0))
    assert sol is not None
    assert 10.0 < sol.miss < 2000.0
    assert sol.m[-1] == pytest.approx(1800.0, rel=1e-4)  # all usable propellant spent
    _check_constraints(sol, cfg)


def test_infeasible_vertical_stop_returns_none():
    pdg = PoweredDescentGuidance(_cfg())
    # 300 m/s down at 200 m: no thrust profile stops in time
    assert pdg.solve([0, 0, 200.0], [0, 0, -300.0], M0, G, 3.0) is None


def test_attitude_rate_limit_and_initial_axis():
    cfg = _cfg(jerk_max=3.0)
    pdg = PoweredDescentGuidance(cfg)
    pdg.axis0 = np.array([0.0, math.sin(0.2), math.cos(0.2)])
    sol = pdg.solve([100.0, 0, 800.0], [0, 0, -60.0], M0, G, 20.0)
    assert sol is not None
    u0 = sol.u[0] / np.linalg.norm(sol.u[0])
    np.testing.assert_allclose(u0, pdg.axis0, atol=1e-4)
    du = np.linalg.norm(np.diff(sol.u[:, :2], axis=0), axis=1)
    assert np.all(du <= 3.0 * sol.tf / cfg.n + 1e-4)


# ---------------------------------------------------------------- drag
def test_drag_by_successive_convexification_is_self_consistent():
    """Fly the planned thrust open loop through the true quadratic drag: the plan, which
    only knows the drag along its previous iterate, must still arrive at the gate."""
    cda, rho = 2.5, 1.1

    def aero(r, v, m, thrust):
        sp = np.linalg.norm(v, axis=1)
        return -(0.5 * rho * cda * sp / m)[:, None] * v

    cfg = _cfg(h_final=10.0, v_final=np.array([0.0, 0.0, -2.0]), accel_max=4.0 * 9.81)
    pdg = PoweredDescentGuidance(cfg)
    r0, v0 = np.array([80.0, 30.0, 900.0]), np.array([-5.0, 0.0, -110.0])
    sol = pdg.solve_free_tf(r0, v0, M0, G, (3.0, 40.0), aero_fn=aero, iterations=3)
    assert sol is not None
    # drag is a large part of the deceleration at the start
    assert np.linalg.norm(sol.a_aero[0]) > 5.0
    r, v, m = r0.copy(), v0.copy(), M0
    a_isp = 1.0 / (ISP * G0)
    dt = 0.002
    t = 0.0
    while t < sol.tf - 1e-9:
        u = np.array([np.interp(t, sol.t, sol.u[:, j]) for j in range(3)])
        drag = aero(r[None], v[None], np.array([m]), None)[0]
        v = v + (u + drag + G) * dt
        r = r + v * dt
        m -= a_isp * np.linalg.norm(u) * m * dt
        t += dt
    assert np.linalg.norm(r - sol.r[-1]) < 5.0
    assert np.linalg.norm(v - sol.v[-1]) < 1.5
    # without the drag model the same thrust profile would be far off
    plain = pdg.solve(r0, v0, M0, G, sol.tf)
    assert plain is None or abs(plain.fuel - sol.fuel) > 1.0
