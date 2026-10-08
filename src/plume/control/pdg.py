"""Convex powered-descent guidance (G-FOLD style) for the landing burn.

Minimum-fuel thrust profile from the current state to the pad with zero velocity, after
Acikmese & Ploen (2007) "Convex programming approach to powered descent guidance for Mars
landing" (lossless convexification of the non-convex thrust lower bound) and Blackmore,
Acikmese & Scharf (2010) "Minimum-landing-error powered-descent guidance for Mars landing
using convex optimization" (minimum miss when the pad is out of reach).

Problem (local frame at the pad: x, y horizontal, z up; ``u = T / m``, ``z = ln m``,
``sigma`` the slack on ``|u|``)::

    minimise   -z_N  +  w_miss * |r_N,xy|                  (max final mass, min miss)
    subject to r' = v,  v' = u + a_aero + g,  z' = -alpha sigma - beta / m
               |u_k| <= sigma_k                              (lossless convexification)
               rho1 e^-z0 (1 - dz + dz^2/2) <= sigma <= rho2 e^-z0 (1 - dz)
               u_k . e_z >= sigma_k cos(theta_max)            (thrust tilt limit)
               |u_k + a_aero,k| <= a_max                      (cargo g-limit, sensed)
               |r_k,xy - r_N,xy| <= (r_k,z - r_N,z + h_apex) / tan(gamma_gs)   (glide slope)
               r_k,z >= h_f                                   (stay above the gate)
               |u_k+1,xy - u_k,xy| <= j_max dt,  u_0 || current axis   (attitude rate)
               z_N >= ln(m_dry)                               (propellant)
               r_0, v_0, z_0 given;  r_N,z = h_f, v_N = v_f, u_N,xy = 0

``dz = z - z0(t)`` with ``z0(t) = ln(m0 - alpha rho2 t)`` (the Taylor expansions of the
thrust bounds in log-mass are exact at z0 and conservative elsewhere). The dynamics are
discretised with first-order-hold controls (trapezoidal velocity, exact position for a
piecewise-linear acceleration).

The aerodynamic acceleration ``a_aero`` is not convex in the state. It is handled by
successive convexification: the problem is solved with the aero acceleration evaluated
along the previous iterate's trajectory (velocity, altitude, mass and thrust - the thrust
matters because a retro-propulsive plume shields the forebody), and re-solved a few
times (Szmuk, Acikmese & Berning 2016; Mao, Szmuk & Acikmese 2016). The final time is
free: a golden-section line search over ``t_f`` (Acikmese & Ploen 2007, section V).

``cvxpy`` is an optional dependency (``pip install plume[gnc]``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from plume.constants import G0

# aero acceleration model: (r (N+1,3), v (N+1,3), m (N+1,), thrust N (N+1,)) -> (N+1,3)
AeroFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the installed extras
        return False
    return True


@dataclass
class PDGConfig:
    """Vehicle and constraint data for the powered-descent problem (SI units)."""

    thrust_min: float  # N, lower thrust bound while the engine burns (throttle_min x max)
    thrust_max: float  # N, upper thrust bound used by the plan
    isp: float  # s, vacuum Isp: mdot = (T + p A_e) / (isp g0)
    m_dry: float  # kg, mass below which the propellant is gone (incl. any reserve)
    accel_max: float | None = None  # m/s^2, sensed (thrust + aero) acceleration limit
    tilt_max: float = math.radians(25.0)  # rad, thrust axis from vertical
    glide_slope: float = math.radians(15.0)  # rad, minimum approach elevation
    glide_apex: float = 5.0  # m, cone apex below the landing point (tolerance)
    mdot_offset: float = 0.0  # kg/s, p_amb A_e / (isp g0): flow not producing thrust
    jerk_max: float | None = None  # m/s^3, horizontal thrust-acceleration rate limit
    n: int = 24  # discretisation intervals
    w_miss: float = 0.05  # objective weight of the horizontal miss (per m, vs ln mass)
    h_final: float = 0.0  # m, height of the end point (a terminal gate above the pad)
    v_final: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    solver: str = "CLARABEL"


@dataclass
class PDGSolution:
    tf: float
    t: np.ndarray  # (N+1,)
    r: np.ndarray  # (N+1, 3) position relative to the pad
    v: np.ndarray  # (N+1, 3)
    u: np.ndarray  # (N+1, 3) thrust acceleration
    m: np.ndarray  # (N+1,) mass
    a_aero: np.ndarray  # (N+1, 3) aero acceleration assumed by the plan
    cost: float
    status: str

    @property
    def thrust(self) -> np.ndarray:
        return np.linalg.norm(self.u, axis=1) * self.m

    @property
    def fuel(self) -> float:
        return float(self.m[0] - self.m[-1])

    @property
    def miss(self) -> float:
        return float(np.linalg.norm(self.r[-1, :2]))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolated (r, v, u, a_aero) at plan time ``t``."""
        tt = float(np.clip(t, 0.0, self.tf))
        out = []
        for arr in (self.r, self.v, self.u, self.a_aero):
            out.append(np.array([np.interp(tt, self.t, arr[:, j]) for j in range(3)]))
        return out[0], out[1], out[2], out[3]


class PoweredDescentGuidance:
    """Builds the parametrised SOCP once (DPP) and re-solves it for new data."""

    def __init__(self, cfg: PDGConfig):
        import cvxpy as cp

        self.cfg = cfg
        n = cfg.n
        self.n = n
        self._cp = cp
        r = cp.Variable((n + 1, 3))
        v = cp.Variable((n + 1, 3))
        u = cp.Variable((n + 1, 3))
        s = cp.Variable(n + 1)
        z = cp.Variable(n + 1)
        self.var = {"r": r, "v": v, "u": u, "s": s, "z": z}
        P = cp.Parameter
        # every parameter enters affinely (DPP), so cvxpy compiles the problem once
        self.p = {
            "hdt": P(nonneg=True),  # dt / 2
            "dt": P(nonneg=True),
            "dt2a": P(nonneg=True),  # dt^2 / 3 (weight of a_k in the position update)
            "dt2b": P(nonneg=True),  # dt^2 / 6 (weight of a_k+1)
            "adt": P(nonneg=True),  # alpha dt / 2
            "dv_ext": P((n, 3)),  # velocity change from gravity + aero over each interval
            "dr_ext": P((n, 3)),  # position change from gravity + aero
            "dz_ext": P(n),  # log-mass change from the non-thrust flow (p_amb A_e)
            "aero": P((n + 1, 3)),
            "r0": P(3),
            "v0": P(3),
            "lnm0": P(),
            "z0": P(n + 1),  # ln(m0 - alpha rho2 t_k)
            "inv_mu1": P(n + 1, nonneg=True),  # 1 / (rho1 e^-z0)
            "mu2": P(n + 1, nonneg=True),  # rho2 e^-z0
            "c2": P(n + 1),  # rho2 e^-z0 (1 + z0)
            "zmax": P(n + 1),  # ln(m0 - alpha rho1 t_k): upper mass bound
            "lnmdry": P(),
            "vf": P(3),
            "jdt": P(nonneg=True),  # jerk_max * dt: thrust-direction rate limit per step
            "ax0": P(3),  # current thrust axis: the plan starts along it
        }
        p = self.p
        cons = [
            r[0] == p["r0"],
            v[0] == p["v0"],
            z[0] == p["lnm0"],
            v[1:] == v[:-1] + p["hdt"] * (u[:-1] + u[1:]) + p["dv_ext"],
            r[1:]
            == r[:-1] + p["dt"] * v[:-1] + p["dt2a"] * u[:-1] + p["dt2b"] * u[1:] + p["dr_ext"],
            z[1:] == z[:-1] - p["adt"] * (s[:-1] + s[1:]) - p["dz_ext"],
        ]
        dz = z - p["z0"]
        cons += [
            cp.norm(u, axis=1) <= s,
            cp.multiply(p["inv_mu1"], s) >= 1 - dz + cp.square(dz) / 2,
            s <= p["c2"] - cp.multiply(p["mu2"], z),
            u[:, 2] >= math.cos(cfg.tilt_max) * s,
            z >= p["z0"],
            z <= p["zmax"],
            z[n] >= p["lnmdry"],
            r[n, 2] == cfg.h_final,
            r[:, 2] >= cfg.h_final,  # never below the gate (terrain clearance)
            v[n] == p["vf"],
            u[n, :2] == 0.0,
        ]
        if cfg.jerk_max is not None:
            # the attitude loop turns the thrust vector at a finite rate: limit the change
            # of the horizontal thrust acceleration between nodes, and start the plan along
            # the current thrust axis (no step in the commanded direction)
            cons += [
                cp.norm(u[1:, :2] - u[:-1, :2], axis=1) <= p["jdt"],
                p["ax0"][2] * u[0, 0] == p["ax0"][0] * u[0, 2],
                p["ax0"][2] * u[0, 1] == p["ax0"][1] * u[0, 2],
            ]
        if cfg.accel_max is not None:
            cons.append(cp.norm(u + p["aero"], axis=1) <= cfg.accel_max)
        tg = math.tan(cfg.glide_slope)
        if tg > 0:
            land = cp.Constant(np.ones((n, 1))) @ r[n : n + 1, :2]
            cons.append(
                cp.norm(r[:n, :2] - land, axis=1) * tg <= r[:n, 2] - r[n, 2] + cfg.glide_apex
            )
        obj = -z[n] + cfg.w_miss * cp.norm(r[n, :2])
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.solves = 0
        self.axis0: np.ndarray | None = None  # current thrust axis (pad frame), if limited

    # ------------------------------------------------------------------ single solve
    def _solve_fixed(self, r0, v0, m0, g, tf, aero, beta_m) -> PDGSolution | None:
        cfg, n, p = self.cfg, self.n, self.p
        dt = tf / n
        alpha = 1.0 / (cfg.isp * G0)
        tk = np.linspace(0.0, tf, n + 1)
        m_lo = m0 - alpha * cfg.thrust_max * tk - cfg.mdot_offset * tk
        if m_lo[-1] <= 1e-3 * m0:
            return None
        m_hi = m0 - alpha * cfg.thrust_min * tk
        z0 = np.log(m_lo)
        g = np.asarray(g, dtype=float)
        aero = np.asarray(aero, dtype=float)
        ext = aero + g[None, :]
        p["hdt"].value = dt / 2.0
        p["dt"].value = dt
        p["dt2a"].value = dt * dt / 3.0
        p["dt2b"].value = dt * dt / 6.0
        p["adt"].value = alpha * dt / 2.0
        p["dv_ext"].value = dt / 2.0 * (ext[:-1] + ext[1:])
        p["dr_ext"].value = dt * dt / 3.0 * ext[:-1] + dt * dt / 6.0 * ext[1:]
        p["dz_ext"].value = dt / 2.0 * (beta_m[:-1] + beta_m[1:])
        p["aero"].value = aero
        p["r0"].value = np.asarray(r0, dtype=float)
        p["v0"].value = np.asarray(v0, dtype=float)
        p["lnm0"].value = math.log(m0)
        p["z0"].value = z0
        mu2 = cfg.thrust_max * np.exp(-z0)
        p["inv_mu1"].value = 1.0 / (cfg.thrust_min * np.exp(-z0))
        p["mu2"].value = mu2
        p["c2"].value = mu2 * (1.0 + z0)
        p["zmax"].value = np.log(m_hi)
        p["lnmdry"].value = math.log(cfg.m_dry)
        p["vf"].value = np.asarray(cfg.v_final, dtype=float)
        p["jdt"].value = (cfg.jerk_max or 0.0) * dt
        ax = np.asarray(self.axis0 if self.axis0 is not None else [0.0, 0.0, 1.0], dtype=float)
        tilt = math.acos(float(np.clip(ax[2] / np.linalg.norm(ax), -1.0, 1.0)))
        lim = 0.95 * cfg.tilt_max
        if tilt > lim:  # start inside the tilt cone
            h = float(np.hypot(ax[0], ax[1]))
            ax = np.array([ax[0] / h * math.sin(lim), ax[1] / h * math.sin(lim), math.cos(lim)])
        p["ax0"].value = ax / np.linalg.norm(ax)
        self.solves += 1
        try:
            self.prob.solve(solver=getattr(self._cp, cfg.solver), warm_start=True)
        except self._cp.error.SolverError:
            return None
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        V = self.var
        return PDGSolution(
            tf=tf,
            t=tk,
            r=np.array(V["r"].value),
            v=np.array(V["v"].value),
            u=np.array(V["u"].value),
            m=np.exp(np.array(V["z"].value)),
            a_aero=aero,
            cost=float(self.prob.value),
            status=self.prob.status,
        )

    def solve(
        self,
        r0,
        v0,
        m0: float,
        g,
        tf: float,
        aero_fn: AeroFn | None = None,
        iterations: int = 3,
        guess: PDGSolution | None = None,
    ) -> PDGSolution | None:
        """Fixed final time, with successive convexification of the aero acceleration."""
        n = self.n
        r0 = np.asarray(r0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        tk = np.linspace(0.0, tf, n + 1)
        if guess is not None:  # re-sample a previous plan onto this grid
            s = np.clip(tk * guess.tf / tf, 0, guess.tf)
            rr = np.column_stack([np.interp(s, guess.t, guess.r[:, j]) for j in range(3)])
            vv = np.column_stack([np.interp(s, guess.t, guess.v[:, j]) for j in range(3)])
            mm = np.interp(s, guess.t, guess.m)
            th = np.interp(s, guess.t, guess.thrust)
        else:  # straight-line, constant-deceleration guess
            f = 1.0 - tk / tf
            rr = r0[None, :] * (f * f)[:, None]
            vv = v0[None, :] * f[:, None]
            dv = float(np.linalg.norm(v0)) / tf + float(np.linalg.norm(g))
            mm = m0 - tk * dv * m0 / (self.cfg.isp * G0)
            th = mm * dv
        sol = None
        for _ in range(max(iterations, 1) if aero_fn is not None else 1):
            aero = np.zeros((n + 1, 3)) if aero_fn is None else aero_fn(rr, vv, mm, th)
            beta_m = self.cfg.mdot_offset / np.maximum(mm, 1.0)
            new = self._solve_fixed(r0, v0, m0, g, tf, aero, beta_m)
            if new is None:
                return None
            sol = new
            rr, vv, mm, th = sol.r, sol.v, sol.m, sol.thrust
        return sol

    def solve_free_tf(
        self,
        r0,
        v0,
        m0: float,
        g,
        tf_range: tuple[float, float],
        aero_fn: AeroFn | None = None,
        iterations: int = 2,
        grid: int = 6,
        refine: int = 5,
        guess: PDGSolution | None = None,
    ) -> PDGSolution | None:
        """Free final time: coarse grid over ``tf_range`` then golden-section refinement of
        the cost (fuel + miss penalty). Returns None if no grid time is feasible."""
        lo, hi = tf_range
        ts = np.linspace(lo, hi, grid)
        best: PDGSolution | None = None
        costs = []
        for tf in ts:
            sol = self.solve(r0, v0, m0, g, float(tf), aero_fn, iterations, guess)
            costs.append(math.inf if sol is None else sol.cost)
            if sol is not None and (best is None or sol.cost < best.cost):
                best = sol
        if best is None:
            return None
        k = int(np.argmin(costs))
        step = ts[1] - ts[0] if grid > 1 else 0.0
        a_, b_ = max(ts[k] - step, lo), min(ts[k] + step, hi)
        gr = 0.5 * (math.sqrt(5.0) - 1.0)
        cache: dict[float, PDGSolution | None] = {}

        def f(tf: float) -> float:
            sol = self.solve(r0, v0, m0, g, tf, aero_fn, iterations, best)
            cache[tf] = sol
            return math.inf if sol is None else sol.cost

        x1, x2 = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        f1, f2 = f(x1), f(x2)
        for _ in range(max(refine - 2, 0)):
            if f1 < f2:
                b_, x2, f2 = x2, x1, f1
                x1 = b_ - gr * (b_ - a_)
                f1 = f(x1)
            else:
                a_, x1, f1 = x1, x2, f2
                x2 = a_ + gr * (b_ - a_)
                f2 = f(x2)
        for sol in cache.values():
            if sol is not None and sol.cost < best.cost:
                best = sol
        return best
