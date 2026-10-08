"""Grid fins: deployable lattice control surfaces near the top of the hull.

Fin ``k`` sits at azimuth ``theta_k`` on a ring of radius ``r`` at height ``z``. Its
force acts along the local tangential direction ``t_k = (-sin, cos, 0)`` (body
frame). Treating the lattice as a flat plate deflected by ``delta`` about its radial
hinge, the normal force is

    F_t = 0.5 rho |V| CN_alpha A (delta * v_ax - u_t)

where ``v_ax`` is the axial and ``u_t`` the tangential component of the fin's
velocity relative to the air. With ``delta = 0`` the fins are a passive stabiliser
(the force opposes the local crossflow); the control effect ``delta * v_ax`` flips
sign automatically in engine-first (reversed) flow, as real grid fins do. Each fin
also adds axial drag ``0.5 rho A (cd0 + cd_delta delta^2) |V| v_ax``.

Opposite fins deflected the same way pitch/yaw the vehicle; all fins deflected the
same way roll it.
"""

from __future__ import annotations

import math

import numpy as np

from plume.config import GeometrySpec, GridFinSpec


class GridFins:
    def __init__(self, spec: GridFinSpec, geometry: GeometrySpec):
        self.spec = spec
        n = spec.count
        self.n = n
        self.radius = spec.radius if spec.radius is not None else geometry.radius + 0.5 * spec.span
        self.area = spec.area if spec.area is not None else spec.span * spec.chord
        th = 2 * math.pi * np.arange(n) / n
        self.pos = np.column_stack(
            [self.radius * np.cos(th), self.radius * np.sin(th), np.full(n, spec.z)]
        )
        self.tan = np.column_stack([-np.sin(th), np.cos(th), np.zeros(n)])
        self.max_defl = math.radians(spec.max_deflection_deg)
        self.rate = math.radians(spec.rate_deg_s)
        self._mach = None  # (mach grid, cn_alpha multiplier, cd0 multiplier) when enabled
        self.reset()

    def enable_mach_effects(self, open_area_ratio: float | None = None) -> None:
        """High fidelity: transonic choking / supersonic lattice behaviour (see
        ``plume.physics.aero_gen.gridfins``) scales ``cn_alpha`` and ``cd0`` with Mach."""
        from plume.physics.aero_gen.gridfins import gridfin_cd0, gridfin_cn_alpha

        oar = open_area_ratio if open_area_ratio is not None else self.spec.open_area_ratio
        m = np.linspace(0.0, 8.0, 801)
        self._mach = (m, gridfin_cn_alpha(m, oar), gridfin_cd0(m, oar))

    def mach_factors(self, mach: float) -> tuple[float, float]:
        if self._mach is None:
            return 1.0, 1.0
        m, cn, cd = self._mach
        return float(np.interp(mach, m, cn)), float(np.interp(mach, m, cd))

    def reset(self) -> None:
        self.delta = np.zeros(self.n)
        self.cmd = np.zeros(self.n)
        self.deployed = self.spec.deploy == "always"

    def deploy(self, deployed: bool = True) -> None:
        self.deployed = deployed
        if not deployed:
            self.cmd[:] = 0.0

    def command(self, normalised) -> None:
        """Deflection commands in [-1, 1] (fraction of the maximum deflection)."""
        self.cmd = np.clip(np.asarray(normalised, dtype=float), -1.0, 1.0) * self.max_defl

    def update(self, dt: float) -> None:
        step = np.clip(self.cmd - self.delta, -self.rate * dt, self.rate * dt)
        self.delta = np.clip(self.delta + step, -self.max_defl, self.max_defl)

    def _lever(self, cg_z: float) -> np.ndarray:
        return self.pos - np.array([0.0, 0.0, cg_z])

    def forces(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        cg_z: float,
        rho: float,
        delta: np.ndarray | None = None,
        sound_speed: float | None = None,
    ):
        """Force and torque (about the CG) in body axes (``delta`` overrides the current
        deflections, e.g. for prediction). With Mach effects enabled and ``sound_speed``
        given, the slope and drag follow the transonic/supersonic multipliers."""
        if not self.deployed or rho <= 0.0:
            return np.zeros(3), np.zeros(3)
        r = self._lever(cg_z)
        v_loc = v_air_body[None, :] + np.cross(omega_body[None, :], r)  # (n, 3)
        speed = float(np.linalg.norm(v_air_body))
        if speed < 1e-6:
            return np.zeros(3), np.zeros(3)
        s = self.spec
        v_ax = v_loc[:, 2]
        u_t = np.einsum("ij,ij->i", v_loc, self.tan)
        k = 0.5 * rho * speed * self.area
        d = self.delta if delta is None else delta
        kcn, kcd = (1.0, 1.0)
        if self._mach is not None and sound_speed:
            kcn, kcd = self.mach_factors(speed / sound_speed)
        f_t = k * s.cn_alpha * kcn * (d * v_ax - u_t)
        f_ax = -k * (s.cd0 * kcd + s.cd_delta * d**2) * v_ax
        f = f_t[:, None] * self.tan + f_ax[:, None] * np.array([0.0, 0.0, 1.0])
        return f.sum(axis=0), np.cross(r, f).sum(axis=0)

    def torque_matrix(
        self, v_air_body: np.ndarray, cg_z: float, rho: float, sound_speed: float | None = None
    ) -> np.ndarray:
        """3 x n: body torque per radian of deflection of each fin (current flow)."""
        if not self.deployed or rho <= 0.0:
            return np.zeros((3, self.n))
        speed = float(np.linalg.norm(v_air_body))
        kcn = self.mach_factors(speed / sound_speed)[0] if sound_speed else 1.0
        k = 0.5 * rho * speed * self.area * self.spec.cn_alpha * kcn * float(v_air_body[2])
        return (np.cross(self._lever(cg_z), self.tan) * k).T

    def capability(
        self, v_air_body: np.ndarray, cg_z: float, rho: float, sound_speed: float | None = None
    ) -> np.ndarray:
        """Approximate max body torque per axis at full deflection."""
        B = self.torque_matrix(v_air_body, cg_z, rho, sound_speed)
        return np.abs(B).sum(axis=1) * self.max_defl

    def allocate(
        self,
        torque: np.ndarray,
        v_air_body: np.ndarray,
        cg_z: float,
        rho: float,
        sound_speed: float | None = None,
    ):
        """Deflections (normalised) that best produce ``torque``; returns (cmd, achieved torque)."""
        B = self.torque_matrix(v_air_body, cg_z, rho, sound_speed)
        if not np.any(B):
            return np.zeros(self.n), np.zeros(3)
        lam = 1e-6 * float(np.abs(B).max()) ** 2
        delta = B.T @ np.linalg.solve(B @ B.T + lam * np.eye(3), torque)
        peak = float(np.abs(delta).max())
        if peak > self.max_defl:
            delta *= self.max_defl / peak  # keep the torque direction, scale down
        return delta / self.max_defl, B @ delta
