"""Aerodynamics: axial drag + slender-body crossflow strip theory.

* Axial force uses Mach tables for nose-first and tail-first (engine-first) flow.
* Lateral force is integrated over ``stations`` strips along the hull. Each strip
  sees its own crossflow velocity (including the ``omega x r`` term), which yields
  the centre-of-pressure moment and pitch/yaw damping without separate
  coefficients. Every strip force opposes its local velocity, so aerodynamics can
  only remove energy (with zero wind) -- a property the tests check.
"""

from __future__ import annotations

import math

import numpy as np

from plume.config import AeroSpec, GeometrySpec


class Aero:
    def __init__(self, spec: AeroSpec, geometry: GeometrySpec):
        self.spec = spec
        self.enabled = spec.enabled
        r = geometry.radius
        self.ref_area = spec.reference_area if spec.reference_area else math.pi * r * r
        self._ca_nose = np.asarray(spec.ca_nose_first, dtype=float).T
        self._ca_tail = np.asarray(spec.ca_tail_first, dtype=float).T
        self._cdc = np.asarray(spec.crossflow_cd, dtype=float).T
        n = spec.stations
        L = geometry.length
        dz = L / n
        self.z = (np.arange(n) + 0.5) * dz
        width = np.full(n, geometry.diameter)
        if geometry.nose_length > 0:
            nose_start = L - geometry.nose_length
            in_nose = self.z > nose_start
            width[in_nose] = geometry.diameter * (L - self.z[in_nose]) / geometry.nose_length
        self.strip_area = width * dz  # projected side area per strip
        fins = spec.fins
        self.fin_cn = fins.cn_alpha * self.ref_area if fins and fins.count > 0 else 0.0
        self.fin_z = fins.z if fins else 0.0

    @property
    def cd_scale(self) -> float:
        return self.spec.cd_scale

    def axial_coefficient(self, mach: float, nose_first: bool) -> float:
        tab = self._ca_nose if nose_first else self._ca_tail
        return float(np.interp(mach, tab[0], tab[1])) * self.spec.cd_scale

    def crossflow_coefficient(self, mach_cross: float) -> float:
        return float(np.interp(mach_cross, self._cdc[0], self._cdc[1])) * self.spec.cd_scale

    def forces(
        self,
        v_air_body: np.ndarray,
        omega_body: np.ndarray,
        cg_z: float,
        rho: float,
        sound_speed: float,
    ):
        """Aerodynamic force and torque (about the CG), body frame.

        ``v_air_body`` is the velocity of the CG relative to the air, in body axes.
        Returns (force, torque, dynamic pressure, Mach).
        """
        vx, vy, vz = float(v_air_body[0]), float(v_air_body[1]), float(v_air_body[2])
        speed2 = vx * vx + vy * vy + vz * vz
        q = 0.5 * rho * speed2
        mach = math.sqrt(speed2) / sound_speed if sound_speed > 0 else 0.0
        if not self.enabled or rho <= 0.0 or (speed2 == 0.0 and not np.any(omega_body)):
            return np.zeros(3), np.zeros(3), q, mach

        # axial (along body z); nose-first when moving toward +z relative to the air
        ca = self.axial_coefficient(mach, vz >= 0.0)
        f_axial = -0.5 * rho * self.ref_area * ca * vz * abs(vz)

        # crossflow strips: local lateral velocity = v_lat + omega x (z - cg) e_z
        dz = self.z - cg_z
        wx, wy = float(omega_body[0]), float(omega_body[1])
        ux = vx + wy * dz
        uy = vy - wx * dz
        un = np.sqrt(ux * ux + uy * uy)
        mach_c = math.sqrt(vx * vx + vy * vy) / sound_speed if sound_speed > 0 else 0.0
        k = -0.5 * rho * self.crossflow_coefficient(mach_c) * self.strip_area * un
        fx = k * ux
        fy = k * uy
        force = np.array([fx.sum(), fy.sum(), f_axial])
        # torque of lateral strip forces about the CG: (0,0,dz) x (fx,fy,0)
        torque = np.array([-(dz * fy).sum(), (dz * fx).sum(), 0.0])
        if self.fin_cn > 0:
            # fins: linear normal force q * CN_alpha * alpha * A at the fin CP
            # (alpha ~ u / |v| with u the local crossflow) -> -0.5 rho |v| CN A u
            fz = self.fin_z - cg_z
            u_x = vx + wy * fz
            u_y = vy - wx * fz
            k_f = -0.5 * rho * math.sqrt(speed2) * self.fin_cn
            force[0] += k_f * u_x
            force[1] += k_f * u_y
            torque[0] -= fz * k_f * u_y
            torque[1] += fz * k_f * u_x
        return force, torque, q, mach
