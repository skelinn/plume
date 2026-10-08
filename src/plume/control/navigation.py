"""Navigation: a 15-state error-state extended Kalman filter (INS/GNSS/baro/radar).

The flight software never sees the simulator's true state in high fidelity. Instead:

* the strapdown INS integrates the IMU (``plume.physics.sensors.Imu``) in the simulator's
  world frame, including gravity (the same field model the flight software carries) and,
  on a rotating Earth, the Coriolis/centrifugal terms and the Earth rate the gyros sense;
* an error-state EKF with states [dp, dv, dtheta, b_gyro, b_accel] (world-frame position,
  velocity and attitude errors, Gauss-Markov sensor biases) corrects the INS with GNSS
  position/velocity (latency compensated), barometric altitude and, near the ground, the
  radar altimeter.

``Navigator.estimate()`` returns a ``State`` built from the estimate. Quantities the
vehicle measures directly keep their sensed values: propellant (gauging), thrust
(chamber pressure), gimbal and fin positions (resolvers), leg contact switches and the
accelerometer g-load. Air data come from the estimated velocity and the forecast wind
(there is no air-data probe). Assumptions: the IMU sits at the CG (no lever-arm
effects), terrain under the vehicle is known (onboard map).
"""

from __future__ import annotations

import math

import numpy as np

from plume.physics.sensors import Barometer, Gnss, Imu, RadarAltimeter

DEG = math.pi / 180.0


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _exp(phi: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        return np.eye(3) + _skew(phi)
    k = _skew(phi / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def _log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (inverse of ``_exp``)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    return th / (2.0 * math.sin(th)) * w


def _orthonormal(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    return u @ vt


class Navigator:
    def __init__(
        self,
        sim,
        seed: int | None = None,
        init_pos_sigma: float = 2.0,
        init_vel_sigma: float = 0.05,
        init_att_sigma_deg: tuple[float, float, float] = (0.05, 0.05, 0.3),
    ):
        self.sim = sim
        spec = sim.vehicle.sensors
        rng = np.random.default_rng(None if seed is None else seed + 104_729)
        self.rng = rng
        self.imu = Imu(spec.imu, rng)
        self.gnss = Gnss(spec.gnss, rng)
        self.baro = Barometer(spec.baro, rng)
        self.radar = RadarAltimeter(spec.radar, rng) if spec.radar is not None else None
        self.spec = spec
        g = sim.gravity
        self._rot = bool(getattr(g, "rotating", False))
        self._w_e = np.asarray(g.omega_w, dtype=float) if self._rot else np.zeros(3)

        st = sim.state
        att0 = np.array(init_att_sigma_deg) * DEG
        # initial alignment errors (pad alignment: levelling + gyrocompass / survey)
        self.p = st.com + init_pos_sigma * rng.standard_normal(3)
        self.v = st.vel_com + init_vel_sigma * rng.standard_normal(3)
        self.R = _exp(att0 * rng.standard_normal(3)) @ st.rot
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.omega = st.omega.copy()
        s = spec.imu
        sg0 = math.hypot(s.gyro_bias_deg_h, s.gyro_bias_instability_deg_h) * DEG / 3600.0
        sa0 = math.hypot(s.accel_bias_mg, s.accel_bias_instability_mg) * 1e-3 * 9.80665
        self.P = np.diag(
            [init_pos_sigma**2] * 3
            + [init_vel_sigma**2] * 3
            + list(att0**2)
            + [sg0**2] * 3
            + [sa0**2] * 3
        )
        self._prev_v = st.vel_com.copy()
        self._prev_R = st.rot.copy()
        self._prev_t = st.t
        self._f_w = np.zeros(3)
        self.history: list[tuple[float, float, float, float]] = []  # t, |dp|, |dv|, att err deg

    # ------------------------------------------------------------------ truth -> IMU
    def _true_imu(self, dt: float):
        sim = self.sim
        st = sim.state
        g = sim.gravity
        r_mid = st.com - 0.5 * dt * st.vel_com
        v_mid = 0.5 * (st.vel_com + self._prev_v)
        a_rel = (st.vel_com - self._prev_v) / dt
        f_w = a_rel - g.accel(r_mid)
        if self._rot:
            f_w = f_w - g.fictitious_accel(r_mid, v_mid)
        R = st.rot
        # the IMU delivers delta-angles: the average body rate over the interval
        w_avg = _log(self._prev_R.T @ R) / dt
        R_mid = self._prev_R @ _exp(0.5 * w_avg * dt)
        w_in = w_avg + R_mid.T @ self._w_e
        return R_mid.T @ f_w, w_in

    # ------------------------------------------------------------------ filter
    def update(self, dt: float) -> None:
        """Advance by one flight-software cycle (after the simulator stepped ``dt``)."""
        if dt <= 0:
            return
        sim = self.sim
        f_b_true, w_true = self._true_imu(dt)
        m = self.imu.measure(w_true, f_b_true, dt)
        self._prev_v = sim.state.vel_com.copy()
        self._prev_R = sim.state.rot.copy()
        self._propagate(m, dt)
        self._measurements(dt)
        self._log()

    def _propagate(self, m, dt: float) -> None:
        g = self.sim.gravity
        s = self.spec.imu
        R0 = self.R
        w_rel = m.gyro - self.bg - R0.T @ self._w_e
        self.omega = w_rel
        self.R = _orthonormal(R0 @ _exp(w_rel * dt))
        R_mid = R0 @ _exp(0.5 * w_rel * dt)
        f_w = R_mid @ (m.accel - self.ba)
        self._f_w = f_w
        a = f_w + g.accel(self.p)
        if self._rot:
            a = a + g.fictitious_accel(self.p, self.v)
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt

        # error-state transition (first order)
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -_skew(f_w)
        F[3:6, 12:15] = -R_mid
        F[6:9, 9:12] = -R_mid
        tau = s.bias_correlation_s
        F[9:12, 9:12] = -np.eye(3) / tau
        F[12:15, 12:15] = -np.eye(3) / tau
        Phi = np.eye(15) + F * dt
        arw = s.gyro_arw_deg_rt_h * DEG / 60.0
        vrw = s.accel_vrw_m_s_rt_h / 60.0
        # scale-factor / misalignment errors act like extra, force-proportional noise
        sf = (s.scale_factor_ppm * 1e-6 + s.misalignment_mrad * 1e-3) * float(np.linalg.norm(f_w))
        q_bg = 2.0 * (s.gyro_bias_instability_deg_h * DEG / 3600.0) ** 2 / tau
        q_ba = 2.0 * (s.accel_bias_instability_mg * 1e-3 * 9.80665) ** 2 / tau
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * (vrw**2 + sf**2 * 1.0) * dt
        Q[6:9, 6:9] = np.eye(3) * arw**2 * dt
        Q[9:12, 9:12] = np.eye(3) * q_bg * dt
        Q[12:15, 12:15] = np.eye(3) * q_ba * dt
        Q[0:3, 0:3] = np.eye(3) * 1e-4 * dt  # position model error (lever arm, CG motion)
        self.P = Phi @ self.P @ Phi.T + Q

    def _correct(
        self, z: np.ndarray, h: np.ndarray, H: np.ndarray, Rm: np.ndarray, gate: float = 25.0
    ):
        y = z - h
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > gate * len(y):  # chi-square gate against outliers
            return
        K = self.P @ H.T @ Si
        dx = K @ y
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = _orthonormal(_exp(dx[6:9]) @ self.R)
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]

    def _measurements(self, dt: float) -> None:
        sim = self.sim
        st = sim.state
        g = sim.gravity
        fix = self.gnss.step(sim.t, st.com, st.vel_com, st.altitude, dt)
        if fix is not None:
            lag = sim.t - fix.t
            a = self._f_w + g.accel(self.p)
            p_then = self.p - self.v * lag + 0.5 * a * lag * lag
            v_then = self.v - a * lag
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            # the receiver reports in local ENU; its errors are horizontal/vertical there
            E = g.enu_at(st.com) if hasattr(g, "enu_at") else np.eye(3)
            Rp = E @ np.diag(fix.sigma_pos**2) @ E.T
            Rm = np.zeros((6, 6))
            Rm[0:3, 0:3] = Rp
            Rm[3:6, 3:6] = np.eye(3) * fix.sigma_vel**2
            self._correct(
                np.concatenate([fix.pos, fix.vel]), np.concatenate([p_then, v_then]), H, Rm
            )
        up = g.up(self.p)
        p_amb = sim.atmosphere.at(st.altitude).pressure
        hb = self.baro.measure(p_amb)
        if hb is not None:
            b = self.spec.baro
            H = np.zeros((1, 15))
            H[0, 0:3] = up
            # off-standard-day bias grows with altitude: weight accordingly
            r = b.sigma_m**2 + b.bias_m**2 + (0.03 * max(hb, 0.0)) ** 2
            self._correct(np.array([hb]), np.array([g.altitude(self.p)]), H, np.array([[r]]))
        if self.radar is not None:
            h_true = st.altitude - sim._ground_height(st.com)
            hr = self.radar.measure(h_true, st.tilt)
            if hr is not None:
                s = self.spec.radar
                h_est = g.altitude(self.p) - sim._ground_height(self.p)
                H = np.zeros((1, 15))
                H[0, 0:3] = up
                r = (s.sigma_m + s.sigma_frac * hr) ** 2
                self._correct(np.array([hr]), np.array([h_est]), H, np.array([[r]]))

    def _log(self) -> None:
        st = self.sim.state
        dR = self.R @ st.rot.T
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        self.history.append(
            (
                self.sim.t,
                float(np.linalg.norm(self.p - st.com)),
                float(np.linalg.norm(self.v - st.vel_com)),
                ang,
            )
        )

    # ------------------------------------------------------------------ estimated state
    def estimate(self):
        """The ``State`` the flight software acts on."""
        from dataclasses import replace

        from plume.physics.sim import _cross_z, mat_to_quat

        sim = self.sim
        st = sim.state
        g = sim.gravity
        R = self.R
        com = self.p
        cg = sim.mp.cg
        up = g.up(com)
        axis = R[:, 2]
        tilt = math.acos(max(-1.0, min(1.0, float(axis @ up))))
        alt = g.altitude(com)
        legs = sim.vehicle.legs
        h_cg = alt - sim._ground_height(com)
        agl = h_cg - (st.cg_z + legs.height) * math.cos(tilt) - legs.span * math.sin(tilt)
        wind = g.local_to_frame(com, sim.wind.mean_at(alt))
        atm = sim.atmosphere.at(alt)
        v_air = float(np.linalg.norm(self.v - wind))
        return replace(
            st,
            pos=com - R @ cg,
            quat=mat_to_quat(R),
            vel=self.v - R @ _cross_z(self.omega, st.cg_z),
            omega=self.omega.copy(),
            com=com.copy(),
            vel_com=self.v.copy(),
            altitude=alt,
            agl=agl if st.legs_down == 0 else st.agl,
            mach=v_air / atm.speed_of_sound if atm.speed_of_sound > 0 else 0.0,
            q_dyn=0.5 * atm.density * v_air * v_air,
            wind=wind,
            tilt=tilt,
            up=up,
        )

    def summary(self) -> dict:
        if not self.history:
            return {}
        h = np.array(self.history)
        return {
            "pos_err_max_m": float(h[:, 1].max()),
            "pos_err_final_m": float(h[-1, 1]),
            "vel_err_max_m_s": float(h[:, 2].max()),
            "att_err_max_deg": float(h[:, 3].max()),
            "pos_err_rms_m": float(np.sqrt(np.mean(h[:, 1] ** 2))),
        }


def navigation_mode(world) -> str:
    nav = getattr(world, "navigation", "auto")
    if nav == "auto":
        return "ekf" if world.fidelity == "high" else "truth"
    return nav
