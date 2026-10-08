"""Drone ship: a landing barge moving in a seaway.

* **Sea state.** Long- or short-crested irregular waves from a JONSWAP (or
  Pierson-Moskowitz, ``gamma = 1``) spectrum for a significant wave height ``Hs`` and
  peak period ``Tp`` (DNV-RP-C205 Sec. 3.5.5), synthesised as a sum of linear deep-water
  components with random phases (frequencies jittered inside their bins so the record
  does not repeat).
* **Ship response.** Heave and pitch from the closed-form transfer functions of a
  homogeneously loaded box barge (Jensen, Mansour & Olsen 2004); roll as a single-DOF
  oscillator driven by the beam-averaged wave slope; surge/sway from the hull-averaged
  orbital motion; a small yaw. Station keeping (thrusters holding position) is a slow,
  smooth wander of position and heading.
* **In the simulator** the deck is a box on its own body with a free joint that is
  driven *kinematically*: position and velocity are prescribed every physics step, so
  contacts see the true deck velocity (friction carries the vehicle with the deck).
  The ship is ~10^4 times heavier than the lander; its reaction to the vehicle is
  neglected.

Equations, sources and limitations: ``docs/models/ship_landing.md``.
"""

from __future__ import annotations

import math

import numpy as np

from plume.config import ShipSpec
from plume.constants import G0


# --------------------------------------------------------------------------- spectrum
def jonswap(omega, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP spectral density S(omega) (m^2 s/rad), DNV-RP-C205 eq. 3.5.5.
    ``gamma = 1`` gives the Pierson-Moskowitz spectrum."""
    w = np.asarray(omega, dtype=float)
    wp = 2.0 * math.pi / tp
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s_pm = (5.0 / 16.0) * hs * hs * wp**4 * w**-5.0 * np.exp(-1.25 * (w / wp) ** -4.0)
        sigma = np.where(w <= wp, 0.07, 0.09)
        a_g = 1.0 - 0.287 * math.log(gamma) if gamma > 0 else 1.0
        s = a_g * s_pm * gamma ** np.exp(-0.5 * ((w - wp) / (sigma * wp)) ** 2)
    return np.where(w > 0, np.nan_to_num(s), 0.0)


def _sinc(x):
    """sin(x)/x, safe at 0."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def _moment_sinc(x):
    """3 (sin x - x cos x) / x^3 -> 1 at x = 0 (slope averaged over a length)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-3
    xn = x[nz]
    out[nz] = 3.0 * (np.sin(xn) - xn * np.cos(xn)) / xn**3
    small = ~nz
    out[small] = 1.0 - x[small] ** 2 / 10.0
    return out


def barge_heave_pitch(omega, beta, L: float, B: float, T: float):
    """Heave (m/m) and pitch (rad/m) amplitude per unit wave amplitude for a box barge at
    zero speed, Jensen, Mansour & Olsen (2004), eqs. 5-17. ``beta`` = wave heading
    relative to the bow (pi = head seas). Signed: a negative value is 180 deg out of
    phase with the wave elevation (heave) or slope (pitch) at midship."""
    w = np.asarray(omega, dtype=float)
    k = w * w / G0
    ke = np.abs(k * np.cos(beta))
    alpha = 1.0  # zero forward speed
    A = 2.0 * np.sin(k * B * alpha**2 / 2.0) * np.exp(-k * T * alpha**2)
    kb = np.maximum(k * B, 1e-12)
    f = np.sqrt((1.0 - k * T) ** 2 + (A * A / (kb * alpha**3)) ** 2)
    kappa = np.exp(-ke * T)
    x = ke * L / 2.0
    F = kappa * f * _sinc(x)
    # 24/((ke L)^2 L) [sin(x) - x cos(x)] = ke * moment_sinc(x)
    G = kappa * f * ke * _moment_sinc(x)
    eta = 1.0 / np.sqrt((1.0 - 2.0 * k * T * alpha**2) ** 2 + (A * A / (kb * alpha**2)) ** 2)
    return eta * F, eta * G


class SeaState:
    """Wave components ``a_i cos(omega_i t - k_i (x cos th_i + y sin th_i) + phi_i)``."""

    def __init__(self, spec, mean_dir: float, seed: int | None = 0):
        self.spec = spec
        self.mean_dir = mean_dir  # propagation direction (ENU, rad from east, CCW)
        self.reset(seed)

    def reset(self, seed: int | None) -> None:
        sp = self.spec
        rng = np.random.default_rng(seed)
        wp = 2.0 * math.pi / sp.tp
        n = sp.components
        edges = np.linspace(0.55 * wp, 3.5 * wp, n + 1)
        dw = np.diff(edges)
        gamma = 1.0 if sp.spectrum == "pierson_moskowitz" else sp.gamma
        # directional spreading cos^(2s)(th - th0), discretised into `directions` bins
        if sp.spreading_s is None or sp.directions <= 1:
            dirs, wts = np.array([0.0]), np.array([1.0])
        else:
            dirs = np.linspace(-math.pi / 2, math.pi / 2, sp.directions + 2)[1:-1]
            wts = np.cos(dirs) ** (2.0 * sp.spreading_s)
            wts /= wts.sum()
        amp, om, th, ph = [], [], [], []
        for d, wt in zip(dirs, wts, strict=True):
            # frequencies jittered inside their bins, independently per direction: no
            # repeat period, and no two components share a frequency (otherwise they
            # would interfere coherently at a point and bias the local variance)
            w = edges[:-1] + dw * rng.uniform(0.05, 0.95, n)
            S = jonswap(w, sp.hs, sp.tp, gamma)
            amp.append(np.sqrt(2.0 * S * dw * wt))
            om.append(w)
            th.append(np.full(n, self.mean_dir + d))
            ph.append(rng.uniform(0.0, 2.0 * math.pi, n))
        self.a = np.concatenate(amp)
        self.w = np.concatenate(om)
        self.k = self.w * self.w / G0
        self.theta = np.concatenate(th)
        self.phase = np.concatenate(ph)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)

    def elevation(self, x, y, t) -> np.ndarray:
        x = np.asarray(x, dtype=float)[..., None]
        y = np.asarray(y, dtype=float)[..., None]
        arg = self.w * t - self.k * (x * self.cos_t + y * self.sin_t) + self.phase
        return np.sum(self.a * np.cos(arg), axis=-1)

    @property
    def hs_spectral(self) -> float:
        """Significant wave height of the synthesised components, 4 sqrt(m0)."""
        return 4.0 * math.sqrt(float(np.sum(0.5 * self.a**2)))


# --------------------------------------------------------------------------- rotations
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _qaxis(axis: int, ang: float):
    q = np.zeros(4)
    q[0] = math.cos(ang / 2)
    q[1 + axis] = math.sin(ang / 2)
    return q


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------- ship
class ShipModel:
    """Kinematic drone-ship motion and its MuJoCo deck body.

    Ship body frame: origin at the waterline amidships (the centre of rotation), +x to
    the bow, +y to port, +z up; the deck top is at ``z = freeboard``. ``motion(t)``
    returns the pose and velocity of that origin in the (flat, ENU) world frame.
    """

    def __init__(self, spec: ShipSpec, seed: int | None = 0):
        self.spec = spec
        # compass heading (clockwise from north) -> yaw from east, CCW
        self.yaw0 = math.pi / 2 - math.radians(spec.heading_deg)
        self.mean_pos = np.array([spec.position[0], spec.position[1], 0.0])
        sea = spec.sea
        self.sea = SeaState(sea, self.yaw0 + math.radians(sea.direction_deg), seed)
        self.target_local = np.array([spec.target_offset[0], spec.target_offset[1], spec.freeboard])
        self.qadr = self.dadr = None
        self.eq_id = None
        self.t = 0.0
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = 0) -> None:
        self.sea.reset(seed)
        sp = self.spec
        sea = self.sea
        L, B, T = sp.length, sp.hull_beam, sp.draft
        beta = sea.theta - self.yaw0  # encounter angle of each component (pi = head seas)
        cb, sb = np.cos(beta), np.sin(beta)
        heave, pitch = barge_heave_pitch(sea.w, beta, L, B, T)
        k = sea.k
        kappa = np.exp(-k * T)
        # heave in phase with the elevation at midship, pitch with the slope along x
        self.h_amp = sea.a * heave
        self.p_amp = sea.a * pitch * np.sign(cb)
        # roll: 1-DOF oscillator driven by the beam-averaged slope k sin(beta)
        gm = T / 2.0 + B * B / (12.0 * T) - sp.kg
        kxx = sp.roll_gyradius
        wn = math.sqrt(G0 * max(gm, 0.1)) / kxx
        self.roll_period = 2.0 * math.pi / wn
        r = sea.w / wn
        zeta = sp.roll_damping
        H = 1.0 / np.sqrt((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2)
        self.r_eps = np.arctan2(2.0 * zeta * r, 1.0 - r * r)
        self.r_amp = sea.a * k * sb * _moment_sinc(k * np.abs(sb) * B / 2.0) * kappa * H
        # surge / sway: hull-averaged orbital displacement; yaw: differential sway
        self.x_amp = sea.a * cb * kappa * _sinc(k * np.abs(cb) * L / 2.0)
        self.y_amp = sea.a * sb * kappa * _sinc(k * np.abs(sb) * B / 2.0)
        self.yaw_amp = 0.5 * sea.a * k * sb * cb * kappa * _moment_sinc(k * np.abs(cb) * L / 2.0)
        # station keeping: smooth wander (3 sinusoids per channel, periods P, 0.61 P, 0.37 P)
        rng = np.random.default_rng(None if seed is None else seed + 7919)
        P = sp.station_keeping_period
        self.dr_w = 2.0 * math.pi / (P * np.array([1.0, 0.61, 0.37]))
        self.dr_ph = rng.uniform(0, 2 * math.pi, (3, 3))  # x, y, heading
        a = math.sqrt(2.0 / 3.0)
        self.dr_amp = np.array(
            [sp.station_keeping_sigma * a] * 2 + [math.radians(sp.heading_sigma_deg) * a]
        )
        self.t = 0.0
        self._cache(0.0)

    # ------------------------------------------------------------------ motion
    def _channels(self, t: float):
        sea = self.sea
        psi = (
            sea.w * t
            + sea.phase
            - sea.k * (sea.cos_t * self.mean_pos[0] + sea.sin_t * self.mean_pos[1])
        )
        c, s = np.cos(psi), np.sin(psi)
        w = sea.w
        heave = float(self.h_amp @ c)
        heave_d = float(-(self.h_amp * w) @ s)
        pitch = float(-(self.p_amp @ s))
        pitch_d = float(-(self.p_amp * w) @ c)
        sr, cr = np.sin(psi - self.r_eps), np.cos(psi - self.r_eps)
        roll = float(self.r_amp @ sr)
        roll_d = float((self.r_amp * w) @ cr)
        surge = float(self.x_amp @ s)
        surge_d = float((self.x_amp * w) @ c)
        sway = float(self.y_amp @ s)
        sway_d = float((self.y_amp * w) @ c)
        yaw = float(self.yaw_amp @ c)
        yaw_d = float(-(self.yaw_amp * w) @ s)
        arg = self.dr_w[None, :] * t + self.dr_ph
        dr = self.dr_amp * np.sin(arg).sum(axis=1)
        dr_d = self.dr_amp * (self.dr_w[None, :] * np.cos(arg)).sum(axis=1)
        return (
            np.array([surge, sway, heave, roll, pitch, yaw]),
            np.array([surge_d, sway_d, heave_d, roll_d, pitch_d, yaw_d]),
            dr,
            dr_d,
        )

    def motion(self, t: float):
        """(pos, quat [w,x,y,z], vel (world), omega (body)) of the ship origin."""
        q6, d6, dr, dr_d = self._channels(t)
        psi = self.yaw0 + dr[2] + q6[5]
        psi_d = dr_d[2] + d6[5]
        Rz = _rz(psi)
        pos = self.mean_pos + np.array([dr[0], dr[1], 0.0]) + Rz @ np.array([q6[0], q6[1], 0.0])
        pos[2] = q6[2]
        # d/dt (Rz s) = Rz s' + psi_d (e_z x Rz s)
        s_h = Rz @ np.array([q6[0], q6[1], 0.0])
        vel = np.array([dr_d[0], dr_d[1], 0.0]) + Rz @ np.array([d6[0], d6[1], 0.0])
        vel += psi_d * np.array([-s_h[1], s_h[0], 0.0])
        vel[2] = d6[2]
        roll, pitch = q6[3], q6[4]
        quat = _qmul(_qmul(_qaxis(2, psi), _qaxis(1, pitch)), _qaxis(0, roll))
        # body angular velocity of R = Rz(psi) Ry(pitch) Rx(roll)
        Rx, Ry = _rx(roll), _ry(pitch)
        omega = (
            np.array([d6[3], 0.0, 0.0])
            + Rx.T @ np.array([0.0, d6[4], 0.0])
            + Rx.T @ Ry.T @ np.array([0.0, 0.0, psi_d])
        )
        return pos, quat, vel, omega

    @staticmethod
    def _rot(quat) -> np.ndarray:
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _cache(self, t: float) -> None:
        self.t = t
        pos, quat, vel, omega = self.motion(t)
        self.pos, self.quat, self.vel, self.omega_b = pos, quat, vel, omega
        self.R = self._rot(quat)
        self.omega_w = self.R @ omega

    def target_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Deck target position and velocity at time ``t`` (does not touch the cache)."""
        pos, quat, vel, omega = self.motion(t)
        R = self._rot(quat)
        r = R @ self.target_local
        return pos + r, vel + np.cross(R @ omega, r)

    # ------------------------------------------------------------------ queries
    def point_velocity(self, p) -> np.ndarray:
        return self.vel + np.cross(self.omega_w, np.asarray(p, dtype=float) - self.pos)

    def target(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and velocity of the deck landing target (cached time)."""
        p = self.pos + self.R @ self.target_local
        return p, self.point_velocity(p)

    def deck_normal(self) -> np.ndarray:
        return self.R[:, 2].copy()

    def on_deck(self, p) -> bool:
        lp = self.R.T @ (np.asarray(p, dtype=float) - self.pos)
        sp = self.spec
        return abs(lp[0]) <= 0.5 * sp.length and abs(lp[1]) <= 0.5 * sp.deck_width

    def ground_height(self, p) -> float:
        """Deck height below ``p`` when over the deck, else mean sea level (0)."""
        p = np.asarray(p, dtype=float)
        n = self.R[:, 2]
        c = self.pos + self.R[:, 2] * self.spec.freeboard
        z = c[2] - (n[0] * (p[0] - c[0]) + n[1] * (p[1] - c[1])) / max(n[2], 1e-6)
        if self.on_deck(np.array([p[0], p[1], z])):
            return float(z)
        return 0.0

    # ------------------------------------------------------------------ MuJoCo
    def mjcf(self, fmt, priority: bool = False) -> tuple[str, str]:
        """(deck body for <worldbody>, <equality> block for the hold-down clamp).
        ``priority``: the deck's friction and contact softness govern pad contacts (as
        for the ground with crushable legs)."""
        _f, _v = fmt
        sp = self.spec
        L, W, fb = sp.length, sp.deck_width, sp.freeboard
        m = sp.mass
        th = 1.0
        ixx = m * (W * W + 4 * fb * fb) / 12.0
        iyy = m * (L * L + 4 * fb * fb) / 12.0
        izz = m * (L * L + W * W) / 12.0
        mu = sp.deck_friction
        prio = ' priority="1"' if priority else ""
        body = (
            f'<body name="ship" pos="{_v(*self.mean_pos)}">'
            '<freejoint name="ship_root"/>'
            f'<inertial pos="0 0 0" mass="{_f(m)}" diaginertia="{_v(ixx, iyy, izz)}"/>'
            f'<geom name="ground_deck" type="box" pos="0 0 {_f(fb - th / 2)}" '
            f'size="{_v(L / 2, W / 2, th / 2)}" friction="{_f(mu)} 0.005 0.0001" '
            f'solref="0.02 1"{prio} rgba="0.25 0.25 0.27 1"/>'
            "</body>"
        )
        eq = (
            '<equality><weld name="clamp" body1="ship" body2="rocket" active="false" '
            'solref="0.02 1"/></equality>'
        )
        return body, eq

    def bind(self, model) -> None:
        jid = model.joint("ship_root").id
        self.qadr = int(model.jnt_qposadr[jid])
        self.dadr = int(model.jnt_dofadr[jid])
        self.bid = model.body("ship").id
        self.eq_id = model.equality("clamp").id

    def drive(self, data, t: float) -> None:
        """Prescribe the deck pose and velocity at time ``t``."""
        self._cache(t)
        qa, da = self.qadr, self.dadr
        data.qpos[qa : qa + 3] = self.pos
        data.qpos[qa + 3 : qa + 7] = self.quat
        data.qvel[da : da + 3] = self.vel
        data.qvel[da + 3 : da + 6] = self.omega_b

    def clamp(self, model, data, rocket_bid: int) -> None:
        """Engage the hold-down clamp: weld the vehicle to the deck where it stands."""
        import mujoco

        Rr = data.xmat[rocket_bid].reshape(3, 3)
        pr = data.xpos[rocket_bid]
        # relpose: body2 (rocket) pose in the body1 (ship) frame
        rel_p = self.R.T @ (pr - self.pos)
        qr = np.zeros(4)
        mujoco.mju_mat2Quat(qr, np.ascontiguousarray(self.R.T @ Rr).reshape(9))
        d = model.eq_data[self.eq_id]
        d[0:3] = 0.0  # anchor at the rocket origin (body2 frame)
        d[3:6] = rel_p
        d[6:10] = qr
        d[10] = 1.0
        data.eq_active[self.eq_id] = 1

    def clamped(self, data) -> bool:
        return self.eq_id is not None and bool(data.eq_active[self.eq_id])

    # ------------------------------------------------------------------ replay
    def meta(self, max_components: int = 48) -> dict:
        """Ship geometry, sea state and the strongest wave components (for the viewer's
        ocean surface, consistent with the recorded deck motion)."""
        sp = self.spec
        sea = self.sea
        idx = np.argsort(-sea.a)[:max_components]
        return {
            "name": sp.name,
            "length": sp.length,
            "deck_width": sp.deck_width,
            "hull_beam": sp.hull_beam,
            "freeboard": sp.freeboard,
            "draft": sp.draft,
            "heading_deg": sp.heading_deg,
            "target_offset": list(sp.target_offset),
            "target_radius": sp.target_radius,
            "roll_period_s": self.roll_period,
            "sea": {
                "hs": sp.sea.hs,
                "tp": sp.sea.tp,
                "spectrum": sp.sea.spectrum,
                "gamma": sp.sea.gamma,
                "direction_deg": sp.sea.direction_deg,
                "hs_synth": self.sea.hs_spectral,
                "waves": [
                    [
                        round(float(sea.a[i]), 4),
                        round(float(sea.w[i]), 5),
                        round(float(sea.theta[i]), 5),
                        round(float(sea.phase[i]), 5),
                    ]
                    for i in idx
                ],
            },
        }


class DeckLink:
    """Ship-to-vehicle data link: the deck target's position/velocity as the flight
    software receives it - sampled at ``rate_hz``, delayed by ``latency_s``, with white
    noise - and extrapolated over the latency with the received velocity."""

    def __init__(self, ship: ShipModel, seed: int | None = 0):
        self.ship = ship
        self.spec = ship.spec.link
        self.rng = np.random.default_rng(seed)
        self.last_t = -math.inf
        self.msg = None

    def reset(self, seed: int | None = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_t = -math.inf
        self.msg = None

    def estimate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        sp = self.spec
        if self.msg is None or t - self.last_t >= 1.0 / sp.rate_hz - 1e-9:
            tm = max(t - sp.latency_s, 0.0)
            p, v = self.ship.target_at(tm)
            p = p + self.rng.normal(0.0, sp.sigma_pos, 3)
            v = v + self.rng.normal(0.0, sp.sigma_vel, 3)
            self.msg = (tm, p, v)
            self.last_t = t
        tm, p, v = self.msg
        return p + v * (t - tm), v
