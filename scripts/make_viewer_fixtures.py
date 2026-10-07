"""Generate synthetic replays + terrain for viewer development (no physics needed).

uv run python scripts/make_viewer_fixtures.py  # writes runs/fixtures/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plume.recording import Recorder
from plume.terrain import flatten_sites, generate_detail_tile, generate_regional_map

OUT = Path("runs/fixtures")
R_EARTH = 6_371_000.0

VEHICLE_LANDER = {
    "name": "lander_small",
    "length": 9.0,
    "diameter": 1.2,
    "nose_length": 1.2,
    "legs": {"count": 4, "span": 2.2, "height": 1.0, "attach_z": 1.6},
    "engine": {"nozzle_radius": 0.35, "gimbal_z": 0.6, "thrust_max": 45000.0},
    "rcs_z": 8.2,
    "cargo_mass": 0.0,
    "dry_mass": 1400.0,
    "prop_mass_initial": 900.0,
}


def quat_from_tilt(ax: float, ay: float) -> list[float]:
    """Small tilt: rotate about x by ax then about y by ay."""
    cx, sx = np.cos(ax / 2), np.sin(ax / 2)
    cy, sy = np.cos(ay / 2), np.sin(ay / 2)
    # q = qy * qx
    w = cy * cx
    x = cy * sx
    y = sy * cx
    z = -sy * sx
    return [w, x, y, z]


def quat_from_axis(axis: np.ndarray) -> list[float]:
    """Quaternion rotating +z onto unit vector ``axis``."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    c = float(np.dot(z, a))
    if c < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]
    v = np.cross(z, a)
    w = 1.0 + c
    q = np.array([w, *v])
    return (q / np.linalg.norm(q)).tolist()


def landing_demo() -> Path:
    rec = Recorder(
        {
            "title": "Synthetic landing (fixture)",
            "source": "sim",
            "controller": "fixture",
            "vehicle": VEHICLE_LANDER,
            "scene": {
                "frame": "flat",
                "ground": {"type": "plane"},
                "pads": [{"name": "LZ-1", "pos": [0, 0, 0], "radius": 10.0}],
                "target": {"pos": [0, 0, 0], "radius": 10.0},
            },
        }
    )
    T = 40.0
    dt = 0.05
    t = np.arange(0, T + dt, dt)
    burn_t = 22.0
    for ti in t:
        s = min(ti / T, 1.0)
        z = 2000 * (1 - s) ** 2 + 1.0
        x = 120 * (1 - s) ** 3
        y = -60 * (1 - s) ** 3
        burning = ti > burn_t
        throttle = 0.0 if not burning else 0.6 + 0.3 * np.sin(ti)
        rec.record(
            {
                "t": ti,
                "pos": [x, y, max(z, 1.0)],
                "quat": quat_from_tilt(0.05 * np.sin(ti / 3), 0.04 * np.cos(ti / 4)),
                "vel": [-360 * (1 - s) ** 2 / T, 180 * (1 - s) ** 2 / T, -4000 * (1 - s) / T],
                "omega": [0.0, 0.0, 0.0],
                "alt": max(z - 1.0, 0.0),
                "throttle": throttle,
                "thrust": throttle * 45000,
                "gimbal": [0.03 * np.sin(ti * 2), 0.02 * np.cos(ti * 1.5)],
                "rcs": [0.0, 0.0, 0.0],
                "prop_mass": 900 - max(0.0, ti - burn_t) * 15,
                "mass": 2300 - max(0.0, ti - burn_t) * 15,
                "g_load": 1.0 + throttle * 1.2,
                "mach": 4000 * (1 - s) / T / 340,
                "q_dyn": 0.5 * 1.1 * (4000 * (1 - s) / T) ** 2,
                "wind": [3.0, 1.0, 0.0],
                "phase": "landing_burn" if burning else "descent",
            }
        )
    rec.event(burn_t, "ignition", "Landing burn")
    rec.event(T, "touchdown", "Touchdown")
    rec.set_outcome(True, "landed", {"landing_error_m": 1.4, "fuel_used_kg": 270})
    return rec.save(OUT / "landing_demo.plume.json.gz")


def surface_point(u: float, v: float, h: float) -> np.ndarray:
    """Map coordinates (east, north arc length) + height -> tangent frame at origin."""
    s = np.hypot(u, v)
    if s < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    else:
        th = s / R_EARTH
        n = np.array([np.sin(th) * u / s, np.sin(th) * v / s, np.cos(th)])
    return np.array([0.0, 0.0, -R_EARTH]) + (R_EARTH + h) * n


def hop_demo() -> Path:
    base = generate_regional_map((-50_000, 850_000), (-150_000, 150_000), 2000.0, seed=3)
    site_a, site_b = (0.0, 0.0), (750_000.0, 40_000.0)
    base = flatten_sites(base, [site_a, site_b], radius=3000, blend=6000)
    base.name = "fixture_region"
    base.save(OUT / "terrain" / "fixture_region.yaml")
    detail = generate_detail_tile(base, site_b, 1000.0, 4.0, seed=4, name="fixture_lz")
    detail.save(OUT / "terrain" / "fixture_lz.yaml")

    hA = base.height(*site_a)
    hB = base.height(*site_b)
    vehicle = dict(VEHICLE_LANDER, name="cargo_hopper", length=10.0, diameter=1.1, cargo_mass=250)
    rec = Recorder(
        {
            "title": "Synthetic cargo hop (fixture)",
            "source": "sim",
            "controller": "fixture",
            "vehicle": vehicle,
            "scene": {
                "frame": "spherical",
                "earth_radius": R_EARTH,
                "ground": {
                    "type": "terrain",
                    "terrain_id": "fixture_region",
                    "detail_terrain_ids": ["fixture_lz"],
                },
                "pads": [{"name": "A", "pos": surface_point(*site_a, hA).tolist(), "radius": 15}],
                "target": {"pos": surface_point(*site_b, hB).tolist(), "radius": 50},
            },
        }
    )
    T = 900.0
    dt = 0.5
    prev = None
    for ti in np.arange(0, T + dt, dt):
        s = ti / T
        u = site_b[0] * (3 * s**2 - 2 * s**3)
        v = site_b[1] * (3 * s**2 - 2 * s**3)
        h_ground = base.height(u, v)
        h = h_ground + 4 * 180_000 * s * (1 - s) + 0.5
        p = surface_point(u, v, h)
        vel = (p - prev) / dt if prev is not None else np.zeros(3)
        prev = p
        ascent = ti < 160
        landing = ti > T - 30
        axis = (
            vel
            if ascent and ti > 5
            else (-vel if np.linalg.norm(vel) > 1 else p - [0, 0, -R_EARTH])
        )
        throttle = 1.0 if ascent else (0.7 if landing else 0.0)
        rec.record(
            {
                "t": ti,
                "pos": p.tolist(),
                "quat": quat_from_axis(np.asarray(axis, dtype=float)),
                "vel": vel.tolist(),
                "alt": h - h_ground,
                "throttle": throttle,
                "thrust": throttle * 70000,
                "gimbal": [0.0, 0.0],
                "prop_mass": 3300 * (1 - min(ti, 160) / 180) - (60 if landing else 0),
                "mass": 4700 - 3300 * min(ti, 160) / 180,
                "g_load": 1.5 + 2 * s if ascent else (2.5 if landing else 0.0),
                "mach": float(np.linalg.norm(vel)) / 320,
                "q_dyn": 1000.0
                * np.exp(-(h - h_ground) / 8000)
                * float(np.linalg.norm(vel)) ** 2
                / 2000,
                "phase": "ascent" if ascent else ("landing_burn" if landing else "coast"),
            }
        )
    rec.event(0, "ignition", "Liftoff")
    rec.event(160, "cutoff", "MECO")
    rec.event(T - 30, "ignition", "Landing burn")
    rec.event(T, "touchdown", "Touchdown")
    return rec.save(OUT / "hop_demo.plume.json.gz")


def real_vs_sim() -> tuple[Path, Path]:
    vehicle = {
        "name": "hobby_rocket",
        "length": 1.6,
        "diameter": 0.075,
        "nose_length": 0.3,
        "legs": {"count": 0, "span": 0, "height": 0},
        "engine": {"nozzle_radius": 0.02, "gimbal_z": 0.0, "thrust_max": 300.0},
    }
    paths = []
    for source, cd, noise in (("sim", 0.45, 0.0), ("real", 0.52, 1.5)):
        rng = np.random.default_rng(1)
        rec = Recorder(
            {
                "title": f"Hobby H-motor flight ({source}) (fixture)",
                "source": source,
                "vehicle": vehicle,
                "scene": {
                    "frame": "flat",
                    "ground": {"type": "plane"},
                    "pads": [{"name": "Rail", "pos": [0, 0, 0], "radius": 1.0}],
                },
            }
        )
        z, vz, dt = 0.0, 0.0, 0.02
        for ti in np.arange(0, 25, dt):
            thrust = 250.0 if ti < 1.8 else 0.0
            m = 2.2 - 0.18 * min(ti, 1.8) / 1.8
            drag = 0.5 * 1.2 * cd * 0.0044 * vz * abs(vz)
            a = (thrust - drag) / m - 9.81
            if z <= 0 and a < 0 and ti < 1:
                a = 0.0
            vz += a * dt
            z = max(0.0, z + vz * dt)
            rec.record(
                {
                    "t": ti,
                    "pos": [0.02 * z, 0.0, z + noise * rng.standard_normal()],
                    "quat": quat_from_tilt(0.0, 0.02 if vz > 0 else np.pi * 0.9),
                    "vel": [0, 0, vz],
                    "alt": z,
                    "throttle": 1.0 if thrust else 0.0,
                    "thrust": thrust,
                    "g_load": abs(a + 9.81) / 9.81,
                    "mach": abs(vz) / 340,
                }
            )
        paths.append(rec.save(OUT / f"hobby_{source}.plume.json.gz"))
    return paths[0], paths[1]


if __name__ == "__main__":
    for p in (landing_demo(), hop_demo(), *real_vs_sim()):
        print("wrote", p)
