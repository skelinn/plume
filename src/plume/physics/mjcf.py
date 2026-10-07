"""Build a MuJoCo model (MJCF) for a vehicle and its ground.

MuJoCo integrates the rigid body and resolves leg/ground contact. Its own gravity,
density and viscosity are disabled: Plume applies gravity, thrust, RCS and aero
forces itself through ``xfrc_applied``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from plume.config import VehicleSpec, WorldSpec


@dataclass
class GroundTile:
    """A heightfield patch. ``heights[j, i]`` (row 0 = local -y edge) in metres above ``origin``."""

    name: str
    heights: np.ndarray
    origin: np.ndarray  # world position of the tile centre at height 0
    half_x: float
    half_y: float
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def hfield_params(self):
        h = np.asarray(self.heights, dtype=float)
        h_min = float(h.min())
        elev = max(float(h.max()) - h_min, 1e-3)
        norm = (h - h_min) / elev
        return h_min, elev, norm


LEG_PHASE = math.pi / 4  # legs at 45 deg so they sit between RCS pods


def leg_angles(n: int) -> list[float]:
    return [LEG_PHASE + 2 * math.pi * k / n for k in range(n)]


def _f(x: float) -> str:
    return f"{x:.6g}"


def _v(*xs: float) -> str:
    return " ".join(_f(x) for x in xs)


def build_mjcf(
    vehicle: VehicleSpec,
    world: WorldSpec,
    tiles: list[GroundTile] | None = None,
    dry_inertia=(1.0, 1.0, 1.0),
) -> str:
    """MJCF for the vehicle.

    The dry structure (with all collision geoms) is the free body ``rocket`` with a
    *fixed* inertial frame -- MuJoCo's collision bounding volumes are tied to it, so
    it must never move. Everything that changes (propellant, RCS gas, cargo) lives in
    the welded, geom-less child body ``wet`` whose mass/CG/inertia are rewritten
    every step.
    """
    g = vehicle.geometry
    r = g.radius
    L = g.length
    body_top = max(L - g.nose_length, 0.1 * L)
    legs = vehicle.legs
    eng = vehicle.engine
    tiles = tiles or []

    assets = []
    ground = []
    if world.ground == "plane" and not tiles:
        ground.append(
            f'<geom name="ground" type="plane" size="0 0 1" friction="{_f(world.ground_friction)} 0.005 0.0001" '
            'solref="0.02 1" rgba="0.3 0.3 0.3 1"/>'
        )
    for k, tile in enumerate(tiles):
        h_min, elev, norm = tile.hfield_params()
        ny, nx = norm.shape
        assets.append(
            f'<hfield name="hf{k}" nrow="{ny}" ncol="{nx}" '
            f'size="{_v(tile.half_x, tile.half_y, elev, 1.0)}"/>'
        )
        up = _rotate(tile.quat, np.array([0.0, 0.0, h_min]))
        pos = np.asarray(tile.origin, dtype=float) + up
        ground.append(
            f'<geom name="ground_{tile.name}" type="hfield" hfield="hf{k}" pos="{_v(*pos)}" '
            f'quat="{_v(*tile.quat)}" friction="{_f(world.ground_friction)} 0.005 0.0001" solref="0.02 1"/>'
        )

    geoms = [
        f'<geom name="hull" type="cylinder" fromto="0 0 0.05 0 0 {_f(body_top)}" size="{_f(r)}" '
        'rgba="0.9 0.9 0.92 1"/>',
        f'<geom name="engine" type="cylinder" fromto="0 0 -0.02 0 0 0.4" size="{_f(eng.nozzle_radius)}" '
        'rgba="0.2 0.2 0.2 1"/>',
    ]
    if g.nose_length > 0:
        geoms.append(
            f'<geom name="nose" type="ellipsoid" pos="0 0 {_f(body_top)}" '
            f'size="{_v(r, r, g.nose_length)}" rgba="0.9 0.9 0.92 1"/>'
        )
    for k, th in enumerate(leg_angles(legs.count)):
        c, s = math.cos(th), math.sin(th)
        fr = legs.footpad_radius
        foot = (legs.span * c, legs.span * s, -legs.height + fr)
        attach = (r * c, r * s, legs.attach_z)
        geoms.append(
            f'<geom name="leg{k}" type="capsule" fromto="{_v(*attach, *foot)}" size="{_f(0.35 * fr)}" '
            'rgba="0.15 0.15 0.15 1"/>'
        )
        geoms.append(
            f'<geom name="foot{k}" type="sphere" pos="{_v(*foot)}" size="{_f(fr)}" '
            'friction="1.0 0.005 0.0001" solref="0.05 1" rgba="0.1 0.1 0.1 1"/>'
        )

    asset_xml = f"<asset>{''.join(assets)}</asset>" if assets else ""
    return f"""<mujoco model="plume_{vehicle.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_f(world.dt)}" integrator="RK4" gravity="0 0 0" density="0" viscosity="0">
    <flag warmstart="enable"/>
  </option>
  <size memory="4M"/>
  {asset_xml}
  <worldbody>
    {"".join(ground)}
    <body name="rocket" pos="0 0 0">
      <freejoint name="root"/>
      <inertial pos="0 0 {_f(vehicle.mass.dry_cg_z)}" mass="{_f(vehicle.mass.dry)}" diaginertia="{_v(*dry_inertia)}"/>
      {"".join(geoms)}
      <body name="wet" pos="0 0 0">
        <inertial pos="0 0 {_f(vehicle.mass.dry_cg_z)}" mass="1" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ v
