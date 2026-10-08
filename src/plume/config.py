"""Configuration models (pydantic v2) and YAML loading.

All quantities are SI unless a field name says otherwise (``*_deg``). Lengths along
the vehicle axis are measured from the base of the hull (body-frame z, +z toward
the nose).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def config_root() -> Path:
    """Directory holding the bundled ``configs/`` tree."""
    for candidate in (Path.cwd() / "configs", REPO_ROOT / "configs"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate the Plume configs/ directory")


def data_root() -> Path:
    for candidate in (Path.cwd() / "data", REPO_ROOT / "data"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate the Plume data/ directory")


Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Table = list[Annotated[list[float], Field(min_length=2, max_length=2)]]


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- vehicle
class GeometrySpec(Spec):
    length: float = Field(gt=0, description="hull base to nose tip, m")
    diameter: float = Field(gt=0)
    nose_length: float = Field(0.0, ge=0)

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter


class LegsSpec(Spec):
    count: int = Field(4, ge=0)
    span: float = Field(2.0, ge=0, description="radial distance of footpad centres from axis")
    height: float = Field(1.0, ge=0, description="footpad depth below the hull base")
    attach_z: float = Field(1.5, description="leg attachment height on the hull")
    footpad_radius: float = Field(0.15, gt=0)
    max_touchdown_speed: float = Field(5.0, gt=0, description="legs fail above this speed")


class MassSpec(Spec):
    dry: float = Field(gt=0, description="dry mass excluding cargo and propellants, kg")
    dry_cg_z: float
    dry_inertia: Vec3 | None = Field(
        None, description="principal inertia about the dry CG [Ixx, Iyy, Izz]; estimated if null"
    )


class CargoSpec(Spec):
    mass: float = Field(0.0, ge=0)
    cg_z: float = 0.0
    max_mass: float | None = Field(None, ge=0)


class TankSpec(Spec):
    name: str = "main"
    capacity: float = Field(gt=0, description="kg")
    initial: float | None = Field(None, ge=0, description="kg; defaults to full")
    z_bottom: float
    z_top: float
    radius: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.z_top <= self.z_bottom:
            raise ValueError(f"tank {self.name!r}: z_top must exceed z_bottom")
        if self.initial is not None and self.initial > self.capacity + 1e-9:
            raise ValueError(f"tank {self.name!r}: initial load exceeds capacity")
        return self

    @property
    def initial_mass(self) -> float:
        return self.capacity if self.initial is None else self.initial


class EngineSpec(Spec):
    type: Literal["liquid", "solid"] = "liquid"
    thrust_vac: float = Field(0.0, ge=0, description="liquid: full-throttle vacuum thrust, N")
    isp_vac: float = Field(300.0, gt=0)
    isp_sl: float | None = Field(None, gt=0, description="sea-level Isp; defaults to isp_vac")
    throttle_min: float = Field(0.4, ge=0, le=1)
    throttle_max: float = Field(1.0, gt=0, le=1.5)
    throttle_tau: float = Field(0.1, ge=0, description="first-order throttle response, s")
    max_ignitions: int | None = Field(None, ge=0)
    gimbal_max_deg: float = Field(0.0, ge=0)
    gimbal_rate_deg_s: float = Field(30.0, gt=0)
    gimbal_tau: float = Field(0.05, ge=0)
    gimbal_z: float = Field(0.5, description="gimbal pivot height in body frame")
    nozzle_radius: float = Field(0.3, gt=0)
    # solid motors
    thrust_curve: Table | None = Field(None, description="[[t, F], ...] for solid motors")
    motor_file: str | None = Field(None, description="RASP .eng file (solid motors)")
    ignition_time: float = Field(0.0, ge=0, description="solid motors: ignition time, s")

    @model_validator(mode="after")
    def _check(self):
        if self.type == "liquid" and self.thrust_vac <= 0:
            raise ValueError("liquid engine needs thrust_vac > 0")
        if self.type == "solid" and not (self.thrust_curve or self.motor_file):
            raise ValueError("solid motor needs thrust_curve or motor_file")
        if self.throttle_min > self.throttle_max:
            raise ValueError("throttle_min must not exceed throttle_max")
        return self

    @property
    def isp_sea_level(self) -> float:
        return self.isp_sl if self.isp_sl is not None else self.isp_vac


class ThrusterSpec(Spec):
    pos: Vec3
    dir: Vec3 = Field(description="thrust direction (force on vehicle), body frame")


class RCSSpec(Spec):
    enabled: bool = True
    thrust: float = Field(200.0, ge=0, description="per thruster, N")
    isp: float = Field(70.0, gt=0)
    propellant: float = Field(20.0, ge=0, description="cold-gas load, kg")
    z: float = Field(8.0, description="pod ring height (layout=ring)")
    radius: float | None = Field(None, description="pod ring radius; defaults to hull radius")
    pods: int = Field(4, ge=2)
    thrusters: list[ThrusterSpec] | None = Field(None, description="custom layout overrides ring")

    @property
    def gas(self) -> float:
        """Cold-gas mass actually carried (zero when the RCS is disabled)."""
        return self.propellant if self.enabled else 0.0


class FinSpec(Spec):
    """Fin set: linear normal force F = q * cn_alpha * alpha * A_ref acting at ``z``."""

    count: int = Field(3, ge=0)
    cn_alpha: float = Field(8.0, ge=0, description="normal-force slope per radian (body ref. area)")
    z: float = Field(0.1, description="fin centre of pressure, body z")


class AeroSpec(Spec):
    enabled: bool = True
    fins: FinSpec | None = None
    reference_area: float | None = Field(None, description="defaults to hull cross-section")
    ca_nose_first: Table = Field(
        default_factory=lambda: [
            [0.0, 0.30],
            [0.8, 0.30],
            [1.1, 0.55],
            [1.5, 0.48],
            [3.0, 0.35],
            [6.0, 0.28],
        ]
    )
    ca_tail_first: Table = Field(
        default_factory=lambda: [
            [0.0, 0.90],
            [0.8, 0.95],
            [1.1, 1.25],
            [1.5, 1.20],
            [3.0, 1.05],
            [6.0, 0.95],
        ]
    )
    crossflow_cd: Table = Field(
        default_factory=lambda: [[0.0, 1.2], [0.8, 1.3], [1.2, 1.6], [3.0, 1.5], [6.0, 1.4]]
    )
    cd_scale: float = Field(1.0, gt=0, description="multiplier on all coefficients (calibration)")
    stations: int = Field(10, ge=2, description="strip-theory stations along the hull")


class ChuteSpec(Spec):
    name: str = "main"
    cd_area: float = Field(gt=0, description="drag coefficient x canopy area, m^2")
    deploy: Literal["apogee", "altitude"] = "apogee"
    delay: float = Field(0.0, ge=0, description="seconds after apogee (or after crossing altitude)")
    altitude: float | None = Field(None, description="deploy='altitude': height above launch, m")
    inflation_time: float = Field(0.5, ge=0)


class RecoverySpec(Spec):
    chutes: list[ChuteSpec] = Field(default_factory=list)


class VehicleSpec(Spec):
    name: str
    description: str = ""
    geometry: GeometrySpec
    legs: LegsSpec = Field(default_factory=LegsSpec)
    mass: MassSpec
    cargo: CargoSpec = Field(default_factory=CargoSpec)
    tanks: list[TankSpec] = Field(default_factory=list)
    engine: EngineSpec
    rcs: RCSSpec = Field(default_factory=lambda: RCSSpec(enabled=False))
    aero: AeroSpec = Field(default_factory=AeroSpec)
    recovery: RecoverySpec = Field(default_factory=RecoverySpec)

    @model_validator(mode="after")
    def _check(self):
        if self.cargo.max_mass is not None and self.cargo.mass > self.cargo.max_mass + 1e-9:
            raise ValueError(f"cargo mass {self.cargo.mass} exceeds max {self.cargo.max_mass}")
        return self

    @property
    def prop_capacity(self) -> float:
        return sum(t.capacity for t in self.tanks)

    @property
    def prop_initial(self) -> float:
        return sum(t.initial_mass for t in self.tanks)

    def with_cargo(self, mass: float) -> VehicleSpec:
        v = self.model_copy(deep=True)
        v.cargo.mass = mass
        return VehicleSpec.model_validate(v.model_dump())


# --------------------------------------------------------------------------- world
class WindSpec(Spec):
    speed: float = Field(0.0, ge=0, description="mean wind at ref_height, m/s")
    from_deg: float = 270.0
    shear_exponent: float = 0.143
    ref_height: float = 10.0
    shear_top: float = 2000.0
    fade_top: float = 30_000.0
    turbulence: float = Field(0.0, ge=0, description="turbulence intensity (1-sigma), m/s")
    turbulence_tau: float = Field(1.5, gt=0)
    gust_rate: float = Field(0.0, ge=0, description="discrete gusts per second")
    gust_max: float = Field(0.0, ge=0)
    gust_duration: tuple[float, float] = (1.0, 4.0)


class WorldSpec(Spec):
    gravity: Literal["flat", "spherical"] = "flat"
    g: float = Field(9.80665, ge=0, description="flat gravity only: acceleration, m/s^2")
    atmosphere: bool = True
    temperature_offset: float = 0.0
    wind: WindSpec = Field(default_factory=WindSpec)
    ground: Literal["plane", "none"] = "plane"
    ground_friction: float = Field(0.8, ge=0)
    dt: float = Field(0.005, gt=0, le=0.05)

    @field_validator("dt")
    @classmethod
    def _dt(cls, v):
        if not math.isfinite(v):
            raise ValueError("dt must be finite")
        return v


# --------------------------------------------------------------------------- loading
def _resolve(path_or_name: str | Path, kind: str) -> Path:
    p = Path(path_or_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = config_root() / kind / f"{p.stem if p.suffix else p.name}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no {kind[:-1]} config {path_or_name!r} (looked in {candidate.parent})"
    )


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vehicle(path_or_name: str | Path) -> VehicleSpec:
    """Load a vehicle by file path or preset name (``configs/vehicles/<name>.yaml``)."""
    path = _resolve(path_or_name, "vehicles")
    spec = VehicleSpec.model_validate(load_yaml(path))
    if spec.engine.motor_file and not Path(spec.engine.motor_file).is_absolute():
        motor = (path.parent / spec.engine.motor_file).resolve()
        if not motor.exists():
            motor = data_root() / "motors" / Path(spec.engine.motor_file).name
        spec.engine.motor_file = str(motor)
    return spec


def load_world(data: dict | None) -> WorldSpec:
    return WorldSpec.model_validate(data or {})


def dump_yaml(model: BaseModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))
    return path


# --------------------------------------------------------------------------- landing env
Range = tuple[float, float]


class StageSpec(Spec):
    """Initial-condition distribution for one curriculum stage (uniform ranges)."""

    name: str
    altitude: Range = (50.0, 100.0)  # m above the pad (footpads)
    offset: Range = (0.0, 5.0)  # horizontal distance from the pad, m
    descent_speed: Range = (0.0, 5.0)  # m/s, positive = down
    horizontal_speed: Range = (0.0, 1.0)  # deviation from the mean wind, m/s
    tilt_deg: Range = (0.0, 3.0)  # attitude error from engine-first
    rate_deg_s: Range = (0.0, 2.0)
    prop_fraction: Range = (0.3, 0.5)
    wind_speed: Range = (0.0, 0.0)
    gust_max: float = 0.0
    gust_rate: float = 0.1
    turbulence: float = 0.0


class CurriculumSpec(Spec):
    promote_success: float = Field(0.8, gt=0, le=1)
    window: int = Field(200, ge=1, description="episodes in the rolling success window")
    min_episodes: int = Field(200, ge=1, description="episodes before a promotion is allowed")
    replay_fraction: float = Field(0.2, ge=0, le=1, description="chance to sample an earlier stage")
    stages: list[StageSpec]


class SuccessSpec(Spec):
    max_vertical_speed: float = 2.0
    max_horizontal_speed: float = 1.0
    max_tilt_deg: float = 10.0
    settle_time: float = 1.0


class RewardSpec(Spec):
    success_bonus: float = 100.0
    accuracy_bonus: float = 50.0
    crash_penalty: float = 100.0
    crash_speed_penalty: float = 2.0  # per m/s of impact speed above the limit
    fail_penalty: float = 150.0  # timeout / out of bounds: worse than crashing
    partial_landing: float = 50.0  # gentle upright touchdown that misses a criterion
    fuel_weight: float = 0.01  # per kg
    rcs_weight: float = 0.005  # per step at full RCS
    track_weight: float = 0.3  # dense: per (m/s of error vs the reference descent) per second
    time_weight: float = 1.0  # dense: per second airborne (finish the job)
    w_distance: float = 1.0  # potential: per 50 m horizontal miss
    w_velocity: float = 1.0  # potential: per 20 m/s velocity error vs reference profile
    w_tilt: float = 1.0  # potential: per radian
    w_rate: float = 0.3  # potential: per rad/s


class LandingEnvSpec(Spec):
    vehicle: str = "lander_small"
    action_mode: Literal["guidance", "direct"] = Field(
        "guidance",
        description="guidance: throttle + thrust-axis tilt (inner attitude loop); "
        "direct: throttle + gimbal + RCS",
    )
    max_tilt_deg: float = Field(30.0, gt=0, le=89, description="guidance mode: axis tilt limit")
    control_dt: float = 0.05
    physics_dt: float = 0.01
    max_time: float = 150.0  # hard cap; each episode also gets 20 s + altitude / 8 m/s
    pad_radius: float = 10.0
    world: WorldSpec = Field(default_factory=WorldSpec)
    success: SuccessSpec = Field(default_factory=SuccessSpec)
    reward: RewardSpec = Field(default_factory=RewardSpec)
    curriculum: CurriculumSpec


def load_landing_env(path_or_name: str | Path = "landing") -> LandingEnvSpec:
    path = _resolve(path_or_name, "envs")
    data = load_yaml(path)
    cur = data.get("curriculum")
    if isinstance(cur, str):  # reference to a separate curriculum file
        data["curriculum"] = load_yaml(_resolve(cur, "envs"))
    return LandingEnvSpec.model_validate(data)


# --------------------------------------------------------------------------- cargo hop missions
class SiteSpec(Spec):
    name: str
    u: float  # map east (m) from the launch-site origin
    v: float  # map north (m)


class MissionTerrainSpec(Spec):
    base: str = "demo_region"
    launch_tile: str | None = "demo_pad_a"
    landing_tile: str | None = "demo_lz_b"


class HopGuidanceSpec(Spec):
    cargo_g_limit: float = Field(6.0, gt=1, description="throttle back to keep cargo below this")
    rise_time: float = Field(
        6.0, description="vertical rise before the kick (optimised if kick is null)"
    )
    kick_time: float = 6.0
    kick_angle_deg: float | None = Field(
        None, description="null = optimise rise + kick before flight"
    )
    entry_altitude: float = Field(60_000.0, description="start the entry burn below this")
    entry_speed: float = Field(1500.0, description="entry burn target speed, m/s")
    meco_bias: float = Field(
        400.0, description="aim this far past the target at MECO (the entry burn trims it), m"
    )
    max_flight_path_deg: float = Field(
        45.0, description="planner: cap on the MECO flight-path angle"
    )


class ScoringSpec(Spec):
    success_points: float = 100.0
    accuracy_points: float = 50.0  # scaled by 1 - error / target radius
    fuel_points: float = 30.0  # scaled by remaining-propellant fraction
    g_penalty_per_g: float = 10.0  # per g above the cargo limit
    time_penalty_per_min: float = 0.0


class MissionSpec(Spec):
    name: str
    description: str = ""
    vehicle: str = "cargo_hopper"
    cargo_mass: float | None = None
    terrain: MissionTerrainSpec = Field(default_factory=MissionTerrainSpec)
    launch: SiteSpec
    target: SiteSpec
    target_radius: float = 50.0
    world: WorldSpec = Field(
        default_factory=lambda: WorldSpec(gravity="spherical", ground="none", dt=0.01)
    )
    guidance: HopGuidanceSpec = Field(default_factory=HopGuidanceSpec)
    scoring: ScoringSpec = Field(default_factory=ScoringSpec)
    max_time: float = 1500.0
    record_every: int = 5  # physics steps between replay frames (x control decimation)


def load_mission(path_or_name: str | Path) -> MissionSpec:
    return MissionSpec.model_validate(load_yaml(_resolve(path_or_name, "missions")))


# --------------------------------------------------------------------------- flight logs
class ColumnSpec(Spec):
    column: str
    unit: str = "si"  # see plume.flightdata.importer.UNITS
    scale: float = 1.0
    offset: float = 0.0


class AccelSpec(ColumnSpec):
    includes_gravity: bool = Field(
        True, description="raw accelerometer (reads +1 g on the pad) vs. gravity-removed"
    )


class GyroSpec(Spec):
    columns: Annotated[list[str], Field(min_length=3, max_length=3)]
    unit: str = "deg/s"


class GpsSpec(Spec):
    lat: str
    lon: str
    alt: str | None = None
    alt_unit: str = "m"


class LaunchDetectSpec(Spec):
    accel_threshold_g: float = 2.5  # sustained axial load that marks liftoff
    altitude_threshold: float = 15.0  # fallback when there is no accelerometer
    pad_window: float = 1.0  # seconds of pre-launch data used to zero the baro


class LogMappingSpec(Spec):
    """Column mapping for a hobby flight-computer CSV export."""

    name: str = "custom"
    delimiter: str = ","
    comment: str = "#"
    skip_rows: int = 0
    time: ColumnSpec
    altitude: ColumnSpec
    acceleration: AccelSpec | None = None
    gyro: GyroSpec | None = None
    gps: GpsSpec | None = None
    launch_detect: LaunchDetectSpec = Field(default_factory=LaunchDetectSpec)
    resample_hz: float = Field(50.0, gt=0)


def load_log_mapping(path_or_name: str | Path) -> LogMappingSpec:
    return LogMappingSpec.model_validate(load_yaml(_resolve(path_or_name, "flightlogs")))
