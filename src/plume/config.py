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
    slosh: bool = Field(False, description="model the first lateral slosh mode (spring-mass)")
    slosh_damping: float = Field(0.02, ge=0, description="slosh damping ratio (baffles: 0.03-0.1)")

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
    misalignment_deg: tuple[float, float] = Field(
        (0.0, 0.0), description="thrust-axis misalignment about body x / y (build tolerance), deg"
    )
    # high-fidelity actuator and start-up dynamics (fast fidelity uses gimbal_tau)
    gimbal_wn_hz: float = Field(8.0, gt=0, description="gimbal actuator natural frequency")
    gimbal_zeta: float = Field(0.7, gt=0, description="gimbal actuator damping ratio")
    gimbal_accel_deg_s2: float = Field(800.0, gt=0, description="gimbal acceleration limit")
    gimbal_delay_s: float = Field(0.015, ge=0, description="command transport delay")
    gimbal_backlash_deg: float = Field(0.05, ge=0, description="total free play")
    ignition_delay_s: float = Field(0.35, ge=0, description="liquid: command to thrust onset")
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
    # high fidelity: valves are pulse-width modulated (fast fidelity averages the duty)
    pwm_period_s: float = Field(0.05, gt=0, description="PWM frame (= flight-software cycle)")
    min_on_time_s: float = Field(0.01, ge=0, description="minimum valve open time (impulse bit)")
    valve_delay_s: float = Field(0.005, ge=0, description="valve opening latency")
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


class GridFinSpec(Spec):
    """Deployable lattice fins near the top of the hull (see ``plume.physics.gridfins``)."""

    count: int = Field(4, ge=2)
    z: float = Field(description="hinge height in body z")
    span: float = Field(0.5, gt=0, description="radial extent of each fin, m")
    chord: float = Field(0.45, gt=0, description="panel width across the flow (tangential), m")
    depth: float = Field(0.12, gt=0, description="lattice depth along the flow (axial), m")
    radius: float | None = Field(
        None, description="fin centre radius; default hull radius + span/2"
    )
    area: float | None = Field(
        None, description="effective lifting area per fin; default span*chord"
    )
    cn_alpha: float = Field(3.0, ge=0, description="normal-force slope per radian, fin area")
    cd0: float = Field(1.2, ge=0, description="lattice drag coefficient (deployed), fin area")
    cd_delta: float = Field(2.0, ge=0, description="extra drag per rad^2 of deflection")
    max_deflection_deg: float = Field(20.0, gt=0)
    rate_deg_s: float = Field(45.0, gt=0)
    open_area_ratio: float = Field(
        0.9,
        gt=0,
        le=1,
        description="lattice open area / frontal area (transonic choking, high fidelity)",
    )
    deploy: Literal["always", "on_command"] = "always"


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
    model: Literal["auto", "strip", "database"] = Field(
        "auto", description="auto: strip theory in fast fidelity, aero database in high"
    )
    database: str | None = Field(
        None, description="aero database (.npz from plume.physics.aerodb); generated if absent"
    )
    reverse_potential: float = Field(
        1.0, ge=0, description="database generator: potential CN for base-first flow (Jorgensen)"
    )
    reynolds_effect: bool = Field(True, description="database generator: crossflow drag crisis")
    roughness: float = Field(20e-6, ge=0, description="equivalent sand-grain roughness, m")
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


class ImuSpec(Spec):
    """Inertial measurement unit (defaults: tactical-grade, HG1700-class)."""

    rate_hz: float = Field(200.0, gt=0)
    gyro_arw_deg_rt_h: float = Field(0.125, ge=0, description="angle random walk, deg/sqrt(h)")
    gyro_bias_deg_h: float = Field(3.0, ge=0, description="turn-on bias 1-sigma, deg/h")
    gyro_bias_instability_deg_h: float = Field(1.0, ge=0)
    accel_vrw_m_s_rt_h: float = Field(0.06, ge=0, description="velocity random walk, m/s/sqrt(h)")
    accel_bias_mg: float = Field(1.0, ge=0, description="turn-on bias 1-sigma, milli-g")
    accel_bias_instability_mg: float = Field(0.05, ge=0)
    bias_correlation_s: float = Field(600.0, gt=0)
    scale_factor_ppm: float = Field(300.0, ge=0)
    misalignment_mrad: float = Field(0.5, ge=0)
    gyro_quantum_rad: float = Field(1e-7, ge=0, description="delta-angle LSB")
    accel_quantum_m_s: float = Field(1e-5, ge=0, description="delta-velocity LSB")
    gyro_range_deg_s: float = Field(1000.0, gt=0)
    accel_range_g: float = Field(40.0, gt=0)


class GnssSpec(Spec):
    rate_hz: float = Field(10.0, gt=0)
    latency_s: float = Field(0.05, ge=0)
    sigma_h_m: float = Field(1.5, ge=0, description="horizontal position 1-sigma")
    sigma_v_m: float = Field(3.0, ge=0, description="vertical position 1-sigma")
    sigma_vel_m_s: float = Field(0.05, ge=0)
    bias_correlation_s: float = Field(300.0, gt=0)
    max_altitude_m: float | None = Field(
        None, description="outage above this altitude (None = never)"
    )


class BaroSpec(Spec):
    sigma_m: float = Field(1.0, ge=0)
    bias_m: float = Field(5.0, ge=0, description="1-sigma calibration bias")
    min_pressure_pa: float = Field(1000.0, ge=0, description="no reading below this pressure")


class RadarAltimeterSpec(Spec):
    max_range_m: float = Field(2500.0, gt=0)
    sigma_m: float = Field(0.1, ge=0)
    sigma_frac: float = Field(0.005, ge=0, description="noise proportional to range")
    max_tilt_deg: float = Field(30.0, gt=0)


class SensorsSpec(Spec):
    imu: ImuSpec = Field(default_factory=ImuSpec)
    gnss: GnssSpec = Field(default_factory=GnssSpec)
    baro: BaroSpec = Field(default_factory=BaroSpec)
    radar: RadarAltimeterSpec | None = Field(default_factory=RadarAltimeterSpec)


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
    grid_fins: GridFinSpec | None = None
    sensors: SensorsSpec = Field(default_factory=SensorsSpec)

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
    profile: str | None = Field(
        None, description="sounding/forecast CSV (path or name in configs/winds/) for the mean wind"
    )
    turbulence_model: Literal["auto", "dryden", "von_karman"] = "auto"
    turbulence_severity: Literal["none", "light", "moderate", "severe"] = Field(
        "none", description="MIL-F-8785C continuous turbulence (replaces `turbulence` when set)"
    )


class AtmosphereModelSpec(Spec):
    """Which atmosphere model to use (``WorldSpec.atmosphere`` still switches air off)."""

    model: Literal["us76", "nrlmsise00", "sounding"] = "us76"
    epoch: str = Field("2025-06-21T18:00:00Z", description="UTC date/time for NRLMSISE-00")
    f107: float = Field(150.0, ge=0, description="previous-day F10.7 solar flux, sfu")
    f107a: float = Field(150.0, ge=0, description="81-day mean F10.7, sfu")
    ap: float = Field(4.0, ge=0, description="daily geomagnetic Ap index")
    sounding: str | None = Field(None, description="CSV with altitude + temperature (+ pressure)")


class EarthSpec(Spec):
    """Earth model used when ``WorldSpec.gravity == "wgs84"``."""

    origin_lat_deg: float = Field(0.0, ge=-90, le=90, description="world-frame origin, geodetic")
    origin_lon_deg: float = Field(0.0, ge=-180, le=360)
    origin_height: float = Field(0.0, description="origin height above the ellipsoid, m")
    shape: Literal["wgs84", "sphere"] = "wgs84"
    radius: float = Field(6_371_008.8, gt=0, description="sphere radius when shape == sphere")
    rotating: bool = True
    zonal_degree: int = Field(6, ge=0, le=6, description="highest zonal harmonic (0 = point mass)")
    gm: float | None = Field(None, description="override gravitational parameter, m^3/s^2")
    omega: float | None = Field(None, description="override rotation rate, rad/s")
    j2: float | None = Field(None, description="override J2")


class WorldSpec(Spec):
    fidelity: Literal["fast", "high"] = Field(
        "fast",
        description="fast: simplified models (RL, iteration); high: verification-grade models",
    )
    gravity: Literal["flat", "spherical", "wgs84"] = "flat"
    earth: EarthSpec = Field(default_factory=EarthSpec)
    g: float = Field(9.80665, ge=0, description="flat gravity only: acceleration, m/s^2")
    atmosphere: bool = True
    temperature_offset: float = 0.0
    atmosphere_model: AtmosphereModelSpec = Field(default_factory=AtmosphereModelSpec)
    wind: WindSpec = Field(default_factory=WindSpec)
    navigation: Literal["auto", "truth", "ekf"] = Field(
        "auto", description="state the flight software sees: auto = EKF in high fidelity"
    )
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
    u: float = 0.0  # map east (m) from the launch-site origin (azimuthal equidistant)
    v: float = 0.0  # map north (m)
    lat: float | None = Field(None, ge=-90, le=90, description="geodetic latitude, deg (WGS-84)")
    lon: float | None = Field(None, ge=-180, le=360, description="geodetic longitude, deg (WGS-84)")


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
    max_divert_m: float | None = Field(
        250.0,
        description="landing-burn divert reach; beyond it land at the closest reachable point",
    )
    closed_loop_ascent: bool = Field(
        True, description="track the planned flight-path angle vs speed in the gravity turn"
    )
    aero_steer_max_mach: float = Field(
        5.0, gt=0, description="high fidelity: steer the descent with body lift below this Mach"
    )
    supersonic_tilt_deg: float = Field(
        15.0, ge=0, description="high fidelity: tilt cap above Mach 1"
    )
    subsonic_tilt_deg: float = Field(20.0, ge=0, description="high fidelity: tilt cap below Mach 1")
    divert_gate: bool = Field(
        False,
        description="experimental: brake to a slow gate above the pad and fly the divert at "
        "low speed (did not improve Monte Carlo results yet; docs/models/guidance.md)",
    )
    # ---- ascent load relief (docs/models/guidance.md, "Ascent load relief")
    load_relief: bool = Field(
        False,
        description="limit the commanded angle of attack in the gravity turn so the predicted "
        "aerodynamic trim uses at most load_relief_gimbal_fraction of the gimbal range",
    )
    load_relief_gimbal_fraction: float = Field(
        0.4, gt=0, le=1, description="gimbal-range budget for aerodynamic trim (load relief)"
    )
    ascent_attitude_bandwidth: float | None = Field(
        None,
        gt=0,
        description="powered ascent: attitude-loop natural frequency, rad/s (None = the "
        "default 2.5 rad/s controller)",
    )
    crossrange_max_deg: float = Field(
        3.44,
        ge=0,
        description="gravity turn: cap on the cross-range steering angle off the velocity "
        "(the AoA limiter still applies; larger values re-target faster after max-q)",
    )
    # ---- landing burn guidance
    landing_guidance: Literal["hoverslam", "convex"] = Field(
        "hoverslam",
        description="landing burn: constant-deceleration hoverslam + ZEM, or convex "
        "minimum-fuel powered-descent guidance (needs the gnc extra: cvxpy)",
    )
    convex_accel_g: float = Field(
        4.0, gt=1, description="convex landing: sensed-acceleration limit of the plan, g"
    )
    convex_ignition_margin: float = Field(
        0.8,
        gt=0,
        le=1,
        description="convex landing: ignite when the plan is only just feasible with this "
        "fraction of the acceleration limit (the rest is tracking margin)",
    )
    convex_glide_slope_deg: float = Field(
        15.0, ge=0, lt=90, description="convex landing: minimum approach elevation angle"
    )
    convex_max_tilt_deg: float = Field(
        25.0, gt=0, lt=90, description="convex landing: thrust tilt limit from vertical"
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
