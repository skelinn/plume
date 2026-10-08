"""Multi-stage launch vehicles and orbital launch missions (configuration).

A :class:`LauncherSpec` is a stack of ordinary :class:`~plume.config.VehicleSpec` stages,
listed bottom-up, plus a payload and an optional fairing. Every stage keeps its own body
frame (z from its own hull base, +z toward the nose); stage ``k + 1`` sits with its hull
base on top of stage ``k`` (at ``geometry.length`` of the stage below), and the payload
and fairing sit on top of the last stage. Single-stage vehicles are unaffected: nothing
in :mod:`plume.config` changes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from plume.config import (
    HopGuidanceSpec,
    MissionTerrainSpec,
    SiteSpec,
    Spec,
    VehicleSpec,
    WorldSpec,
    _resolve,
    load_yaml,
)


class LauncherStageSpec(Spec):
    name: str
    vehicle: VehicleSpec = Field(description="the stage as a stand-alone vehicle (own body frame)")
    reserve: float = Field(
        0.0,
        ge=0,
        description="propellant (kg) kept in the tanks at cutoff, e.g. for a booster's return",
    )


class PayloadSpec(Spec):
    mass: float = Field(0.0, ge=0, description="kg")
    cg_z: float = Field(0.5, description="above the top of the last stage, m")
    length: float = Field(1.0, gt=0, description="drawn / aero length above the last stage, m")


class FairingSpec(Spec):
    mass: float = Field(gt=0, description="both halves, kg")
    length: float = Field(gt=0, description="base (top of the last stage) to tip, m")
    nose_length: float = Field(gt=0, description="ogive part of the fairing, m")
    diameter: float | None = Field(None, gt=0, description="defaults to the last stage's diameter")
    jettison_altitude: float = Field(
        105_000.0, description="jettison above this altitude (after upper-stage ignition), m"
    )
    jettison_speed: float = Field(2.0, ge=0, description="lateral speed of each half, m/s")
    jettison_rate_deg_s: float = Field(12.0, ge=0, description="outward tumble rate, deg/s")

    @model_validator(mode="after")
    def _check(self):
        if self.nose_length > self.length:
            raise ValueError("fairing nose_length exceeds its length")
        return self


class SeparationSpec(Spec):
    coast: float = Field(2.0, ge=0, description="MECO to stage separation, s")
    delta_v: float = Field(1.0, ge=0, description="relative separation speed (springs), m/s")
    ignition_delay: float = Field(3.0, ge=0, description="separation to upper-stage ignition, s")


class LauncherSpec(Spec):
    name: str
    description: str = ""
    stages: list[LauncherStageSpec] = Field(min_length=2, description="bottom-up")
    payload: PayloadSpec = Field(default_factory=PayloadSpec)
    fairing: FairingSpec | None = None
    separation: SeparationSpec = Field(default_factory=SeparationSpec)

    @model_validator(mode="after")
    def _check(self):
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        for s in self.stages:
            if s.reserve > s.vehicle.prop_initial:
                raise ValueError(f"stage {s.name!r}: reserve exceeds its propellant load")
        return self

    def stage_base(self, k: int) -> float:
        """Height of stage ``k``'s hull base above the bottom of the stack (stage 0)."""
        return sum(s.vehicle.geometry.length for s in self.stages[:k])

    @property
    def top(self) -> float:
        """Top of the last stage, in the stack frame."""
        return self.stage_base(len(self.stages))

    def with_payload(self, mass: float) -> LauncherSpec:
        out = self.model_copy(deep=True)
        out.payload.mass = float(mass)
        return LauncherSpec.model_validate(out.model_dump())


# --------------------------------------------------------------------------- missions
class OrbitTargetSpec(Spec):
    perigee_altitude: float = Field(200_000.0, gt=0, description="insertion altitude, m")
    apogee_altitude: float = Field(250_000.0, gt=0, description="m")
    inclination_deg: float | None = Field(
        None, ge=0, le=180, description="null = due east (inclination = launch latitude)"
    )
    min_perigee_altitude: float = Field(
        150_000.0, description="'in orbit' when the osculating perigee is above this"
    )

    @model_validator(mode="after")
    def _check(self):
        if self.apogee_altitude < self.perigee_altitude:
            raise ValueError("apogee_altitude must not be below perigee_altitude")
        return self


class AscentSpec(Spec):
    """First-stage (stack) ascent: vertical rise, pitch kick, gravity turn, MECO."""

    rise_time: float = Field(6.0, gt=0, description="vertical rise before the kick, s")
    kick_time: float = Field(6.0, gt=0, description="duration of the pitch kick, s")
    kick_angle_deg: float | None = Field(
        None, gt=0, lt=45, description="null = planned for the staging flight-path angle"
    )
    staging_flight_path_deg: float = Field(
        30.0, gt=0, lt=89, description="planner: flight-path angle at MECO"
    )
    closed_loop: bool = Field(
        True, description="track the planned flight-path angle vs speed in the gravity turn"
    )
    max_g: float = Field(5.0, gt=1, description="throttle back above this sensed acceleration")


class UpperGuidanceSpec(Spec):
    """Closed-loop upper-stage guidance (linear-acceleration terminal guidance)."""

    freeze_time: float = Field(6.0, ge=0, description="stop re-targeting below this time-to-go, s")
    max_off_tangent_deg: float = Field(
        60.0, gt=0, le=89, description="thrust direction limit from the local horizontal, deg"
    )
    coast_after_cutoff: float = Field(60.0, ge=0, description="coast simulated after SECO, s")


class BoostbackSpec(Spec):
    pitch_deg: float = Field(10.0, ge=-30, le=60, description="thrust elevation above horizontal")
    max_g: float = Field(4.0, gt=0, description="throttle back above this thrust acceleration, g")
    flip_rate_deg_s: float = Field(12.0, gt=0, description="RCS flip rate limit, deg/s")
    align_deg: float = Field(15.0, gt=0, description="ignite when within this of the attitude")
    coast_before_flip: float = Field(2.0, ge=0, description="after separation, s")


class LaunchMissionSpec(Spec):
    name: str
    description: str = ""
    kind: str = Field("launch", pattern="^launch$")
    vehicle: str = "launcher_two_stage"
    payload_mass: float | None = None
    terrain: MissionTerrainSpec = Field(default_factory=MissionTerrainSpec)
    launch: SiteSpec
    landing: SiteSpec = Field(description="booster landing zone (return to launch site)")
    landing_radius: float = Field(30.0, gt=0, description="booster landing success radius, m")
    orbit: OrbitTargetSpec = Field(default_factory=OrbitTargetSpec)
    ascent: AscentSpec = Field(default_factory=AscentSpec)
    upper_guidance: UpperGuidanceSpec = Field(default_factory=UpperGuidanceSpec)
    boostback: BoostbackSpec = Field(default_factory=BoostbackSpec)
    booster_guidance: HopGuidanceSpec = Field(
        default_factory=HopGuidanceSpec, description="booster entry / descent / landing"
    )
    world: WorldSpec = Field(
        default_factory=lambda: WorldSpec(gravity="spherical", ground="none", dt=0.01)
    )
    max_time: float = 1500.0
    record_every: int = Field(5, ge=1, description="control cycles between replay frames")


def load_launcher(path_or_name: str | Path) -> LauncherSpec:
    """Load a multi-stage vehicle (``configs/vehicles/<name>.yaml`` or a path)."""
    path = _resolve(path_or_name, "vehicles")
    data = load_yaml(path)
    if "stages" not in data:
        raise ValueError(f"{path} is a single-stage vehicle, not a launcher (no 'stages')")
    return LauncherSpec.model_validate(data)


def load_launch_mission(path_or_name: str | Path) -> LaunchMissionSpec:
    return LaunchMissionSpec.model_validate(load_yaml(_resolve(path_or_name, "missions")))
