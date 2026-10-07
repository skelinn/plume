from __future__ import annotations

import pytest

from plume.config import VehicleSpec, WorldSpec, load_vehicle


@pytest.fixture
def lander() -> VehicleSpec:
    return load_vehicle("lander_small")


def make_test_vehicle(**overrides) -> VehicleSpec:
    """A simple vehicle for conservation tests: no legs, optional aero/RCS."""
    data = {
        "name": "test_body",
        "geometry": {"length": 8.0, "diameter": 1.0, "nose_length": 1.0},
        "legs": {"count": 0},
        "mass": {"dry": 800.0, "dry_cg_z": 3.0, "dry_inertia": [3000.0, 3400.0, 600.0]},
        "tanks": [
            {"name": "main", "capacity": 600.0, "z_bottom": 1.0, "z_top": 5.0, "radius": 0.45}
        ],
        "engine": {
            "type": "liquid",
            "thrust_vac": 20000.0,
            "isp_vac": 310.0,
            "isp_sl": 280.0,
            "throttle_min": 0.4,
            "throttle_tau": 0.0,
            "gimbal_max_deg": 6.0,
            "gimbal_rate_deg_s": 1000.0,
            "gimbal_tau": 0.0,
            "gimbal_z": 0.5,
        },
        "rcs": {"enabled": True, "thrust": 100.0, "isp": 70.0, "propellant": 10.0, "z": 7.0},
        "aero": {"enabled": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return VehicleSpec.model_validate(data)


@pytest.fixture
def test_vehicle() -> VehicleSpec:
    return make_test_vehicle()


@pytest.fixture
def vacuum_world() -> WorldSpec:
    return WorldSpec(atmosphere=False, ground="none")
