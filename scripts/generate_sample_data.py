"""Generate the bundled sample flight logs (synthetic "real" flights).

    uv run python scripts/generate_sample_data.py

The "truth" is the 6-DOF simulator flying ``hobby_rocket`` with parameters that
differ from the nominal model (more drag, a weaker and longer motor burn, a
smaller parachute), on a launch rail, in gusty wind. Sensor models add noise,
bias, quantisation and range limits, and the logs are written in two common
hobby-altimeter styles, so the import + calibration pipeline has a known answer:

* ``data/flights/sample_flight.csv``  (ms / ft / g, gyro, sparse GPS) ->
  ``configs/flightlogs/generic_altimeter.yaml``
* ``data/flights/sample_flight_b.csv`` (semicolon, s / m / m/s^2 gravity-removed) ->
  ``configs/flightlogs/simple_altimeter.yaml``
* ``data/flights/sample_truth.json``  the true parameters
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from plume.config import WindSpec, WorldSpec, load_vehicle
from plume.constants import G0
from plume.physics.propulsion import read_eng
from plume.physics.sim import RocketSim, quat_from_axis_angle

OUT = Path("data/flights")
TRUTH = {
    "cd_scale": 1.22,
    "impulse_scale": 0.94,
    "time_scale": 1.07,
    "chute_cd_area": 0.47,
    "rail_length": 1.5,
    "rail_tilt_deg": 4.0,
}


def truth_vehicle():
    v = load_vehicle("hobby_rocket")
    curve, _ = read_eng(v.engine.motor_file)
    curve = curve.copy()
    curve[:, 0] *= TRUTH["time_scale"]
    curve[:, 1] *= TRUTH["impulse_scale"] / TRUTH["time_scale"]
    v.engine.thrust_curve = curve.tolist()
    v.engine.motor_file = None
    v.aero.cd_scale = TRUTH["cd_scale"]
    v.recovery.chutes[0].cd_area = TRUTH["chute_cd_area"]
    return type(v).model_validate(v.model_dump())


def fly(seed: int, wind: float, rate_hz: float = 100.0):
    v = truth_vehicle()
    world = WorldSpec(
        ground="none",
        dt=0.002,
        wind=WindSpec(
            speed=wind, from_deg=300, turbulence=0.25 * wind, gust_rate=0.05, gust_max=wind
        ),
    )
    sim = RocketSim(v, world, seed=seed)
    tilt = math.radians(TRUTH["rail_tilt_deg"])
    sim.reset(pos=(0, 0, 0), quat=quat_from_axis_angle([1, 0.3, 0], tilt), seed=seed)
    sim.set_rail(TRUTH["rail_length"])
    steps = round(1.0 / rate_hz / world.dt)
    rows = []
    v_prev = sim.state.vel_com.copy()
    g = np.array([0.0, 0.0, -G0])
    while sim.t < 400:
        sim.step(steps)
        st = sim.state
        a = (st.vel_com - v_prev) / (steps * world.dt) - g
        v_prev = st.vel_com.copy()
        rows.append((sim.t, st.com.copy(), float(a @ st.axis), st.omega.copy()))
        if st.altitude < -0.5 and sim.t > 5:
            break
    return rows


def sensors(rows, rng, pad_s=3.0, post_s=4.0, rate_hz=100.0, t_boot=12.345, ground_asl=312.0):
    dt = 1.0 / rate_hz
    pre = [(-pad_s + k * dt, np.zeros(3), G0, np.zeros(3)) for k in range(int(pad_s * rate_hz))]
    last = rows[-1]
    post = [
        (last[0] + (k + 1) * dt, last[1] * [1, 1, 0], G0, np.zeros(3))
        for k in range(int(post_s * rate_hz))
    ]
    out = []
    acc_bias = rng.normal(0, 0.03 * G0)
    for t, pos, acc, w in pre + rows + post:
        alt = ground_asl + float(pos[2]) + rng.normal(0, 0.6)
        a = acc + acc_bias + rng.normal(0, 0.08 * G0)
        a = float(np.clip(a, -24 * G0, 24 * G0))
        gyro = np.degrees(w) + rng.normal(0, 0.3, 3)
        out.append((t + t_boot, alt, a, gyro, pos.copy()))
    return out


def write_a(samples, path: Path, rng):
    lat0, lon0 = 32.9901, -106.9752
    lines = [
        "# Plume sample flight A -- synthetic log in a generic hobby-altimeter export style",
        "time_ms,baro_alt_ft,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps,gps_lat,gps_lon,gps_alt_m",
    ]
    for k, (t, alt, a, gyro, pos) in enumerate(samples):
        alt_ft = round(alt / 0.3048)  # 1 ft resolution
        gps = ["", "", ""]
        if k % 10 == 0:  # 10 Hz GPS fixes
            north = pos[1] + rng.normal(0, 2.0)
            east = pos[0] + rng.normal(0, 2.0)
            lat = lat0 + math.degrees(north / 6_371_000.0)
            lon = lon0 + math.degrees(east / (6_371_000.0 * math.cos(math.radians(lat0))))
            gps = [f"{lat:.7f}", f"{lon:.7f}", f"{312.0 + pos[2] + rng.normal(0, 4.0):.1f}"]
        lines.append(
            f"{t * 1000:.0f},{alt_ft},{a / G0:.3f},{gyro[0]:.2f},{gyro[1]:.2f},{gyro[2]:.2f},{','.join(gps)}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_b(samples, path: Path):
    lines = ["Time (s);Altitude (m);Accel (m/s2)"]
    for k, (t, alt, a, _, _) in enumerate(samples):
        if k % 2:  # 50 Hz
            continue
        lines.append(f"{t:.3f};{alt:.2f};{a - G0:.2f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    a = sensors(fly(seed=3, wind=4.0), rng)
    write_a(a, OUT / "sample_flight.csv", rng)
    b = sensors(fly(seed=8, wind=2.0), rng, t_boot=0.0, ground_asl=1401.0)
    write_b(b, OUT / "sample_flight_b.csv")
    (OUT / "sample_truth.json").write_text(json.dumps(TRUTH, indent=2) + "\n")
    print(
        "wrote",
        OUT / "sample_flight.csv",
        len(a),
        "rows;",
        OUT / "sample_flight_b.csv",
        len(b) // 2,
        "rows",
    )


if __name__ == "__main__":
    main()
