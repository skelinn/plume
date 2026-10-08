# Sensors and navigation

**Code:** `src/plume/physics/sensors.py`, `src/plume/control/navigation.py`
**Selection:** `WorldSpec.navigation: auto | truth | ekf`. With `auto`, high fidelity uses the EKF and fast mode uses the true state.
**Verification status:** verified, covering noise statistics, the GNSS rate and latency, the barometric inversion, and filter accuracy on the pad and in powered flight (`tests/test_navigation.py`). Validation against flight data is pending.

## What the flight software sees

In high fidelity the guidance and control loops (cargo-hop autopilot, landing autopilot, attitude control) act on `Navigator.estimate()`, not on the simulator's state. Scoring uses the truth.

| Quantity | Source |
|---|---|
| position, velocity, attitude, body rates | EKF (IMU + GNSS + baro + radar altimeter) |
| altitude, height above ground | EKF position plus the onboard terrain map |
| airspeed, Mach, dynamic pressure | EKF velocity minus the *forecast* wind (no air-data probe) |
| mass, propellant | propellant gauging (true value) |
| thrust, gimbal angle, fin angles | chamber pressure and resolvers (true value) |
| leg contact | touchdown switches (true value) |
| g-load | accelerometers |

## Sensor models (defaults in `SensorsSpec`)

**IMU:** a tactical-grade, HG1700-class default, at the CG.
- **Error sources (gyro and accelerometer):**
  - turn-on bias
  - first-order Gauss–Markov bias instability, τ = 600 s
  - white noise (ARW 0.125°/√h, VRW 0.06 m/s/√h)
  - 300 ppm scale factor
  - 0.5 mrad misalignment
  - quantisation and range saturation
- **Outputs:**
  - The outputs are the averages over each interval, matching delta-angle and delta-velocity sensors.
  - The gyro senses the inertial rate, Earth rate included.
  - The accelerometer senses specific force.

**GNSS:**
- 10 Hz, 50 ms latency.
- Noise: 1.5 m horizontal / 3 m vertical. About half of that variance is white; the rest is a Gauss–Markov bias with τ = 300 s, standing in for multipath and atmospheric errors.
- 0.05 m/s velocity noise.
- An optional outage above a given altitude models COCOM-limited receivers.

**Barometric altimeter:**
- Static pressure converted to altitude through the standard-atmosphere inverse.
- On an off-standard day, or with NRLMSISE-00 or a sounding, that conversion produces a physical altitude bias.
- 1 m noise, a 5 m calibration bias, and no readings below 1 kPa.

**Radar altimeter:** height of the CG above the terrain. Range up to 2.5 km, tilt up to 30°, noise σ = 0.1 m + 0.5 % of range.

## Filter

Error-state EKF, 15 states: world-frame δp, δv and δθ, gyro bias and accelerometer bias.

**Propagation (each 50 ms flight-software cycle):**
- Strapdown integration with mid-interval attitude.
- Gravity uses the same field model as the simulator.
- On a rotating Earth, the Coriolis and centrifugal terms are added and the Earth rate is removed from the gyro.
- Process noise comes from ARW and VRW, plus force-proportional scale-factor and misalignment terms and the Gauss–Markov bias driving noise.

**Updates:**
- **GNSS:** position and velocity, latency-compensated by back-propagating the estimate to the fix time.
- **Baro:** weight drops with altitude (3 % of altitude).
- **Radar altimeter:** near the ground.
- All updates use Joseph-form covariance updates with a χ² outlier gate.

**Initial alignment:** 2 m, 0.05 m/s, and 0.05°/0.05°/0.3° in roll/pitch/heading. This represents pad levelling plus a surveyed or gyrocompassed heading.

## Results

- **On the pad:** about 0.3 m position error after 60 s, and 0.24° attitude error (the initial heading).
- **750 km high-fidelity hop:** at most 4.8 m position error (rms 3.4 m), 0.09 m/s velocity and 0.24° attitude.
- **Effect on accuracy:** navigation is not the limiting factor for landing accuracy (see `guidance.md`).

## Limitations

- **IMU location:** no lever arm between the IMU and the CG, and no structural vibration or coning/sculling environment.
- **GNSS:** no carrier-phase or RTK positioning; there is no RTK/landing-pad beacon option yet.
- **Terrain map:** treated as perfect for the radar altimeter.
- **Directly sensed quantities:** mass and propellant gauging have no error yet.
