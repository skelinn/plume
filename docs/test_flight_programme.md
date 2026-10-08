# Test-flight programme: from simulation to first real data

Plume's models are **verified** (the maths is implemented correctly) but **not validated**: no model has yet been compared with a real flight. This guide is the plan for getting that evidence, written for a small team without specialist flight-test staff. It covers what to fly, what to measure, how to log it, and how to turn a log into a validation record that `plume vv-report` shows.

The programme has three phases. Each phase validates the models the next one depends on, so skipping ahead wastes the expensive flights.

| Phase | Vehicle | What it validates | Cost and risk |
|---|---|---|---|
| 1 | Instrumented hobby rocket on a commercial solid motor | drag, solid-motor thrust curve, parachute descent, wind drift, barometer and accelerometer processing, the whole data pipeline | low; can fly within weeks |
| 2 | Tethered hop rig (`configs/vehicles/hop_rig.yaml`) | liquid engine thrust, Isp and throttle response, gimbal actuator, inertia, attitude control, tether loads | medium; needs a test site and an engine |
| 3 | Free hops of the rig, up to about 50 m | guidance and landing, navigation filter, landing-leg loads, propellant budget | high; needs airspace and range-safety approval |

The honest validation method is the same in every phase:

1. **Predict before you fly.** Run the simulator for the planned flight, with uncertainty bands, and commit the prediction to git *before* the flight. The commit timestamp proves the prediction was not adjusted afterwards.
2. **Fly and log** the quantities the prediction covers.
3. **Compare** the flight with the bands. A model passes only if the flight falls inside the bands it predicted.
4. **Calibrate** the model to the flight, then predict the *next* flight with the calibrated model. A calibrated model is not validated by the flight it was calibrated on; only the next blind prediction can validate it.

---

## Phase 1: instrumented hobby rocket

### Why start here

A high-power hobby rocket (for example a 1–1.5 m airframe on a 29 mm H motor, like `configs/vehicles/hobby_rocket.yaml`) flies to several hundred metres in about 10 seconds, costs little, and exercises the same drag, thrust, parachute, wind and sensor-processing code as the cargo vehicle. It also proves the team's data pipeline (logging, export, import, calibration) before anything expensive depends on it.

### Flights and objectives

| Flight | Configuration | Objective | Plume model (docs/models) |
|---|---|---|---|
| 1-1 | nominal rocket, motor A (e.g. H-class) | first blind prediction; drag, thrust curve, parachute | `aero.md` (axial drag), `propulsion.md` (solid motor), recovery (`physics/recovery.py`) |
| 1-2 | identical repeat | repeatability: how much do two "identical" flights differ? This sets realistic uncertainties | all of the above |
| 1-3 | same airframe, motor B (different impulse, e.g. G-class) | separates drag from motor impulse (one flight cannot fully tell "more drag" from "weaker motor") | `aero.md`, `propulsion.md` |
| 1-4 | ballast added (+20 % mass, CG unchanged) | checks the mass model and the drag fit at a different speed | `mass.md`, `aero.md` |
| 1-5 | windy day (5–8 m/s, within the safety code) | weathercocking and drift; tests the wind model and the landing-dispersion prediction | `atmosphere.md` (wind) |

Before flight 1-2, predict it with the model calibrated on 1-1. That second prediction is the first genuine validation test.

### What each quantity tells you

| Measured | From | Validates |
|---|---|---|
| apogee, time to apogee | barometer | drag and total impulse together |
| boost acceleration profile | axial accelerometer | thrust curve shape and burn time |
| coast deceleration | axial accelerometer | drag coefficient on its own |
| descent rate | barometer after apogee | parachute drag area (Cd·A) |
| landing point | GNSS | wind drift (and the landing-dispersion prediction) |
| body rates during boost | gyro | stability and coning (Plume's 3-DOF prediction does not model these; large rates mean the point-mass assumption is weak) |

### Instrumentation

One flight computer that logs to onboard memory is enough. Recommended specification:

| Sensor | Range | Sample rate | Notes |
|---|---|---|---|
| accelerometer (3-axis, or at least the axial axis) | at least ±24 g for an H motor (the bundled sample flight peaks near 18 g); a separate high-g channel is ideal | 100 Hz minimum, 400–1000 Hz during boost | a clipped accelerometer ruins the thrust-curve fit; check the range against the predicted peak |
| gyro (3-axis) | ±2000 deg/s | 100 Hz or more | only for checking stability and for the replay attitude |
| barometer | full scale to well above the predicted apogee | 20–50 Hz | the main altitude source |
| GNSS | – | 5–10 Hz | set the receiver's dynamic model to an airborne mode if it has one; consumer receivers may drop out during boost, which is fine, but they must reacquire for the landing point |

**Mounting**

- Mount the flight computer rigidly (no foam, no loose sled). Vibration and flex corrupt the accelerometer.
- Align the accelerometer's axial axis with the airframe axis and **record which way it points**. If it points toward the tail it reads −1 g on the pad; the mapping then needs `scale: -1` (`plume flightlog inspect` warns about this).
- Mount it as close to the CG as is practical; measure and record the distance.
- Vent the avionics bay with several small static-port holes around the circumference, away from the nose-cone shoulder and other disturbances, sized according to the altimeter maker's guidance. Badly sized ports cause pressure lag and false apogee detection.

**Time synchronisation**

- Use the flight computer's own clock for all its channels (one time column).
- Record a UTC reference: a GNSS time column, or note the clock-to-UTC offset at power-up. The validation record needs the flight's UTC date and time to prove the prediction came first.
- If you film the flight, make the logger beep or flash at a known log time, or film the logger's display, so video and log can be aligned.

### Data to log and the CSV columns Plume needs

Log from **at least 5 seconds before liftoff until after landing**. Plume zeroes the barometer on the pad data (the last 1 s before liftoff) and measures the descent rate and landing point after apogee.

| Column (example name) | Unit | Required? | Used for |
|---|---|---|---|
| `time_ms` | ms, s or µs | **required** | everything |
| `baro_alt_ft` | m, ft or km (any datum) | **required** | apogee, descent rate, calibration |
| `accel_z_g` | g, m/s² or ft/s² (state whether gravity is included) | strongly recommended | liftoff detection, thrust curve, drag |
| `gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps` | deg/s or rad/s | optional | replay attitude, stability check |
| `gps_lat`, `gps_lon` | degrees | optional (needed for the landing-point check) | landing point, drift |
| `gps_alt_m` | m | optional | cross-check of the barometer |

The column names and units of your flight computer go in a small YAML **mapping**. Plume does not ship mappings for commercial flight computers, because their CSV layouts change between models and firmware versions and a wrong unit silently corrupts a calibration. Instead:

```bash
uv run plume flightlog inspect my_flight.csv
```

prints every column with its range, guesses which column is which and in which unit (from the column names and the values: a clock step of 10 means milliseconds, a pad reading of 1.0 means g with gravity included, and so on), and writes a **draft** mapping to `runs/flightlogs/my_flight_draft.yaml`. It tries the draft at once and prints the apogee it gives. Check every line against the flight computer's manual, then save it as `configs/flightlogs/<your_logger>.yaml`. [`configs/flightlogs/template.yaml`](../configs/flightlogs/template.yaml) documents every field. Do this with a **ground test log** (power the logger, carry it up a flight of stairs) before the first flight, not after.

### Pre-flight checklist (hobby rocket)

**Days before**

- [ ] Rocket weighed (with recovery gear, without motor) and CG measured; numbers entered in the vehicle YAML (`mass.dry`, `mass.dry_cg_z`).
- [ ] Motor chosen; its `.eng` thrust-curve file in `data/motors/` (from the manufacturer or the certification data).
- [ ] Mapping file tested on a ground log (`plume import-log ground_test.csv --mapping <yours>`).
- [ ] Flight computer: accelerometer range adequate for the predicted peak g; logging rate set; memory erased.

**Day before / morning of the flight**

- [ ] Wind forecast or measurement at the field.
- [ ] Prediction recorded and committed:
  ```bash
  uv run plume predict hobby_rocket --motor H180 --wind 4 --wind-from 250 --rail 1.5 --rail-tilt 3 --flight-id flight_1-1
  git add docs/predictions/flight_1-1.* && git commit -m "Prediction for flight 1-1" && git push
  ```
  The report lists the apogee band, the descent rate and the **95 % landing ellipse**: brief the recovery team and keep spectators and roads clear of it.
- [ ] Motor lot number and the rocket's final mass (with motor) written down.

**At the pad**

- [ ] Logger armed and confirmed logging (beeps / LED) before the igniter goes in.
- [ ] Rail length and actual tilt and direction measured (phone inclinometer is fine).
- [ ] Wind speed and direction measured at the pad; temperature noted.
- [ ] Clock-to-UTC note or GNSS fix confirmed.

### Post-flight checklist

- [ ] Download the raw log the same day. **Keep the raw file unchanged**; make a copy before any editing. Plume records its SHA-256 hash in the validation record.
- [ ] Write down: flight UTC time, weather, anything unusual (late deployment, coning, broken fin, motor anomaly), the landing GPS position if the logger had none.
- [ ] Inspect the rocket and motor casing; photograph damage.
- [ ] Run the validation (next section) and commit the record.

### Calibration and validation workflow

```bash
# 1. look at the flight (liftoff detection, apogee, fused velocity) and make a replay
uv run plume import-log flight_1-1.csv --mapping my_logger --vehicle hobby_rocket

# 2. compare with the prediction, calibrate, write the validation record
uv run plume validate flight_1-1.csv --mapping my_logger \
    --prediction docs/predictions/flight_1-1.json --flight-id flight_1-1 \
    --flight-date 2026-11-14T15:42Z --notes "wind 4 m/s gusting 6, clean deployment"

# 3. regenerate the V&V report (now lists the record)
uv run plume vv-report
```

`plume validate` does the following:

1. **Measures the flight**: apogee, time to apogee, peak vertical speed and acceleration, descent rate, flight time and landing distance.
2. **Places each value in the prediction's distribution**, giving its percentile, and checks it lies inside the 95 % band.
3. **Calibrates** drag (`cd_scale`), total impulse, burn time and parachute Cd·A by least squares on the 3-DOF model (the same fit as `plume calibrate`). It then checks each fitted factor lies within two standard deviations of 1, using the uncertainty the prediction assumed. For example, a fitted drag factor of 1.25 with an assumed 10 % uncertainty is a discrepancy: the drag model, or its stated uncertainty, is wrong for this airframe.
4. **Gives a verdict per model**:
   - **validated**: every criterion passed on real data with a prediction recorded before the flight. This applies only within the stated envelope (this vehicle and motor, the Mach range flown).
   - **consistent**: the criteria passed but the test was not blind.
   - **discrepancy**: a criterion failed.
   - **inconclusive**: the data needed was not logged.
5. **Writes** `docs/validation/<flight-id>.md` and `docs/validation/<flight-id>/` (`record.json`, an overlay plot of real against simulated flight, the calibrated vehicle and a copy of the prediction).

`plume vv-report` reads the records. A model page's row changes from "Validation: pending real data" to, for example, "flight_1-1: validated (hobby_rocket on the H180 motor; subsonic ...); elsewhere pending". The "models validated" count only counts real, blind, passing flights. Records made with `--synthetic` (simulated logs, such as the bundled samples) are listed as rehearsals and never change a model's status. Rehearse the whole workflow on the bundled synthetic log before the first flight:

```bash
uv run plume predict hobby_rocket --motor H180 --wind 0 --rail-tilt 4 --flight-id rehearsal --out-dir runs/rehearsal
uv run plume validate data/flights/sample_flight.csv --mapping generic_altimeter \
    --prediction runs/rehearsal/rehearsal.json --flight-id rehearsal --synthetic --out-root runs/rehearsal
```

The rehearsal gives a **discrepancy** for drag and motor. The synthetic "truth" was built with 22 % more drag and a motor 6 % weaker and 7 % longer-burning than the model, deliberately outside the assumed uncertainties. This is how a real discrepancy looks.

**After a discrepancy:**
- Fly the next flight with the calibrated vehicle (`docs/validation/<id>/calibrated_vehicle.yaml`) and a new prediction.
- If the calibrated factors repeat from flight to flight, the model is biased for this airframe. Keep the correction and note it in the model page.
- If they scatter, the uncertainty is larger than assumed: widen it (`--cd-sigma`).

**Updating the model documentation.** When a model is validated, add one sentence to the **Verification status** line of its page in `docs/models/` naming the flight record and the envelope, for example: "Axial drag validated against flight 1-2 (docs/validation/flight_1-2.md) for M < 0.5, small angle of attack, 66 mm finned airframe." Leave "validation pending" for everything outside that envelope.

---

## Phase 2: tethered hop rig

The rig (`configs/vehicles/hop_rig.yaml`) is **representative, not a design**: about 200 kg, a 4 kN throttleable pressure-fed engine, a 6° gimbal, nitrogen cold-gas RCS and four legs. Replace every number with weighed and test-stand data before using any prediction for a real test. The test campaign, the rig-log format and the system-identification workflow are in [hop_rig.md](hop_rig.md). In short:

| Step | Validates | Plume |
|---|---|---|
| weighing, CG, swing tests | mass properties | `mass.md` |
| static hot fire on a load cell | thrust law, Isp, throttle lag, minimum throttle | `propulsion.md` |
| gimbal bench sweep | gimbal actuator (lag, natural frequency, delay, backlash) | `propulsion.md` |
| tethered hover with throttle doublets and gimbal chirps | installed thrust, throttle lag, pitch/yaw inertia | `plume rig calibrate` |
| tethered hover, closed loop | attitude control, tether loads | `plume sim hop_rig --script tethered_hover` |
| deliberate stuck-throttle test (on the tether, low height) | tether sizing and the catch | `--script tether_catch` |

**Predict each rig test the same way:** run the scenario with the current (calibrated) rig file, save the replay and outcome, and commit them before the test.

**Safety and legal (general).**
- Liquid-propellant and pressurised systems are hazardous. Proof-test every pressure vessel and line. Operate remotely from behind a barrier with a defined exclusion zone. Have written abort and safing procedures.
- Storing and handling propellants may need local permits; involve the local fire authority early.
- The tether and its anchor must hold the full thrust with a margin, as `tether_catch` shows: a 2 s stuck throttle loads the default rope to about 10 kN, five times the rig's weight. Size the rope, the shackles and the ground anchor for the worst case, not for the hover.
- Tethered tests on private land may still need notifications to the local aviation authority if the rig can leave the ground; check before the first test.

## Phase 3: free hops

Only after the tethered tests have calibrated the engine, actuators and inertia, and the closed-loop tethered hover matched its prediction:

- `plume sim hop_rig --script translation_step`: a 5 m sideways step at 10 m height, reporting rise time, overshoot and settling time to compare with the logged response.
- `plume sim hop_rig --script free_hop`: climb to 50 m, move 15 m to a second pad, land. Predict landing accuracy, touchdown speed and propellant left, and fly it in Monte Carlo with wind before the real hop.
- A free-flying vehicle needs an independent flight-termination method, a defined hazard area, and airspace authorisation from the national aviation authority. Allow months for this.

What free hops validate: guidance and landing (`guidance.md`), the navigation filter (`navigation.md`: log the raw IMU, GNSS and altimeter data, not just the filter output), leg loads, and the propellant budget.

## Safety and legal notes for hobby rockets (general)

This is orientation, not legal advice. Rules differ by country; check the current rules before every season.

- **Join the national rocketry association** and fly at its sanctioned launches. Associations publish a safety code: minimum distances by motor class, maximum launch-rail angle, wind limits, recovery requirements and range procedures. High-power motors normally need a **certification** from the association (step by step from the smaller classes), and the motors can usually only be bought by certified flyers.
- **Airspace.** Above a size or impulse threshold, rocket flights need permission or a notification to the national aviation authority (usually arranged by the launch organiser as a waiver or notice to airmen for the launch window). Never fly outside the authorised window, altitude or area. Use the predicted apogee band to check the flight stays below the ceiling, including the upper end of the band.
- **The prediction is also a safety tool.** Keep the landing ellipse clear of roads, buildings and spectators; do not fly if the wind moves it outside the field.
- **Motors** are explosives in many jurisdictions: storage and transport rules apply. Use only certified commercial motors for the validation flights. Their thrust curves are documented, which is exactly what the calibration compares against.
- **Range Safety Officer.** At sanctioned launches the RSO inspects the rocket; the logged data does not replace that inspection.

## What Plume's prediction does not cover (yet)

- The 3-DOF prediction assumes the rocket points into the relative wind instantly after leaving the rail. Real weathercocking is slower: in wind, expect slightly lower apogee and the landing slightly upwind of the prediction.
- Coning, roll and fin flutter are not modelled. Large gyro rates in the log mean the point-mass comparison is weaker.
- The parachute opens exactly as configured; tangles and late or early deployments are not modelled. Note them in the record, because they invalidate the descent-rate check for that flight.
- On the hop rig: no ground effect or plume recirculation near the pad, no engine start transient beyond the ignition delay, no tether mass or sag, and no leg-stroke dynamics (others are working on landing-gear models).
