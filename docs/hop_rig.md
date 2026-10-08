# Hop-test rig: model, scenarios and test campaign

**Vehicle:** [`configs/vehicles/hop_rig.yaml`](../configs/vehicles/hop_rig.yaml)
**Code:** `src/plume/hoprig/scenarios.py` (test flights), `src/plume/hoprig/sysid.py` (system identification), `src/plume/physics/tether.py` (tether, [model page](models/tether.md))

The rig is a **representative** small VTVL test article, not a design:

| | value |
|---|---|
| wet mass | 198 kg (140 kg dry, 55 kg propellant, 3 kg nitrogen) |
| engine | 4 kN vacuum / 3.65 kN sea-level, Isp 230 / 210 s, 30–100 % throttle, 0.15 s lag |
| gimbal | ±6°, 40 deg/s; high fidelity: 6 Hz, ζ 0.7, 20 ms delay, 0.1° backlash |
| RCS | 4 pods of 25 N nitrogen thrusters at 2.6 m |
| legs | four, 0.9 m footprint radius, 0.5 m below the hull base, 3 m/s touchdown limit |
| lift-off T/W | 1.9; hover throttle about 58 % full, 45 % nearly empty |

Every number is a placeholder. Weigh, measure and test-stand the real rig, then calibrate the file from rig logs (below) before trusting any prediction.

## Scenarios

```bash
uv run plume sim hop_rig --script tethered_hover      # 1.5 m hover on a 3 m ground tether, +0.5 m step, land
uv run plume sim hop_rig --script tether_catch        # throttle stuck at 100 % for 2 s: the tether arrests the climb
uv run plume sim hop_rig --script translation_step    # free flight: 5 m sideways step at 10 m, then land
uv run plume sim hop_rig --script free_hop --wind 3   # climb to 50 m, move 15 m to pad B, land
uv run plume sim hop_rig --script free_hop --fidelity high   # actuator dynamics, aero database
```

`data/replays/hop_rig_free_hop.plume.json.gz` is the free hop at high fidelity in 3 m/s wind, viewable with `plume viz`. The viewer does not draw the tether. The tether tension is recorded in each frame (`tether_tension`, `tether_stretch`) and its geometry in the replay metadata (`meta.tether`).

Results with the representative rig (fast fidelity, no wind, seed 0):

| scenario | outcome | key numbers |
|---|---|---|
| tethered_hover | landed | footpads peak 1.94 m (tether limit 2.5 m), rope never taut, touchdown 0.33 m/s |
| tether_catch | landed | rope peaks at about 9.6 kN (5× the rig's weight), footpads stop at 2.85 m; rated load set to 15 kN |
| translation_step | landed 0.002 m from pad B | rise time 5.2 s, no overshoot, 5 % settling 8 s |
| free_hop | landed 0.15 m from pad B | 50.0 m peak, 37 kg propellant used, 18 kg left |

The controller is the same waypoint guidance and attitude control as the lander's `hop_test`. It inverts the engine's thrust law (thrust = u·F_vac − p·A_e) for the throttle command, so that the back-pressure loss at part throttle does not leave a steady height error.

### Tether

The default tether runs from a ground anchor at the pad centre to the hull base: 3 m long, stiffness 2×10⁴ N/m (a few metres of 6–8 mm Dyneema or steel cable with a shock-absorbing link) and damping at 0.4 of critical. It is slack throughout a nominal hover and only pulls when the rig climbs above its length. A crane tether (anchor above, attached at the nose) catches the rig if it falls instead; build one with `TetherSpec(anchor=(0, 0, 12), attach=(0, 0, 3.0), length=...)`. See [models/tether.md](models/tether.md) for the equations and their verification.

## Test campaign: which test identifies which parameter

`plume rig plan` prints this table:

| test | where | inputs | identifies (vehicle YAML) | measure |
|---|---|---|---|---|
| weighing and CG | ground | weigh on three scales, dry and with known loads | `mass.dry`, `mass.dry_cg_z`, tank geometry | scale readings |
| swing test (bifilar/trifilar pendulum) | ground | small free swings about each axis | `mass.dry_inertia` | swing period |
| static hot fire, throttle steps | test stand | steps 30–100 % and back, 3–5 s each | `engine.thrust_vac`, `isp_vac`/`isp_sl`, `throttle_tau`, `throttle_min` | load cell, tank load cells or flow meter, chamber pressure |
| gimbal bench sweep | test stand | chirp 0.2–5 Hz at ±1° and ±4°; steps; slow triangle | `gimbal_tau`, `gimbal_wn_hz`, `gimbal_zeta`, `gimbal_delay_s`, `gimbal_rate_deg_s`, `gimbal_backlash_deg` | commanded vs measured nozzle angle |
| tethered hover: throttle doublets | tethered | ±0.10 doublets (1 s) and a 3-2-1-1 multistep | installed thrust, throttle lag, Isp | accelerometer, tether load cell, tank load cells, height |
| tethered hover: gimbal chirps | tethered | ±1° chirp 0.3–3 Hz, one axis at a time, RCS pitch/yaw off | pitch/yaw inertia, gimbal dynamics under thrust | gyro, measured gimbal angles, throttle |
| RCS pulse train | tethered or stand | single-thruster pulses of 20–200 ms | `rcs.thrust`, `min_on_time_s`, `valve_delay_s` | gyro (roll), valve commands, bottle pressure |
| translation step | free flight (or long tether) | 5 m waypoint step at constant height | closed-loop response | GNSS/RTK position, attitude |
| leg drop test | ground | drops from increasing heights | `legs.max_touchdown_speed`, leg stroke | accelerometer peak, stroke, video |
| free hop | free flight | the planned profile | everything together | the full log, predicted beforehand |

**Why these inputs.**
- A *doublet* (up, then down, the same size) excites the throttle and vertical dynamics without changing the average height, so the rig stays inside the tether.
- A *3-2-1-1* multistep spreads energy over a wider band of frequencies.
- A *chirp* (a sine sweep whose frequency rises) excites the gimbal actuator and the attitude dynamics across the whole control bandwidth. One axis at a time keeps the two axes separate.
- Turning off RCS pitch and yaw during a chirp means the gimbal alone produces the torque, so torque and response can be paired cleanly.

Abort a chirp on an attitude limit (for example 10°) or if the tether goes taut.

## Rig-log format

The rig's flight software should log these columns at 100 Hz or faster (CSV, header row, `#` comments allowed). `plume rig sysid` writes the same format from the simulator:

| column | content |
|---|---|
| `time_s` | time since logging started, s |
| `manoeuvre` | test-sequencer step (`hover`, `throttle_doublet`, `throttle_3211`, `gimbal_chirp_x`, `gimbal_chirp_y`, ...) |
| `throttle_cmd` | commanded throttle 0–1, as sent to the valves |
| `gimbal_cmd_x_deg`, `gimbal_cmd_y_deg` | commanded nozzle angles |
| `gimbal_x_deg`, `gimbal_y_deg` | measured nozzle angles (LVDT/encoder) |
| `gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps` | body rates |
| `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2` | specific force at the CG, body axes (z reads +9.8 at rest) |
| `prop_mass_kg` | propellant on board (tank load cells or integrated flow meter) |
| `tether_tension_n` | tether load cell (0 untethered) |
| `rcs_cmd_x`, `rcs_cmd_y`, `rcs_cmd_z` | RCS torque commands, −1 to 1 of capability |
| `height_m` | footpad height above the pad (laser or barometer) |

Required: `time_s`, `throttle_cmd`, `accel_z_mps2` and `prop_mass_kg`. The gimbal, gyro and RCS columns are needed for the actuator and inertia fits. Convention: the command in a row is the one issued at that sample time, which acts until the next row; the sensors in a row were sampled at that time.

## Calibration path

```bash
uv run plume rig sysid --out runs/rig/sysid_log.csv     # rehearse on a simulated log (or skip: use the real one)
uv run plume rig calibrate runs/rig/sysid_log.csv --vehicle hop_rig
```

`plume rig calibrate` fits, and writes a calibrated vehicle YAML:

1. **Engine.** Measured thrust is (m·a_z + tether tension) / cos(nozzle angle), with m the weighed dry mass plus the logged propellant, airborne samples only. It is fitted to F = u·F_vac − p_amb·A_e, where u is the commanded throttle passed through the engine's first-order lag. A one-dimensional search over the lag finds `throttle_tau`; for each lag, a linear least-squares fit gives F_vac and p·A_e. The propellant record then gives the mass flow (prop = P₀ − ṁ_max ∫u dt), and from that Isp_vac = F_vac/(ṁ_max g₀) and Isp_sl. These are the same equations as `plume.physics.propulsion.Engine`.
2. **Gimbal actuator.** A first-order lag with a rate limit (fast fidelity: `gimbal_tau`) and a second-order model with delay (high fidelity: `gimbal_wn_hz`, `gimbal_zeta`, `gimbal_delay_s`) are both fitted to the commanded and measured nozzle angles in the chirp windows. The second-order values are only written when the data show second-order behaviour: a clearly better fit than first order, not pinned at a bound.
3. **Inertia.** Over 0.3 s windows in the chirps, the change in body rate is fitted to s·∫(τ/I_model)dt. The torque τ is the gimbal arm times thrust times nozzle angle, plus the RCS command times its capability. I_model is the vehicle file's inertia at the logged propellant load, so I_true = I_model / s. Integrating instead of differentiating avoids amplifying gyro noise. The scale is applied to `mass.dry_inertia` (pitch and yaw).

**Checked on simulated data.** The test `tests/test_rig_sysid.py` flies a "true" rig with thrust 5 % lower, Isp 4 % lower, a 47 % slower throttle, a 50 % slower gimbal and 10–16 % more inertia than the nominal file, with sensor noise. Calibrating the nominal file on that log recovers:
- thrust and Isp within 1.5 %;
- throttle lag within 10 %;
- gimbal lag within 15 %;
- inertia within 5 % (one seeded run: thrust 3810 against 3800 N, Isp 220.6 against 220 s, lag 0.223 against 0.22 s, gimbal 0.060 against 0.06 s, inertia 108.6 against 110 kg m²).

At high fidelity the second-order fit gives 5.3 Hz and ζ 0.80 against a true 4.5 Hz and 0.70: the 0.1° backlash and the acceleration limit are not in the fitted model. Measure backlash separately on the bench (slow triangle input).

**Then close the loop:** refly `tethered_hover` and `translation_step` with the calibrated file and compare them with the logged response. The calibrated file is not validated by the log it was fitted to; the next test, predicted beforehand, is the check.

**Limits.**
- Aerodynamic and ground-effect forces near the pad are ignored in the thrust estimate. Use data with the footpads more than about one engine diameter above the ground; the default cut is 0.3 m.
- The inertia fit needs thrust well above zero and a slack tether.
- The fits assume the mass model (dry mass, CG and inertia against propellant load) from weighing; errors there go straight into thrust and inertia.
