# Guidance, navigation and control: the cargo hop

**Code:** `src/plume/missions/hop.py` (`HopAutopilot`, `plan_ascent`), `src/plume/missions/targeting.py` (`ImpactPredictor`, `kepler_impact`), `src/plume/control/autopilot.py` (`LandingAutopilot`), `src/plume/control/attitude.py`
**Verification status:** not applicable: this is the *vehicle's* flight software, not a physics model. The simulator's job is to show how well it works; the Monte Carlo campaigns below measure it. On the real route at high fidelity, 87.5 % of runs (95 % CI 77–94 %) land the cargo on target.

## Flight phases

1. **Liftoff and vertical rise:** `rise_time` from the pre-flight plan.
2. **Pitch kick and kick hold:** hold the kick attitude until the flight path has pitched over as far as the kick. Only the elevation is compared. Requiring a full 3-D match timed out in crosswinds and lofted the trajectory to 380 km; Monte Carlo found this.
3. **Gravity turn:**
   - Follow the ground velocity, with a cross-range correction from the vacuum impact point.
   - The throttle is limited to the cargo g-limit.
   - MECO is timed at the zero crossing of the predicted along-track error, using a drag-aware 3-DOF prediction that includes the entry burn.
4. **Coast:** grid fins deploy, and RCS flips the vehicle engine-first.
5. **Entry burn:**
   - Starts below `entry_altitude` while faster than `entry_speed`; flown retrograde and g-limited.
   - The cutoff is timed on the *coast-from-now* impact prediction: cut when it reaches the target, once below the entry-speed (heating) limit, with a floor at 0.75 × `entry_speed`.
   - The cross-track miss is steered by tilting off retrograde.
   - Earlier versions predicted with the burn continuing to `entry_speed` while cutting on a different rule. That inconsistency alone cost about 300 m.
6. **Aerodynamic descent:** the vehicle is steered with body lift onto the predicted miss, and the drag-scale estimate (measured versus modelled deceleration) feeds the predictor.
   - The allocator requests a *change* from the zero-tilt aerodynamic force. On an inclined supersonic descent the drag has a large horizontal component; asking for an absolute force made the allocator fight it, and the impact point ran away (−44 m to +1.7 km).
   - **High fidelity:** steering is active up to Mach 5, with tilt caps of 15° supersonic and 20° subsonic. The aero database gives a monotonic side force when the engine-first body is tilted (the body moves *away* from the tilt): 2–5 m/s² at Mach 1.5 and 15–20°, about 0.4 m/s² subsonic.
   - **Fast fidelity:** subsonic only, 12° cap. Strip theory gives a non-monotonic, sign-flipping side force beyond about 7°, and supersonic steering made every test case worse there.
7. **Landing burn:**
   - Constant-deceleration hoverslam profile in the vertical, ZEM/ZEV feedback in the horizontal, and aero-aware thrust/tilt allocation.
   - If the predicted miss exceeds `max_divert_m` (default 250 m), the vehicle retargets to the closest reachable point rather than spending its landing propellant on an impossible divert.
   - Legs deploy below 50 m/s or 200 m.

In high fidelity all of this acts on the navigation estimate (see `navigation.md`). Planning uses the *nominal* vehicle and the *forecast* environment, while the simulator flies the dispersed truth.

## Monte Carlo results: `real_hop`, high fidelity, 64 runs

| version | change | success (95 % CI) | recovered | CEP50 |
|---|---|---|---|---|
| v4 | hoverslam, open-loop gravity turn, subsonic steering | 22 % (14 / 64) | | |
| v6 | closed-loop ascent; steering allocation fix; supersonic aero steering | **87.5 % (77.2–93.5 %)** | 93.8 % | 5 m (CEP90 28 m) |

Remaining failures:
- 4 safe landings off target
- 3 terrain impacts
- 1 tip-over

Other findings:
- **Sensitivity:** wind speed ρ = 0.78, temperature −0.32, dry mass −0.30, throttle lag −0.26, dry CG 0.21.
- **Cargo load:** the peak is 5.97 g (median) and 6.28 g (p95) against the 6 g limit. It occurs in the unpowered descent, at about 31 kPa of drag after the entry burn, not under thrust. Peak drag deceleration scales roughly with the square of `entry_speed`, so the trade is cargo load against entry-burn propellant (5th-percentile margin 85 kg).
- **Propellant left:** 85 kg at p5, 133 kg median.

### Ascent loss of control at max-q (vehicle design finding)

All three terrain impacts in v6 are ascent losses of control near max-q. The sequence:

1. The vehicle is at Mach 1.0–1.6 and about 50 kPa, with 7.5–10.7 m/s wind and moderate turbulence.
2. A gust raises the angle of attack to 5–6°.
3. The nose-first hull is aerodynamically unstable in the database aerodynamics, and the 7° gimbal saturates.
4. The vehicle tumbles.

Re-flying the same three draws:

| change | run 15 | run 20 | run 33 |
|---|---|---|---|
| v6 | lost at max-q | lost at max-q | lost at max-q |
| + load relief (fly along the air-relative velocity at high q) and a 1° AoA cap above 30 kPa (rejected; see v7) | lost | lost | lost |
| + 40 kPa throttle bucket | lost | lost | lost |
| + 9° gimbal range (vehicle change) | lost | ascent OK, landed off site | ascent OK, tipped over |

Cutting thrust in a throttle bucket also cuts gimbal authority, so it doesn't help an unstable vehicle. Thrust-vector authority margin at max-q is a **vehicle design** issue: more gimbal range, a lower-max-q trajectory, or aerodynamic stability on ascent (fins or deployed grid fins).

**v7, tried and rejected:** load relief plus reduced cross-range correction at high dynamic pressure scored **39 %** over 64 runs (versus 87.5 %). Following the air-relative velocity turns the trajectory into the wind, and the velocity heading drifted 9° off the track through max-q. With cross-range correction suppressed, MECO left tens to hundreds of kilometres of cross-track error. Load relief needs ascent re-targeting after max-q before it can be used. The code is back to v6.

### Propellant slosh

64 runs with the first lateral slosh mode in both tanks and damping dispersed between 0.005 and 0.03 (`configs/dispersions/real_hop_slosh.yaml`). Mission success was **identical**: the same 56 of 64 runs landed. Vehicle recovery was 95.3 % against 93.8 % rigid. For this vehicle and controller, first-mode slosh is not a dependability driver.

## Monte Carlo results: `demo_hop`, fast fidelity, 200 runs

Dispersions are in `configs/dispersions/demo_hop.yaml`:
- engine: thrust 1.5 %, Isp 0.5 %, misalignment 0.15°, throttle lag
- mass: dry mass 1 %, CG 5 cm, cargo 2 %
- aero: CD 10 %, grid-fin CNα 15 %
- RCS: 5 %
- environment: wind 2.5 m/s and 30° error against the forecast, gusts 0–6 m/s, temperature 6 K

| Version | Change | Mission success (95 % CI) | Vehicle recovered | CEP50 |
|---|---|---|---|---|
| v1 | first campaign | 11.5 % (7.8–16.7 %) | 69 % | 243 m |
| v2 | kick-hold fix, g-limited planner and predictor, drag estimation, Mach-aware fins | 19.0 % (14.2–25.0 %) | 74 % | 234 m |
| v3 | consistent entry-burn cutoff (coast prediction) | stopped at 33 runs: 2 landed, so worse; reverted | | |
| v4 | v2 plus divert-limit safe landing and off-site landing classification | 19.0 % (14.2–25.0 %) | 78 % (72–83 %) | 238 m |
| v6 | closed-loop ascent and steering allocation fix (high fidelity gains supersonic steering) | 12.5 % (8.6–17.8 %) | 81.5 % | 257 m |

In fast fidelity the correct allocation does *worse* than the buggy one did. The strip-theory side force flips sign with tilt, so this table measures the fast aero model's limits more than the guidance.

All versions use the same 200 seeded draws, so the comparison is run by run. On the first 63 runs, v2 and v4 land exactly the same 9.

Also tried and measured on six reference draws, then left off by default:
- early "divert-aware" ignition
- limited supersonic steering
- a drag-lag coast steering law
- a divert-gate landing profile (brake to 15 m/s, then divert slowly)

The divert gate shrinks the miss but runs out of propellant, or its fuel guard aborts the divert. A guidance law that optimises the divert and the propellant together (convex optimisation) is the next step.

## Fast versus high fidelity for dependability numbers

The two fidelities disagree about how well this vehicle can steer during descent. Measured side-force response to a commanded tilt:

| | Mach 1.5–2.4, 15–20° tilt | Mach 0.5, 5° | Mach 0.5, 15–20° |
|---|---|---|---|
| fast (strip theory) | −1.2 to −1.5 m/s² (fins saturated, sign flips near 7°) | +0.15 m/s² | −0.25 to −0.39 m/s² |
| high (aero database) | −2.0 to −4.9 m/s² | −0.12 m/s² | −0.31 to −0.44 m/s² |

Dependability figures for the cargo hop should therefore come from **high-fidelity** campaigns. Fast-mode campaigns are a quick, pessimistic screen.

## Findings

- **Terminal divert authority was the weak point** before the high-fidelity steering fix, and it still is in fast fidelity. Near terminal velocity, drag carries the weight. The landing burn therefore runs near minimum throttle, where tilting the vehicle produces more aerodynamic crossflow force (in the opposite direction) than lateral thrust. Body lift during the unpowered descent is small (net lateral force well under 1 kN), because the grid-fin trim force cancels most of the hull's normal force.
  - Tried and rejected: early "divert" ignition and supersonic steering both made results worse.
  - **Needed:** a powered-descent guidance law that plans the divert at high thrust, for example a convex-optimisation (G-FOLD-type) landing burn, plus more lateral control authority.
- **Sensitivity** (Spearman ρ with miss distance, v4, 200 runs; |ρ| > 0.14 is significant):
  - wind speed 0.52
  - engine misalignment (yaw) −0.20
  - cargo mass 0.19
  - temperature offset −0.16
  - throttle lag −0.14
  - drag coefficient 0.13

  The drag coefficient ranks lower across the campaign than in individual draws, but a 12 % axial-force error can still move the touchdown point by kilometres after the entry burn, and nothing downstream recovers that. Aero coefficients should be known to about 2–3 % (wind tunnel, CFD, then calibration against flight data).
- **Propellant margin is thin.** The nominal mission lands with about 2–3 % of propellant. Combined 1 % mass and 0.5 % Isp errors consume it, so size the tanks to the Monte Carlo 99th percentile.
- **Wind:** success falls from about 44 % in near-calm air to 0 % above 8 m/s. Even with a perfect forecast, the miss grows with wind speed through gusts and the divert limitation above.

## Reproducing

```bash
uv run plume mc demo_hop --workers 8     # fast fidelity, about 25 min on 8 cores
uv run plume mc real_hop --workers 8     # high fidelity (WGS-84, NRLMSISE-00, von Karman, aero database, EKF), slower
```
