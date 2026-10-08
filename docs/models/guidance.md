# Guidance, navigation and control: the cargo hop

**Code:** `src/plume/missions/hop.py` (`HopAutopilot`, `plan_ascent`), `src/plume/missions/targeting.py` (`ImpactPredictor`, `kepler_impact`), `src/plume/control/autopilot.py` (`LandingAutopilot`), `src/plume/control/attitude.py`
**Verification status:** not applicable: this is the *vehicle's* flight software, not a physics model. The simulator's job is to show how well it works. The Monte Carlo campaigns below do that, and they show that this guidance is **not yet dependable**.

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
6. **Aerodynamic descent:**
   - Subsonic only: steered with body and grid-fin lift (12° tilt cap) on the predicted miss.
   - The drag-scale estimate (measured versus modelled deceleration) feeds the predictor.
7. **Landing burn:**
   - Constant-deceleration hoverslam profile in the vertical, ZEM/ZEV feedback in the horizontal, and aero-aware thrust/tilt allocation.
   - If the predicted miss exceeds `max_divert_m` (default 250 m), the vehicle retargets to the closest reachable point rather than spending its landing propellant on an impossible divert.
   - Legs deploy below 50 m/s or 200 m.

In high fidelity all of this acts on the navigation estimate (see `navigation.md`). Planning uses the *nominal* vehicle and the *forecast* environment, while the simulator flies the dispersed truth.

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

All versions use the same 200 seeded draws, so the comparison is run by run. On the first 63 runs, v2 and v4 land exactly the same 9.

Also tried and measured on six reference draws, then left off by default:
- early "divert-aware" ignition
- limited supersonic steering
- a drag-lag coast steering law
- a divert-gate landing profile (brake to 15 m/s, then divert slowly)

The divert gate shrinks the miss but runs out of propellant, or its fuel guard aborts the divert. A guidance law that optimises the divert and the propellant together (convex optimisation) is the next step.

## Findings

- **Terminal divert authority is the weak point.** Near terminal velocity, drag carries the weight. The landing burn therefore runs near minimum throttle, where tilting the vehicle produces more aerodynamic crossflow force (in the opposite direction) than lateral thrust. Body lift during the unpowered descent is small (net lateral force well under 1 kN), because the grid-fin trim force cancels most of the hull's normal force.
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
