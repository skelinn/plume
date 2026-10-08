# Multi-stage vehicles, staging and orbital flight

**Code:** `src/plume/launcher/` — `spec.py` (`LauncherSpec`, `LaunchMissionSpec`), `stack.py` (stacked mass properties, separation, fairing halves), `orbit.py` (inertial frame, orbital elements, target plane), `guidance.py` (`StackAscent`, `UpperStageGuidance`, `BoosterReturn`), `mission.py` (`run_launch`); CLI `plume launch`.
**Configs:** `configs/vehicles/launcher_two_stage.yaml`, `configs/missions/demo_orbit.yaml`, landing-zone tile `data/terrain/demo_lz_rtls` (`scripts/generate_rtls_tile.py`).
**Fidelity:** both. Fast fidelity flies a non-rotating spherical Earth; high fidelity the rotating WGS-84 Earth with J2–J6, the aero database and the actuator models (see `earth.md`, `aero.md`, `propulsion.md`).
**Verification status:** verified: stacked mass properties against a brute-force sum, exact conservation of mass, linear and angular momentum at separation in both fidelities, orbital elements against analytic two-body orbits (also through the rotating Earth-fixed frame), closed-loop insertion into the target orbit (`tests/test_launcher.py`). The guidance is flight software, not a physics model: the demo shows it working, nothing more. Validation is pending, and the launcher is representative, not a real product.

## The vehicle

A `LauncherSpec` is a list of ordinary `VehicleSpec` stages, **bottom-up**, plus a payload, an optional fairing and the separation parameters. Every stage keeps its own body frame (z from its own hull base, +z toward the nose); stage k+1 sits with its hull base on top of stage k, at height `stage_base(k+1) = Σ_{j≤k} L_j` in the stack frame (the frame of stage 0). The payload rides on the last stage as its cargo (`payload.cg_z` above the stage top); the fairing's base is the top of the last stage. Each stage may keep a propellant `reserve` at cutoff (the booster's return propellant). Single-stage vehicles are unaffected: `VehicleSpec` and every existing preset load and fly exactly as before.

The demo vehicle `launcher_two_stage` is a **representative** 23 t kerolox launcher of the Electron / Falcon 1 class with round-number masses and engine data. It is not a model of any real product.

| | booster | upper stage |
|---|---|---|
| length, diameter | 13.2 m (incl. interstage), 1.6 m | 3.6 m + 1.2 m payload, 1.6 m |
| dry mass | 2 100 kg (legs, grid fins) | 310 kg |
| propellant | 17 500 kg (3 300 kg return reserve) | 3 200 kg |
| engine | 370 kN vac, Isp 282 / 312 s, 7° gimbal, throttle to 6 % | 36 kN vac, Isp 343 s, 5° gimbal |
| RCS | 4 × 500 N cold gas, 80 kg | 4 × 60 N, 12 kg |
| payload, fairing | | 100 kg; 60 kg fairing, 3.2 m long |

## Stacked flight: one rigid body

Before separation the whole stack is **one** `RocketSim`. The lowest stage is active: its engine, tanks, RCS, legs, grid fins and aerodynamic options are the stack's. Everything above it (the upper stage with its unburnt propellant and RCS gas, the payload, the fairing) is folded into the stack's dry mass as rigid dead mass (`stack_vehicle`). The geometry is the whole stack (length to the fairing tip, the fairing's nose), so drag, the crossflow moment and in high fidelity the generated aero database see the full vehicle.

The dead-mass parts are combined on the body axis with the parallel-axis theorem:

  m = Σ mᵢ,  z_cg = Σ mᵢ zᵢ / m,  I_lat = Σ (I_lat,i + mᵢ (zᵢ − z_cg)²),  I_ax = Σ I_ax,i

where each loaded upper stage contributes its `MassModel` mass, CG (shifted by its `stage_base`) and inertia at its initial load, and the fairing is a thin shell of revolution with its CG at 40 % of its length, I_lat = m (r²/2 + L²/12), I_ax = m r². The active stage's tanks drain and move the CG exactly as in a single-stage vehicle (`mass.md`). Liftoff mass is the sum of all stages, payload and fairing (checked to 1e-12).

## Separation

`MECO → coast (2 s) → separation`. Each part becomes its **own `RocketSim`**, started from the parent's rigid-body state (`spawn_from`):

  r_child = r_parent + R e_z z_off,  v_child = v_parent + R (ω × e_z z_off),  q_child = q_parent,  ω_child = ω_parent

where `r`, `v` are the body-origin position and velocity, `R` the attitude and `z_off` the child's hull base in the parent's body frame (0 for the booster, `stage_base(1)` for the upper stage). The children's masses add up to the parent's and their CGs move with the parent's rigid velocity field, so mass, linear momentum and angular momentum about any point carry over exactly (tested to 1e-10 in fast and high fidelity, including the RK4-stage force callback). The booster takes over the stack's engine object, so its ignition count, throttle state and gimbal actuator continue.

Then a spring separation applies equal and opposite impulses along the stack axis through both CGs,

  J = Δv m₁ m₂ / (m₁ + m₂),  Δv_booster = −J / m₁ e_axis,  Δv_upper = +J / m₂ e_axis

(`separation.delta_v` = 1 m/s relative). Momentum is unchanged (collinear impulses through the CGs; tested). After that the simulators run independently and in lock-step on the 20 Hz flight-software cycle; each has its own atmosphere/wind sampling, aerodynamics and contacts (the booster keeps the terrain tiles for its landing).

**Fairing jettison** works the same way: above `jettison_altitude` (105 km) during the upper-stage burn, the upper stage is re-spawned without the fairing mass (the burning engine object carries over), and the two halves leave sideways at `jettison_speed` with an outward tumble. The halves are tracked kinematically for 40 s (ballistic CG in the same gravity and rotating-frame model, constant body rate, no aerodynamics: they separate above 100 km) for the replay only.

## Inertial state and orbital elements

The world frame is East-North-Up at the launch site (`earth.md`). `InertialFrame.to_inertial` maps a world state to an Earth-centred inertial (ECI) frame:

* **High fidelity (rotating WGS-84):** r = r_ECEF, v = v_ECEF + ω_E × r_ECEF, both rotated by the Earth rotation angle ω_E t. The ECI frame coincides with ECEF at t = 0, so the right ascension of the ascending node is measured from the launch-epoch Greenwich meridian.
* **Fast fidelity (non-rotating sphere):** already inertial. The gravity model has no pole, so one is assigned from the launch latitude φ: k = cos φ · north + sin φ · up at the pad, x on the launch meridian. Inclinations in fast fidelity are relative to that pole.

`elements(r, v, μ)` returns the osculating two-body elements (Vallado, alg. 9): a = −μ/2ε from the specific energy ε = v²/2 − μ/r; the eccentricity vector e = ((v² − μ/r) r − (r·v) v)/μ; i, Ω, ω, ν from h = r × v and the node vector; r_p = p/(1+e), r_a = p/(1−e), p = h²/μ. **Altitudes** of perigee and apogee are r − R_eq with the equatorial radius (6 378 137 m on WGS-84). "In orbit" means the osculating perigee is above `min_perigee_altitude` (150 km). J2 makes the osculating elements oscillate slightly in high fidelity; they are not mean elements.

## Guidance

### Target plane and launch heading

For a target inclination i, the orbit normal n must satisfy n·k = cos i and pass through the launch position r̂: with the local north N and east E at the pad, n = cos β N − sin β E with cos β = cos i / cos φ (the ascending, north-east pass). If i < φ it is replaced by φ (no dog-leg). The heading flown off the pad is the ground-relative direction of V·(n × r̂) − ω_E × r with V = 7.8 km/s, projected onto the horizontal: on a rotating Earth this corrects the azimuth for the pad's eastward speed.

### Stack ascent (`StackAscent`)

Vertical rise (6 s), linear pitch kick to the kick angle along the heading (6 s), kick hold until the flight path has pitched over by the kick angle, then a gravity turn that follows the *ground* velocity (a 6 m/s wind turned a 3° kick around when following the air velocity). The kick is **planned**: a 3-DOF point-mass ascent of the nominal stack (calm air, same direction law, same g-limited thrust, stop at the reserve) is bisected on the kick angle until the flight-path angle at MECO equals `staging_flight_path_deg` (35°). The planned flight-path angle vs speed is then tracked in closed loop during the gravity turn (as in the cargo hop; AoA limited to 1.5° above 15 kPa and 4° below). Above 20 km and below 5 kPa a small yaw trim nulls the inertial out-of-plane velocity. Thrust is limited to `max_g` = 5 g sensed. MECO when the booster is down to its reserve (plus the thrust tail-off ṁ τ).

The staging angle trades upper-stage performance against the booster's return: a lofted staging leaves less horizontal speed to cancel. In fast fidelity (before the boost-back g-limit below), 30° / 35° / 40° gave boost-back burns of 2 720 / 2 440 / 2 270 kg with the upper stage's margin almost unchanged (77–90 kg).

### Upper stage (`UpperStageGuidance`)

Closed-loop terminal guidance in the ECI frame, re-solved every 50 ms. With r̂, the target-plane normal n and the downrange direction t̂ = n × r̂:

* **radial channel:** r̈ = a_r − g_eff with g_eff = μ/r² − v_t²/r. The net radial acceleration is given the profile A + B τ (τ = time from now), with A, B chosen so that r(t_go) = r_T and ṙ(t_go) = 0:
  B = (−12 Δr − 6 ṙ t_go)/t_go³, A = (−ṙ − B t_go²/2)/t_go, Δr = r_T − r − ṙ t_go; commanded a_r = g_eff + A.
* **out-of-plane channel:** the same law drives the distance from the target plane z = r·n and its rate to zero: a_n = A_n.
* **downrange:** the rest of the thrust, a_t = √(a_T² − a_r² − a_n²); a_r, a_n are limited to `max_off_tangent_deg` (60°) of the thrust.
* **time to go:** from the rocket equation, t_go = τ_m (1 − e^(−Δv/v_e)), τ_m = m v_e / T, iterated with Δv = |(v_T − v_t, −ṙ + ½ (g_eff + g_eff,T) t_go, −v_n)| (the radial term carries the gravity loss still to be paid).
* **freeze:** below `freeze_time` (6 s) to go, A, B are held and only advanced in time (the solution is singular as t_go → 0).
* **cutoff:** when the specific orbital energy reaches the target's, ε_T = −μ / (2 a_T), less the energy the thrust tail-off still adds (|v| (T/m) τ) and with half a control cycle of lead. The tail-off is the sampled engine lag's shutdown impulse (dt a/(1 − a), a = e^(−dt/τ), less the 2 % cut), and dε/dt uses the actual thrust direction (v·a_T), which near burnout is well off the velocity. In the last ~3 s of full-thrust energy gain the throttle is set between minimum and full so the remaining energy takes a whole number of 50 ms cycles: the cut then falls on a cycle boundary. Without that, the half-cycle quantisation alone was ±1 m/s, ±3.5 km of apogee; with it the demo's apogee is within 1 km.

The target state is the perigee of the target orbit: r_T = R_eq + h_p, v_T = √(μ (2/r_T − 1/a_T)), flight-path angle 0. This is a simplified linear-acceleration ("linear tangent family", PEG-like) law: robust and simple, not propellant-optimal, and it does not model J2 in its prediction (the closed loop absorbs it).

### Booster return to launch site (`BoosterReturn`)

1. **Coast and flip** (2 s after separation): the RCS turns the booster at up to 12°/s to the boost-back attitude.
2. **Boost-back:** thrust horizontally opposite the predicted miss of the vacuum impact point (`kepler_impact`, rotating-frame aware) on the landing zone, pitched up 5°, throttled to ≤ 4 g (a nearly empty stage at full thrust would pull 14 g). Once within 40 km, the drag- and entry-burn-aware 3-DOF predictor of the cargo hop (`ImpactPredictor`) runs at 2 Hz and the cut is scheduled at the zero crossing of its along-track error (aim point `meco_bias` = 400 m past the LZ; the entry burn trims it). The burn is also cut if the propellant falls to 1.6 × the landing reserve.
3. **Hand-over:** grid fins deploy and the unchanged cargo-hop autopilot (`HopAutopilot`, from its `coast` phase) flies the retrograde flip, entry burn, aerodynamic descent with grid-fin steering on the predicted miss, and the landing burn (`LandingAutopilot`) onto LZ-1, a flat 300 m tile 1.5 km up-range of Pad A. The landing zone is placed opposite the launch heading so that the hop autopilot's along-track sign conventions (A → B) match the booster's return direction.

## Replays

A launch replay is a normal Plume replay whose primary vehicle (`frames`, `meta.vehicle`) is the upper stage, extended compatibly (`recording/schema.json`):

* `meta.vehicles`: every vehicle (id, name, role, `persist`, `attach_z`, `t_start`/`t_end`, and a `vehicle` description like `meta.vehicle`; `kind: fairing_half` selects the fairing-half model);
* `tracks[id].frames`: the frames of the other vehicles (booster, `fairing_a`, `fairing_b`). While stacked, the upper stage and the fairing halves are recorded at their stack positions, so every track covers the whole flight from liftoff;
* `events[].vehicle`: which vehicle an event belongs to.

Old single-vehicle replays have none of these and load unchanged; old viewers show the primary vehicle only. The viewer draws every vehicle with its own model and trail, hides jettisoned hardware after its track ends, labels the vehicles it is not following, and has a **Follow** selector (and the `V` key / `?vehicle=<id>`) that moves the camera, HUD, telemetry and engineering overlay to another vehicle.

## Verification

| Check | Result | Test |
|---|---|---|
| Stack mass, CG, inertia vs brute-force sum of booster, loaded upper stage, payload, fairing | 1e-12 (mass, CG), 1e-9 (inertia) | `test_stack_mass_properties_match_brute_force` |
| Separation: mass, linear and angular momentum, fast and high fidelity | exact to 1e-10; spring Δv exactly 1 m/s along the axis; momentum unchanged by the impulse and in free flight after it | `test_separation_conserves_mass_and_momentum` |
| Orbital elements vs analytic two-body orbits (a, e, i, Ω, ω, ν, r_p, r_a, h, vis-viva) | 1e-10 | `test_elements_roundtrip_analytic`, `test_circular_orbit_elements` |
| Elements through the rotating WGS-84 world frame at t = 0, 437 s, 5000 s; a point at rest on the ground has i = geocentric latitude | 1e-9 | `test_elements_from_rotating_world_frame` |
| Target-plane normal has the requested inclination and contains the position | 1e-9 | `test_non_rotating_frame_inclination_and_target_plane` |
| Upper stage alone from a typical staging state into 200 × 250 km, 40° | perigee ± 3 km, apogee ± 6 km, i ± 0.05° | `test_upper_stage_guidance_reaches_target_orbit` |
| Coarse full demo (fast, 20 ms): orbit and booster landing, replay validates against the schema | pass | `test_demo_orbit_coarse_flight` (slow) |

## Demo results (`plume launch demo_orbit`)

<!-- ORBIT:START -->
| | fast fidelity | high fidelity |
|---|---:|---:|
| staging | T+120 s, 53.6 km, 1 740 m/s, 35° | T+120 s, 53.6 km, 1 743 m/s, 34° |
| max q (stack) | 31.9 kPa | 32.6 kPa |
| orbit (target 200 × 250 km, 40°) | 199.8 × 250.1 km, 40.00° | 199.6 × 249.4 km, 39.98° |
| upper-stage propellant at SECO | 79 kg (2.5 %) | 102 kg (3.2 %) |
| booster boost-back propellant | 2 553 kg | 2 581 kg |
| booster apogee after boost-back | 117 km | 117 km |
| booster landing error on LZ-1 | 2.8 m | 3.2 m |
| booster touchdown | 0.96 m/s down, 0.42 m/s across | 0.71 m/s down, 0.12 m/s across |
| booster propellant at touchdown | 226 kg | 183 kg |
<!-- ORBIT:END -->

The high-fidelity elements are osculating (J2) and the bundled replay `data/replays/orbit_demo_high.plume.json.gz` is this run. Fast fidelity has no Earth rotation, which is why it needs more of the upper stage's propellant.

Single runs at the nominal conditions with light wind; no Monte Carlo campaign has been flown for the launcher yet, so these are not success probabilities.

## Known limitations

- **Two stages:** `run_launch` flies stack + upper stage. `LauncherSpec` and `stack_vehicle` accept more stages, but the sequencing for a third stage is not written.
- **Navigation:** the flight software of every vehicle flies on the true state in launch missions (no per-vehicle EKF yet), also in high fidelity.
- **Separation dynamics:** the spring push is an instantaneous impulse; there is no interstage clearance, recontact, plume impingement on the booster or separation transient. The stages separate in near vacuum (54 km, 0.7 kPa), where aerodynamic interference is small but not zero.
- **Stack aerodynamics:** one body of revolution with the stack's length and the fairing's nose; no diameter steps, no interstage flow. The upper stage's propellant is rigid dead mass in the stack (no slosh, no CG travel).
- **Fairing halves:** kinematic, no aerodynamics or recontact check; drawn for the replay only.
- **Guidance:** not optimal (a PEG/UPFG solution would save propellant); no J2 in its prediction; no launch window, plane change beyond the reachable inclination, or abort modes. Fast fidelity has no Earth rotation, so the fast-fidelity orbit costs about 350 m/s more than at high fidelity at this site and inclination.
- **Booster return:** boost-back steering on the vacuum impact point with a drag-aware cutoff; the margins of the demo are single-run margins. The landing zone must lie opposite the launch heading (the cargo-hop autopilot's along-track convention).
- **Thermal and structural:** no heating or load limits on the stages, no engine-out.
