# Landing gear: crushable shock absorbers, soil and tip-over

**Code:** `src/plume/physics/gear.py` (model, soil presets, tip-over margin), `src/plume/physics/mjcf.py` (leg bodies and joints), `src/plume/physics/sim.py` (`RocketSim.gear`, `leg_failure()`)
**Fidelity:** `high` by default (`LegsSpec.model: auto`); `fast` keeps the rigid legs, so RL training and the fast benchmarks are unchanged. `model: crush` or `model: rigid` forces either.
**Verification status:** verified against closed-form results (crush-curve energy integral, sizing rule, Bekker inverse, sudden-load sinkage, tip-over geometry) and by energy accounting in drop tests (`tests/test_landing_gear.py`). Validation against drop-test or flight data is pending.

## What it answers

With rigid legs, a touchdown either passes (vertical speed below `max_touchdown_speed`) or breaks the legs. The crush model replaces that threshold with physics: how much of each leg's stroke a landing uses, the peak leg load, the energy absorbed, whether the core bottoms out, how deep the pads sink into soft ground, and how close the vehicle comes to tipping over. These are reported in touchdown results, replays and Monte Carlo summaries.

## Model

### Leg kinematics

Each leg's lower strut and footpad form a separate MuJoCo body on a **slide joint** (the stroke `s`, `0 ≤ s ≤ stroke`). The pad sits on a second slide joint (the soil sinkage `z`) when any ground in the world can sink; otherwise that joint is omitted.

Both joints move along the **body axis**. This idealises the leg linkage as one that moves the pad straight up relative to the hull. A telescoping strut along the inclined leg would scrub the pads sideways across the ground on a level touchdown. In early tests this dissipated about a third of the impact energy in pad friction, which is a kinematic artefact rather than gear behaviour.

The stroking mass (`foot_mass`, default 8 kg per leg) is moved out of the hull body. The hull body's mass, CG and inertia are reduced so that hull plus legs at zero stroke has exactly the vehicle's dry mass properties (`dry_body_split`). Plume's applied forces act at the hull body's CG (`RocketSim.xfrc_z`), and the torque is shifted accordingly.

### Crushable core

The core is a **rigid-plastic element**, implemented with MuJoCo's joint dry friction (`frictionloss`). The joint moves only when the axial load exceeds the crush force `F_c`, and the constraint solver resolves this implicitly (no stiff explicit spring). After each step, `F_c` is set from the largest stroke reached so far, so crushing is **non-recoverable**: the leg never springs back, and unloading follows a vertical line in the force-stroke plane (hysteresis).

```
F_c(s) = P · [ f0 + (1 − f0) · min(s / s_e, 1) ]                      onset ramp (elastic rise)
       + P · (r_d − 1) · clamp((s − s_d) / (S − s_d), 0, 1)           densification
```

| symbol | field | default |
|---|---|---|
| `S` | `stroke` | 0.4 m |
| `P` | `crush_force` (plateau, per leg) | sized, see below |
| `f0` | `crush_onset` | 0.3 (raised to the break-out load) |
| `s_e` | `elastic_stroke` | 0.01 m |
| `s_d = densification · S` | `densification` | 0.8 |
| `r_d` | `densified_ratio` | 2.0 |
| `k`, `c` | `spring_rate`, `damping` (parallel, MuJoCo joint stiffness and damping) | 0 N/m, 1500 N·s/m |
| `F_max` | `max_load` | 1.6 `P` |

**Sizing (default `P`).** With all legs on the plateau, stop the design touchdown speed `v_d = max_touchdown_speed` within 75 % of the stroke: `n P = m (g + v_d² / (1.5 S))`. Here `m` is the nominal landing mass (dry + cargo + RCS gas + 10 % of the propellant capacity). For `lander_small` this gives `P` = 18.3 kN per leg, so the gear absorbs a 5 m/s landing in about 0.28 m of stroke (measured: 0.283 m, 71 %).

**Break-out load.** A vehicle that stands on its legs fully fuelled, such as the cargo hopper on its launch pad, must not crush them. The force at zero stroke is therefore at least 1.2 × the fully fuelled weight per leg, and the plateau is raised to match if needed (`crush_breakout`, `crush_params`). This plays the role of a preloaded core or break-out stage.

**Failure.** A leg fails with:
- `gear_bottomed` when the stroke reaches its end stop (`s ≥ S − 1 mm`). The joint range is a hard limit, and the load then spikes.
- `gear_overload` when the peak axial load exceeds `F_max`.
- `pad_buried` when the sinkage reaches the soil's `max_sinkage`.

The landing environment and the cargo-hop mission call `RocketSim.leg_failure()`. For rigid legs it returns `crash_legs` above `max_touchdown_speed`, exactly as before.

**Leg load.** The axial load is the ground contact force on the footpad (`mj_contactForce`), projected on the stroke axis. It is evaluated by an extra `mj_forward` with the step's forces, and only while a pad is in contact.

**Energy.** The absorbed energy is computed per leg each step as the work against the resistances that acted during that step:
- crush: `F_c(s_prev) Δs`
- damper: `c ṡ² Δt`
- soil: `F_soil(z_prev) Δz`

### Soil

The pressure under a pad of radius `b` follows Bekker's pressure-sinkage relation. It is capped by an optional ultimate bearing strength (general shear failure):

```
p(z) = min( (k_c / b + k_φ) zⁿ ,  q_ult )          F_soil(z) = p(z) · π b²
```

Soil is plastic in the same way as the core: the pad sinks while the load exceeds `F_soil(z)` and never rises back. Presets (`SOIL_PRESETS`) come from Wong (2008), Table 2.3, plus a lunar-regolith set used for Apollo-era mobility analyses:

| preset | n | k_c (kN/m^(n+1)) | k_φ (kN/m^(n+2)) | q_ult | μ |
|---|---|---|---|---|---|
| `rigid` / `concrete` / `steel_deck` | – | rigid | rigid | – | world / 0.8 / 0.6 |
| `compacted_gravel` | 1.0 | 0 | 60 000 | – | 0.7 |
| `dry_sand` | 1.1 | 0.99 | 1528.4 | – | 0.55 |
| `sandy_loam` | 0.7 | 5.27 | 1515.0 | – | 0.6 |
| `clay` | 0.5 | 13.19 | 692.2 | 150 kPa | 0.45 |
| `lunar_regolith` | 1.0 | 1.4 | 820 | – | 0.65 |

**Where the soil applies.** `WorldSpec.soil` sets the default for all ground (default `rigid`). A heightfield tile can override it (`GroundTile.soil`), and a mission sets its landing tile with `terrain.landing_soil`. With crush legs, the ground geoms take contact priority, so the soil (or deck) friction coefficient governs the pad contact. With rigid legs, the original contact parameters are unchanged.

### Tip-over margin

The static tip-over margin is the smallest rotation about a support-polygon edge (the line between adjacent footpads) that brings the CG over that edge:

```
margin = min over edges  atan( d_edge / h_cg )
```

- `d_edge` is the horizontal distance from the CG to the edge.
- `h_cg` is the CG height above the edge.
- The margin is negative when the CG is already outside the polygon.

Sinkage, stroke differences and deck roll all reduce it through the actual pad positions. It is tracked every step after touchdown: `tipover_margin_deg` is the final value and `min_tipover_margin_deg` the lowest seen. For `lander_small` upright on level ground it is 21.7°.

## Outputs

| where | fields |
|---|---|
| `sim.gear.report()` (`GearReport`) | per leg: `stroke_used_m`, `stroke_fraction`, `peak_load_n`, `energy_crush_j`, `energy_damper_j`, `sinkage_m`, `energy_soil_j`; `tipover_margin_deg`, `min_tipover_margin_deg`, `failure` |
| touchdown results (`MissionResult`, landing-env replay outcome) | `stroke_used_max_m`, `stroke_fraction_max`, `leg_load_peak_kn`, `gear_energy_kj`, `sinkage_max_m`, `tipover_margin_deg`, `min_tipover_margin_deg` |
| Monte Carlo summary `metrics` | percentiles of `stroke_fraction_max`, `leg_load_peak_kn`, `sinkage_max_m`, `min_tipover_margin_deg` |
| replay frames | `stroke[n]`, `sinkage[n]`; the viewer moves each footpad up by its stroke |

## Verification

`tests/test_landing_gear.py`:

| check | result |
|---|---|
| crush-curve energy integral vs closed form (onset ramp + plateau + densification triangle) | 0.1 % |
| sizing rule: plateau stops `v_d` in exactly 75 % of the stroke; the core does not crush at rest | exact |
| Bekker: `static_sinkage` inverts `soil_force`; beyond `q_ult` the pad keeps sinking | 1e-9 |
| drop test, rigid ground, 3 and 5 m/s: vehicle energy lost vs crush + damper energy | within 5 % (measured 0.6–3.7 %) |
| crush energy vs the curve integrated over the stroke used | within 3 % |
| 5 m/s design touchdown uses 55–85 % of the stroke | 71 % |
| static leg load = weight / 4 | within 3 % |
| 7 m/s drop fails | `gear_overload` (load passes 1.6 P while the core densifies, then it bottoms out) |
| dry sand, 3 m/s: energy balance including soil work; sinkage above static (impact peak) | within 10 % (measured 5 %); 0.13 m sinkage |
| sudden placement on linear soil sinks 2 × the static value (`W z = k z²/2`, no rebound) | within 15 % |
| tip-over margin upright = `atan(span · cos(π/n) / h_cg)` | 0.3° |

The rest of the energy, a few per cent, goes to contact softness and the pad masses' impact.

## Assumptions and limitations

- **Idealised linkage.** Pads stroke along the body axis, the vertical-equivalent stroke. The real leg geometry (A-frame hinge, inclined strut, mechanical advantage) is not modelled, so `peak_load_n` is the pad's axial load along the body axis, not the strut load.
- **Plastic elements use MuJoCo soft dry friction.** Below the crush force, the joints creep by a few millimetres over seconds. The impedance is set to 0.9999 to minimise this.
- **No load-rate effects.** Honeycomb crush strength rises with impact speed by 10–30 %; this is not modelled. Lateral strength and buckling of the legs, structural failure of the pads, and pad tilt are also not modelled.
- **Simplified soil.** Bekker parameters are for plate sinkage at low speed. There is no dynamic bearing capacity, slope or layer effects, or friction that depends on sinkage. Sinkage is along the body axis, a good approximation only while the vehicle is near upright.
- **Static tip-over margin.** It ignores inertial loads, from deck motion or residual rates. Treat it as a stability indicator, not a criterion.
- **High fidelity only.** The fast models keep rigid legs, which bounds the RL tasks' realism at touchdown.
