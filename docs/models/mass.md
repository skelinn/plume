# Mass properties

**Code:** `src/plume/physics/massprops.py` (`MassModel`, `MassProps`), `src/plume/physics/mjcf.py` (`build_mjcf`), `src/plume/physics/sim.py` (`RocketSim._apply_mass`, mid-step update)
**Fidelity:** both (identical in fast and high fidelity).
**Verification status:** verified, covering the CG against a brute-force sum, the CG travel as the tanks drain, exact mass bookkeeping, and the Tsiolkovsky Δv and variable-mass work–energy balance that depend on the mid-step update (`tests/test_physics_models.py`, `tests/test_physics_conservation.py`). Validation against measured vehicle mass properties (weighing and swing tests) is pending.

## Components

All components lie on the body z axis (+z toward the nose, origin at the dry-body reference point), so the CG is (0, 0, z_cg). Inertias are principal and diagonal about the CG, [Ixx, Iyy, Izz].

| Component | Spec | Mass | Position | Own inertia |
|---|---|---|---|---|
| Dry structure | `mass.dry`, `mass.dry_cg_z`, `mass.dry_inertia` | fixed | `dry_cg_z` | as given; if omitted, a thin-walled cylinder estimate: I_lat = m(r²/2 + L²/12), I_z = m r² |
| Cargo | `cargo.mass`, `cargo.cg_z` | fixed per run (can be set per mission) | `cargo.cg_z` | solid cylinder of radius 0.4 D and height 0.8 D |
| Propellant tanks | `tanks[]`: `capacity`, `initial`, `z_bottom`, `z_top`, `radius` | variable | centre of the liquid column | solid cylinder: I_lat = m(r²/4 + h²/12), I_z = ½ m r² |
| RCS gas | `rcs.propellant` (0 when the RCS is disabled) | variable | pod-ring height `rcs.z` | point mass |

**Tank model.** The liquid is a solid cylinder of the tank radius sitting on the tank bottom. Its height is h = (z_top − z_bottom) × fill fraction. As the tank drains the liquid column shortens from the top, so the CG first moves down. Once the liquid level falls below the dry CG, the vehicle CG rises again (both tested). The main engine draws from all tanks in proportion to their contents.

**Combination.** `MassModel.evaluate(tank_masses, rcs_mass, cargo_mass)` accumulates Σm, Σm z and Σm z², plus the components' own inertias. It then shifts the lateral inertias to the combined CG with the parallel-axis term Σm (z − z_cg)². The axial inertia needs no shift. A non-axisymmetric dry inertia (Ixx ≠ Iyy) is carried through. Products of inertia are always zero.

## MuJoCo body split (dry / wet)

MuJoCo ties a body's collision bounding volumes to its inertial frame, so the body that carries the geoms must keep a fixed mass frame. The vehicle is therefore two bodies (`build_mjcf`):

| Body | Joint | Geoms | Inertial |
|---|---|---|---|
| `rocket` (dry) | free joint | all collision and visual geoms | fixed: `mass.dry` at `dry_cg_z` with the dry inertia; never changes |
| `wet` | none (welded child of `rocket`) | none | rewritten every step: everything that is not dry structure (propellant, RCS gas, cargo) |

`RocketSim._apply_mass` writes the wet body from the total mass properties:
- m_w = m − m_dry
- z_w = (m z_cg − m_dry z_dry) / m_w
- I_w,lat = I_lat − I_dry,lat − m_dry (z_dry − z_cg)² − m_w (z_w − z_cg)²
- I_w,z = I_z − I_dry,z

Each inertia is floored at 1e-9 kg·m². With nothing left beyond dry structure, the wet body becomes a 1e-9 kg placeholder at the dry CG. MuJoCo recombines the two bodies into the composite inertia when it integrates.

**Where forces act.** External forces are applied through `xfrc_applied` on the dry body, which acts at the dry CG. The force models compute torques about the current total CG. The simulator shifts each torque to the dry CG (τ_dry = τ_cg + (z_cg − z_dry) e_z × F) before applying it.

## Mid-step mass properties

On every physics step `RocketSim.step`:
1. advances the engine and RCS and takes this step's mass flows ṁ_e and ṁ_r
2. evaluates the mass properties at the mid-step propellant load, tanks − ½ dt ṁ_e (shared across the tanks) and RCS gas − ½ dt ṁ_r, and writes them to the wet body
3. computes gravity, thrust, RCS and aerodynamic forces about that mid-step CG, then integrates. In high fidelity the forces are re-evaluated at every RK4 stage, while the mass properties stay at the mid-step value.
4. depletes the tanks and RCS gas by the full step

Evaluating the mass at the middle of the step makes the variable-mass integration second-order accurate, which is what makes the rocket equation come out right:
- Δv over a 60 s burn matches Isp g₀ ln(m₀/m₁) within 1e-4.
- The kinetic-energy balance ΔKE = W_thrust + ½ v² dm (the exhaust carries kinetic energy away) holds within 1e-3.
- Mass is accounted for exactly: m + propellant used + RCS gas used = m₀ to 1e-9 kg.

## Verification

| Check | Result | Test |
|---|---|---|
| Mass and CG vs a brute-force sum (dry + cargo + partial tank + RCS gas) | exact | `test_mass_properties_match_brute_force` |
| CG travel while draining | moves down, then up below the dry CG | `test_cg_drops_as_tanks_drain` |
| Mass bookkeeping over a 15 s burn | to 1e-9 kg | `test_mass_conservation_and_total_impulse` |
| Rocket equation (mid-step mass) | within 1e-4 | `test_tsiolkovsky_delta_v`, `test_burn_to_depletion_matches_rocket_equation` |
| Variable-mass work–energy | within 1e-3 | `test_work_energy_with_variable_mass` |
| Angular momentum, torque-free | conserved | `tests/test_physics_conservation.py::test_torque_free_angular_momentum_conserved` |

## Known limitations

- **Liquid propellant:** always settled on the tank bottom along the body axis, whatever the acceleration direction. Coast phases, negative-g and slosh dynamics are not modelled (see `propulsion.md`).
- **Tank shape:** cylindrical only. Domed ends, internal hardware, residuals and unusable propellant are not modelled. A tank can be drained to zero.
- **Off-axis mass:** none. There is no lateral CG offset and there are no products of inertia, so the lateral CG offsets from manufacturing tolerance cannot be represented.
- **Moving engine mass:** the gimballing engine and the deploying legs do not change the mass properties.
- **Inertia rate:** the İω term from the changing inertia tensor is not applied separately. MuJoCo integrates with the mid-step inertia held constant over each step. This is consistent with the second-order mass treatment, but there is no explicit jet-damping moment.
- **Cargo:** modelled as a rigid uniform cylinder fixed to the structure. There is no cargo shift and no compliant mounting.
