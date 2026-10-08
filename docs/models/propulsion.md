# Propulsion: main engine, gimbal and RCS

**Code:** `src/plume/physics/propulsion.py` (`Engine`, `RCS`, `read_eng`), `src/plume/config.py` (`EngineSpec`, `RCSSpec`)
**Fidelity:** both. High fidelity (`WorldSpec.fidelity: high`) adds the second-order gimbal actuator, the ignition delay and RCS pulse-width modulation.
**Verification status:** verified, covering the Isp and throttle limits, ignition count, flame-out when dry, gimbal limits and torque signs, RCS allocation, mass conservation, total impulse, Tsiolkovsky Δv and burn to depletion (`tests/test_physics_models.py`, `tests/test_physics_conservation.py`), plus the high-fidelity gimbal actuator, ignition delay and RCS minimum impulse bit (`tests/test_actuators.py`). Validation against engine test-stand or flight data is pending.

## Liquid engine (`engine.type: liquid`)

**Mass flow and thrust.** At full throttle the vacuum mass flow is ṁ_max = F_vac / (Isp_vac g₀). At throttle u:

ṁ = u ṁ_max,  F = ṁ Isp_vac g₀ − p_amb A_e  (clamped at 0)

The nozzle exit area comes from the two Isp values: A_e = ṁ_max g₀ (Isp_vac − Isp_sl) / p₀, with p₀ = 101 325 Pa. At full throttle the effective Isp is therefore exactly `isp_vac` in vacuum and exactly `isp_sl` at sea-level pressure (tested). Because A_e is fixed, the back-pressure loss is a larger share of thrust at low throttle, as on a real fixed nozzle. `p_amb` is the atmosphere model's pressure at the CG altitude.

**Throttle.**
- Command u_cmd in [0, `throttle_max`]. A command below ½ `throttle_min` means "engine off"; any other command is clamped to [`throttle_min`, `throttle_max`].
- The delivered throttle follows a first-order lag with time constant `throttle_tau`, discretised exactly (`exp(−dt/τ)`). After shutdown the tail-off is cut once the throttle falls below 2 %.

**Ignitions.** Each off → on transition uses one ignition. With `max_ignitions` set, the engine will not relight once they are used up. It cannot ignite with no propellant.

**Propellant depletion.** The engine draws from all tanks in proportion to their contents. When the propellant runs out the engine shuts down. On the last step, thrust and mass flow are scaled so that exactly the remaining propellant is used, with no overdraw.

| Parameter (`EngineSpec`) | Default | Meaning |
|---|---|---|
| `thrust_vac` | required (> 0) | full-throttle vacuum thrust, N |
| `isp_vac` / `isp_sl` | 300 s / = `isp_vac` | vacuum and sea-level specific impulse |
| `throttle_min` / `throttle_max` | 0.4 / 1.0 (≤ 1.5) | deep-throttle limit and overthrottle |
| `throttle_tau` | 0.1 s | first-order throttle response |
| `max_ignitions` | unlimited | relight budget |

## Gimbal (both fidelities)

- **Commands:** normalised to [−1, 1] per axis and scaled by `gimbal_max_deg`. The deflection is limited to a circle of radius `gimbal_max_deg` (the default of 0 means a fixed nozzle).
- **Thrust direction:** angles (a, b) rotate the thrust about body x, then y: d = (cos a sin b, −sin a, cos a cos b). Thrust acts at the pivot height `gimbal_z` (default 0.5 m, body frame), so the torque about the CG is (z_gimbal − z_cg) e_z × F.
- **Fast fidelity:** a first-order lag (`gimbal_tau`, 0.05 s) with a rate limit (`gimbal_rate_deg_s`, 30°/s).
- **Engine misalignment:** `misalignment_deg` (x, y; default 0, 0) is a fixed build tolerance added to the gimbal angles. The flight software does not know about it. The Monte Carlo dispersion files sample it (σ = 0.15° per axis in `configs/dispersions/*.yaml`).

## High-fidelity additions

**Gimbal actuator.** A second-order servo per axis replaces the first-order lag:

θ̈ = ω_n² (θ_cmd,delayed − θ) − 2 ζ ω_n θ̇

- The command reaches the actuator after a transport delay. The actuator acts on the newest command that is at least `gimbal_delay_s` old.
- The acceleration is limited to `gimbal_accel_deg_s2` and the rate to `gimbal_rate_deg_s`.
- The explicit update is sub-stepped so that ω_n h ≤ 0.2, which keeps it stable at any physics dt.
- At the circular angle limit the piston stops and its rate is zeroed.
- The nozzle follows the piston through a backlash band. While the piston moves within ±½ `gimbal_backlash_deg` of the nozzle, the nozzle does not move.

**Ignition delay.** A liquid start produces thrust `ignition_delay_s` after the command. This stands for the igniter, valve sequencing and chamber fill. If the command is withdrawn first, the start is aborted and the timer resets. Fast fidelity has no delay.

**RCS pulse-width modulation.** Each thruster's duty cycle d is flown as one valve pulse per PWM frame (`pwm_period_s`, equal to the 50 ms flight-software cycle):
- A pulse shorter than `min_on_time_s` is dropped. This gives the minimum impulse bit F × t_min (2 N·s at the defaults).
- The valve opens `valve_delay_s` after the frame starts and stays open for d × period.
- Each physics step applies the fraction of that step during which the valve is open, so the impulse per frame matches the averaged (fast) model to within the latency effect (tested within 15 %).

| Parameter | Default | Meaning |
|---|---|---|
| `gimbal_wn_hz` | 8 Hz | actuator natural frequency |
| `gimbal_zeta` | 0.7 | actuator damping ratio |
| `gimbal_rate_deg_s` | 30°/s | rate limit (both fidelities) |
| `gimbal_accel_deg_s2` | 800°/s² | acceleration limit |
| `gimbal_delay_s` | 0.015 s | command transport delay |
| `gimbal_backlash_deg` | 0.05° | total free play |
| `ignition_delay_s` | 0.35 s | liquid start: command to thrust onset |
| `RCSSpec.pwm_period_s` | 0.05 s | PWM frame |
| `RCSSpec.min_on_time_s` | 0.01 s | minimum valve open time |
| `RCSSpec.valve_delay_s` | 0.005 s | valve opening latency |

## Reaction control system

**Layout:**
- **Default ring:** `pods` (4) pods at height `z` (8 m) on radius `radius` (the hull radius by default). Each pod has a pair of opposed tangential thrusters.
- **Custom layout:** `thrusters` (position and direction in the body frame) replaces the ring.

**Thruster defaults:** cold gas, 200 N each, Isp 70 s, 20 kg of gas.

**Allocation:**
- **Command:** a body torque request, as a fraction of the per-axis capability.
- **Per-axis capability:** computed at the nominal CG by linear programming. It is the largest torque about one axis with zero torque about the other two, with duties in [0, 1].
- **Duty cycles:** solved by non-negative least squares on the torque matrix about the current CG, then scaled down if any duty exceeds 1.
- **Translation:** the net force of the firing thrusters is applied too, so RCS use also translates the vehicle slightly.
- **Gas use:** ṁ = Σ d F / (Isp g₀).

## Solid motors (`engine.type: solid`)

**Thrust curve:** `thrust_curve` ([[t, F], ...]) or a RASP `.eng` file (`motor_file`; it is looked up next to the vehicle file, then in `data/motors/`). The reader skips `;` comments, reads the header (name, diameter, length, propellant and total mass, maker) and prepends a (0, 0) point if needed.

**Burn:**
- The motor ignites at `ignition_time`. Thrust is sampled from the curve at the mid-point of each step.
- The burn ends at the last curve point, or when the propellant runs out.

**Mass flow:** total impulse I = ∫F dt (trapezoid). Effective Isp = I / (m_prop g₀), where m_prop is the vehicle's tank capacity, so set the tank to the motor's propellant mass. ṁ = F / (Isp_eff g₀), which burns exactly m_prop over the curve.

**No ambient-pressure correction:** RASP curves are static sea-level test data, so the altitude gain in thrust is not modelled. There is no throttle; the reported "throttle" is F / F_max.

## Verification

| Check | Result | Test |
|---|---|---|
| Isp at p = 0 and p = p₀ | equals `isp_vac`, `isp_sl` | `test_engine_isp_and_throttle_limits` |
| Thrust / ṁ in vacuum | = Isp_vac g₀ to 1e-9 | same |
| Min-throttle clamp, off threshold | as specified | same |
| Relight budget, flame-out when dry | no thrust | `test_engine_ignition_limit`, `test_engine_flameout_when_dry` |
| Gimbal rate and circular angle limit | as specified | `test_gimbal_rate_and_angle_limits` |
| Gimbal torque signs | a > 0 pitches about −x, b > 0 about −y | `test_gimbal_torque_signs` |
| RCS pure-axis torques | off-axis rate < 1e-6 of on-axis | `test_rcs_allocation_produces_requested_axis` |
| Mass bookkeeping | mass + propellant used = m₀ to 1e-9 kg | `test_mass_conservation_and_total_impulse` |
| Total impulse in vacuum | = Isp g₀ m_used to 1e-9 | same |
| Tsiolkovsky Δv (60 s burn, m₀/m₁ > 1.35) | within 1e-4 | `test_tsiolkovsky_delta_v` |
| Burn to depletion | Δv within 1e-4, thrust 0 after | `test_burn_to_depletion_matches_rocket_equation` |
| Work–energy with mass loss | ΔKE = W_thrust + ½v² dm to 1e-3 | `test_work_energy_with_variable_mass` |
| Gimbal actuator | transport delay, rate limit, settles within backlash, overshoot < 5 % | `test_gimbal_actuator_delay_rate_limit_and_settling` |
| Ignition delay | thrust onset at `ignition_delay_s` (± 2 steps); none in fast mode | `test_ignition_delay` |
| RCS PWM | impulse per frame within 15 % of averaged model; sub-minimum pulse dropped | `test_rcs_pwm_impulse_and_minimum_impulse_bit` |

## Known limitations

- **Propellant slosh:** not modelled. The liquid is a rigid cylinder on the tank bottom (see `mass.md`).
- **Tanks:** no pressurisation, ullage or feed-system model. Isp does not depend on tank pressure, propellant temperature or mixture ratio, and there is no ullage-settling requirement for relights.
- **Combustion:** no thrust oscillation, combustion instability or start/shutdown transients beyond the first-order throttle lag and the ignition delay. Shutdown impulse is not modelled separately.
- **Nozzle:** no side loads during start-up or at flow separation, and no gimbal-angle-dependent loss.
- **Thrust vector:** the thrust axis always passes through the gimbal pivot on the body axis. Lateral thrust offset is not modelled; misalignment is angular only.
- **Moving engine mass:** the mass of the gimballed engine does not move the CG, and there is no "tail-wags-dog" reaction torque.
- **Jet damping:** the jet-damping moment from the mass flow through the rotating vehicle is not modelled.
- **RCS:** cold-gas thrust does not depend on remaining gas pressure or temperature, and plume impingement is not modelled.
- **Solid motors:** no temperature dependence of the burn rate and no ambient-pressure thrust correction.
