# Drone-ship landing: sea state, deck motion and guidance

**Code:** `src/plume/physics/ship.py` (sea state, barge response, deck body, link, clamp), `src/plume/control/ship_autopilot.py` (guidance), `src/plume/envs/landing_env.py` + `configs/envs/ship_landing.yaml` (scenario), `src/plume/viz/static/js/ship.js` (viewer)
**Fidelity:** both. The deck motion is the same at both levels. At `high` the vehicle also lands on crushable legs (see [landing_gear.md](landing_gear.md)).
**Verification status:** verified (spectrum moments and peak, significant wave height of the synthesised sea, transfer-function limits, deck-motion variance vs. the spectral prediction, kinematic consistency, deck-relative touchdown speed, friction carry and clamp; `tests/test_ship.py`). Validation against barge motion records and real landings is pending.

```bash
uv run plume land --stage ship_landing                    # fast fidelity, rigid legs
uv run plume land --stage ship_landing --fidelity high    # crushable legs on a steel deck
```

The bundled replay `data/replays/ship_landing_high.plume.json.gz` is a high-fidelity landing (seed 7). It came down 1.6 m from the centre of the circle at 1.06 m/s relative to the deck, used 9 % of the leg stroke, and then rode out 10 s of a rolling and pitching deck.

## Sea state

Irregular waves come from a **JONSWAP** spectrum (DNV-RP-C205 §3.5.5) with significant wave height `Hs`, peak period `Tp` and peak enhancement `γ` (`γ = 1` is Pierson–Moskowitz):

```
S(ω) = A_γ · (5/16) Hs² ωp⁴ ω⁻⁵ exp(−5/4 (ω/ωp)⁻⁴) · γ^exp(−(ω−ωp)² / (2σ²ωp²))
A_γ = 1 − 0.287 ln γ,   σ = 0.07 (ω ≤ ωp) / 0.09 (ω > ωp),   ωp = 2π/Tp
```

The sea is synthesised as a sum of linear deep-water components:

```
η(e, n, t) = Σ a_i cos(ω_i t − k_i (e cos θ_i + n sin θ_i) + φ_i),   a_i = √(2 S(ω_i) Δω D(θ_i)),   k_i = ω_i²/g
```

- `components` frequencies span 0.55–3.5 ωp. Each frequency is jittered inside its bin, independently for each direction, so the record does not repeat and no two components share a frequency (coherent components would bias the variance at a point).
- Phases are random (seeded).
- Directions are long-crested by default. Optional `cos^2s` spreading is discretised into `directions` bins.

`SeaStateSpec.direction_deg` is the mean wave heading relative to the bow: 180° is head seas, 90° beam seas.

## Barge response

The ship is a 91 m × 30 m hull with wings that widen the deck to 52 m (`ShipSpec`). Its draught is 4 m and its freeboard 3 m. The ship origin is the waterline amidships, which is also the centre of rotation. Each degree of freedom is the sum over components of a transfer function `H(ω, β)` times the wave at the ship (β is the encounter angle):

- **Heave and pitch.** These use the closed-form expressions for a homogeneously loaded box barge at zero speed (Jensen, Mansour & Olsen, *Ocean Engineering* 31, 2004), with `k_e = |k cos β|`:

  ```
  A = 2 sin(kB/2) e^(−kT)                         f = √((1 − kT)² + (A²/(kB))²)        κ = e^(−k_e T)
  F = κ f sin(k_e L/2)/(k_e L/2)                  G = κ f k_e · 3 (sin x − x cos x)/x³,  x = k_e L/2
  η_r = [(1 − 2kT)² + (A²/(kB))²]^(−1/2)          heave = η_r F (m/m),   pitch = η_r G (rad/m)
  ```

  Heave follows the elevation at midship and pitch follows the slope along the hull. In long waves they reach the limits 1 and `k`. The barge averages out waves shorter than about its length.
- **Roll.** A single-DOF oscillator is driven by the beam-averaged slope `k sin β · 3(sin y − y cos y)/y³` (with `y = k|sin β|B/2`), times `κ`. The natural frequency is `ω_φ = √(g GM_T)/k_xx`, with `GM_T = T/2 + B²/(12T) − KG`, and the damping ratio is `ζ = 0.1`. The default ship has a 7.1 s roll period.
- **Surge, sway and yaw.** Surge and sway are the hull-averaged orbital displacement (sinc over length or beam). Yaw is a small differential-sway term.
- **Station keeping.** Thrusters hold the ship near its mean position, modelled as a slow, smooth wander: three sinusoids per channel with periods `P`, 0.61 `P` and 0.37 `P` (`P = 120 s`), 1.5 m 1-σ in position and 1° in heading.

With the shipped scenario (`Hs` 2.5 m, `Tp` 9 s, waves 30° off the bow, spread `s = 4`), one hour of motion gives (three seeds):

| | 1-σ |
|---|---|
| heave | 0.29 m |
| pitch | 0.8° |
| roll | 2.6° (peaks of 8–9°) |
| surge / sway | 0.18 / 0.24 m |
| yaw (waves only) | 0.18° |
| heave rate | 0.20 m/s |

In quartering seas at the roll resonance (`Hs` 3 m, `Tp` 8 s, 120°), roll reaches ±11°.

## Simulation

The deck is a box geom (`ground_deck`) on a body with a **free joint whose pose and velocity are prescribed** every physics step (`ShipModel.drive`). Contacts therefore see the true deck velocity. Friction carries the vehicle with the heaving, swaying deck, and the deck pushes it up through the contact. The ship (1.1 × 10⁷ kg) is not pushed back by the lander (10⁻⁴ of its mass), so the vehicle's reaction on the ship is neglected. Other details:

- **Height above ground** (`State.agl`) is measured to the deck plane while over the deck, and to mean sea level elsewhere.
- **Touchdown speeds** (`Touchdown.vertical_speed` / `horizontal_speed`) are relative to the deck under the vehicle, taken at the start of the contact step (the impact speed).
- **Footpads** grip the steel deck with friction 0.6 (`steel_deck`, rigid soil).
- **Hold-down clamp** (`ShipSpec.clamp_delay_s`, off by default). It is an idealised octagrabber-style robot that locks the vehicle to the deck where it stands, implemented as a MuJoCo weld equality activated at runtime (`RocketSim.clamp_to_deck`). A real robot takes minutes to drive under the vehicle; the delay is a parameter.
- **Ship-to-vehicle link** (`DeckLink`). The deck target's position and velocity arrive at 10 Hz, 0.2 s late, with 5 cm / 3 cm/s noise, as from RTK GNSS on the ship. The flight software extrapolates over the latency.

## Guidance

`ShipLandingAutopilot` flies the unchanged land autopilot (`LandingAutopilot`: hoverslam profile, zero-effort-miss divert, aero-aware allocation) in a frame that moves with the deck (a Galilean change of frame):

- **Target.** The deck target from the link.
- **Vehicle velocity.** Taken relative to a smoothed deck velocity. Horizontally this is the station-keeping drift (low-passed, τ = 4 s): the vehicle neither can nor should follow wave-frequency sway. Vertically it is the heave rate (τ = 0.4 s), faded in over the last 12 m, so the final sink rate is relative to the deck.
- **Wind forecast.** Shifted by the same velocity, so air-relative velocities, and therefore the aerodynamic predictions, are unchanged.

The engine cuts at first leg contact. After that the vehicle stays upright through pad friction and its 21° static tip-over margin. On a rolling deck that margin shrinks by roughly the roll angle (`min_tipover_margin_deg`).

**Judging the landing.** Touchdown under 2 m/s vertical and 1 m/s horizontal relative to the deck, inside the 10 m circle. Then 10 s on the deck at rest relative to it, and within 10° of the deck normal. A vehicle that misses the deck and falls below sea level is a `splashdown`.

## Results

These are measured with the PID/guidance baseline on the `ship_landing` stage: a 0.5–1 km descent at 30–60 m/s, 0–40 m targeting error, 0–8 m/s wind with gusts and turbulence, and the sea state above.

| fidelity | episodes (seeds) | success (95 % CI) | landing error (mean) | touchdown sink rate rel. deck (mean) | fuel (mean) |
|---|---:|---:|---:|---:|---:|
| fast (rigid legs) | 50 (1000–1049) | 50/50 (93–100 %) | 2.5 m | 0.97 m/s | 142 kg |
| high (crush legs; guidance on the true vehicle state, as in the landing env) | 30 (2000–2029) | 30/30 (89–100 %) | 2.2 m | 0.98 m/s | 145 kg |

This scenario is moderate: it starts below 1 km, with the vehicle already in the descent burn envelope. The run-to-run spread comes mostly from the wind and the initial targeting error, not from the deck. Heave (0.29 m 1-σ) and drift are small next to the 10 m circle. The deck motion matters most for the touchdown itself: the sink rate relative to the deck, and the tip-over margin on a rolling deck (17° minimum in the bundled replay, against 21.7° on level ground). Harder seas and deck-motion prediction are not yet assessed by Monte Carlo.

## Verification

`tests/test_ship.py`:

| check | result |
|---|---|
| ∫S dω = Hs²/16 and peak at ωp (JONSWAP and PM) | 2 %, 1 % |
| synthesised sea: `4·std(η)` over 1 h = Hs (3 seeds, spread sea) | within 5 % |
| heave/pitch transfer functions → 1 and `k` in long waves; short waves average out | 1 %, 2 % |
| deck heave and pitch variance = `Σ a²H²/2` over 1 h; head seas give zero roll | within 12 % |
| deck velocity and body rate = finite-difference derivatives of the pose | 1e-3 |
| deck link (latency-compensated) median error | < 0.3 m |
| lander set on a rolling deck (`Hs` 3 m, quartering) rides with it (relative speed < 0.2 m/s, within 3° of the deck normal); the clamp then holds it to 2 cm | pass |
| touchdown speed relative to the deck | 0.08 m/s |
| nominal ship landing, 3 seeded episodes (fast) | 3/3 |
| high-fidelity ship landing with crush legs (`slow`) | pass |

## Assumptions and limitations

- **Linear seas.** Waves are linear and in deep water. There are no breaking waves, spray or green water, and no second-order drift forces; those are represented by the station-keeping wander instead.
- **Simplified barge response.** The closed-form transfer functions describe a homogeneous box barge with no ballast tuning, phase lags of heave and pitch near resonance are neglected (amplitudes only, signed), and roll is a 1-DOF oscillator with linear damping. Validation against model tests or barge motion records is pending.
- **Ship-rocket coupling.** The ship does not respond to the vehicle (thrust impingement, contact loads), and plume-deck interaction (overpressure, heating) is not modelled.
- **Guidance.** The guidance does not predict deck motion. It tracks the smoothed drift and, near the deck, the heave rate. Timing touchdown to a heave or roll phase (quiescent-period prediction) is future work.
- **Tip-over margin.** The margin is static. Inertial loads from deck acceleration are not included; the simulated dynamics are.
- **Viewer.** The ocean is rendered from the 48 strongest wave components recorded in `meta.ship.sea.waves`; the deck pose comes from the recorded frames.
