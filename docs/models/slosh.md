# Propellant slosh

**Code:** `src/plume/physics/slosh.py`, enabled per tank with `tanks[].slosh: true` (damping `slosh_damping`)
**Verification status:** verified. Tests check the analytic first-mode frequency and mass, the free-oscillation period (within 1 %), and closed-loop hover stability with slosh coupling (`tests/test_slosh.py`). Validation against tank tests or flight data is pending.

## Model

Each tank with `slosh: true` adds the first antisymmetric (lateral) slosh mode as an equivalent spring–mass, after Abramson (NASA SP-106, 1966) and Ibrahim (*Liquid Sloshing Dynamics*, 2005). For an upright cylindrical tank of radius R, filled to height h under axial settling acceleration a:

| quantity | expression |
|---|---|
| mode constant | ξ₁ = 1.8412 (first root of J₁′) |
| natural frequency | ω² = (a ξ₁ / R) · tanh(ξ₁ h / R) |
| sloshing mass | m₁ / m_liquid = 2 tanh(ξ₁ h/R) / (ξ₁ (ξ₁² − 1) h/R) |
| location | (R/ξ₁) · tanh(ξ₁ h / 2R) below the free surface |

The fill height follows the propellant mass. The frequency follows the current settling acceleration (thrust and drag), so it changes through each burn.

The rest of the propellant stays in the rigid mass model. Relative to the tank wall, the slosh mass obeys

x″ = −ω² x − 2ζω x′ − a_lat,

where a_lat is the tank's lateral specific force at the slosh-mass height, including the centripetal term of the body rotation. The rigid model already carries m₁ along with the tank, so the vehicle receives only the reaction of the relative motion, F = −m₁ x″, applied at the slosh-mass height.

## Parameters

| parameter | default | notes |
|---|---|---|
| `slosh` | false | opt-in per tank |
| `slosh_damping` (ζ) | 0.02 | smooth wall 0.005–0.02; ring baffles 0.03–0.1 |

## Validity and limitations

- First mode only, and small amplitude: |x| is held below R/2, and nonlinear rotary slosh is not modelled.
- The tank is treated as a flat-bottomed cylinder. Domes shift the mode slightly, and the mass model still treats the propellant as rigid.
- Below a settling acceleration of 0.5 m/s², for example in ballistic coast, the model freezes. The propellant is assumed held by a management device; no free-floating propellant or ullage-bubble dynamics are modelled.
- There is no coupling to structural bending modes.
