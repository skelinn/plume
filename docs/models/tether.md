# Tether (tethered hover tests)

**Code:** `src/plume/physics/tether.py` (`Tether`, `TetherSpec`), installed as a `RocketSim.extra_forces` entry; used by the hop-rig scenarios (`src/plume/hoprig/scenarios.py`)
**Fidelity:** both. High fidelity evaluates the rope force at every RK4 stage; fast fidelity holds it over the step, evaluated at the predicted mid-step stretch.
**Verification status:** verified. Tests cover slack (no force), taut (tension k·s toward the anchor), tension-only damping (it never pushes), the torque about the CG from an offset attach point, point kinematics on a rotated body, energy conservation of the undamped rope over repeated bounces (drift 0.01 % of the drop energy in high fidelity, 0.04 % in fast fidelity; tested below 0.1 % and 0.5 %), dissipation and the static stretch m·g/k (`tests/test_tether.py`), and the tether arresting a stuck-throttle climb (`tests/test_hop_rig.py`). Validation against a load-cell record from a real tethered test is pending.

## Model

A rope between a fixed anchor **a** (world frame) and an attach point **r** on the vehicle (body frame, z from the hull base). With **p** = x + R·r the attach point in the world and **v**ₚ its velocity:

| quantity | expression |
|---|---|
| stretch | s = ‖p − a‖ − L |
| stretch rate | ṡ = u · vₚ,  u = (p − a)/‖p − a‖ |
| tension | T = max(k s + c ṡ, 0) if s > 0, else 0 |
| force on the vehicle | F = −T u (world), at p |
| torque about the CG | τ = (r − r_cg) × Rᵀ F (body) |

`k` is the axial stiffness of the rope (EA/L), `c` a viscous damping coefficient (`TetherSpec.critical_damping(k, m, ζ)` gives c for a damping ratio ζ on a hanging mass m). The `max(·, 0)` makes it tension-only: a recoiling rope goes slack instead of pushing. The elastic energy ½ k s² (s > 0) is available as `Tether.elastic_energy(sim)` for energy checks.

In fast fidelity the simulator holds forces constant over a step. A stiff spring evaluated at the start of the step pumps energy in: about 40 % of the drop energy over a few bounces in the test case. The stretch is therefore evaluated at the predicted mid-step position p + ½ dt vₚ, the same device the simulator uses for gravity, which brings the drift down to 0.04 %.

| parameter (`TetherSpec`) | default | meaning |
|---|---|---|
| `anchor` | (0, 0, 0) | world position of the anchor (ground or crane hook), m |
| `attach` | (0, 0, 0) | body position of the attachment (hull base), m |
| `length` | 3.0 | unstretched length, m |
| `stiffness` | 2×10⁴ | k, N/m |
| `damping` | 1.5×10³ | c, N s/m |
| `breaking_load` | none | flags `overloaded` when exceeded (the rope keeps pulling) |

The hop-rig scenarios use a ground anchor at the pad centre, `length` 3 m (4 m for the identification hover), damping at 0.4 of critical for the wet rig mass, and a 15 kN rated load.

## Assumptions and limits

- Massless, straight rope: no sag, no waves travelling along it, no rope inertia. This is fine for short, light tethers on a ~200 kg rig. A long or heavy cable (a crane line of tens of metres) needs a catenary or lumped-mass model.
- Linear elastic with viscous damping. Real synthetic ropes are stiffer as they stretch and dissipate energy by hysteresis; use the slope and loss from a pull test of the actual rope.
- One anchor, one attach point. A multi-leg bridle or several ground tethers means installing several `Tether` objects.
- No contact between the rope and the vehicle or the ground, and no friction at a pulley.
- The viewer does not draw the rope. The tension and stretch are in the replay frames (`tether_tension`, `tether_stretch`).
