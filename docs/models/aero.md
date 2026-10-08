# High-fidelity aerodynamics: database, generator, verification

**Code:** `src/plume/physics/aerodb.py`, `src/plume/physics/aero_gen/` (files listed below)
**Selection:** `AeroSpec.model: auto | strip | database`. With `auto`, high fidelity uses this database and fast mode uses the strip model (`src/plume/physics/aero.py`).
**Verification status:** verified against public NASA wind-tunnel data at M 2.86 (normal force, centre of pressure, axial force) and against exact theory (Taylor-Maccoll cones, Rossow ogive charts, Love base pressure, slender-body limit) (`tests/test_aerodb.py`). Subsonic/transonic body data and finned-body data are not yet compared (see [Verification gaps](#verification-gaps)). Validation against flight data is pending.

Files:

| file | content |
|---|---|
| `src/plume/physics/aerodb.py` | `AeroDatabase` (tables, IO, Monte Carlo), `AeroModelHiFi` (runtime forces), `GridInterpolator`, `sutherland_viscosity`, `reynolds_per_metre` |
| `src/plume/physics/aero_gen/geometry.py` | body / fin planform / leg geometry |
| `.../aero_gen/body.py` | body CN, Cm for alpha 0-180 deg (Allen & Perkins / Jorgensen) |
| `.../aero_gen/nose.py` | nose wave drag, Taylor-Maccoll solver |
| `.../aero_gen/friction.py` | compressible turbulent skin friction |
| `.../aero_gen/base.py` | base drag, blunt-face (engine-first) drag, stagnation pressure |
| `.../aero_gen/fins.py` | Barrowman / Ackeret fins |
| `.../aero_gen/newtonian.py` | modified Newtonian panel integration |
| `.../aero_gen/gridfins.py` | grid-fin Mach multipliers |
| `.../aero_gen/srp.py` | supersonic retro-propulsion |
| `.../aero_gen/heating.py` | Sutton-Graves heating |
| `.../aero_gen/generate.py` | `generate_database(vehicle)` |
| `.../aero_gen/importers.py` | RASAero II, OpenRocket, Missile DATCOM, generic CSV |
| `tests/test_aerodb.py`, `tests/data/aero_ref/` | tests and digitised reference data |

## 1. Conventions

Body frame as in `physics/aero.py`: +z along the axis toward the nose, origin at
the base of the hull.  `v` is the velocity of the CG relative to the air.

* **Total angle of attack** `alpha_t = atan2(sqrt(vx^2 + vy^2), vz)` in [0, 180] deg:
  0 = nose first, 180 = engine (base) first.
* **Aerodynamic roll angle** `phi = atan2(vy, vx)` (only used by roll-dependent tables).
* Reference area `S` = hull cross-section (or `AeroSpec.reference_area`), reference
  length `d` = hull diameter, moment reference `x_ref` (body z; generated tables use
  0 = the base).
* `CA`: body-axis axial force, `F_z = -q S CA` (positive nose first, negative engine
  first).  `CN`: normal force in the alpha_t plane, positive when it opposes the
  lateral velocity.  `Cm`: moment about `x_ref`, positive when it increases alpha_t;
  centre of pressure `z_cp = x_ref + d Cm/CN` (identical to Jorgensen's convention
  `x_ac = x_m - d Cm/CN` measured from the nose).
* `CY, Cn, Cl`: side force / yaw / roll for roll-dependent tables (zero for bodies of
  revolution and for >= 3 fins in linear theory).
* `Cmq`: pitch damping (Cmq + Cm_alpha_dot) **about the centre of pressure, in
  excess of the quasi-steady lever-arm damping of the normal force** (see 2.2),
  normalised by `q S d (q_rate d / 2V)`; `Clp` by `q S d (p d / 2V)`.  Both <= 0.
* `CA_base`: the power-off base-drag part of `CA` (nose first).

Static stability: nose first stable when `z_cp < z_cg`; engine first stable when
`z_cp > z_cg` (`AeroModelHiFi.static_margin` returns a value that is positive when
stable in either case).

## 2. Database and runtime model

### 2.1 `AeroDatabase`

Rectilinear grid over `mach` (default 25 points, 0-8, dense transonic),
`alpha_deg` (37 points, 0-180, dense near 0 and 180), optional `phi_deg` (with
`meta["phi_period_deg"]`), optional `log10_re` (Reynolds number per metre; default
grid 4 ... 8).  Per coefficient a 1-sigma absolute uncertainty table `sigma`.
`meta` carries provenance per coefficient, methods, references, validity,
geometry and generator options.

* `evaluate(mach, alpha_deg, phi_deg=0, log10_re=None) -> dict`
* `evaluate_signed(mach, alpha_deg)` (CN, Cm odd in alpha)
* `centre_of_pressure(...)`, `zcp_table()`
* `perturbed(rng, scale=1)` - Monte Carlo copy: one standard-normal draw per
  coefficient, applied as `c + z sigma` over the whole grid (fully correlated,
  keeps tables smooth; damping clipped to <= 0)
* `check()` - physical sanity warnings (negative drag, anti-damping, alpha coverage)
* `save(path)` -> `path.npz` (+ `path.json` sidecar), `AeroDatabase.load(path)`,
  `to_csv(path)` (long format, readable by `import_generic_csv`)

Interpolation is multilinear with clamping at the grid edges
(`GridInterpolator`: pure-Python index search, one gather, one dot product;
~4 us for all coefficients on a 3-D grid).

### 2.2 `AeroModelHiFi`

Same interface as `Aero`: `forces(v_air_body, omega_body, cg_z, rho, sound_speed)
-> (force_body, torque_about_cg_body, q, mach)`, attributes `enabled`, `ref_area`,
`cd_scale` (settable; multiplies every coefficient), `axial_coefficient(mach,
nose_first)`, `crossflow_coefficient(mach_cross)` (CN at 90 deg x S / A_planform),
`strip_area`, `z` (compatibility), plus `thrust` / `set_thrust(T)`,
`static_margin(mach, alpha, cg_z)`, `coefficients(...)` and
`AeroModelHiFi.from_vehicle(vehicle, db=None | AeroDatabase | path, **gen_kw)`.

Algorithm per call:

1. Mach, q and Re/m (Sutherland viscosity at T = a^2 / (gamma R)) from the CG airspeed.
2. alpha_t, phi at the CG -> centre of pressure `z_cp` (interpolated table, clipped
   to [-0.5 L, 1.5 L]).
3. Local air velocity at the CP, `u = v + omega x (z_cp - z_cg) e_z`; alpha_t, phi
   from `u` -> CA, CN, CY, Cl, Cmq, Clp, CA_base.
4. Power effects (if `thrust > 0`, `C_T = T / (q S)`): nose first,
   `CA -= CA_base (1 - f_base)`, `f_base = (1 - A_e/A_b) exp(-C_T / 0.5)`; engine
   first, `CA *= exp(-C_T / 0.4)` (SRP).
5. `F = q_u S (-CN e_lat + CY e_side - CA e_z)`, applied **at the CP**; torque `r_cp x F`.
6. Rate damping `tau = q S d (d / 2U) (Cmq omega_x, Cmq omega_y, Clp omega_z)` with
   Cmq, Clp clipped to <= 0, plus the static roll moment `q S d Cl`.

**Dissipativity.**  The aerodynamic power is
`F.v + tau.omega = F.(v + omega x r_cp) + tau_damp.omega = F.u + tau_damp.omega`.
Every generated table has `CN >= 0` and `CA >= 0` (nose first) / `<= 0` (engine
first), so the drag `CN sin a + CA cos a >= 0` and `F.u <= 0`; a guard removes any
positive component along `u` that interpolation or Monte Carlo dispersions could
create.  With Cmq, Clp <= 0 the total power is never positive with zero wind (test
`test_dissipative_no_wind`, random states incl. pure rotation, power-on and
dispersed tables).  The only non-dissipative term is a non-zero static roll moment
`Cl` (fin cant) from imported data, which is physical.

Because the normal force is applied at the CP with the local velocity there, the
runtime already produces the quasi-steady lever-arm damping
`Cmq_lever = -2 CN_alpha ((z_cp - z_cg)/d)^2`; the table's `Cmq` is only the extra
part, which is independent of the CG.  Importers convert total damping about a
reference point with `Cmq_extra = Cmq_total + 2 CN_alpha ((z_cp - z_ref)/d)^2`.

Performance (Windows, Python 3.12, `cargo_hopper` / `hobby_rocket`): **12 us per
`forces` call** (fast strip model: 12-13 us); test limit 50 us.

## 3. Generator (`generate_database(vehicle, mach_grid, alpha_grid, log10_re_grid, options | **overrides)`)

### 3.1 Body normal force and pitching moment (alpha 0-180 deg)

Jorgensen, NASA TN D-6996 (1973), eqs. (1), (4)-(6), after Allen (NACA RM A9I26)
and Allen & Perkins (NACA Rep. 1048):

    CN = (Ab/A) sin(2a') cos(a'/2) + eta Cdn (Ap/A) sin^2(a'),   a' = min(a, 180 - a)

* slender-body (potential) term acts at `x = l - V/Ab` from the nose for a <= 90 deg
  (CN_alpha -> 2 Ab/A per rad, Munk; tested) and at `x = V/Ab` for a > 90 deg;
* viscous crossflow term acts at the planform centroid;
* `Cdn(M_n, Re_n)`: TN D-6996 Fig. 1 (subcritical, M_n = M sin a, digitised; peak
  ~2.1 at M_n ~ 0.95, -> 1.29 hypersonic = modified Newtonian), with the
  crossflow-Reynolds drag crisis of Fig. 2 (Re_n = Re_d sin a; Cdn falls from 1.2 to
  ~0.3 between Re_n 2e5 and 5e5 and recovers to ~0.65 at 1e7) applied fully for
  M_n <= 0.3 and faded out by M_n = 0.5;
* `eta(l/d)` finite-length factor, Fig. 4 (Goldstein), blended to 1 between M 0.8 and
  1.2 (Jorgensen: eta ~ 1 supersonic);
* above M 4 the body CN/Cm are blended linearly into **modified Newtonian** panel
  integration (`Cp = Cp_max sin^2`, shadowed panels 0, Cp_max from the Rayleigh pitot
  formula), fully Newtonian from M 6.

Options: `reynolds_effect` (default True), `reverse_potential` (scale of the
potential term for a > 90 deg; 1 = Jorgensen, 0 = viscous-only like the fast strip
model).

### 3.2 Axial force

Nose first (alpha 0), on S:

* **skin friction**: Karman-Schoenherr incompressible law with **van Driest II**
  compressibility (Hopkins & Inouye 1971 recommendation; Eckert's reference
  temperature method available), adiabatic wall, fully turbulent, Schlichting
  fully-rough limit with roughness `roughness` (default 20 um), times Hoerner's body
  form factor `1 + 1.5 (d/l)^1.5 + 7 (d/l)^3` (faded out between M 0.8 and 1.2), on
  the wetted area;
* **nose wave drag**: cones - Linnell-Bailey (TN D-6996 eq. 10, a fit to exact
  Taylor-Maccoll pressures, checked here against our own TM solver within 6 %);
  tangent ogives - cone value x ogive/cone ratio of Rossow's method-of-characteristics
  correlation (TN D-6996 Fig. 6); von Karman - x 0.83 (minimum-drag curve of Fig. 6,
  proxy); valid once the shock is attached on the equivalent cone (TM detachment) and
  `beta sin t >= 0.05`; subsonic 0.8 sin^2 t (cones) / 0 (smooth noses); PCHIP
  transonic blend through the cone sonic value sin t (Hoerner via Niskanen);
* **base drag, power off** (`CA_base`): Hoerner `0.029/sqrt(C_Df)` x
  `(1 + 1.083 M^2)` subsonic, Love's turbulent-boundary-layer compilation (NACA TN
  3819 via TN D-6996 Fig. 8) supersonic, Gabeaud beyond M 8, blended 0.9-1.1;
* **fins**: friction (both sides, `1 + 2 t/c`), rounded/square leading-edge and square
  trailing-edge pressure drag (Hoerner via Niskanen eqs. 3.89-3.94);
* **legs**: deployed struts by the crossflow principle `1.2 d l sin^3(psi)` and flat
  footpads `1.17 pi r^2` (x 0.5 for the base-wake shielding nose first), grown with
  `Cp_stag(M)`.

Engine first (alpha 180): blunt face `k(M) Cp_stag(M)` (`k` 0.75 subsonic -> 0.85 from
M 1.2; supersonic value calibrated on Jernell's flat-faced cylinders, see 5), plus
friction, fins and legs (footpads leading, unshielded); the trailing pointed nose
adds no pressure drag.

Variation with alpha: `CA(a) = CA(0) cos^2 a` (a <= 90), `CA(180) cos^2 a` (a > 90)
(TN D-6996 eqs. 2-3); `CA_base` follows the nose-first branch.

Power on: base pressure on the annulus outside the nozzle exit rises toward ambient
as the plume fills the base: `f_base(C_T) = (1 - A_e/A_b) exp(-C_T / 0.5)` (engineering
fit, +-50 % on 0.5; at typical burn C_T = 2-10 the base drag is gone, as in RASAero
power-on and OpenRocket practice).

### 3.3 Supersonic retro-propulsion

Engine-first burns: `CA_aero(C_T) = CA_aero(0) exp(-C_T / 0.4)` for a single central
nozzle - the survey of Korzun, Braun & Cruz (JSR 46(5), 2009; Jarvinen & Adams 1970
data, M 2-4) shows the forebody drag largely lost by C_T ~ 1.  Valid M 1.5-4,
1-sigma +-50 % on the decay constant, doubled outside that range (subsonic landing
burns are an extrapolation).  Normal force under SRP unchanged (not modelled).

### 3.4 Fins

Barrowman (1967) subsonic CN_alpha with Diederich's compressibility form, n/2 for 3-4
fins (OpenRocket factors for 5-8), interference `K_T(B) = 1 + R/(s + R)`; Ackeret
`4/beta` per panel area with the rectangular-wing tip correction (slender-wing
`pi AR / 2` limit for `beta AR < 1`) from M 1.2; linear blend 0.9-1.2.  CP at the MAC
quarter chord subsonic, `(AR beta - 0.67)/(2 AR beta - 1)` MAC from M 2 (Niskanen
eq. 3.35).  Beyond the linear range a flat-plate law
`CN_alpha sin a' cos a' + 1.2 sin^2 a'` (Hoerner) acting at the planform centroid
(crossflow part).  Roll damping by strip theory, `Clp = -2 n a_f INT c (R+y)^2 dy / (S d^2)`.

`FinSpec` only has `count`, `cn_alpha`, `z`; `geometry.infer_fin_planform` keeps the
hobby-rocket proportions (root 1.75 s, tip 0.75 s, LE sweep 1.0 s, thickness
0.0375 s), scales the span so Barrowman's incompressible CN_alpha x K_T(B) equals
`cn_alpha`, and places the fin so its CP is at `FinSpec.z`.  For the hobby rocket
this gives a 77 mm span, 134/58 mm root/tip fin (the YAML's 9/rad at z = 0.09 m).

### 3.5 Grid fins (`aero_gen/gridfins.py`, used by `physics/gridfins.py`)

`gridfin_cn_alpha(mach, open_area_ratio=0.9)` and `gridfin_cd0(mach, ...)` return
multipliers equal to 1 at M -> 0 (YAML `cn_alpha`, `cd0` are low-speed values):
mild subsonic growth `(1 - M^2)^-1/4`; cell choking from `A/A*(M_ch) = 1/phi`
(M_ch = 0.68 for phi = 0.9) to the shock-swallowing Mach from the
Kantrowitz-Donaldson criterion (M_sw = 1.57), with a 45 % loss of effectiveness and a
1.9x drag peak in between (Washington & Miller 1993/1998, Simpson & Sadler 1998);
supersonic `CN_alpha ~ 1/beta`, drag `~ (beta_sw/beta)^0.5` (floor 0.6).
`gridfin_uncertainty(mach)`: 15 % subsonic, 35 % transonic, 20 % supersonic.

### 3.6 Heating

`stagnation_heat_flux(rho, V, nose_radius)`: Sutton & Graves (NASA TR R-376),
`q = 1.7415e-4 sqrt(rho/R_n) V^3` W/m^2; `heat_load(t, rho, V, R_n)`; `HeatingMonitor`;
`nose_radius_from_geometry(geometry, bluntness_ratio=0.05)`.  Validity: continuum,
equilibrium, cold wall, V >~ 2-3 km/s (+-10-15 %); order-of-magnitude below 1 km/s.

### 3.7 Rate damping

`Cmq_extra = -2 SUM_i k_i ((z_i - z_cp)/d)^2 - 2 k_visc (r_g/d)^2` over the
components (nose potential, viscous crossflow, fins) with `k_i = |dCN_i/da|` and, for
the quadratic viscous crossflow, at least the describing-function slope
`(8 / 3 pi) eta Cdn (Ap/A) sin(5 deg)`; `r_g` is the planform radius of gyration.
This is the quasi-steady strip argument (the fast model's strip integration in table
form); apparent-mass (Munk) Cm_alpha_dot is not included.  +-50 %.

## 4. Uncertainty model (1-sigma, in the `sigma` tables)

| coefficient | band | basis |
|---|---|---|
| CA | 10 % M < 0.8; 20 % 0.8-1.2; 10 % 1.2-5; 15 % > 5; >= 15 % engine first | component-method accuracy (Hoerner, Hopkins & Inouye +-10 %), transonic correlations, Jernell CA comparison |
| CA_base | 25 % | Love scatter, Hoerner |
| CN | 15 %; 25 % in the critical crossflow-Re band (M_n < 0.5, 1.5e5 < Re_n < 3e6); 30 % for alpha > 100 deg | Jernell comparison (sec. 5): 11 % rms nose first, 20 % base first; Jorgensen Fig. 2 scatter |
| Cm | sqrt((0.15 Cm)^2 + (0.25 CN)^2), i.e. +-0.25 d on the CP | Jernell CP comparison: 0.15 d rms, 0.37 d max |
| Cmq | 50 % | quasi-steady strip estimate |
| Clp | 30 % | strip theory |
| SRP decay constant | 50 % (x2 outside M 1.5-4) | Korzun et al. survey scatter |
| grid-fin multipliers | 15 / 35 / 20 % | literature spread |

## 5. Verification against reference data

Reference data are in `tests/data/aero_ref/` (see its README for exact sources).

**Jernell (NASA TM X-1658, 1968) via Jorgensen TN D-6996 Figs. 9-12**: cylinders,
cone-cylinders and ogive-cylinders, l/d 6-11, M 2.86, Re_d 1.25e5, alpha 5-175 deg.
Generator run with the same geometry (smooth, moments about x_m).

| quantity | range | measured error (model - data) | test tolerance |
|---|---|---|---|
| CN, 8 bodies | 15-105 deg (65 pts) | mean +8.3 %, rms 11.0 %, max 43 % (flat-faced body 1 / ogive body 7 at 15-25 deg, |dCN| <= 0.62) | each <= max(20 %, 0.7); rms < 12 % |
| CN, 6 bodies | 115-165 deg (base first, 37 pts) | mean +17 %, rms 19.9 %, max 42 % | each <= max(45 %, 0.7); rms < 22 % |
| CP from Cm/CN, bodies 3-5 | 25-165 deg (40 pts) | mean +0.08 d, rms 0.15 d, max 0.37 d | rms < 0.25 d, max < 0.5 d |
| CA engine/face first, bodies 1-5 | 5 deg (flat) / 175 deg | -0 % / +1 % (bodies 1-2, used to calibrate k = 0.85); +6-8 % (bodies 3-5) | 12 % |
| CA nose first, cone-cylinders 3-5 | 5 deg | +24 %, +33 %, +43 % (sting-support interference raises base pressure, TN D-6996 p. 21; Jorgensen's own estimate is also high) | 50 % |

Per-body CN errors (%; alpha in deg):
body 3 (cone-cyl 7): 15:+3 25:+10 45:+8 85:+4 105:+11 125:+26 145:+27 155:+35;
body 5 (cone-cyl 11): 25:0 45:+5 85:+9 125:+11 145:+14 155:+21;
body 9 (ogive-cyl 11): 25:+8 45:+9 85:+9 125:+12 145:+12 165:-5.
The method over-predicts base-first normal force (the published Jorgensen curves
show the same bias) while its CP is good; an empirical reverse-flow CN factor ~0.8
would fit CN but not Cm and is therefore not applied.

**Theory checks**: Taylor-Maccoll solver vs NACA 1135 (M 2, 10 deg cone: shock angle
31.2 deg) and Linnell-Bailey within 6 % (M 1.5-5, 5-20 deg); ogive and cone wave
drag vs Rossow / Ehret-Rossow-Stevens curves (TN D-6996 Fig. 6, K 0.6-1.8) within
6 %; base drag equals Love's curve (it is the implementation) and is continuous
through the transonic blend; Karman-Schoenherr Cf(1e7) = 0.00293; van Driest II at
M 3 = 0.64 x incompressible (classic result), Eckert within 15 % of it; modified
Newtonian flat face `CA = Cp_max`, broadside cylinder `CN = 2/3 Cp_max Ap/A`;
slender-body CN_alpha = 2/rad within 1 % (three nose shapes, M 0.3-3.5); body
geometry (Ap, V, x_c, A_s) equals TN D-6996 Fig. 9 within 0.3 %.

**Vehicles**: all four presets generate (~0.4 s each, cached Taylor-Maccoll),
pass `check()`; hobby rocket nose-first static margin > 1 calibre for every CG,
M 0.1-2, alpha 1-10 deg (CN_alpha 11.9/rad at M 0.1 vs YAML fins 9 + nose 2).

### Engine-first stability - important finding

With Jorgensen's reverse-flow potential term (which matches Jernell's base-first
CP data best: CP rms 0.19 d for `reverse_potential = 1` vs 0.43 d for 0, cone-
cylinders at 135-165 deg) the lander and the cargo hopper are **statically unstable
engine first near alpha_t = 180 deg**: the base-leading potential lift sits near the
base, below the CG.  Stable trim angles off the axis (CG at the heaviest-propellant
case):

| vehicle (z_cg) | M 0.15 | M 0.5 | M 1.5 |
|---|---|---|---|
| lander_small (3.42 m), default | 50 deg | 40 deg | 19 deg |
| lander_small, `reynolds_effect=False` | 27 deg | 27 deg | 18 deg |
| cargo_hopper hull only (5.31 m), default | 60 deg | 48 deg | 24 deg |
| cargo_hopper, `reynolds_effect=False` | 39 deg | 38 deg | 24 deg |

The low-speed values are worse with the default crossflow-Reynolds effect because
the full-scale crossflow is supercritical (Re_d ~ 3e6; Cdn ~ 0.3-0.4 instead of 1.2;
Jorgensen's "red warning flag").  The fast strip model (viscous crossflow only,
Cdn 1.2) puts the CP at the planform centroid and predicts weathercock stability at
all angles; `generate_database(v, reverse_potential=0, reynolds_effect=False)`
reproduces that behaviour (tested).  Physically this is consistent with flight
practice (boosters need grid fins / a ring fin at the top for engine-first
stability).  The hopper's grid fins (separate model) must provide the margin; the
autopilot/RL should be checked with the hi-fi model before relying on passive
stability.

### Verification gaps

* No subsonic/transonic body CN or CA data compared yet.  Next sources (public
  domain, NTRS): Jorgensen & Nelson, NASA TM X-3129 (1975) - ogive-cylinder l/d 10 at
  M 0.6-2.0, alpha 0-58 deg; NASA TM X-3310 (1976) - bodies with fins/wings;
  Perkins, Jorgensen & Sommer, NACA Rep. 1386 (nose drag, M 1.24-7.4);
  Jorgensen & Treon, NASA TM X-580 (booster model M 0.6-4, alpha 0-180 deg).  These
  PDFs exceed the fetch tool's size limit and were not digitised in this pass.
* No finned-body comparison (Barrowman is the de-facto standard; compare with
  TM X-3310 or RASAero/OpenRocket for the hobby rocket).
* Grid-fin multipliers, SRP and power-on base drag are trend models without a
  digitised dataset here.

## 6. Limitations

* Bodies of revolution with a flat base: no boattail, flare, or non-circular section;
  nose bluntness ignored for drag.
* Asymmetric vortex side forces (alpha 25-65 deg, low speed), Magnus forces and
  control-surface increments are not modelled.  Grid fins stay in `gridfins.py`.
* Legs are always deployed (drag only; no leg normal force).  Ascent with stowed legs
  over-predicts drag (hopper: ~0.3 of CA0 at M 1.5).
* Boundary layer fully turbulent; transition and heating effects on friction ignored.
* Static coefficients are quasi-steady; Cm_alpha_dot (apparent mass) not included in
  generated Cmq.
* The Reynolds axis is per metre; interpolation in log10(Re) smooths the sharp
  crossflow drag crisis (0.5-decade grid).
* Nose-first hypersonic CA uses the attached-shock correlations (no real-gas effects).

## 7. Integration (for the main engineer)

Proposed config additions (`config.py`):

```python
class GeometrySpec(Spec):
    ...
    nose_shape: Literal["cone", "tangent_ogive", "von_karman"] = "tangent_ogive"
    nose_tip_radius: float | None = None   # stagnation radius for heating, m

class FinSpec(Spec):
    ...                                      # existing count, cn_alpha, z
    root_chord: float | None = None          # m; None -> inferred planform
    tip_chord: float | None = None
    span: float | None = None
    sweep: float | None = None               # LE sweep distance root->tip, m
    thickness: float | None = None
    le_shape: Literal["rounded", "square"] = "rounded"

class GridFinSpec(Spec):
    ...
    open_area_ratio: float = 0.9             # for gridfin_cn_alpha / gridfin_cd0

class AeroSpec(Spec):
    ...
    model: Literal["strip", "database"] = "strip"
    database: str | None = None              # .npz path; None + model="database" -> generate
    reverse_potential: float = 1.0           # generator option (0 = fast-model-like)
    reynolds_effect: bool = True
    roughness: float = 20e-6
```

If `FinSpec` gets planform fields, build `FinPlanform(count, root_chord, tip_chord,
span, sweep, thickness, z_root_le, body_radius)` and pass it as
`generate_database(v, fin_planform=...)`.

`physics/sim.py` (and `pointmass.py` if desired):

```python
from plume.physics.aerodb import AeroModelHiFi
...
if vehicle.aero.model == "database" or self.world.fidelity == "high":
    self.aero = AeroModelHiFi.from_vehicle(
        vehicle, vehicle.aero.database,           # None -> generated
        reverse_potential=vehicle.aero.reverse_potential,
        reynolds_effect=vehicle.aero.reynolds_effect,
        roughness=vehicle.aero.roughness,
    )   # (gen kwargs only used when generating)
else:
    self.aero = Aero(vehicle.aero, vehicle.geometry)
...
# every physics step, before self.aero.forces(...):
if hasattr(self.aero, "set_thrust"):
    self.aero.set_thrust(thrust)
```

Generation takes ~0.4 s per vehicle; cache it (e.g. `functools.lru_cache` keyed on
`vehicle.model_dump_json()`, or save with `db.save()` next to the YAML).  For Monte
Carlo use `AeroModelHiFi(db.perturbed(rng), ...)` per run.  The returned `q, mach`
and the work bookkeeping (`F.v + tau.omega`) are unchanged.

`physics/gridfins.py`:

```python
from plume.physics.aero_gen.gridfins import gridfin_cn_alpha, gridfin_cd0
...
mach = speed / sound_speed            # forces() needs sound_speed passed in
cn_a = s.cn_alpha * gridfin_cn_alpha(mach, s.open_area_ratio)
cd0 = s.cd0 * gridfin_cd0(mach, s.open_area_ratio)
```

(`GridFins.forces` currently has no `sound_speed` argument; add it with a default of
`None` -> multiplier 1 to keep the existing call sites and tests working.)

Heating (optional telemetry): `HeatingMonitor(nose_radius_from_geometry(v.geometry))`
updated with `(dt, rho, |v_air|)`.
