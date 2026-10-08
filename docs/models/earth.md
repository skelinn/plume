# Earth model, frames and integration

**Code:** `src/plume/physics/earth.py`, `src/plume/physics/gravity.py`, `src/plume/physics/sim.py`
**Fidelity:** `high` (`WorldSpec.gravity: wgs84`). Fast mode uses a flat or non-rotating spherical Earth.
**Verification status:** verified (NASA check cases 1–10, analytic tests). Validation against flight data is pending.

## Frames

| Frame | Definition |
|---|---|
| ECEF | WGS-84 Earth-centred, Earth-fixed |
| World (sim) | East-North-Up at a geodetic origin (the launch site), fixed to the rotating Earth. The MuJoCo world frame is this frame. |
| Local ENU | East-North-Up at the vehicle's current geodetic position (`EarthGravity.enu_at`) |
| Body | Vehicle: +z along the thrust axis (nose), origin at the dry-body reference point |
| Map | Azimuthal-equidistant projection (WGS-84) centred on the origin: `u` east, `v` north, metres. Terrain and mission sites use this. |

## Geodesy

- **Ellipsoid:** WGS-84 (a = 6 378 137 m, f = 1/298.257223563).
- **Geodetic → ECEF:** closed-form.
- **ECEF → geodetic:** Bowring's method plus two Newton refinements. The round-trip error is below 1e-10° and 1 µm (tested).
- **Map projection:** pyproj `+proj=aeqd` on the ellipsoid, falling back to spherical aeqd when pyproj is missing.

## Gravity

Zonal harmonics J2–J6 (EGM96, normalised values converted to unnormalised), evaluated by Legendre recursion:

U = −GM/r · [1 − Σₙ Jₙ (a/r)ⁿ Pₙ(sin φ)],  g = −∇U

- GM = 3.986004418e14 m³/s²
- `zonal_degree` selects 0 (point mass) to 6.
- The analytic gradient is checked against finite differences of U to 1e-7.
- Normal gravity at the equator (with centrifugal) is 9.7803 m/s², matching WGS-84.

## Rotating Earth

The world frame rotates with ω = 7.292115e-5 rad/s. The equations of motion are written in that frame.

- **Linear:** a = F/m + g − 2 ω×v − ω×(ω×r), where r is measured from the Earth's centre.
- **Attitude:** MuJoCo integrates the body rate relative to the world frame, ω_rel. The inertial rate is ω_abs = ω_rel + Rᵀω_e. The torque that makes Euler's equations hold for ω_abs is
  τ_corr = −ω_abs × Iω_abs + ω_rel × Iω_rel + I(ω_rel × ω_e,b).

**Check:** the Jacobi integral (kinetic energy + gravitational potential − ½|ω×r|²) is conserved to < 1e-6 of the kinetic energy over 4 000 s of tumbling vacuum flight on a J6 rotating Earth (`tests/vv/test_analytic.py`).

## Integration

MuJoCo RK4 at dt = 5 ms (default in high fidelity).

- **Fast mode:** forces are computed once per step and held through the RK4 stages. The integration error is first order.
- **High fidelity:** every external force (aero, grid fins, gravity, rotating-frame terms, thrust, RCS, parachute) is re-evaluated at each RK4 stage through MuJoCo's passive-force callback (`mjcb_passive`). Actuator states (throttle, gimbal, fin angle) stay held for the step, as a zero-order-hold flight computer would.

**Convergence** (tumbling body with drag, 30 s): error shrinks about 4× per halving of dt. At 5 ms it is under 1 cm against a 1.25 ms reference.

## NASA check cases (NASA/TM-2015-218675)

`tests/vv/test_nasa_checkcases.py` runs atmospheric check cases 1–10:
- dropped sphere: non-rotating and rotating Earth, J2, drag, wind
- tumbling brick, with and without damping
- cannonball

Each is compared with the published trajectories from up to six NASA simulation tools. The pass criterion is max(floor, 2 × the spread between the NASA tools). All ten pass. For example, the tumbling-brick attitude error is about 0.0001° against a tool spread of 1–4°.

## Mission frame (high fidelity)

`plume hop <mission> --fidelity high` (see `missions/hop.py::mission_frame`):
- puts the world origin at `launch.lat/lon` on the ellipsoid
- projects the target from `target.lat/lon` into map coordinates
- uses this Earth model for the 6-DOF sim, the 3-DOF ascent planner (`PointMassSim`, which includes Coriolis and centrifugal terms) and the ballistic impact predictor (solved in inertial space, with the impact point rotated back by Earth's rotation over the time of flight)

## Known limitations

- **Heights:** DEM heights are orthometric (EGM2008) but the sim uses ellipsoidal heights. The geoid undulation (about −25 m in New Mexico) is not applied. This is consistent within a run, but absolute heights are offset.
- **Gravity field:** no tesseral or sectoral harmonics and no third-body or tidal terms. These are negligible for sub-orbital hops.
- **Earth orientation:** polar motion and UT1 are not modelled. Earth rotation is a constant rate about the ECEF z axis.
