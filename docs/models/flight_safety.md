# Flight-safety analysis export

The flight-safety export turns a simulated flight and a Monte Carlo campaign into the
geographic products a launch-licence application asks for: where the vehicle flies, where it
would come down if the engine stopped at any instant, where it lands, where failed flights hit
the ground, how likely failure is in each flight phase, and hazard areas around all of it. The
outputs are GeoJSON, KML (Google Earth) and a monochrome HTML summary.

**Code:** `src/plume/analysis/safety.py` (analysis and writers), `src/plume/analysis/safety_cli.py` (`plume safety`), `src/plume/viz/static/js/safety_overlay.js` (viewer)
**Selection:** on demand: `plume safety <mc dir or replay>`, and automatically after a planner reliability job
**Verification status:** verified, covering WGS-84 and spherical lat/lon round trips, target-offset inversion out to 2,000 km, the vacuum IIP against a numerical vacuum integration and the Earth-rotation (Coriolis) shift, hazard-geometry areas against closed forms, failure-phase assignment, GeoJSON/KML structure and the CLI (`tests/test_safety.py`). Validation is pending: there is no flight data, and no comparison with an accepted flight-safety tool yet.

> **Not a certified analysis.** This is an engineering input to a licence application (for
> example the flight safety analysis of FAA 14 CFR 450, §450.113–450.139), not a substitute for
> one. There is no breakup or debris model, no population or casualty-expectation (E<sub>c</sub>)
> computation, no flight termination system, and the probabilities carry the Monte Carlo
> sampling uncertainty stated with them.

## Usage

```bash
# Monte Carlo campaign + its nominal flight (ground track and IIP trace)
uv run plume safety runs/mc/real_hop_high --replay data/replays/real_hop_high.plume.json.gz

# one replay only (ground track, IIP trace, IIP corridor)
uv run plume safety runs/hop_real_hop.plume.json.gz --iip-step 1

# also write a copy of the replay with the overlay for the viewer's engineering view (E, camera 3)
uv run plume safety runs/mc/real_hop_high --replay data/replays/real_hop_high.plume.json.gz \
    --embed runs/real_hop_high_safety.plume.json.gz
```

Options: `--out` (default `runs/safety/<name>/`), `--iip-step` (s), `--vacuum-only`,
`--corridor-km` (IIP corridor half-width, default 5), `--impact-buffer-km` (default 5),
`--landing-buffer-m` (default 500), `--origin lat,lon` (fast-fidelity inputs without a
geodetic origin). The planner's reliability jobs run the same analysis on their campaign
([docs/planner.md](../planner.md)).

Output files (`safety.*`):

| file | content |
|---|---|
| `safety.geojson` | RFC 7946 FeatureCollection, WGS-84 lon/lat. `properties.kind`: `launch_site`, `landing_site`, `ground_track` (3-D, height above the ellipsoid), `iip_vacuum`, `iip_drag`, `landing_ellipse_50`, `landing_ellipse_99`, `landing_point`, `impact_point` (with `run`, `reason`, `phase`), `hazard_area` (with `hazard`: `iip_corridor`, `landing_zone`, `failure_impacts`, and `area_km2`). The collection's `properties` carry the phase table, statistics, sources and the disclaimer. |
| `safety.kml` | the same in folders, monochrome styles; the trajectory is drawn at its true altitude |
| `safety.html` | summary: KPIs, map (with Natural Earth coastlines), failure probability by phase, hazard areas, failed-run impact points, method |
| `safety.json` | everything above as plain data |

The reference campaign's export is in [`docs/safety/real_hop_high/`](../safety/real_hop_high/).

## Geodesy

The simulator's world frame is converted to WGS-84 latitude/longitude by `GeoFrame`:

* **High fidelity (`scene.frame = wgs84`):** the world frame is East-North-Up at the launch
  site on the rotating WGS-84 ellipsoid (`EarthGravity`, [earth.md](earth.md)). Conversions are
  exact (ECEF ↔ geodetic).
* **Fast fidelity (`spherical`):** the world is a non-rotating sphere of radius `R_EARTH`; map
  coordinates are the spherical azimuthal-equidistant projection about the launch site
  (`site_projection(..., backend="sphere")`, the same sphere). Fast replays without a geodetic
  origin take it from their mission (`launch.lat/lon`) or `--origin`.

`plume mc` records landing and impact points as local east/north offsets in the tangent plane
at the target. `offset_latlon` inverts that with a fixed-point iteration (find the ground point
whose tangent-plane offset matches), which stays exact for failures hundreds of kilometres away
where a flat-plane conversion would be off by kilometres.

## Instantaneous impact point (IIP)

For each powered-flight sample (thrust > 0, every `--iip-step` s; the terminal landing burn is
excluded because there the IIP is the vehicle itself) the IIP is where the vehicle would hit the
ground if thrust stopped at that instant:

* **Vacuum:** the two-body arc from the state, solved in inertial space and rotated back by the
  Earth's rotation over the time of flight (`kepler_impact`, the same function guidance uses).
  The impact sphere passes through the landing pad. Checked against a numerical vacuum
  integration (within one integration step) and against the Coriolis deflection of a vertical
  shot (about (4/3)·ω·v₀³/g² west).
* **Drag-aware:** the 3-DOF point mass (`ImpactPredictor`) with the vehicle's axial
  aerodynamics flying engine-first with grid fins deployed, the standard (or MSIS) atmosphere,
  calm air, and no further burns. This is the intact-vehicle ballistic fall: a tumbling or
  broken-up vehicle has a different ballistic coefficient (see limitations).

The **IIP corridor** is the drag-aware trace (vacuum if drag is off) widened by
`--corridor-km` on each side, with round end caps. On the reference flight the IIP sweeps from
the pad to the landing site in the 95 s of the ascent burn, then sits on the target during the
entry burn, so the corridor covers the whole ground track (7,800 km² at ±5 km).

## Monte Carlo products

* **Landing points and ellipses:** every intact touchdown (on or off target), and the 50 % and
  99 % bivariate-normal ellipses of `summarize()` ([montecarlo.py](../../src/plume/analysis/montecarlo.py)),
  converted point by point to lat/lon.
* **Impact points:** the final position of every run that lost the vehicle or failed away from
  the pad (`terrain_impact`, `crash_hull`, `crash_legs`, `tipped_over`, `timeout`,
  `no_touchdown`). Simulation errors are excluded and counted separately.
* **Failure probability by phase:** failures per phase over valid runs, with 95 % Wilson
  intervals, split into all failures and vehicle losses. Phases: powered ascent (liftoff to
  MECO), coast/entry/aero descent (MECO to landing-burn ignition), landing burn and touchdown.
  Records do not yet store the failure time, so the phase is assigned by a stated rule: an
  apogee more than max(15 km, 10 %) off the nominal means the ascent went wrong (the ballistic
  arc is fixed at MECO); a loss before the nominal landing-burn time is a descent failure;
  everything else (off-target landings, tip-overs, broken legs) is the landing phase. A record
  with `result.failure_phase` overrides the rule.
* **Hazard areas:** the landing zone (convex hull of all touchdowns, landing-phase losses and the
  99 % ellipse, plus `--landing-buffer-m`), and failure-impact areas (impacts away from the
  landing site, grouped by single-linkage clustering at 50 km, convex hull plus
  `--impact-buffer-km`). Buffering is done in a local azimuthal-equidistant plane about each
  group; areas are reported in km². They are geometric envelopes of the simulated points, **not**
  containment probabilities.

### Reference campaign (real_hop, high fidelity, 64 runs)

| phase | failed | P(failure) (95 % CI) | vehicle lost |
|---|---:|---:|---:|
| powered ascent | 3 | 4.7 % (1.6–12.9 %) | 3 |
| coast, entry and aero descent | 0 | 0.0 % (0.0–5.7 %) | 0 |
| landing burn and touchdown | 5 | 7.8 % (3.4–17.0 %) | 1 (tip-over) |

The three ascent losses are the loss of control at max-q described in
[guidance.md](guidance.md). With no flight termination in the simulator they fly on: one comes
down 14 km from the pad, the other two 760 km and 1,340 km away to the south-west (in the Gulf
of California and in the Pacific off Baja California), opposite to the flight direction. That is the honest output of the model, and it is exactly
why a real vehicle needs a flight termination system and why these impact areas must not be
read as the hazard of a terminated flight.

## Viewer overlay

`--embed` (and every planner reliability job) writes `meta.safety` into a replay: the IIP traces
and hazard rings in world coordinates. The engineering view (E, best with the top camera, 3)
draws them as thin lines (drag-aware IIP dashed amber, vacuum IIP dashed grey, hazard areas red)
with a legend in the engineering panel.

## Limitations

* **No debris model.** The intact vehicle is propagated. A licence-grade analysis needs breakup
  modes, fragment lists with ballistic coefficients and imparted velocities, and their
  dispersions.
* **No flight termination system** in the simulator: failed flights fly until they hit the
  ground, so failure impact points can be far from the route.
* **No population, sheltering or casualty expectation (E<sub>c</sub>)**, no aircraft or ship
  hazard areas, no toxic-release or far-field overpressure analysis.
* **Probabilities are small-sample estimates:** with 64 runs a phase with no failures still has
  an upper 95 % bound of 5.7 %. Rare-event probabilities need far larger campaigns or
  importance sampling.
* **IIP drag model:** engine-first attitude, calm air, nominal vehicle; wind drift of a falling
  vehicle is not included.
* **Failure phase** comes from a heuristic until run records store the failure time and phase.
* **Geometry:** convex hulls over-cover curved or separate clusters within 50 km; polygons near
  the antimeridian or the poles are not split.
* **Fast-fidelity inputs** use a spherical, non-rotating Earth; quote high-fidelity campaigns.
