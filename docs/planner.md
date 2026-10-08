# Mission planner

Pick a launch site and a landing site on a world map, choose a vehicle and a cargo mass, and the
planner tells you within seconds whether the hop is feasible: range, the vehicle's maximum range
for that cargo, propellant margin, apogee, flight time and peak cargo load. With `plume viz`
running it can also fly the full 6-DOF mission and estimate reliability with a small Monte Carlo
campaign, followed by a flight-safety export.

```bash
uv run plume viz            # then open http://localhost:8765/planner (or "Planner" in the viewer)
```

Static demo (no server): [skelinn.github.io/plume/planner.html](https://skelinn.github.io/plume/planner.html).
It interpolates a precomputed table and cannot run simulations.

**Code:** `src/plume/missions/planner.py` (3-DOF feasibility), `src/plume/missions/route.py`
(missions between arbitrary sites), `src/plume/viz/planner_api.py` (endpoints and jobs),
`src/plume/viz/static/planner.html`, `js/planner.js`, `css/planner.css` (page),
`scripts/build_planner_table.py`, `scripts/build_planner_map.py` (bundled data)
**Verification status:** the quick planner reproduces the 6-DOF high-fidelity reference flight
(real_hop, 761 km, 250 kg): 126 kg vs 132 kg propellant at touchdown, 188 vs 186 km apogee, 587
vs 589 s, 6.17 vs 6.06 g (`tests/test_planner.py`). Like every Plume model it is not validated
against real flights.

## Using the page

* **Sites:** click the map (the *Map click sets* switch chooses launch or landing), search the
  bundled list of spaceports, launch sites and cargo airports, or type latitude/longitude.
  The URL keeps the route (`?from=lat,lon&to=lat,lon&vehicle=...&cargo=...`) so a plan can be
  shared.
* **Map:** drag to pan, wheel or +/− to zoom, *Fit* frames the route. The great-circle route is
  solid when feasible and dashed when not; the dashed ring is the maximum range from the launch
  site for the current cargo.
* **Feasibility:** *FEASIBLE* / *NOT FEASIBLE* with the reason. Beyond the vehicle's capability
  the page states the maximum range for this cargo and the largest cargo that would make the
  route feasible, or that no cargo would.
* **Range vs cargo:** the vehicle's capability curve, with the route distance and the current
  cargo marked.
* **Simulation** (server only):
  * *Run flight*: the 6-DOF mission at fast (about 2–3 min) or high fidelity (5–8 min), saved as
    a replay; *Open replay* shows it in the viewer.
  * *Estimate*: a screening Monte Carlo of 16 or 32 runs (fast: about 10–15 min; high: about
    25–45 min on 4 cores) with the dispersions of `configs/dispersions/demo_hop.yaml` (fast) or
    `real_hop.yaml` (high). It reports success probability with a 95 % interval, vehicle
    recovery, CEP and failure modes, writes the Monte Carlo report and a flight-safety export
    (GeoJSON, KML, HTML; [models/flight_safety.md](models/flight_safety.md)), and embeds the IIP
    trace and hazard areas in the replay for the engineering view. Treat 16–32 runs as a
    screening estimate: the intervals are wide, and fast fidelity under-predicts this vehicle
    (12.5 % vs 87.5 % at high fidelity, see the README).
  * Jobs run one at a time in the background; a flight uses one worker process and a reliability
    estimate at most four.

## How feasibility is computed

`plume.missions.planner` flies the mission with the 3-DOF point mass (`PointMassSim`) and the
flight software's own planning logic:

1. **Ascent family.** For rise times of 4 and 7 s and pitch kicks from 6° to 26° in 2° steps,
   the ascent is flown to propellant depletion: vertical rise, kick, hold, gravity turn,
   cargo-g-limited throttle (as `plan_ascent`). Each sample is a candidate engine cutoff.
2. **Kick choice.** For the route distance, the cutoff where each arc's vacuum impact point
   reaches the target is found, and the kick with the most propellant left (after an analytic
   entry-burn estimate and the landing reserve) under the flight-path-angle cap is chosen.
3. **MECO.** A secant search moves the cutoff until the **drag-aware impact prediction with the
   planned entry burn** (the `ImpactPredictor` flight plan: coast, retrograde g-limited entry
   burn to `entry_speed` below `entry_altitude`, aero descent with grid fins) lands on the
   target.
4. **Margins.** The landing burn is costed as the flight software's landing reserve,
   Δv = 1.5 × terminal speed + 150 m/s at sea-level Isp (the 6-DOF reference flight used 195 kg of
   its 197 kg reserve). Propellant margin = propellant after the entry burn − landing burn.
   Flight time adds 12 s for the landing burn to the ballistic fall.
5. **Feasible** means: margin ≥ 0, the entry burn reaches `entry_speed` with the engine, and the
   peak cargo load is within 5 % of the guidance limit (6 g). The 5 % allowance exists because
   the reference 6-DOF flight peaks at 6.06 g in the unpowered descent, where no throttle logic
   can limit it ([models/guidance.md](models/guidance.md)); beyond that the route is reported as
   load-limited.
6. **Maximum range** for a cargo: the latest cutoff on any arc whose simulated descent still
   meets the conditions above, found by bracketing and bisection. **Cargo for a route**: inverse
   of the max-range-vs-cargo curve (monotone envelope).

Guidance settings (entry speed 1400 m/s, entry altitude 60 km, 6 g, 38° arc cap) come from
`configs/missions/real_hop.yaml`. A first query for a vehicle and cargo takes about 5–8 s, later
ones about 1 s (cached).

What limits the cargo hopper: up to about 200 kg of cargo the range is **load-limited** at about
810–830 km (a faster re-entry exceeds the cargo g-limit, even with propellant left); above that
it is **propellant-limited** (803 km at 250 kg, 724 km at 450 kg).

### Static demo table

`src/plume/viz/static/planner/capability.json` holds, per hop-capable preset, the max-range curve
and the feasibility over a grid (ranges 50 km to beyond the max range in 50 km steps, cargo
0–450 kg in 50 kg steps), tagged with a hash of the vehicle file. The static page interpolates it
bilinearly; `plume viz` uses its curve when the hash matches and recomputes otherwise.
Regenerate after changing a vehicle (a few minutes, 4 processes):

```bash
uv run python scripts/build_planner_table.py --workers 4
```

## Full flights between arbitrary sites

`plume.missions.route.build_route_mission` writes a mission from the `real_hop` template
(guidance, 5 m/s forecast wind, scoring) with the chosen sites, vehicle and cargo, and **flat
terrain at sea level**: the planner has no elevation data for arbitrary sites. For a real route,
fetch Copernicus terrain (`plume terrain fetch`, [models/terrain.md](models/terrain.md)) and
write a mission file like `configs/missions/real_hop.yaml`. Fast-fidelity flights use the
spherical azimuthal-equidistant map about the launch site; high fidelity flies the rotating
WGS-84 Earth from lat/lon. Jobs write to `runs/planner/` (missions, terrain, replays), which the
viewer serves.

## Endpoints

| | |
|---|---|
| `GET /planner` | the page |
| `GET /api/planner/vehicles` | hop-capable presets (liquid engine, `cargo.max_mass` set) |
| `GET /api/planner/curve?vehicle=` | max range vs cargo; `source`: bundled table or computed |
| `POST /api/planner/feasibility` | `{launch: {lat, lon}, landing: {lat, lon}, vehicle, cargo_kg}` → feasibility, `result` (margins, apogee, times, loads, MECO), `max_range_km`, `cargo_for_range_kg` when infeasible, `geodesic_km`, `azimuth_deg`, `notes` |
| `POST /api/planner/jobs` | `{..., kind: flight\|reliability, fidelity: fast\|high, runs: 4–64}` |
| `GET /api/planner/jobs[/{id}]` | state, progress, message, result (`flight.replay_id`, `reliability`), files |
| `GET /api/planner/jobs/{id}/files/{name}` | `report.html`, `safety.html`, `safety.geojson`, `safety.kml`, `safety.json` |

## Bundled data

* **World map:** Natural Earth 1:50m coastlines, country borders and first-order (state)
  boundaries, public domain ([naturalearthdata.com](https://www.naturalearthdata.com/)), rounded
  to 0.01° (`static/planner/world.json`, 1.2 MB; `scripts/build_planner_map.py` rebuilds it from
  the GeoJSON files of github.com/nvkelso/natural-earth-vector). No tile services or API keys.
* **Sites:** `static/planner/sites.json`, 35 spaceports, launch sites and cargo airports with
  approximate coordinates (about 100 m) and their source (Wikipedia, FAA airport data). For
  orientation only: a real mission needs surveyed pad and landing-zone coordinates.

## Limitations

* Screening model: 3-DOF point mass, zero angle of attack, calm air, nominal vehicle, one flight
  plan (fixed entry speed). Wind, dispersions and control are only in the 6-DOF flight and the
  Monte Carlo campaign.
* Non-rotating Earth: the quick planner ignores Earth rotation, which changes the achievable
  range by a few per cent with direction. High-fidelity flights include it.
* The kick-angle grid (2°) leaves a wiggle of about ±20 km in the max-range curve.
* Full flights between arbitrary sites use flat terrain at sea level; terrain height, slope and
  obstacles at real sites are not represented.
* Nothing is validated against real flight data (see the README).
