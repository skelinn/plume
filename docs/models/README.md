# Model documentation

Each page documents one physical model: the equations, parameters and defaults, how it was verified, and its known limitations. Every page starts with the same three lines: **Code:** (where it lives), **Fidelity:** or **Selection:** (when it is used), and **Verification status:**. The V&V report reads those lines.

| Page | Model | Verification status |
|---|---|---|
| [earth.md](earth.md) | WGS-84 geodesy, frames, J2–J6 gravity, rotating Earth, RK4 integration, NASA check cases | verified (NASA check cases 1–10, analytic and convergence tests) |
| [atmosphere.md](atmosphere.md) | US Standard Atmosphere 1976, NRLMSISE-00, soundings, mean wind, MIL-spec Dryden / von Kármán turbulence | verified against published tables and reference outputs |
| [aero.md](aero.md) | high-fidelity aerodynamic database: generator (body, nose, friction, base, fins, grid fins, retro-propulsion, heating), importers, runtime model | verified against NASA wind-tunnel data at M 2.86 and exact theory; subsonic/transonic and finned-body data not yet compared |
| [propulsion.md](propulsion.md) | liquid engine (throttle, Isp with nozzle back-pressure, ignitions, depletion), gimbal and high-fidelity actuator, RCS with PWM, solid motors | verified (Isp, limits, Tsiolkovsky, depletion, actuator dynamics) |
| [mass.md](mass.md) | dry, cargo, tank and RCS gas mass properties, CG and inertia vs propellant, dry/wet MuJoCo body split, mid-step update | verified (brute-force CG, mass bookkeeping, rocket equation) |
| [navigation.md](navigation.md) | IMU, GNSS, barometric and radar altimeters; 15-state error-state EKF | verified (noise statistics, latency, filter accuracy) |
| [guidance.md](guidance.md) | the vehicle's flight software for the cargo hop: ascent plan, impact prediction, landing autopilot, attitude control (not a physics model) | assessed by Monte Carlo, not by unit verification; the page states it is not yet dependable |
| [terrain.md](terrain.md) | Copernicus DEM terrain: download, projection, resampling, synthetic detail, landing hazards | verified (projection, resampling against analytic surfaces, hazard maps) |

**Validation is pending for every model.** Verification shows that the code implements the stated equations correctly. Validation, meaning agreement with real flight or test data, needs data that does not exist yet. The Monte Carlo reports in [`docs/mc/`](../mc/) quantify dependability under the stated dispersions. They inherit the model-form error of these unvalidated models.

## V&V report

`plume vv-report` collects all of this into one page, [`docs/vv/report.html`](../vv/report.html):
- runs `pytest -m vv` and the model test modules, and groups the results by model area
- runs the ten NASA 6-DOF check cases and tabulates the error against tolerance for each variable
- lists every page above with its **Verification status** line and its test count
- summarises the Monte Carlo reports

```sh
uv run plume vv-report                 # run everything (a few minutes) and write docs/vv/report.html
uv run plume vv-report --no-nasa       # skip the NASA check-case table (~1-2 min)
uv run plume vv-report --no-run        # re-render from the last runs/vv/junit.xml
uv run plume vv-report --junit ci.xml --no-run --out report.html
```

The command exits with status 1 if any test failed. The raw JUnit results are kept in `runs/vv/junit.xml`.

## Adding a model page

Copy the header of an existing page:

```markdown
# <Model name>

**Code:** `src/plume/physics/<module>.py`
**Fidelity:** <which fidelity levels use it>
**Verification status:** <what is verified, by which tests>. Validation against <data> is pending.
```

Then add the page's file stem and its test areas to `MODEL_AREAS` in `src/plume/analysis/vv.py`, so the report counts its tests.
