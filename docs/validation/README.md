# Validation records

This folder holds the evidence that a Plume model matches reality. It is empty until the first real flight. The workflow is in [../test_flight_programme.md](../test_flight_programme.md):

```bash
uv run plume predict hobby_rocket --motor H180 --wind 4 --flight-id flight_1-1   # BEFORE the flight; commit docs/predictions/
uv run plume validate flight_1-1.csv --mapping my_logger --prediction docs/predictions/flight_1-1.json \
    --flight-id flight_1-1 --flight-date 2026-11-14T15:42Z
uv run plume vv-report
```

Each flight writes:

| file | content |
|---|---|
| `<flight-id>.md` | human-readable record: verdicts, prediction against flight, criteria, calibration |
| `<flight-id>/record.json` | the same, machine-readable (read by `plume vv-report`) |
| `<flight-id>/overlay.png` | real against nominal and calibrated simulation |
| `<flight-id>/calibrated_vehicle.yaml` | the vehicle fitted to this flight (use it to predict the next one) |
| `<flight-id>/prediction.json` | copy of the prediction the flight was compared with |

## `record.json` (format `plume-validation`, version 1)

| key | meaning |
|---|---|
| `flight_id`, `flight_date`, `created_utc`, `plume_version`, `notes` | identification |
| `synthetic` | true for simulated logs: a rehearsal that never counts as validation |
| `log` | file name, SHA-256 of the raw CSV, mapping used, sample count |
| `prediction` | prediction id, its UTC timestamp and git commit, the vehicle hash, and `blind` (true when the prediction predates `flight_date`; null if no date was given) |
| `envelope` | what the evidence covers: vehicle, motor, Mach range, flight type |
| `observed` | measured apogee, timings, peak speed and acceleration, descent rate, landing point |
| `comparison` | for each quantity: observed value, predicted nominal, 95 % band, percentile in the Monte Carlo distribution, inside or not |
| `calibration` | fitted factors (drag, impulse, burn time, time offset, parachute Cd·A) with 1-sigma, and the fit metrics before and after |
| `checks` | each criterion: model, quantity, value, criterion, pass/fail/no data |
| `verdicts` | per model (`drag` → docs/models/aero.md, `motor` → propulsion.md, `parachute`): `validated`, `consistent`, `discrepancy` or `inconclusive` |

A model is **validated** only when every criterion passes, on real data, with a prediction recorded before the flight. The validation applies only to the stated envelope. `plume vv-report` shows it on the model's row with a link to the record, and keeps "pending" for everything else.

Records are evidence: do not edit them by hand. If a log was processed wrongly (a wrong unit in the mapping, say), fix the mapping, write the record again, and explain the correction in `--notes`. Git keeps the history.
