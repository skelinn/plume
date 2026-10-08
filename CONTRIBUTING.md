# Contributing to Plume

Thanks for helping! Plume aims to be a small, readable, well-tested simulator.

## Setup

```bash
uv sync --all-extras
uv run pytest -q -n auto            # full suite (~3 min), add -m "not slow" for ~1.5 min
uv run ruff check src tests scripts && uv run ruff format src tests scripts
```

## Ground rules

- **SI units everywhere.** Anything else is spelled out in the name (`gimbal_max_deg`).
- **Physics changes need a test.** The conservation tests in
  `tests/test_physics_conservation.py` (energy, momentum, mass, Tsiolkovsky) must keep
  passing; add a new check when you add a force model.
- **Configs are validated.** New vehicle/world/mission fields go into the pydantic models in
  `src/plume/config.py` with a description and a sensible default.
- **Replays are a contract** with the viewer (`src/plume/recording/schema.json`). Extend it
  compatibly and update the schema test.
- Keep the viewer build-free (vanilla ES modules, vendored three.js).

## Useful commands

| command | what it does |
|---|---|
| `plume sim lander_small --script hop_test` | scripted 6-DOF flight -> replay |
| `plume land --controller pid --stage full_descent` | one landing episode |
| `plume train --when-idle` / `plume train --status` | idle-aware PPO training |
| `plume hop demo_hop` | 750 km cargo hop |
| `plume calibrate data/flights/sample_flight.csv` | fit drag + thrust to a log |
| `plume bench` | PID vs PPO table into the README |
| `plume viz` | the 3-D viewer on http://localhost:8765 |
