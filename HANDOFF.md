# Handoff: continuing Plume in a new session

If the current Claude session runs out of usage, start a new session (for example Claude Code on the web, with the repository `skelinn/plume`) and paste the prompt below. The **Status** section at the end is kept up to date as work lands, so the new session knows exactly where things stand.

---

## Prompt to paste

```
You are continuing work on Plume (https://github.com/skelinn/plume), an open-source 6-DOF rocket
simulator (MuJoCo, Python 3.12, uv) that is the engineering backbone of a reusable cargo-rocket
startup. The owner is not a software engineer; explain results plainly and honestly.

Start by reading, in this order:
1. HANDOFF.md, especially the Status section: what is done, what is in progress, which branches exist.
2. README.md and docs/models/README.md, which cover the models, fidelity levels and verification.
3. docs/models/guidance.md: the full Monte Carlo history of what worked and what failed. Do not repeat
   rejected approaches without addressing why they failed.

Setup: `uv sync --all-extras`, then `uv run pytest -q -n 4 -m "not slow and not rl and not network"`.
That suite should pass; confirm it before changing anything.

The roadmap (11 items; the Status section says which are done):
 1. Design trade study. Monte Carlo on rocket variants: 9 deg gimbal, +5 % tanks, entry speed
    1300 m/s, combined. Output docs/design/trade_study.md with reliability gain and cost per change.
 2. Convex-optimisation (G-FOLD-style) landing-burn guidance with drag handled by successive
    convexification and hoverslam fallback. Keep it only if Monte Carlo shows it is better.
 3. Ascent load relief with re-targeting after max-q. The previous attempt without re-targeting was
    rejected at 39 % vs 87.5 %.
 4. Test-flight programme: plan, instrumentation and data-logging guide, calibration workflow for a
    first real flight (hobby rocket and hop rig).
 5. Hop-test-rig vehicle model: small VTVL, tethered hops, scenario and replays.
 6. Multi-stage rockets and orbit: staging, upper-stage guidance to a target orbit, booster RTLS,
    multi-vehicle replays and viewer support.
 7. Landing on a moving ship: deck motion from sea state, guidance to a moving pad, viewer drone ship.
 8. Crushable landing-gear stroke and soft-soil (bearing strength, sinkage) models.
 9. Browser mission planner: pick two points on a map, choose rocket and cargo, get feasibility, fuel,
    time and success probability (runs against `plume viz`; static demo on Pages).
10. AI pilot (PPO) training on the landing task, then publish the PID vs PPO results table.
11. Flight-safety export: instantaneous-impact-point trace, landing and impact dispersion ellipses
    as GeoJSON/KML, failure probability by phase.

Rules:
- Honesty first. Report Monte Carlo results as measured, keep anything that does not improve results
  off by default, and never claim validation without real flight data (every model is
  "verified, validation pending").
- Monte Carlo is CPU-heavy, about 5 min per high-fidelity flight. Use paired draws:
  `uv run plume mc real_hop --runs 64` (same seed), and compare run by run against the baseline
  (87.5 % success, 56/64, CEP50 5 m).
- Keep fast fidelity (RL training) behaviour unchanged unless clearly beneficial.
- For every change, add tests and docs (docs/models/*.md), run `uv run ruff check src tests` and
  `uv run ruff format src tests`, and run the fast test suite.
- Commit messages end with: Co-Authored-By: Claude <noreply@anthropic.com>
- Work on a branch and open a PR (or push to main only if the owner asks). Pushing to main redeploys
  the public viewer at https://skelinn.github.io/plume/ via the Pages workflow.
- Update the Status section of HANDOFF.md whenever an item lands, so the next session can continue.

Pick the highest-priority unfinished item from the Status section and continue it.
```

---

## Status

_Last updated: 2026-10-08 (evening)_

**Done and live** (main, https://skelinn.github.io/plume/):
- High-fidelity physics: WGS-84 rotating Earth, J2–J6 gravity, US76/NRLMSISE-00/soundings, MIL-spec turbulence, aero database, actuator dynamics, slosh, IMU/GNSS/baro/radar with an EKF.
- Verification: NASA check cases 10/10, 139 automated checks, `plume vv-report`.
- Monte Carlo (`plume mc`): real route at high fidelity reaches 87.5 % mission success and 93.8 % vehicle recovery, CEP50 5 m.
- Viewer: monochrome UI, realistic models, engineering view, Plume Rocketry logo; static export on GitHub Pages.

**Roadmap items**

| # | Item | Status | Branch / notes |
|---|---|---|---|
| 1 | Design trade study | running: 4 variant campaigns (configs/dispersions/real_hop_{gimbal9,tanks105,entry1300,combined}.yaml); report via `scripts/trade_study.py` -> docs/design/trade_study.md | main. If interrupted: rerun `uv run plume mc real_hop_<variant> --workers 6` (resumable) then the script |
| 2 | Convex landing guidance | in progress | agent branch (pushed to origin as it lands; see `git branch -r`) |
| 3 | Ascent load relief + re-targeting | starting | same agent as 2 |
| 4 | Test-flight programme | starting | worktree agent |
| 5 | Hop-test-rig model | starting | same agent as 4 |
| 6 | Multi-stage + orbit | starting | worktree agent |
| 7 | Ship landing | starting | worktree agent |
| 8 | Landing gear + soil | starting | same agent as 7 |
| 9 | Mission planner | done (branch, PR pending) | `worktree-agent-a73ae095d4518575c`: `/planner` page (map, sites, 3-DOF feasibility, max range and cargo advice, 6-DOF flight and reliability jobs), static demo `planner.html`; docs/planner.md |
| 10 | AI pilot training | queued until the Monte Carlo campaigns finish (CPU) | checkpoint at 5.0M / 50M steps in runs/ppo_landing (local PC only, not in git); `uv run plume train --when-idle` |
| 11 | Safety-analysis export | done (branch, PR pending) | same branch: `plume safety` (IIP vacuum + drag, ellipses, impacts, failure by phase, hazard areas; GeoJSON/KML/HTML), viewer overlay; docs/models/flight_safety.md. Next: store failure time/phase in MC records, debris + FTS model |
