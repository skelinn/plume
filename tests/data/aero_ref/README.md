# Aerodynamic reference data (`tests/data/aero_ref/`)

Two kinds of files live here.

## 1. Digitised public-domain reference data (used for verification)

All values were digitised manually by the Plume contributors from the NTRS scan of

> Jorgensen, Leland H.: *Prediction of Static Aerodynamic Characteristics for
> Space-Shuttle-Like and Other Bodies at Angles of Attack From 0° to 180°.*
> NASA TN D-6996, January 1973 (NTRS 19730006261).

using a calibrated pixel grid over the figure axes (crop + ruler overlay); reading
uncertainty is roughly one symbol radius.  NASA/NACA reports are works of the U.S.
Government and are in the public domain.

| file | content | source in TN D-6996 | original data |
|---|---|---|---|
| `jernell_m286_bodies.csv` | geometry of the 8 bodies (l/d, l_N/d, nose shape, moment centre x_m/d) | Fig. 9 (table) | Jernell, L.S., *Aerodynamic characteristics of bodies of revolution at Mach numbers from 1.50 to 2.86 and angles of attack to 180°*, NASA TM X-1658, 1968 (Langley UPWT) |
| `jernell_m286_cn.csv` | measured CN vs alpha (5-165 deg), M 2.86, Re_d 1.25e5; +-0.3 | Figs. 10, 11, 12 (symbols) | Jernell 1968 |
| `jernell_m286_cm.csv` | measured Cm about x_m vs alpha, cone-cylinders 3-5; +-0.3 | Fig. 11 (symbols) | Jernell 1968 |
| `jernell_m286_ca.csv` | measured CA at alpha 5 and 175 deg; +-0.03 | Figs. 10, 11 | Jernell 1968 |
| `love_base_pressure.csv` | base pressure coefficient vs Mach, turbulent BL, alpha 0; +-0.005 | Fig. 8 ("Analysis of Love") | Love, E.S., NACA TN 3819, 1957 |
| `rossow_wave_drag.csv` | wave-drag parameter (gamma/2) M^2 CA_W vs K = M d/l_N for cones and tangent ogives; +-0.01 | Fig. 6 | Ehret, Rossow & Stevens, NACA TN 2250 (cones, Taylor-Maccoll); Rossow, NACA TN 2399 (ogives, characteristics) |

Notes from TN D-6996 (p. 21): Jernell's data show discontinuities attributed to
support (sting) interference, least at M 2.86, which is why only M 2.86 is used;
the axial force is the total measured value without base-pressure adjustment.
Where two symbols are plotted at one angle (repeat runs) the files give their mean.

The crossflow-drag curves (TN D-6996 Figs. 1, 2, 4) and the stagnation-pressure
curve (Fig. 7) were digitised directly into `src/plume/physics/aero_gen/body.py`
(`CDN_MACH`, `CDN_REYNOLDS`, `ETA_TABLE`) and checked through the tests.

Comparison results are tabulated in `docs/models/aero.md`, section 5.

## 2. SYNTHETIC importer test files (not reference data)

| file | imitates |
|---|---|
| `synthetic_rasaero.csv` | RASAero II "Aero Data" CSV export |
| `synthetic_openrocket.csv` | OpenRocket simulation CSV export |
| `synthetic_datcom_for006.dat` | Missile DATCOM `for006` text output |
| `synthetic_generic.csv` | Plume generic CSV |

These contain made-up numbers produced by `make_synthetic.py` from Plume's own
generator for the hobby rocket with deliberate offsets (e.g. CA x 1.10, CN x 0.95 in
the RASAero file) so the importer tests can check that imported data override the
generated tables.  Each file carries a "SYNTHETIC" header line.  Regenerate with

    uv run python tests/data/aero_ref/make_synthetic.py
