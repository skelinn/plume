"""Regenerate the SYNTHETIC importer test files in this directory.

These files imitate the layouts of RASAero II, OpenRocket and Missile DATCOM exports
but contain made-up numbers derived from Plume's own generator for the hobby rocket
(with deliberate offsets so the tests can see that imported data override the
generated tables).  They are NOT reference data.

    uv run python tests/data/aero_ref/make_synthetic.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from plume.config import load_vehicle
from plume.physics.aero_gen.generate import generate_database

HERE = Path(__file__).resolve().parent
INCH = 0.0254


def main() -> None:
    v = load_vehicle("hobby_rocket")
    db = generate_database(v)
    L = v.geometry.length
    lre = 6.5

    # ---------------------------------------------------------------- RASAero II
    rows = []
    for m in np.round(np.arange(0.1, 2.01, 0.1), 2):
        for a in (0.0, 2.0, 4.0):
            c = db.evaluate(m, a, log10_re=lre)
            ca_off = 1.10 * c["CA"]  # +10 % so the override is visible
            ca_on = ca_off - c["CA_base"]
            cn = 0.95 * c["CN"]
            zcp = db.centre_of_pressure(m, max(a, 2.0), log10_re=lre)
            cp_in = (L - zcp) / INCH
            ar = math.radians(a)
            cd_off = ca_off * math.cos(ar) + cn * math.sin(ar)
            cd_on = ca_on * math.cos(ar) + cn * math.sin(ar)
            cl = cn * math.cos(ar) - ca_off * math.sin(ar)
            cna = 0.95 * db.evaluate(m, 2.0, log10_re=lre)["CN"] / math.radians(2.0)
            rows.append(
                [
                    m,
                    a,
                    cd_off,
                    cd_off,
                    cd_on,
                    ca_off,
                    ca_on,
                    cl,
                    cn,
                    0.0,
                    0.0,
                    cna,
                    cp_in,
                    cp_in,
                    2.0e6,
                ]
            )
    header = (
        "Mach,Alpha,CD,CD Power-Off,CD Power-On,CA Power-Off,CA Power-On,CL,CN,CN Potential,"
        "CN Viscous,CNalpha (0-4 deg)(per rad),CP,CP (0-4 deg),Reynolds Number"
    )
    with open(HERE / "synthetic_rasaero.csv", "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC test file imitating a RASAero II aero-data export (not real data)\n")
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{x:.6g}" for x in r) + "\n")

    # ---------------------------------------------------------------- OpenRocket
    t = np.linspace(0, 6, 241)
    mach = 1.4 * np.sin(np.pi * t / 12) + 0.05
    alpha = 3.0 + 2.5 * np.sin(2 * np.pi * t / 1.3)  # deg
    with open(HERE / "synthetic_openrocket.csv", "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC test file imitating an OpenRocket simulation export (not real data)\n")
        f.write("# Simulation 1 (synthetic)\n")
        f.write(
            "# Time (s),Altitude (m),Mach number (​),Angle of attack (°),"
            "Drag coefficient (​),Axial drag coefficient (​),"
            "Normal force coefficient (​),CP location (cm),Reference area (cm²)\n"
        )
        f.write("# Event LAUNCH occurred at t=0 seconds\n")
        for ti, m, a in zip(t, mach, alpha, strict=True):
            c = db.evaluate(m, a, log10_re=lre)
            c0 = db.evaluate(m, 0.0, log10_re=lre)
            ca = 1.05 * c0["CA"]
            cn = 1.08 * c["CN"]
            cd = ca * math.cos(math.radians(a)) + cn * math.sin(math.radians(a))
            cp_cm = (L - db.centre_of_pressure(m, a, log10_re=lre)) * 100
            f.write(
                f"{ti:.3f},{300 * ti * ti:.2f},{m:.5f},{a:.4f},{cd:.5f},{ca:.5f},{cn:.5f},"
                f"{cp_cm:.4f},{db.ref_area * 1e4:.4f}\n"
            )
        f.write("# Event BURNOUT occurred at t=6 seconds\n")

    # ---------------------------------------------------------------- Missile DATCOM
    xcg = 0.80  # m from the nose
    d = v.geometry.diameter
    lines = [
        "1 SYNTHETIC test file imitating Missile DATCOM for006 output (not real data)",
        "                 ***** THE USAF AUTOMATED MISSILE DATCOM * REV 3/99 *****",
        f"  REFERENCE QUANTITIES: SREF= {db.ref_area:.6f}  LREF= {d:.4f}  LATREF= {d:.4f}"
        f"  XCG= {xcg:.3f}",
    ]
    alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    for m in (0.6, 1.5):
        lines += [
            "",
            "                 ******* FLIGHT CONDITIONS AND REFERENCE QUANTITIES *******",
            f"  MACH NO = {m:.2f}   REYNOLDS NO = 3.000E+06 /M   ALTITUDE = 0.0 M",
            "",
            "             ---------- STATIC AERODYNAMICS FOR BODY-FIN SET 1 ----------",
            "        ---------- LONGITUDINAL ----------      -- LATERAL DIRECTIONAL --",
            " ALPHA      CN        CM        CA        CY       CLN       CLL",
            "",
        ]
        for a in alphas:
            c = db.evaluate(m, abs(a), log10_re=lre)
            s = 1 if a >= 0 else -1
            cn = s * 1.1 * c["CN"]
            # CM about XCG (z_cg = L - xcg), reference length d
            cm = s * 1.1 * (c["Cm"] + c["CN"] * (db.x_ref - (L - xcg)) / d)
            lines.append(
                f"{a:7.2f}  {cn:8.4f}  {cm:8.4f}  {c['CA']:8.4f}  {0.0:8.4f}  {0.0:8.4f}  {0.0:8.4f}"
            )
        lines += [
            "",
            "             ---------- DYNAMIC DERIVATIVES (PER DEGREE) ----------",
            " ALPHA       CNQ        CMQ        CAQ        CNAD       CMAD",
            "",
        ]
        for a in alphas:
            lines.append(
                f"{a:7.2f}  {0.12:9.5f}  {-15.0:9.5f}  {0.0:9.5f}  {0.02:9.5f}  {-2.0:9.5f}"
            )
        lines += [
            "",
            " ALPHA       CYR        CLNR       CLLR       CYP        CLNP       CLLP",
            "",
        ]
        for a in alphas:
            lines.append(
                f"{a:7.2f}  {0.0:9.5f}  {0.0:9.5f}  {0.0:9.5f}  {0.0:9.5f}  {0.0:9.5f}  {-0.4:9.5f}"
            )
    (HERE / "synthetic_datcom_for006.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- generic CSV
    with open(HERE / "synthetic_generic.csv", "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC generic aero table (not real data)\n")
        f.write(f"# ref_area={db.ref_area:.8g} ref_length={d:.6g} x_ref=0.0\n")
        f.write("mach,alpha_deg,CA,CN,Cm,sigma_CN\n")
        for m in (0.3, 0.8, 1.2, 2.0):
            for a in (0.0, 5.0, 10.0, 20.0):
                c = db.evaluate(m, a, log10_re=lre)
                f.write(
                    f"{m},{a},{0.9 * c['CA']:.6g},{0.9 * c['CN']:.6g},{0.9 * c['Cm']:.6g},"
                    f"{0.05 * c['CN']:.6g}\n"
                )


if __name__ == "__main__":
    main()
