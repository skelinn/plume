"""Importers for external aerodynamic tools, merged onto a generated database.

Supported inputs:

* **RASAero II** "Aero Data" CSV export (columns such as ``Mach``, ``Alpha``,
  ``CA Power-Off``, ``CA Power-On``, ``CD Power-Off``, ``CN``, ``CNalpha (0-4 deg)
  (per rad)``, ``CP`` ... ; CP from the nose tip, inches by default);
* **OpenRocket** simulation CSV export (time series with ``Mach number``,
  ``Angle of attack``, ``Axial drag coefficient`` / ``Drag coefficient``,
  ``Normal force coefficient``, ``CP location``; reduced to CA0(M), CN_alpha(M),
  CP(M) by Mach binning of the low-alpha samples);
* **Missile DATCOM** ``for006.dat`` text output (static tables ``ALPHA CN CM CA
  [CY CLN CLL]`` and dynamic-derivative tables ``ALPHA CNQ CMQ CAQ CNAD CMAD`` /
  ``... CLLP``, per Mach block);
* **generic CSV** ``mach, alpha_deg, CA, CN, Cm[, CY, Cn, Cl, Cmq, Clp, CA_base,
  phi_deg, log10_re, sigma_*]``.

Merge rule: imported values override generated ones where the import has data
(inside its Mach x alpha hull) and are blended linearly into the generated tables
over a margin (``margin_mach``, ``margin_alpha_deg``) outside it, so that tool
data (typically 0-15 deg nose-first) and the generator's 0-180 deg coverage join
without steps.  Imported data without a Reynolds axis are applied at every
Reynolds number.  Provenance is recorded per coefficient in
``db.meta['provenance']``; imported values get a 1-sigma uncertainty of
``rel_sigma`` x |value| (default 10 %) unless sigma columns are given.

The parsers were developed against the documented export layouts and tested on
the small synthetic files in ``tests/data/aero_ref/`` (clearly marked as
synthetic); real exports vary between tool versions -- check the column mapping
reported in ``db.meta['imports']``.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import re
from pathlib import Path

import numpy as np

from plume.physics.aerodb import COEFFICIENTS, AeroDatabase

INCH = 0.0254


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find(cols: list[str], *candidates: str, required: bool = True) -> int | None:
    norm = [_norm(c) for c in cols]
    for cand in candidates:
        c = _norm(cand)
        if c in norm:
            return norm.index(c)
    for cand in candidates:  # prefix match (units appended to the header)
        c = _norm(cand)
        for i, n in enumerate(norm):
            if n.startswith(c):
                return i
    if required:
        raise KeyError(f"none of the columns {candidates} found in {cols}")
    return None


# --------------------------------------------------------------------------- merging
def merge_into(
    base: AeroDatabase,
    source: str,
    mach,
    alpha_deg,
    values: dict[str, np.ndarray],
    *,
    margin_mach: float = 0.1,
    margin_alpha_deg: float = 5.0,
    rel_sigma: float = 0.10,
    sigma: dict[str, np.ndarray] | None = None,
    mode: str = "replace",
) -> AeroDatabase:
    """Return a copy of ``base`` with ``values`` (each shaped (len(mach), len(alpha)),
    on the database's reference quantities) overriding the generated tables.

    ``mode="scale"`` treats ``values`` as multiplicative correction factors instead
    (keeps the base table's Reynolds-number and nonlinear alpha dependence)."""
    mach = np.asarray(mach, dtype=float)
    alpha = np.asarray(alpha_deg, dtype=float)
    order_m = np.argsort(mach)
    order_a = np.argsort(alpha)
    mach, alpha = mach[order_m], alpha[order_a]
    db = AeroDatabase(
        axes={k: v.copy() for k, v in base.axes.items()},
        coeffs={k: v.copy() for k, v in base.coeffs.items()},
        sigma={k: v.copy() for k, v in base.sigma.items()},
        ref_area=base.ref_area,
        ref_length=base.ref_length,
        x_ref=base.x_ref,
        meta=copy.deepcopy(base.meta),
    )
    names = list(db.axes)
    im, ia = names.index("mach"), names.index("alpha_deg")
    grid_m = db.axes["mach"]
    grid_a = db.axes["alpha_deg"]

    def weight(x, lo, hi, margin):
        if margin <= 0:
            return ((x >= lo) & (x <= hi)).astype(float)
        return np.clip(1 - np.maximum(lo - x, x - hi) / margin, 0, 1) * 1.0

    wm = (
        weight(grid_m, mach[0], mach[-1], margin_mach)
        if len(mach) > 1
        else (np.abs(grid_m - mach[0]) <= margin_mach) * 1.0
    )
    wa = weight(grid_a, alpha[0], alpha[-1], margin_alpha_deg)
    w2 = wm[:, None] * wa[None, :]
    mm, aa = np.meshgrid(
        np.clip(grid_m, mach[0], mach[-1]), np.clip(grid_a, alpha[0], alpha[-1]), indexing="ij"
    )
    applied = {}
    for name, tab in values.items():
        if name not in COEFFICIENTS:
            raise KeyError(f"unknown coefficient {name!r}")
        tab = np.asarray(tab, dtype=float)[np.ix_(order_m, order_a)]
        if np.isnan(tab).all():
            continue
        tab = _fill_nan(tab)
        vals = _bilinear(mach, alpha, tab, mm, aa)
        sig = (
            np.zeros_like(vals)
            if mode == "scale"
            else _bilinear(
                mach,
                alpha,
                _fill_nan(np.asarray(sigma[name], float)[np.ix_(order_m, order_a)]),
                mm,
                aa,
            )
            if sigma and name in sigma
            else rel_sigma * np.abs(vals)
        )
        # broadcast the (mach, alpha) override over the remaining axes
        shape = [1] * len(names)
        shape[im], shape[ia] = len(grid_m), len(grid_a)
        perm_vals = vals.reshape(shape)  # AXIS_ORDER puts mach before alpha
        perm_sig = sig.reshape(shape)
        perm_w = w2.reshape(shape)
        if mode == "scale":
            factor = (1 - perm_w) + perm_w * perm_vals
            db.coeffs[name] = db.coeffs[name] * factor
            db.sigma[name] = (1 - perm_w) * db.sigma[name] + perm_w * rel_sigma * np.abs(
                db.coeffs[name]
            )
        else:
            db.coeffs[name] = (1 - perm_w) * db.coeffs[name] + perm_w * perm_vals
            db.sigma[name] = (1 - perm_w) * db.sigma[name] + perm_w * perm_sig
        applied[name] = (
            f"{source} (M {mach[0]:g}-{mach[-1]:g}, alpha {alpha[0]:g}-{alpha[-1]:g} deg), "
            f"blended into '{base.meta.get('provenance', {}).get(name, 'base')}' over "
            f"+-{margin_mach:g} M / +-{margin_alpha_deg:g} deg"
        )
    db.meta.setdefault("provenance", {}).update(applied)
    db.meta.setdefault("imports", []).append({"source": source, "coefficients": sorted(applied)})
    db._interp = None
    db._zcp_interp = None
    return db


def _fill_nan(tab: np.ndarray) -> np.ndarray:
    """Fill NaNs along alpha (then Mach) by linear interpolation / edge values."""
    tab = tab.copy()
    for axis in (1, 0):
        t = np.moveaxis(tab, axis, -1)
        for idx in np.ndindex(t.shape[:-1]):
            row = t[idx]
            ok = np.isfinite(row)
            if ok.any() and not ok.all():
                row[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), row[ok])
    return tab


def _bilinear(x, y, tab, xq, yq):
    out = np.empty_like(xq, dtype=float)
    for i in range(xq.shape[0]):
        for j in range(xq.shape[1]):
            col = (
                np.array([np.interp(yq[i, j], y, tab[k]) for k in range(len(x))])
                if len(y) > 1
                else tab[:, 0]
            )
            out[i, j] = np.interp(xq[i, j], x, col) if len(x) > 1 else col[0]
    return out


def _cm_from_cp(cn, z_cp, x_ref, d):
    return cn * (z_cp - x_ref) / d


def _grid_from_rows(m, a, cols: dict[str, np.ndarray]):
    """Rectilinear grid (unique M x unique alpha) from scattered rows; missing = NaN."""
    mu = np.unique(np.round(m, 6))
    au = np.unique(np.round(a, 6))
    out = {k: np.full((len(mu), len(au)), np.nan) for k in cols}
    im = np.searchsorted(mu, np.round(m, 6))
    ia = np.searchsorted(au, np.round(a, 6))
    for k, v in cols.items():
        out[k][im, ia] = v
    return mu, au, out


# --------------------------------------------------------------------------- RASAero II
def import_rasaero(
    path,
    base: AeroDatabase,
    *,
    body_length: float,
    cp_unit: float = INCH,
    ref_area: float | None = None,
    **merge_kw,
) -> AeroDatabase:
    """RASAero II aerodynamic CSV -> database (``base`` supplies everything missing).

    ``body_length`` (m) converts the CP (measured from the nose tip, ``cp_unit``
    metres per file unit; RASAero exports inches) to Plume body z.  Coefficients
    are rescaled from ``ref_area`` (the file's reference area, default = the
    database's) to the database reference area."""
    text = Path(path).read_text(encoding="utf-8-sig")
    body_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.reader(body_lines))
    header = [h.strip() for h in rows[0]]
    data = np.array(
        [[float(x) if x.strip() else np.nan for x in r] for r in rows[1:] if r and r[0].strip()]
    )
    im = _find(header, "Mach Number", "Mach")
    ia = _find(header, "Alpha", "Angle of Attack")
    ica_off = _find(header, "CA Power-Off", "CA PowerOff", required=False)
    ica_on = _find(header, "CA Power-On", required=False)
    icd_off = _find(header, "CD Power-Off", "CD", required=False)
    icn = _find(header, "CN")
    icp = _find(header, "CP", required=False)
    scale = (ref_area or base.ref_area) / base.ref_area
    m, a = data[:, im], data[:, ia]
    if ica_off is not None:
        ca = data[:, ica_off]
    elif icd_off is not None:  # CA ~ CD cos a - CL sin a; with CN: CA = (CD - CN sin a)/cos a
        ar = np.radians(a)
        ca = (data[:, icd_off] - data[:, icn] * np.sin(ar)) / np.cos(ar)
    else:
        raise KeyError("RASAero file has neither CA Power-Off nor CD Power-Off")
    cols = {"CA": ca * scale, "CN": data[:, icn] * scale}
    if ica_on is not None and ica_off is not None:
        cols["CA_base"] = np.maximum(data[:, ica_off] - data[:, ica_on], 0.0) * scale
    if icp is not None:
        z_cp = body_length - data[:, icp] * cp_unit
        cols["Cm"] = _cm_from_cp(cols["CN"], z_cp, base.x_ref, base.ref_length)
    mu, au, grid = _grid_from_rows(m, a, cols)
    db = merge_into(base, f"RASAero II {Path(path).name}", mu, au, grid, **merge_kw)
    db.meta["imports"][-1]["columns"] = {
        "mach": header[im], "alpha": header[ia], "CN": header[icn],
        "CA": header[ica_off] if ica_off is not None else header[icd_off],
        "CP": header[icp] if icp is not None else None,
    }  # fmt: skip
    return db


# --------------------------------------------------------------------------- OpenRocket
def import_openrocket(
    path,
    base: AeroDatabase,
    *,
    body_length: float,
    mach_bin: float = 0.05,
    max_alpha_deg: float = 10.0,
    **merge_kw,
) -> AeroDatabase:
    """OpenRocket simulation CSV export -> low-alpha correction of CA, CN, Cm.

    Rows are binned in Mach; per bin, CA0 = median axial drag coefficient (or total
    drag coefficient if the axial one is absent) of rows with alpha < 2 deg,
    CN_alpha = median CN/alpha of rows with 0.5 < alpha < ``max_alpha_deg``,
    CP = median CP location.  These become multiplicative corrections of the base
    tables over the observed alpha range (CA by CA0 ratio, CN by CN_alpha ratio, Cm
    so that the low-alpha CP matches), which keeps the base Reynolds and nonlinear
    alpha dependence.
    Units are taken from the header (``cm``/``mm``/``m``, ``°``/``rad``)."""
    header = None
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "mach" in s.lower() and "," in s:
                header = [c.strip() for c in s.lstrip("#").split(",")]
            continue
        rows.append([float(x) if x.strip() not in ("", "NaN") else np.nan for x in s.split(",")])
    if header is None:
        raise ValueError("no OpenRocket header line (with 'Mach number') found")
    data = np.array(rows)
    im = _find(header, "Mach number")
    ia = _find(header, "Angle of attack")
    ica = _find(header, "Axial drag coefficient", required=False)
    icd = _find(header, "Drag coefficient", required=False)
    icn = _find(header, "Normal force coefficient")
    icp = _find(header, "CP location", required=False)
    iar = _find(header, "Reference area", required=False)
    a_unit = math.pi / 180 if ("°" in header[ia] or "deg" in header[ia].lower()) else 1.0
    a = data[:, ia] * a_unit  # rad
    m = data[:, im]
    ca = data[:, ica] if ica is not None else data[:, icd]
    cn = data[:, icn]
    scale = 1.0
    if iar is not None:
        unit = header[iar]
        f = 1e-4 if "cm" in unit else 1e-6 if "mm" in unit else 1.0
        area = np.nanmedian(data[:, iar]) * f
        scale = area / base.ref_area
    cp = None
    if icp is not None:
        unit = header[icp]
        f = 0.01 if "(cm)" in unit else 0.001 if "(mm)" in unit else 1.0
        cp = data[:, icp] * f
    ok = np.isfinite(m) & np.isfinite(a) & np.isfinite(ca) & np.isfinite(cn) & (m > 0.02)
    edges = np.arange(0.0, np.nanmax(m[ok]) + mach_bin, mach_bin)
    mach_c, ca0, cna, zcp = [], [], [], []
    for lo, hi in itertools.pairwise(edges):
        sel = ok & (m >= lo) & (m < hi)
        small = sel & (np.abs(a) < math.radians(2))
        lift = sel & (np.abs(a) > math.radians(0.5)) & (np.abs(a) < math.radians(max_alpha_deg))
        if small.sum() == 0 and lift.sum() == 0:
            continue
        mach_c.append(float(np.median(m[sel])))
        ca0.append(float(np.median(ca[small])) * scale if small.any() else np.nan)
        cna.append(float(np.median(cn[lift] / np.abs(a[lift]))) * scale if lift.any() else np.nan)
        if cp is not None and lift.any():
            zcp.append(body_length - float(np.nanmedian(cp[lift])))
        else:
            zcp.append(np.nan)
    if not mach_c:
        raise ValueError("no usable OpenRocket rows")
    mach_c = np.array(mach_c)
    a_obs = float(np.degrees(np.nanmax(np.abs(a[ok]))))
    alpha_g = np.array([a_ for a_ in base.axes["alpha_deg"] if a_ <= min(max_alpha_deg, a_obs)])
    # correction factors relative to the base table (keeps its Re / nonlinear shape)
    lre = base.meta.get("nominal_log10_re")
    f_ca = np.ones((len(mach_c), len(alpha_g)))
    f_cn = np.ones_like(f_ca)
    f_cm = np.ones_like(f_ca)
    a_ref = 2.0
    for i, mc in enumerate(mach_c):
        g0 = base.evaluate(mc, 0.0, log10_re=lre)
        gref = base.evaluate(mc, a_ref, log10_re=lre)
        cna_gen = gref["CN"] / math.radians(a_ref)
        if np.isfinite(ca0[i]) and g0["CA"] > 0:
            f_ca[i] = ca0[i] / g0["CA"]
        if np.isfinite(cna[i]) and cna_gen > 0:
            f_cn[i] = cna[i] / cna_gen
            if np.isfinite(zcp[i]):
                zgen = base.centre_of_pressure(mc, a_ref, log10_re=lre)
                den = zgen - base.x_ref
                if abs(den) > 1e-9:
                    f_cm[i] = f_cn[i] * (zcp[i] - base.x_ref) / den
    values = {"CA": f_ca, "CN": f_cn}
    if cp is not None:
        values["Cm"] = f_cm
    merge_kw.setdefault("mode", "scale")
    db = merge_into(base, f"OpenRocket {Path(path).name}", mach_c, alpha_g, values, **merge_kw)
    db.meta["imports"][-1]["reduced"] = {
        "mach": mach_c.tolist(), "CA0": ca0, "CN_alpha": cna, "z_cp": zcp,
    }  # fmt: skip
    return db


# --------------------------------------------------------------------------- DATCOM
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def parse_datcom(path) -> dict:
    """Parse Missile DATCOM ``for006`` output into per-Mach tables.

    Returns ``{"mach": [...], "blocks": [{"mach": M, "static": {col: [...]},
    "dynamic": {col: [...]}, "deriv_unit": "deg"|"rad"}], "ref": {...}}``."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[dict] = []
    ref: dict[str, float] = {}
    cur = None
    deriv_unit = "deg"
    i = 0
    while i < len(lines):
        ln = lines[i]
        up = ln.upper()
        for key in ("SREF", "LREF", "XCG", "LATREF"):
            mm = re.search(rf"\b{key}\s*=\s*({_NUM})", up)
            if mm:
                ref[key.lower()] = float(mm.group(1).replace("D", "E"))
        mm = re.search(rf"MACH\s*(?:NO\.?|NUMBER)?\s*=\s*({_NUM})", up)
        if mm:
            mval = float(mm.group(1).replace("D", "E"))
            if cur is None or abs(cur["mach"] - mval) > 1e-9:
                cur = {"mach": mval, "static": {}, "dynamic": {}, "deriv_unit": deriv_unit}
                blocks.append(cur)
        if "PER DEGREE" in up:
            deriv_unit = "deg"
            if cur:
                cur["deriv_unit"] = "deg"
        elif "PER RADIAN" in up:
            deriv_unit = "rad"
            if cur:
                cur["deriv_unit"] = "rad"
        toks = up.split()
        if toks and toks[0] == "ALPHA" and len(toks) > 1 and cur is not None:
            names = toks[1:]
            dynamic = any(
                n in ("CNQ", "CMQ", "CAQ", "CNAD", "CMAD", "CLLP", "CLNP", "CYP") for n in names
            )
            table = cur["dynamic" if dynamic else "static"]
            j = i + 1
            rows = []
            while j < len(lines):
                t = lines[j].split()
                if not t:
                    if rows:
                        break
                    j += 1
                    continue
                if not re.fullmatch(_NUM, t[0]):
                    if rows:
                        break
                    j += 1
                    continue
                vals = []
                for x in t[: len(names) + 1]:
                    try:
                        vals.append(float(x.replace("D", "E")))
                    except ValueError:
                        vals.append(np.nan)
                rows.append(vals + [np.nan] * (len(names) + 1 - len(vals)))
                j += 1
            if rows:
                arr = np.array(rows)
                table.setdefault("ALPHA", list(arr[:, 0]))
                for k, n in enumerate(names):
                    table[n] = list(arr[:, k + 1])
            i = j
            continue
        i += 1
    return {"mach": [b["mach"] for b in blocks], "blocks": blocks, "ref": ref}


def import_datcom(
    path,
    base: AeroDatabase,
    *,
    body_length: float,
    xcg_from_nose: float | None = None,
    sref: float | None = None,
    lref: float | None = None,
    **merge_kw,
) -> AeroDatabase:
    """Missile DATCOM ``for006`` -> database (static CN, CM, CA [, CY, CLN, CLL] and
    dynamic CMQ + CMAD, CLLP; derivatives converted to per rad).

    DATCOM moments are about XCG (from the nose) with reference length LREF and
    area SREF; they are transferred to the database reference point / length /
    area.  Total pitch damping about XCG is converted to the database convention
    (damping in excess of the CP lever-arm term): Cmq_extra = Cmq + 2 CN_a ((z_cp -
    z_cg)/d)^2, clipped to <= 0."""
    parsed = parse_datcom(path)
    ref = parsed["ref"]
    sref = sref or ref.get("sref") or base.ref_area
    lref = lref or ref.get("lref") or base.ref_length
    xcg = xcg_from_nose if xcg_from_nose is not None else ref.get("xcg")
    if xcg is None:
        raise ValueError("DATCOM moment centre unknown: pass xcg_from_nose")
    z_cg = body_length - xcg
    d = base.ref_length
    sa = sref / base.ref_area
    blocks = [b for b in parsed["blocks"] if "CN" in b["static"]]
    if not blocks:
        raise ValueError("no static DATCOM tables found")
    alpha_all = sorted({round(a, 6) for b in blocks for a in b["static"]["ALPHA"] if a >= 0})
    mach = np.array([b["mach"] for b in blocks])
    shape = (len(mach), len(alpha_all))
    out = {k: np.full(shape, np.nan) for k in ("CN", "Cm", "CA", "CY", "Cn", "Cl", "Cmq", "Clp")}
    for i, b in enumerate(blocks):
        st = b["static"]
        al = np.array(st["ALPHA"])
        sel = al >= 0
        idx = [alpha_all.index(round(a, 6)) for a in al[sel]]
        cn = np.array(st["CN"])[sel] * sa
        out["CN"][i, idx] = cn
        if "CM" in st:
            cm_ref = np.array(st["CM"])[sel] * sa * lref / d  # about XCG, on d
            out["Cm"][i, idx] = cm_ref + cn * (z_cg - base.x_ref) / d
        if "CA" in st:
            out["CA"][i, idx] = np.array(st["CA"])[sel] * sa
        for src, dst in (("CY", "CY"), ("CLN", "Cn"), ("CLL", "Cl")):
            if src in st:
                out[dst][i, idx] = np.array(st[src])[sel] * sa * (1.0 if dst == "CY" else lref / d)
        dyn = b["dynamic"]
        if "CMQ" in dyn and "ALPHA" in dyn:
            per = 180 / math.pi if b["deriv_unit"] == "deg" else 1.0
            ad = np.array(dyn["ALPHA"])
            dsel = ad >= 0
            didx = [alpha_all.index(round(a, 6)) for a in ad[dsel] if round(a, 6) in alpha_all]
            cmq = np.array(dyn["CMQ"])[dsel] * per
            if "CMAD" in dyn:
                cmq = cmq + np.nan_to_num(np.array(dyn["CMAD"])[dsel] * per)
            cmq = cmq * sa * (lref / d) ** 2
            # lever-arm term at the static CP for this Mach
            a_r = np.radians(np.array(alpha_all))
            cn_row = out["CN"][i]
            small = (a_r > 0) & (a_r < math.radians(6)) & np.isfinite(cn_row)
            cna = float(np.median(cn_row[small] / a_r[small])) if small.any() else 0.0
            cm_row = out["Cm"][i]
            zc = np.where(
                np.abs(cn_row) > 1e-9,
                base.x_ref + d * cm_row / np.where(np.abs(cn_row) > 1e-9, cn_row, 1),
                np.nan,
            )
            zc = _fill_nan(zc[None, :])[0]
            extra = cmq[: len(didx)] + 2 * cna * ((zc[didx] - z_cg) / d) ** 2
            out["Cmq"][i, didx] = np.minimum(extra, 0.0)
        if "CLLP" in dyn and "ALPHA" in dyn:
            per = 180 / math.pi if b["deriv_unit"] == "deg" else 1.0
            ad = np.array(dyn["ALPHA"])
            dsel = ad >= 0
            didx = [alpha_all.index(round(a, 6)) for a in ad[dsel] if round(a, 6) in alpha_all]
            out["Clp"][i, didx] = np.minimum(
                np.array(dyn["CLLP"])[dsel][: len(didx)] * per * sa * (lref / d) ** 2, 0.0
            )
    values = {k: v for k, v in out.items() if np.isfinite(v).any()}
    db = merge_into(
        base, f"Missile DATCOM {Path(path).name}", mach, np.array(alpha_all), values, **merge_kw
    )
    db.meta["imports"][-1]["reference"] = {"sref": sref, "lref": lref, "xcg_from_nose": xcg}
    return db


# --------------------------------------------------------------------------- generic CSV
def import_generic_csv(
    path,
    base: AeroDatabase | None = None,
    *,
    ref_area: float | None = None,
    ref_length: float | None = None,
    x_ref: float | None = None,
    body_length: float | None = None,
    **merge_kw,
) -> AeroDatabase:
    """Generic long-format CSV (``mach, alpha_deg, CA, CN, Cm[, ...]``).

    Comment lines ``# key=value`` may give ``ref_area``, ``ref_length``, ``x_ref``.
    With ``base`` the data are merged onto it; without, the file must hold a full
    rectilinear grid and becomes the database on its own (missing coefficients 0)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    meta_kv: dict[str, float] = {}
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("#"):
            for k, v in re.findall(r"(\w+)\s*=\s*([-+0-9.eE]+)", ln):
                meta_kv[k] = float(v)
        elif ln.strip():
            lines.append(ln)
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    data = np.array([[float(x) for x in r] for r in reader])
    col = {_norm(h): i for i, h in enumerate(header)}
    ra = ref_area or meta_kv.get("ref_area")
    rl = ref_length or meta_kv.get("ref_length")
    xr = x_ref if x_ref is not None else meta_kv.get("x_ref")
    axes_present = [a for a in ("mach", "alphadeg", "phideg", "log10re") if a in col]
    if "mach" not in col or "alphadeg" not in col:
        raise KeyError("generic CSV needs 'mach' and 'alpha_deg' columns")
    exact = {h: i for i, h in enumerate(header)}  # CN vs Cn: coefficient names are case-sensitive
    coef_cols = {c: exact[c] for c in COEFFICIENTS if c in exact}
    sig_cols = {c: exact["sigma_" + c] for c in COEFFICIENTS if "sigma_" + c in exact}
    if base is not None:
        if any(a in axes_present for a in ("phideg", "log10re")):
            raise ValueError("merging needs a (mach, alpha) table; load roll/Re tables standalone")
        scale_a = (ra or base.ref_area) / base.ref_area
        scale_l = (rl or base.ref_length) / base.ref_length
        m, a = data[:, col["mach"]], data[:, col["alphadeg"]]
        vals = {k: data[:, i] * scale_a for k, i in coef_cols.items()}
        for k in ("Cm", "Cn", "Cl"):
            if k in vals:
                vals[k] = vals[k] * scale_l
        if "Cm" in vals and xr is not None and abs(xr - base.x_ref) > 0:
            vals["Cm"] = vals["Cm"] + vals["CN"] * (xr - base.x_ref) / base.ref_length
        sig = {k: data[:, i] * scale_a for k, i in sig_cols.items()}
        mu, au, grid = _grid_from_rows(m, a, vals)
        sgrid = _grid_from_rows(m, a, sig)[2] if sig else None
        return merge_into(base, f"CSV {Path(path).name}", mu, au, grid, sigma=sgrid, **merge_kw)
    if ra is None or rl is None:
        raise ValueError("standalone generic CSV needs ref_area and ref_length")
    axis_name = {
        "mach": "mach",
        "alphadeg": "alpha_deg",
        "phideg": "phi_deg",
        "log10re": "log10_re",
    }
    axes = {axis_name[a]: np.unique(data[:, col[a]]) for a in axes_present}
    shape = tuple(len(v) for v in axes.values())
    if np.prod(shape) != len(data):
        raise ValueError(f"generic CSV is not a full grid ({len(data)} rows for shape {shape})")
    idx = tuple(np.searchsorted(axes[axis_name[a]], data[:, col[a]]) for a in axes_present)
    coeffs, sigma = {}, {}
    for k, i in coef_cols.items():
        arr = np.zeros(shape)
        arr[idx] = data[:, i]
        coeffs[k] = arr
    for k, i in sig_cols.items():
        arr = np.zeros(shape)
        arr[idx] = data[:, i]
        sigma[k] = arr
    meta = {
        "name": Path(path).stem,
        "provenance": {k: f"CSV {Path(path).name}" for k in coeffs},
    }
    if body_length:
        meta["body_length"] = body_length
    return AeroDatabase(axes, coeffs, sigma, float(ra), float(rl), float(xr or 0.0), meta)
