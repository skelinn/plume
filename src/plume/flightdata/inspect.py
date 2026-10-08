"""Inspect an unknown flight-computer CSV and draft a column mapping for it.

``plume flightlog inspect <csv>`` reads the file (delimiter, comment lines and header
row are detected), prints every column with its value range, guesses which columns
hold time, altitude, axial acceleration, gyro rates and GPS - and in which units - from
the column names *and* the values, and writes a draft mapping YAML.

The guesses are heuristics. Every unit in the draft is marked with how it was decided
(name hint or value range) and the file starts with a "confirm before use" banner:
check each line against the flight computer's manual. Rules used:

* time: increasing column; unit from the name ("ms", "us", "s") or the sample step;
* altitude: name contains alt/height/agl; feet if the name says so, else metres
  (ambiguous from values alone - flagged);
* acceleration: the accel column with the largest range is the axial one; the pad
  reading tells whether gravity is included (about 1 g or 9.8 m/s^2 at rest) and the
  unit (g, m/s^2, ft/s^2);
* gyro: three rate columns; deg/s unless the name says rad;
* GPS: latitude/longitude by name and range.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plume.constants import G0


@dataclass
class ColumnInfo:
    name: str
    numeric_fraction: float
    n_valid: int
    vmin: float = math.nan
    vmax: float = math.nan
    median: float = math.nan
    first: float = math.nan  # median of the first 2 % of valid samples (the pad)
    step: float = math.nan  # median positive increment
    monotonic: bool = False
    role: str = ""
    unit: str = ""
    why: str = ""
    includes_gravity: bool = True  # acceleration only


@dataclass
class Inspection:
    path: Path
    delimiter: str
    comment: str
    skip_rows: int
    rows: int
    columns: list[ColumnInfo]
    mapping: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_role(self, role: str) -> list[ColumnInfo]:
        return [c for c in self.columns if c.role == role]


def _sniff(path: Path) -> tuple[str, str, int, str]:
    """(delimiter, comment char, rows to skip before the header, decoded text).

    The header is taken to be the first non-comment line (pandas skips comment lines);
    files with banner lines that are not commented need ``skip_rows`` set by hand."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    comment = "#" if any(ln.lstrip().startswith("#") for ln in lines[:50]) else ""
    body = [ln for ln in lines if ln.strip() and not (comment and ln.lstrip().startswith("#"))]
    sample = "\n".join(body[:50])
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        delim = max(counts, key=counts.get)
    return delim, comment, 0, text


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _profile(name: str, s: pd.Series) -> ColumnInfo:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    info = ColumnInfo(name, float(ok.mean()) if len(x) else 0.0, int(ok.sum()))
    if ok.sum() < 3:
        return info
    v = x[ok]
    info.vmin, info.vmax, info.median = float(v.min()), float(v.max()), float(np.median(v))
    info.first = float(np.median(v[: max(3, len(v) // 50)]))
    d = np.diff(v)
    pos = d[d > 0]
    info.step = float(np.median(pos)) if len(pos) else math.nan
    info.monotonic = bool(len(d) and (d >= 0).mean() > 0.995 and v[-1] > v[0])
    return info


def _has(name: str, *words: str) -> bool:
    n = name.lower()
    return any(w in n for w in words)


def _unit_hint(name: str) -> str | None:
    """Unit written in the column name, e.g. 'alt_ft', 'Altitude (m)', 'accel [g]'."""
    n = name.lower()
    tokens = re.split(r"[^a-z0-9/^]+", n)
    for unit, keys in (
        ("ft", ("ft", "feet", "foot")),
        ("m", ("m", "meter", "meters", "metre", "metres")),
        ("ms", ("ms", "msec", "millis")),
        ("us", ("us", "usec", "micros")),
        ("s", ("s", "sec", "secs", "seconds")),
        ("g", ("g", "gs")),
        ("m/s2", ("m/s2", "m/s^2", "mps2", "ms2")),
        ("ft/s2", ("ft/s2", "ft/s^2", "fps2")),
        ("deg/s", ("deg/s", "dps", "degs", "deg")),
        ("rad/s", ("rad/s", "rads", "rad")),
    ):
        if any(k in tokens for k in keys):
            return unit
    return None


def _guess_time(cols: list[ColumnInfo]) -> ColumnInfo | None:
    cands = [c for c in cols if c.monotonic and c.numeric_fraction > 0.95]
    named = [
        c for c in cands if _has(c.name, "time", "tick", "clock", "millis") or c.name in ("t", "T")
    ]
    pick = (named or cands or [None])[0]
    if pick is None:
        return None
    pick.role = "time"
    hint = _unit_hint(pick.name)
    if hint in ("ms", "us", "s"):
        pick.unit, pick.why = hint, "column name"
    else:
        # sample step: 1-100 counts per sample of a ms clock, 1e-3-0.1 for seconds
        st = pick.step
        if st >= 200:
            pick.unit = "us"
        elif st >= 0.5:
            pick.unit = "ms"
        else:
            pick.unit = "s"
        pick.why = f"sample step {st:g}"
    return pick


def inspect_csv(path: str | Path) -> Inspection:
    path = Path(path)
    delim, comment, skip, text = _sniff(path)
    df = pd.read_csv(
        io.StringIO(text),
        sep=delim,
        comment=comment or None,
        skiprows=skip,
        skipinitialspace=True,
    )
    df.columns = [str(c).strip() for c in df.columns]
    cols = [_profile(c, df[c]) for c in df.columns]
    ins = Inspection(path, delim, comment, skip, len(df), cols)
    numeric = [c for c in cols if c.n_valid >= 3]  # sparse GPS columns count too

    t = _guess_time(numeric)
    if t is None:
        ins.warnings.append("no increasing time column found")
    rest = [c for c in numeric if c is not t]

    # GPS first (lat/lon look like plain numbers otherwise)
    for c in rest:
        if _has(c.name, "lat") and -90 <= c.vmin and c.vmax <= 90:
            c.role, c.why = "gps_lat", "name + range"
        elif _has(c.name, "lon", "lng") and -180 <= c.vmin and c.vmax <= 360:
            c.role, c.why = "gps_lon", "name + range"
    for c in rest:
        if c.role or not (_has(c.name, "gps", "gnss") and _has(c.name, "alt", "height")):
            continue
        c.role, c.unit = "gps_alt", _unit_hint(c.name) if _unit_hint(c.name) in ("m", "ft") else "m"
        c.why = "name"

    # altitude (barometric)
    alts = [
        c
        for c in rest
        if not c.role
        and _has(c.name, "alt", "height", "agl", "asl", "baro")
        and not _has(c.name, "press", "temp")
    ]
    if alts:
        a = max(alts, key=lambda c: c.vmax - c.vmin)
        a.role = "altitude"
        hint = _unit_hint(a.name)
        if hint in ("ft", "m"):
            a.unit, a.why = hint, "column name"
        else:
            a.unit, a.why = "m", "ASSUMED metres: confirm (feet look the same from values alone)"
            ins.warnings.append(f"altitude unit of {a.name!r} is a guess (m); confirm it")
    else:
        press = [c for c in rest if _has(c.name, "press", "baro")]
        ins.warnings.append(
            "no altitude column; Plume needs baro altitude"
            + (f" (pressure column {press[0].name!r} found: convert it first)" if press else "")
        )

    # acceleration: the axial channel has the largest range
    accs = [c for c in rest if not c.role and _has(c.name, "acc", "accel")]
    if accs:
        a = max(accs, key=lambda c: c.vmax - c.vmin)
        a.role = "acceleration"
        hint = _unit_hint(a.name)
        pad = a.first
        if hint in ("g", "m/s2", "ft/s2"):
            a.unit, a.why = hint, "column name"
        elif abs(pad - 1.0) < 0.25 or (abs(pad) < 0.25 and a.vmax < 60):
            a.unit, a.why = "g", f"pad reading {pad:.2f}"
        elif abs(pad - 32.17) < 5 or a.vmax > 400:
            a.unit, a.why = "ft/s2", f"pad reading {pad:.2f}"
        else:
            a.unit, a.why = "m/s2", f"pad reading {pad:.2f}"
        scale = {"g": 1.0, "m/s2": G0, "ft/s2": G0 / 0.3048}[a.unit]
        a.includes_gravity = abs(abs(pad / scale) - 1.0) < 0.3
        if abs(pad / scale) > 0.3 and not a.includes_gravity:
            ins.warnings.append(
                f"pad reading of {a.name!r} ({pad:.2f}) is neither ~0 nor ~1 g: "
                "is it the axial axis? is the unit right?"
            )
        if pad / scale < -0.7:
            ins.warnings.append(
                f"{a.name!r} reads -1 g on the pad: axis points down; use scale: -1"
            )

    # gyro
    gyros = [
        c
        for c in rest
        if not c.role
        and (_has(c.name, "gyr", "rate", "omega") or re.fullmatch(r"g[xyz].*", c.name.lower()))
    ]
    if len(gyros) >= 3:
        for c in gyros[:3]:
            c.role = "gyro"
            hint = _unit_hint(c.name)
            c.unit = hint if hint in ("deg/s", "rad/s") else "deg/s"
            c.why = "column name" if hint in ("deg/s", "rad/s") else "ASSUMED deg/s"
    ins.mapping = draft_mapping(ins)
    return ins


def draft_mapping(ins: Inspection) -> dict:
    """Mapping dict in the ``LogMappingSpec`` layout (``configs/flightlogs/*.yaml``)."""
    m: dict = {"name": f"{ins.path.stem}_draft", "delimiter": ins.delimiter}
    if ins.comment:
        m["comment"] = ins.comment
    if ins.skip_rows:
        m["skip_rows"] = ins.skip_rows
    t = ins.by_role("time")
    alt = ins.by_role("altitude")
    if t:
        m["time"] = {"column": t[0].name, "unit": t[0].unit}
    if alt:
        m["altitude"] = {"column": alt[0].name, "unit": alt[0].unit}
    acc = ins.by_role("acceleration")
    if acc:
        a = acc[0]
        m["acceleration"] = {
            "column": a.name,
            "unit": a.unit,
            "includes_gravity": a.includes_gravity,
        }
        if a.includes_gravity and a.first < 0:  # axis mounted pointing down
            m["acceleration"]["scale"] = -1.0
    gy = ins.by_role("gyro")
    if len(gy) == 3:
        m["gyro"] = {"columns": [c.name for c in gy], "unit": gy[0].unit}
    lat, lon = ins.by_role("gps_lat"), ins.by_role("gps_lon")
    if lat and lon:
        m["gps"] = {"lat": lat[0].name, "lon": lon[0].name}
        galt = ins.by_role("gps_alt")
        if galt:
            m["gps"].update(alt=galt[0].name, alt_unit=galt[0].unit)
    m["resample_hz"] = 50
    return m


def write_draft(ins: Inspection, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    why = "\n".join(
        f"#   {c.name}: {c.role} [{c.unit or '-'}] ({c.why})" for c in ins.columns if c.role
    )
    warn = "\n".join(f"#   WARNING: {w}" for w in ins.warnings)
    head = (
        f"# DRAFT column mapping for {ins.path.name}, written by `plume flightlog inspect`.\n"
        "# CONFIRM EVERY COLUMN AND UNIT against the flight computer's manual before use:\n"
        "# the guesses come from column names and value ranges.\n"
        f"# How each column was decided:\n{why}\n" + (f"{warn}\n" if warn else "")
    )
    out.write_text(head + yaml.safe_dump(ins.mapping, sort_keys=False), encoding="utf-8")
    return out
