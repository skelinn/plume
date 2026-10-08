"""Validation records: a real flight compared with its pre-flight prediction.

``plume validate <log.csv> --prediction <prediction.json> --flight-id <id>``:

1. imports the log and measures the flight (apogee, timing, speeds, descent rate,
   landing point when GPS is logged);
2. places each measurement in the prediction's Monte Carlo distribution (its percentile)
   and checks it falls inside the 95 % band;
3. calibrates drag, motor impulse/burn time and parachute to the log
   (``plume.flightdata.calibrate``) and checks each fitted factor is within twice the
   1-sigma uncertainty the prediction assumed;
4. writes ``docs/validation/<id>.md`` (human-readable) and ``docs/validation/<id>/``
   (``record.json``, the overlay plot, the calibrated vehicle, a copy of the prediction).

Per model, the verdict is

* ``validated``   every criterion passed **and** the prediction was recorded before the
                  flight (a blind test), on real data. The record states the envelope
                  (vehicle, motor, Mach range) the evidence covers - nothing beyond it;
* ``consistent``  every criterion passed but the test was not blind (no flight date
                  given, or the prediction was made afterwards);
* ``discrepancy`` a criterion failed: the model or its stated uncertainty is wrong for
                  this vehicle. Use the calibrated vehicle and fly again with a new prediction;
* ``inconclusive`` the log lacks the data for the criterion.

Synthetic logs (``--synthetic``) produce the same record, marked as a rehearsal: they
never count as validation (``plume vv-report`` lists them separately).

``plume vv-report`` reads ``docs/validation/*/record.json`` (``read_records``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plume import __version__
from plume.config import VehicleSpec, dump_yaml
from plume.constants import G0
from plume.flightdata.importer import FlightLog, load_log
from plume.flightdata.predict import METRICS, Prediction, resolve_motor

FORMAT = "plume-validation"
VERSION = 1

# model checks: (record key, docs/models page stem or "" when there is none, description)
MODEL_DOCS = {
    "drag": ("aero", "Aerodynamic drag (axial force)"),
    "motor": ("propulsion", "Solid-motor thrust curve"),
    "parachute": ("", "Parachute descent (recovery)"),
}


def observe(log: FlightLog) -> dict[str, float]:
    """What the flight did, in the prediction's metric names."""
    out = {
        "apogee_m": log.apogee,
        "t_apogee_s": log.t_apogee,
        # the baro/accel Kalman velocity is vertical; near-vertical boosts make it ~speed
        "max_speed_mps": float(log.velocity.max()),
    }
    if log.accel is not None:
        out["max_accel_g"] = float(np.nanmax(log.accel) / G0)
    after = (log.t > log.t_apogee + 5.0) & (log.altitude > 20.0)
    if after.sum() > 20:
        out["descent_rate_mps"] = float(np.median(-log.velocity[after]))
    down = np.nonzero((log.t > log.t_apogee) & (log.altitude < 2.0))[0]
    if len(down):
        out["flight_time_s"] = float(log.t[down[0]])
        if log.east is not None and log.north is not None:
            k = down[0]
            out["landing_east_m"] = float(log.east[k])
            out["landing_north_m"] = float(log.north[k])
            out["landing_distance_m"] = float(math.hypot(log.east[k], log.north[k]))
    return out


def percentile_of(pred: Prediction, key: str, value: float) -> float | None:
    x = np.array([r.get(key, np.nan) for r in pred.runs], dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return float(100.0 * (np.sum(x < value) + 0.5 * np.sum(x == value)) / len(x))


def compare(pred: Prediction, obs: dict[str, float]) -> list[dict]:
    rows = []
    for key in METRICS:
        if key not in obs or key not in pred.summary:
            continue
        b = pred.summary[key]
        v = obs[key]
        rows.append(
            {
                "metric": key,
                "observed": v,
                "nominal": pred.nominal.get(key),
                "p2.5": b["p2.5"],
                "p97.5": b["p97.5"],
                "percentile": percentile_of(pred, key, v),
                "inside_95": bool(b["p2.5"] <= v <= b["p97.5"]),
            }
        )
    return rows


def _parse_date(s: str | None) -> _dt.datetime | None:
    if not s:
        return None
    d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=_dt.UTC)


@dataclass
class Check:
    model: str
    quantity: str
    value: float | None
    criterion: str
    passed: bool | None  # None = inconclusive

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "doc": MODEL_DOCS[self.model][0],
            "quantity": self.quantity,
            "value": self.value,
            "criterion": self.criterion,
            "passed": self.passed,
        }


def model_checks(pred: Prediction, comp: list[dict], params: dict, sigma: dict) -> list[Check]:
    d = pred.settings["dispersions"]
    by = {r["metric"]: r for r in comp}
    checks: list[Check] = []

    def band(model, key):
        r = by.get(key)
        checks.append(
            Check(
                model,
                f"{METRICS[key][0]} inside the predicted 95 % band",
                r["observed"] if r else None,
                f"{r['p2.5']:.4g} to {r['p97.5']:.4g}" if r else "not measured",
                r["inside_95"] if r else None,
            )
        )

    def factor(model, key, label, s1):
        v = params.get(key)
        checks.append(
            Check(
                model,
                f"fitted {label} (x nominal)",
                v,
                f"within 2 sigma of 1 (sigma {s1:g})",
                None if v is None else bool(abs(v - 1.0) <= 2.0 * s1),
            )
        )

    band("drag", "apogee_m")
    factor("drag", "cd_scale", "drag coefficient", d["cd"])
    factor("motor", "impulse_scale", "total impulse", d["impulse"])
    factor("motor", "time_scale", "burn time", d["burn_time"])
    band("parachute", "descent_rate_mps")
    nominal_cda = sum(c["cd_area"] for c in pred.meta["vehicle_spec"]["recovery"]["chutes"])
    cs = params.get("chute_cd_area")
    checks.append(
        Check(
            "parachute",
            "fitted parachute Cd*A (x nominal)",
            None if cs is None or not nominal_cda else cs / nominal_cda,
            f"within 2 sigma of 1 (sigma {d['chute_cd_area']:g})",
            None
            if cs is None or not nominal_cda
            else bool(abs(cs / nominal_cda - 1.0) <= 2.0 * d["chute_cd_area"]),
        )
    )
    return checks


def verdicts(checks: list[Check], blind: bool | None, synthetic: bool) -> dict[str, str]:
    out = {}
    for model in MODEL_DOCS:
        cs = [c for c in checks if c.model == model]
        if not cs or any(c.passed is None for c in cs):
            v = "inconclusive"
        elif not all(c.passed for c in cs):
            v = "discrepancy"
        elif blind and not synthetic:
            v = "validated"
        else:
            v = "consistent"
        out[model] = v
    return out


def vehicle_from_prediction(pred: Prediction) -> VehicleSpec:
    """The exact vehicle the prediction flew (motor file re-resolved on this machine and
    checked against the recorded hash)."""
    spec = json.loads(json.dumps(pred.meta["vehicle_spec"]))
    motor = pred.meta.get("motor") or {}
    if motor.get("file"):
        path = resolve_motor(motor["file"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if motor.get("sha256") and sha != motor["sha256"]:
            raise ValueError(
                f"motor file {path} differs from the one the prediction used (sha256 mismatch)"
            )
        spec["engine"]["motor_file"] = str(path)
    return VehicleSpec.model_validate(spec)


def build_record(
    csv: str | Path,
    mapping,
    prediction: Prediction,
    flight_id: str,
    flight_date: str | None = None,
    synthetic: bool = False,
    rail_length: float | None = None,
    notes: str = "",
):
    """Returns (record dict, calibration result, flight log)."""
    from plume.flightdata.calibrate import calibrate

    csv = Path(csv)
    log = load_log(csv, mapping)
    vehicle = vehicle_from_prediction(prediction)
    obs = observe(log)
    comp = compare(prediction, obs)
    rail = rail_length if rail_length is not None else prediction.settings["launch"]["rail_length"]
    cal = calibrate(log, vehicle, rail_length=rail)
    checks = model_checks(prediction, comp, cal.params, cal.sigma)
    created = _parse_date(prediction.meta["created_utc"])
    fdate = _parse_date(flight_date)
    blind = None if fdate is None else bool(created < fdate)
    v = verdicts(checks, blind, synthetic)
    mach = prediction.summary.get("max_mach", {})
    envelope = (
        f"{vehicle.name} on the {prediction.meta.get('motor', {}).get('name', '?')} motor; "
        f"subsonic (Mach up to about {mach.get('p50', float('nan')):.2f}), near-vertical "
        "flight at small angle of attack, single parachute at apogee"
    )
    record = {
        "format": FORMAT,
        "version": VERSION,
        "flight_id": flight_id,
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "plume_version": __version__,
        "synthetic": bool(synthetic),
        "flight_date": flight_date,
        "notes": notes,
        "log": {
            "file": csv.name,
            "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "mapping": log.meta["mapping"],
            "samples": len(log.t),
        },
        "prediction": {
            "flight_id": prediction.meta["flight_id"],
            "created_utc": prediction.meta["created_utc"],
            "git": prediction.meta.get("git"),
            "vehicle_sha256": prediction.meta["vehicle"]["sha256"],
            "blind": blind,
        },
        "envelope": envelope,
        "observed": obs,
        "comparison": comp,
        "calibration": {
            "params": cal.params,
            "sigma": {k: (s if math.isfinite(s) else None) for k, s in cal.sigma.items()},
            "before": cal.before,
            "after": cal.after,
        },
        "checks": [c.to_dict() for c in checks],
        "verdicts": v,
    }
    return record, cal, log


_VERDICT_TEXT = {
    "validated": "**validated** (blind prediction, all criteria met)",
    "consistent": "consistent (all criteria met, but not a blind test)",
    "discrepancy": "**discrepancy** (a criterion failed)",
    "inconclusive": "inconclusive (data missing)",
}


def record_markdown(rec: dict, folder: str) -> str:
    fid = rec["flight_id"]
    p = rec["prediction"]
    blind = {
        True: "yes",
        False: "NO (prediction made after the flight)",
        None: "unknown (no flight date)",
    }[p["blind"]]
    lines = [
        f"# Validation record: {fid}",
        "",
    ]
    if rec["synthetic"]:
        lines += [
            "> **Synthetic data - a rehearsal of the workflow, not validation.** "
            "This record does not count toward any model's validation status.",
            "",
        ]
    lines += [
        f"**Flight:** {rec['flight_date'] or 'date not given'} · log `{rec['log']['file']}` "
        f"(sha256 `{rec['log']['sha256'][:12]}`, mapping `{rec['log']['mapping']}`)",
        f"**Prediction:** `{p['flight_id']}` recorded {p['created_utc']} · blind test: {blind}",
        f"**Envelope covered:** {rec['envelope']}",
        f"**Record written:** {rec['created_utc']} by Plume {rec['plume_version']}",
        "",
        "## Verdicts",
        "",
        "| model | verdict |",
        "|---|---|",
    ]
    for model, v in rec["verdicts"].items():
        text = _VERDICT_TEXT[v]
        if rec["synthetic"] and v == "consistent":
            text = "consistent (synthetic rehearsal: never counts as validation)"
        lines.append(f"| {MODEL_DOCS[model][1]} | {text} |")
    lines += [
        "",
        "## Prediction vs flight",
        "",
        "| quantity | predicted nominal | predicted 95 % band | flight | percentile | inside |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rec["comparison"]:
        label, unit, fmt = METRICS[r["metric"]]
        u = f" {unit}" if unit else ""
        pct = "-" if r["percentile"] is None else f"{r['percentile']:.0f}"
        nom = r["nominal"] if r["nominal"] is not None else float("nan")
        lines.append(
            f"| {label} | {nom:{fmt}}{u} | {r['p2.5']:{fmt}} – {r['p97.5']:{fmt}}{u} | "
            f"{r['observed']:{fmt}}{u} | {pct} | {'yes' if r['inside_95'] else '**no**'} |"
        )
    c = rec["calibration"]
    lines += [
        "",
        "Maximum speed from the log is the vertical speed of the baro/accelerometer filter.",
        "",
        "## Criteria",
        "",
        "| model | check | value | criterion | result |",
        "|---|---|---:|---|---|",
    ]
    for ch in rec["checks"]:
        val = "-" if ch["value"] is None else f"{ch['value']:.4g}"
        res = {True: "pass", False: "**fail**", None: "no data"}[ch["passed"]]
        lines.append(
            f"| {MODEL_DOCS[ch['model']][1]} | {ch['quantity']} | {val} | {ch['criterion']} | {res} |"
        )
    lines += [
        "",
        "## Calibration (least squares on the 3-DOF model)",
        "",
        "| parameter | fit | 1 sigma |",
        "|---|---:|---:|",
    ]
    for k, val in c["params"].items():
        s = c["sigma"].get(k)
        lines.append(f"| {k} | {val:.4f} | {'-' if s is None else f'{s:.4f}'} |")
    lines += [
        "",
        f"Apogee error of the nominal model {c['before']['apogee_error_m']:+.1f} m, after "
        f"calibration {c['after']['apogee_error_m']:+.1f} m.",
        "",
        f"![real vs simulated]({folder}/overlay.png)",
        "",
        f"Data: [`{folder}/record.json`]({folder}/record.json), calibrated vehicle "
        f"[`{folder}/calibrated_vehicle.yaml`]({folder}/calibrated_vehicle.yaml), prediction "
        f"[`{folder}/prediction.json`]({folder}/prediction.json).",
        "",
    ]
    if rec.get("notes"):
        lines += ["## Notes", "", rec["notes"], ""]
    return "\n".join(lines)


def write_record(
    rec: dict, cal, log: FlightLog, prediction_path: str | Path, out_root: str | Path
) -> dict[str, Path]:
    from plume.flightdata.compare import plot_comparison

    out_root = Path(out_root)
    fid = rec["flight_id"]
    folder = out_root / fid
    folder.mkdir(parents=True, exist_ok=True)
    js = folder / "record.json"
    js.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    png = plot_comparison(
        log,
        [cal.nominal_trace, cal.calibrated_trace],
        folder / "overlay.png",
        f"{fid}: real vs simulated",
    )
    yml = dump_yaml(cal.vehicle, folder / "calibrated_vehicle.yaml")
    pred_copy = folder / "prediction.json"
    if Path(prediction_path).resolve() != pred_copy.resolve():
        shutil.copyfile(prediction_path, pred_copy)
    md = out_root / f"{fid}.md"
    md.write_text(record_markdown(rec, fid), encoding="utf-8")
    return {"markdown": md, "record": js, "overlay": png, "vehicle": yml}


# ----------------------------------------------------------------------------- reading
def read_records(root: str | Path) -> list[dict]:
    """All validation records under ``root`` (``docs/validation``), newest first. Each
    gets ``_md`` (path of its markdown page) added."""
    root = Path(root)
    out = []
    for p in sorted(root.glob("*/record.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("format") != FORMAT:
            continue
        rec["_md"] = str(root / f"{rec.get('flight_id', p.parent.name)}.md")
        out.append(rec)
    out.sort(key=lambda r: r.get("created_utc", ""), reverse=True)
    return out


def status_by_doc(records: list[dict]) -> dict[str, list[dict]]:
    """docs/models page stem -> [{flight_id, verdict, envelope, md, synthetic}], real
    flights only (synthetic rehearsals never change a model's status)."""
    out: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("synthetic"):
            continue
        for model, verdict in rec.get("verdicts", {}).items():
            doc = MODEL_DOCS.get(model, ("", ""))[0]
            if not doc:
                continue
            out.setdefault(doc, []).append(
                {
                    "flight_id": rec["flight_id"],
                    "verdict": verdict,
                    "envelope": rec.get("envelope", ""),
                    "md": rec["_md"],
                }
            )
    return out
