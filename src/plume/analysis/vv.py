"""Verification & validation report (``plume vv-report``).

Runs the verification test suite (``pytest -m vv`` plus the model test modules), the
NASA 6-DOF check cases, reads the model documentation (``docs/models/*.md``) and the
Monte Carlo summaries (``runs/mc/*/summary.json``), and writes one self-contained
monochrome HTML page (default ``docs/vv/report.html``).

Validation (agreement with real flight or test data) is reported as *pending* for every
model: no real data exists yet, and this report never claims otherwise.
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# model test modules run in addition to ``pytest -m vv``
MODEL_TEST_MODULES = [
    "tests/test_aerodb.py",
    "tests/test_navigation.py",
    "tests/test_actuators.py",
    "tests/test_slosh.py",
    "tests/test_dem.py",
    "tests/test_landing_gear.py",
    "tests/test_ship.py",
    "tests/test_montecarlo.py",
    # propulsion and mass properties are verified here (Tsiolkovsky, depletion, CG)
    "tests/test_physics_models.py",
    "tests/test_physics_conservation.py",
]

AREAS: dict[str, str] = {
    "earth": "Earth & gravity",
    "integration": "Integration & convergence",
    "nasa": "NASA check cases",
    "atmosphere": "Atmosphere, wind & turbulence",
    "aero": "Aerodynamics",
    "navigation": "Navigation & sensors",
    "propulsion": "Propulsion & actuators",
    "mass": "Mass properties",
    "slosh": "Propellant slosh",
    "terrain": "Terrain",
    "gear": "Landing gear & soil",
    "ship": "Drone ship & sea state",
    "mc": "Monte Carlo machinery",
    "other": "Other",
}

# (module regex, test-name regex, area): the first match wins
RULES: list[tuple[str, str, str]] = [
    (r"test_nasa_checkcases", r"", "nasa"),
    (r"test_analytic", r"convergence|step_size", "integration"),
    (r"test_analytic", r"", "earth"),
    (r"test_environment", r"", "atmosphere"),
    (r"test_aerodb", r"", "aero"),
    (r"test_navigation", r"", "navigation"),
    (r"test_actuators", r"", "propulsion"),
    (r"test_slosh", r"", "slosh"),
    (r"test_dem", r"", "terrain"),
    (r"test_landing_gear", r"", "gear"),
    (r"test_ship", r"", "ship"),
    (r"test_montecarlo", r"", "mc"),
    (r"test_physics_models", r"atmosphere|wind", "atmosphere"),
    (r"test_physics_models", r"gravity|spherical_map", "earth"),
    (r"test_physics_models", r"mass_properties|cg_drops", "mass"),
    (r"test_physics_models", r"engine|gimbal|rcs", "propulsion"),
    (r"test_physics_models", r"pointmass", "integration"),
    (r"test_physics_conservation", r"tsiolkovsky|rocket_equation", "propulsion"),
    (r"test_physics_conservation", r"mass_conservation|variable_mass", "mass"),
    (r"test_physics_conservation", r"drag_work", "aero"),
    (r"test_physics_conservation", r"wind", "atmosphere"),
    (r"test_physics_conservation", r"", "integration"),
    # tests outside the listed modules (e.g. new ``vv`` tests): guess from the name
    (r"", r"nasa|checkcase", "nasa"),
    (r"", r"atmos|wind|turbulen|us76|sounding", "atmosphere"),
    (r"", r"aero|drag|fin", "aero"),
    (r"", r"gravity|geodetic|earth|kepler", "earth"),
    (r"", r"convergence|energy|momentum", "integration"),
    (r"", r"engine|gimbal|rcs|thrust|ignition", "propulsion"),
    (r"", r"mass|cg_", "mass"),
    (r"", r"terrain|dem|hazard", "terrain"),
]

# model page (file stem) -> test areas that verify it
MODEL_AREAS: dict[str, tuple[str, ...]] = {
    "earth": ("earth", "integration", "nasa"),
    "atmosphere": ("atmosphere",),
    "aero": ("aero",),
    "navigation": ("navigation",),
    "propulsion": ("propulsion",),
    "mass": ("mass",),
    "slosh": ("slosh",),
    "terrain": ("terrain",),
    "landing_gear": ("gear",),
    "ship_landing": ("ship",),
}


# ----------------------------------------------------------------------------- junit
@dataclass
class TestResult:
    module: str  # dotted, e.g. tests.vv.test_analytic
    name: str  # e.g. test_us76_lower_matches_published_table[0]
    outcome: str  # passed | failed | error | skipped
    time: float
    message: str = ""
    area: str = "other"

    @property
    def ok(self) -> bool:
        return self.outcome == "passed"


def classify(module: str, name: str) -> str:
    base = name.split("[", 1)[0]
    for mod_re, name_re, area in RULES:
        if re.search(mod_re, module) and re.search(name_re, base):
            return area
    return "other"


def _module_of(classname: str) -> str:
    parts = classname.split(".")
    idx = max((i for i, p in enumerate(parts) if p.startswith("test_")), default=len(parts) - 1)
    return ".".join(parts[: idx + 1])


def parse_junit(path: str | Path) -> tuple[list[TestResult], dict]:
    """Per-test results from a JUnit XML file (pytest ``--junitxml``), de-duplicated."""
    root = ET.parse(path).getroot()
    out: dict[tuple[str, str], TestResult] = {}
    stamps = [s.get("timestamp") for s in root.iter("testsuite") if s.get("timestamp")]
    for tc in root.iter("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (el.get("message") or el.text or "").strip()
                break
        module = _module_of(classname)
        try:
            t = float(tc.get("time") or 0.0)
        except ValueError:
            t = 0.0
        out[(classname, name)] = TestResult(
            module, name, outcome, t, message, classify(module, name)
        )
    meta = {"timestamp": min(stamps) if stamps else None}
    return list(out.values()), meta


def area_counts(tests: list[TestResult]) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    for t in tests:
        c = counts.setdefault(
            t.area, {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "time": 0.0}
        )
        key = "failed" if t.outcome in ("failed", "error") else t.outcome
        c[key] += 1
        c["total"] += 1
        c["time"] += t.time
    return {a: counts[a] for a in AREAS if a in counts}


# ----------------------------------------------------------------------------- running
def _pytest_cmd() -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if importlib.util.find_spec("xdist") is not None:
        cmd += ["-n", "4"]
    return cmd


def run_tests(junit: Path, root: Path = REPO, log: Callable[[str], None] = print) -> int:
    """Run ``pytest -m vv`` and the model test modules; merge both JUnit files into
    ``junit``. Returns the worst pytest exit code (0 = all passed, 1 = failures)."""
    junit = Path(junit)
    junit.parent.mkdir(parents=True, exist_ok=True)
    flags = {}
    if sys.platform == "win32":  # be polite to interactive use
        flags["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    parts = []
    codes = []
    modules = [m for m in MODEL_TEST_MODULES if (root / m).exists()]
    for missing in sorted(set(MODEL_TEST_MODULES) - set(modules)):
        log(f"warning: {missing} not found, skipped")
    for tag, args in (("vv", ["-m", "vv"]), ("models", modules)):
        part = junit.with_name(f"{junit.stem}_{tag}.xml")
        part.unlink(missing_ok=True)
        cmd = [*_pytest_cmd(), *args, f"--junitxml={part}"]
        log("$ " + " ".join(cmd[1:]))
        codes.append(subprocess.call(cmd, cwd=root, **flags))
        if not part.exists():
            raise RuntimeError(f"pytest ({tag}) wrote no JUnit file (exit code {codes[-1]})")
        parts.append(part)
    merged = ET.Element("testsuites")
    for part in parts:
        r = ET.parse(part).getroot()
        merged.extend(r.iter("testsuite") if r.tag == "testsuites" else [r])
    ET.ElementTree(merged).write(junit, encoding="utf-8", xml_declaration=True)
    bad = [c for c in codes if c not in (0, 1, 5)]
    return max(bad) if bad else max(codes)


def _nasa_case(number: int) -> dict:
    from plume.analysis.nasa_checkcases import CASES, FLOORS, compare, passed

    case = CASES[number]
    try:
        res = compare(number)
    except Exception as exc:  # report, don't abort the whole report
        return {"number": number, "name": case.name, "error": repr(exc), "vars": []}
    rows = [
        {
            "var": v,
            "error": err,
            "spread": res.spread[v],
            "tol": max(FLOORS[v], 2.0 * res.spread[v]),
            "passed": passed(res, v),
        }
        for v, err in res.errors.items()
    ]
    return {"number": number, "name": case.name, "error": None, "vars": rows}


def run_nasa(workers: int = 4, log: Callable[[str], None] = print) -> list[dict]:
    """All NASA check cases (NASA/TM-2015-218675), in parallel processes."""
    from plume.analysis.nasa_checkcases import CASES

    numbers = sorted(CASES)
    log(f"NASA check cases 1-{numbers[-1]} ({workers} workers) ...")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(numbers)))) as pool:
        return list(pool.map(_nasa_case, numbers))


# ----------------------------------------------------------------------------- inputs
@dataclass
class ModelDoc:
    path: Path
    title: str
    code: str = ""
    status: str = ""
    areas: tuple[str, ...] = field(default_factory=tuple)


def read_model_docs(models_dir: Path) -> list[ModelDoc]:
    docs = []
    for p in sorted(Path(models_dir).glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        text = p.read_text(encoding="utf-8")
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), p.stem)
        code = re.search(r"^\*\*Code:\*\*\s*(.+)$", text, re.M)
        status = re.search(r"^\*\*Verification status:\*\*\s*(.+)$", text, re.M) or re.search(
            r"^\*\*Status:\*\*\s*(.+)$", text, re.M
        )
        docs.append(
            ModelDoc(
                p,
                title,
                code.group(1).strip() if code else "",
                status.group(1).strip() if status else "",
                MODEL_AREAS.get(p.stem, ()),
            )
        )
    return docs


def _mc_from_summary(s: dict) -> dict:
    land = s.get("landing") or {}
    p, ci, cep = s.get("success_probability"), s.get("success_ci95"), land.get("cep50_m")
    return {
        "mission": str(s.get("mission", "-")),
        "fidelity": str(s.get("fidelity", "-")),
        "runs": str(s.get("runs", "-")),
        "success": f"{100 * p:.1f} %" if p is not None else "-",
        "ci": f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f} %" if ci else "-",
        "cep50": f"{cep:,.0f} m" if cep is not None else "-",
        "source": "summary.json",
    }


def _mc_from_html(text: str) -> dict | None:
    """Fallback: the headline tiles of a report written by ``montecarlo.write_report``."""
    kpi = dict(
        (k.strip(), html.unescape(v).strip())
        for v, k in re.findall(r'<div class="kpi"><b>([^<]*)</b><span>([^<]*)</span>', text)
    )
    sub = re.search(r"mission <code>([^<]*)</code> · fidelity (\w+) · (\d+) runs", text)
    if not kpi and not sub:
        return None
    cep = kpi.get("CEP50", "-").replace(" m", "")
    try:
        cep = f"{float(cep.replace(',', '')):,.0f} m"
    except ValueError:
        cep = "-"
    return {
        "mission": sub.group(1) if sub else "-",
        "fidelity": sub.group(2) if sub else "-",
        "runs": sub.group(3) if sub else "-",
        "success": kpi.get("success", "-"),
        "ci": kpi.get("95 % interval", "-"),
        "cep50": cep,
        "source": "report page",
    }


def read_mc_reports(root: Path) -> list[dict]:
    """Monte Carlo reports in docs/mc/*.html with their headline numbers, from
    runs/mc/<same stem>/summary.json, else from the report page itself."""
    out = []
    for p in sorted((root / "docs" / "mc").glob("*.html")):
        head = None
        sp = root / "runs" / "mc" / p.stem / "summary.json"
        try:
            head = _mc_from_summary(json.loads(sp.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            try:
                head = _mc_from_html(p.read_text(encoding="utf-8"))
            except OSError:
                head = None
        out.append({"html": p, "name": p.stem, "head": head})
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        )
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment(root: Path) -> dict:
    commit = _git(root, "rev-parse", "--short", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": bool(dirty) if commit else False,
        "date": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip(),
        "python": platform.python_version(),
        "mujoco": _version("mujoco"),
        "pytest": _version("pytest"),
        "platform": f"{platform.system()} {platform.machine()}",
    }


# ----------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fff;--fg:#0a0a0a;--mute:#666;--rule:#d0d0d0}
@media (prefers-color-scheme: dark){:root{--bg:#0a0a0a;--fg:#f2f2f2;--mute:#8a8a8a;--rule:#2a2a2a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}
h2{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:40px 0 8px;font-weight:500}
p{margin:0 0 8px} p.sub{color:var(--mute);margin:0 0 24px}
a{color:inherit;text-decoration:underline;text-decoration-color:var(--rule);text-underline-offset:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid var(--rule);border-left:1px solid var(--rule)}
.kpi{padding:12px 16px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.kpi b{display:block;font:500 22px "IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.kpi span{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{font-weight:500;color:var(--mute);font-size:11px;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
td.n,th.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
code,pre{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
td code{overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 0;color:var(--mute)}
.note{color:var(--mute);font-size:12px}
.fail{font-weight:600;background:var(--fg);color:var(--bg);padding:0 4px}
.skip,.pend{color:var(--mute)}
dl{margin:0;border-top:1px solid var(--rule)}
dl div{display:grid;grid-template-columns:140px 1fr;gap:16px;padding:8px 0;border-bottom:1px solid var(--rule)}
dt{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);padding-top:2px}
dd{margin:0}
details{border-bottom:1px solid var(--rule)}
details:first-of-type{border-top:1px solid var(--rule)}
summary{cursor:pointer;padding:8px 0;display:flex;justify-content:space-between;gap:16px;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--mute);width:1em;flex:none}
details[open]>summary::before{content:"\\2212"}
summary span:first-of-type{flex:1}
summary .n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;color:var(--mute);white-space:nowrap}
details table{margin-bottom:12px}
table.models td:nth-child(1){min-width:140px} table.models td:nth-child(2){min-width:160px}
@media (max-width:600px){dl div{grid-template-columns:1fr;gap:2px} td,th{padding:6px 4px}
table.stack tr:first-child{display:none} table.stack tr{display:block;border-bottom:1px solid var(--rule);padding:8px 0}
table.stack td{display:block;border:0;padding:2px 0;text-align:left;min-width:0!important;white-space:normal}
table.stack td::before{content:attr(data-l);display:block;font:500 11px/1.6 "IBM Plex Sans",system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}}
@media print{details>*{display:block}}
"""

E = html.escape


def _md_inline(text: str) -> str:
    """Escape, then render `code`, **bold** and [links](..) (as plain text)."""
    s = E(text)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    return s


def _rel(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:  # different drives on Windows
        return target.resolve().as_uri()


def _g(x: float) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.3g}"


def _natural(t: TestResult) -> tuple:
    """Sort key: module, then name with embedded numbers compared numerically."""
    parts = re.split(r"(\d+)", t.name)
    return (t.module, [int(p) if p.isdigit() else p for p in parts])


def _outcome(o: str) -> str:
    if o == "passed":
        return "pass"
    if o == "skipped":
        return '<span class="skip">skip</span>'
    return f'<span class="fail">{"FAIL" if o == "failed" else "ERROR"}</span>'


def _count_cell(c: dict | None) -> str:
    if not c:
        return '<td class="n" data-l="tests">-</td>'
    extra = f' <span class="fail">{c["failed"]} failed</span>' if c["failed"] else ""
    return f'<td class="n" data-l="tests passed">{c["passed"]} / {c["total"]}{extra}</td>'


def _merge_counts(counts: dict[str, dict], areas: tuple[str, ...]) -> dict | None:
    cs = [counts[a] for a in areas if a in counts]
    if not cs:
        return None
    return {k: sum(c[k] for c in cs) for k in ("passed", "failed", "skipped", "total")}


def render_html(
    *,
    env: dict,
    tests: list[TestResult],
    junit_meta: dict,
    junit_path: Path | None,
    models: list[ModelDoc],
    nasa: list[dict] | None,
    mc: list[dict],
    out_dir: Path,
    root: Path,
    ran: bool,
) -> str:
    counts = area_counts(tests)
    n_pass = sum(t.outcome == "passed" for t in tests)
    n_fail = sum(t.outcome in ("failed", "error") for t in tests)
    n_skip = sum(t.outcome == "skipped" for t in tests)

    # --- KPIs
    if nasa is None:
        nasa_kpi = "not run"
    else:
        ok = sum(
            1 for c in nasa if not c["error"] and c["vars"] and all(v["passed"] for v in c["vars"])
        )
        nasa_kpi = f"{ok} / {len(nasa)}"
    kpis = "".join(
        f'<div class="kpi"><b>{v}</b><span>{k}</span></div>'
        for v, k in (
            (f"{n_pass} / {len(tests)}", "tests passed"),
            (str(n_fail), "tests failed"),
            (str(n_skip), "tests skipped"),
            (nasa_kpi, "NASA cases passed"),
            (f"0 / {len(models)}", "models validated"),
        )
    )

    # --- failures
    failed = [t for t in tests if t.outcome in ("failed", "error")]
    fail_html = ""
    if failed:
        rows = "".join(
            f"<tr><td><code>{E(t.module)}::{E(t.name)}</code>"
            f"<pre>{E(t.message[:600])}</pre></td><td>{E(AREAS[t.area])}</td>"
            f"<td>{_outcome(t.outcome)}</td></tr>"
            for t in failed
        )
        fail_html = (
            "<h2>Failures</h2><div class=tw><table><tr><th>test</th><th>area</th><th></th></tr>"
            f"{rows}</table></div>"
        )

    # --- models
    model_rows = "".join(
        f'<tr><td data-l="model"><a href="{E(_rel(m.path, out_dir))}">{E(m.title)}</a></td>'
        f"<td data-l=code>{_md_inline(m.code) or '<span class=pend>not stated</span>'}</td>"
        f"<td data-l=verification>{_md_inline(m.status) or '<span class=pend>not stated</span>'}</td>"
        f"{_count_cell(_merge_counts(counts, m.areas))}"
        '<td data-l="validation" class="pend">Validation: pending real data</td></tr>'
        for m in models
    )
    models_html = (
        "<table class='stack models'><tr><th>model</th><th>code</th><th>verification status</th>"
        f'<th class="n">tests</th><th>validation</th></tr>{model_rows}</table>'
        if models
        else '<p class="note">No model pages found in docs/models.</p>'
    )

    # --- areas
    area_rows = "".join(
        f"<tr><td>{E(AREAS[a])}</td><td class=n>{c['passed']}</td>"
        f"<td class=n>{c['failed'] if not c['failed'] else '<span class=fail>' + str(c['failed']) + '</span>'}</td>"
        f"<td class=n>{c['skipped']}</td><td class=n>{c['total']}</td><td class=n>{c['time']:.1f}</td></tr>"
        for a, c in counts.items()
    )
    areas_html = (
        '<div class=tw><table><tr><th>area</th><th class="n">passed</th><th class="n">failed</th>'
        '<th class="n">skipped</th><th class="n">total</th><th class="n">time, s</th></tr>'
        f"{area_rows}</table></div>"
    )

    # --- per-test detail
    details = []
    for a, c in counts.items():
        rows = "".join(
            f"<tr><td><code>{E(t.name)}</code><br><span class=note>{E(t.module)}</span>"
            + (f"<pre>{E(t.message[:300])}</pre>" if t.outcome == "skipped" and t.message else "")
            + f"</td><td>{_outcome(t.outcome)}</td><td class=n>{t.time:.2f}</td></tr>"
            for t in sorted((t for t in tests if t.area == a), key=_natural)
        )
        state = (
            f"{c['passed']} / {c['total']} passed"
            + (f", {c['failed']} failed" if c["failed"] else "")
            + (f", {c['skipped']} skipped" if c["skipped"] else "")
        )
        details.append(
            f"<details><summary><span>{E(AREAS[a])}</span><span class=n>{state}</span></summary>"
            '<div class=tw><table><tr><th>test</th><th>result</th><th class="n">time, s</th></tr>'
            f"{rows}</table></div></details>"
        )
    tests_html = "".join(details) or '<p class="note">No test results.</p>'

    # --- NASA
    if nasa is None:
        nasa_html = (
            '<p class="note">Not run for this report (<code>--no-nasa</code>). The same cases '
            "also run as <code>tests/vv/test_nasa_checkcases.py</code> (see the NASA check "
            "cases area above).</p>"
        )
    else:
        srows, drows = [], []
        for c in nasa:
            label = f"{c['number']}. {E(c['name'])}"
            if c["error"]:
                srows.append(
                    f"<tr><td>{label}</td><td colspan=5><span class=fail>ERROR</span> "
                    f"<code>{E(c['error'][:300])}</code></td></tr>"
                )
                continue
            vs = c["vars"]
            worst = max(vs, key=lambda v: v["error"] / v["tol"] if v["tol"] else math.inf)
            npass = sum(v["passed"] for v in vs)
            srows.append(
                f"<tr><td data-l=case>{label}</td>"
                f"<td class=n data-l='variables passed'>{npass} / {len(vs)}</td>"
                f"<td data-l='worst variable'><code>{E(worst['var'])}</code></td>"
                f"<td class=n data-l='max error'>{_g(worst['error'])}</td>"
                f"<td class=n data-l=tolerance>{_g(worst['tol'])}</td>"
                f"<td data-l=pass>{_outcome('passed' if npass == len(vs) else 'failed')}</td></tr>"
            )
            for v in vs:
                drows.append(
                    f"<tr><td class=n>{c['number']}</td><td><code>{E(v['var'])}</code></td>"
                    f"<td class=n>{_g(v['error'])}</td><td class=n>{_g(v['spread'])}</td>"
                    f"<td class=n>{_g(v['tol'])}</td><td>{_outcome('passed' if v['passed'] else 'failed')}</td></tr>"
                )
        nasa_html = (
            '<p class="note">Plume (high fidelity) against the median of the NASA simulation '
            "tools over 30 s, 0.5 s samples. Tolerance = max(absolute floor, 2 &times; the largest "
            "deviation of any NASA tool from that median). Units: ft, ft/s, deg, deg/s. The "
            "worst variable is the one with the largest error-to-tolerance ratio.</p>"
            '<table class=stack><tr><th>case</th><th class="n">variables passed</th>'
            '<th>worst variable</th><th class="n">max error</th><th class="n">tolerance</th>'
            f"<th>pass</th></tr>{''.join(srows)}</table>"
        )
        if drows:
            nasa_html += (
                f"<details><summary><span>All comparisons</span><span class=n>{len(drows)} "
                "rows</span></summary><div class=tw><table><tr><th class=n>case</th>"
                '<th>variable</th><th class="n">max error</th><th class="n">NASA spread</th>'
                f'<th class="n">tolerance</th><th>pass</th></tr>{"".join(drows)}</table></div>'
                "</details>"
            )

    # --- Monte Carlo
    if mc:
        mrows = []
        for r in mc:
            h = r["head"] or dict.fromkeys(
                ("mission", "fidelity", "runs", "success", "ci", "cep50", "source"), "-"
            )
            mrows.append(
                f'<tr><td data-l=report><a href="{E(_rel(r["html"], out_dir))}">{E(r["name"])}</a></td>'
                + "".join(
                    f"<td data-l='{lbl}'{' class=n' if n else ''}>{E(h[k])}</td>"
                    for k, lbl, n in (
                        ("mission", "mission", False),
                        ("fidelity", "fidelity", False),
                        ("runs", "runs", True),
                        ("success", "success", True),
                        ("ci", "95 % CI", True),
                        ("cep50", "CEP50", True),
                        ("source", "source", False),
                    )
                )
                + "</tr>"
            )
        mc_html = (
            "<table class=stack><tr><th>report</th><th>mission</th><th>fidelity</th>"
            '<th class="n">runs</th><th class="n">success</th><th class="n">95 % CI</th>'
            f'<th class="n">CEP50</th><th>source</th></tr>{"".join(mrows)}</table>'
            '<p class="note">Headline numbers come from <code>runs/mc/&lt;name&gt;/summary.json</code>, '
            "or from the report page itself when that file is missing (for example while a "
            "campaign is being re-run). Success probability uses the Wilson score interval; CEP50 "
            "is the median miss distance of the landed runs. Monte Carlo results inherit the "
            "model-form error of the unvalidated models.</p>"
        )
    else:
        mc_html = (
            '<p class="note">No Monte Carlo reports in docs/mc. Run <code>plume mc</code>.</p>'
        )

    # --- method
    if ran:
        how = (
            f"<code>{E(' '.join(_pytest_cmd()[1:]))} -m vv</code> and "
            f"<code>{E(' '.join(_pytest_cmd()[1:]))} {E(' '.join(MODEL_TEST_MODULES))}</code>"
        )
    else:
        how = "read from an existing JUnit file (<code>--no-run</code>)"
    src = E(_rel(junit_path, root)) if junit_path else "-"
    ts = E(junit_meta.get("timestamp") or "unknown")
    commit = E(env["commit"]) + (" + uncommitted changes" if env["dirty"] else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Plume V&amp;V report</title>
<style>{CSS}</style></head><body><main>
<h1>Plume - verification &amp; validation report</h1>
<p class="sub">commit <code>{commit}</code> · {E(env["date"])} · Python {E(env["python"])} ·
MuJoCo {E(env["mujoco"])} · pytest {E(env["pytest"])} · {E(env["platform"])}</p>
<div class="kpis">{kpis}</div>

<h2>What this report means</h2>
<dl>
<div><dt>Verification</dt><dd>The math is implemented correctly: the code reproduces analytic
solutions, NASA reference trajectories, published tables and the expected convergence with
time step. Everything on this page is verification.</dd></div>
<div><dt>Validation</dt><dd>The models match reality. This needs real flight or test data,
which does not exist yet, so validation is <b>pending</b> for every model. Nothing here
claims that the simulator predicts real flights.</dd></div>
<div><dt>Monte Carlo</dt><dd>Quantified dependability under stated uncertainties: the
probability of success and the landing dispersion when vehicle and environment parameters
are dispersed as specified. It is only as good as the models and the dispersions.</dd></div>
</dl>
{fail_html}
<h2>Models</h2>
{models_html}
<p class="note">Status text is quoted from each page's <b>Verification status</b> line. The
tests column counts the test areas mapped to that model (Earth: Earth &amp; gravity,
integration, NASA check cases).</p>

<h2>Tests by area</h2>
{areas_html}

<h2>NASA 6-DOF check cases (NASA/TM-2015-218675)</h2>
{nasa_html}

<h2>Monte Carlo reports</h2>
{mc_html}

<h2>All tests</h2>
{tests_html}

<h2>Method</h2>
<p class="note">Tests: {how}. JUnit source <code>{src}</code>, run started {ts}.
Regenerate with <code>uv run plume vv-report</code> (<code>--no-run</code> re-reads the last
JUnit file, <code>--no-nasa</code> skips the check-case table).</p>
</main></body></html>
"""


# ----------------------------------------------------------------------------- driver
def build_report(
    *,
    run: bool = True,
    junit: str | Path | None = None,
    out: str | Path | None = None,
    nasa: bool = True,
    root: str | Path = REPO,
    nasa_workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (optionally) the verification suite and the NASA check cases and write the
    report. Returns a summary dict (paths, counts, exit code)."""
    root = Path(root)
    junit = Path(junit) if junit else root / "runs" / "vv" / "junit.xml"
    out = Path(out) if out else root / "docs" / "vv" / "report.html"
    junit, out = junit.resolve(), out.resolve()
    code = None
    if run:
        code = run_tests(junit, root, log)
    if not junit.exists():
        raise FileNotFoundError(f"no JUnit file at {junit} (run without --no-run first)")
    tests, meta = parse_junit(junit)
    nasa_res = run_nasa(nasa_workers, log) if nasa else None
    models = read_model_docs(root / "docs" / "models")
    mc = read_mc_reports(root)
    env = environment(root)
    page = render_html(
        env=env,
        tests=tests,
        junit_meta=meta,
        junit_path=junit,
        models=models,
        nasa=nasa_res,
        mc=mc,
        out_dir=out.parent,
        root=root,
        ran=run,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {
        "out": out,
        "junit": junit,
        "exit_code": code,
        "areas": area_counts(tests),
        "passed": sum(t.outcome == "passed" for t in tests),
        "failed": sum(t.outcome in ("failed", "error") for t in tests),
        "skipped": sum(t.outcome == "skipped" for t in tests),
        "total": len(tests),
        "nasa": nasa_res,
        "models": len(models),
    }
