// Plume mission planner: pick two sites on a world map, a vehicle and a cargo mass; get the
// 3-DOF feasibility (range, propellant margin, apogee, flight time, cargo load) and, with
// `plume viz` running, queue full 6-DOF flights and reliability estimates.
// Static export (GitHub Pages): window.PLUME_STATIC -> feasibility is interpolated from the
// bundled precomputed table (static/planner/capability.json).

import { el, clamp } from './util.js';

const STATIC = window.PLUME_STATIC === true;
const asset = (p) => (STATIC ? `static/planner/${p}` : `/static/planner/${p}`);
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const G_TOL = 0.05; // allowed peak cargo load over the limit (plume.missions.planner.G_TOLERANCE)

const S = {
  launch: null, // {lat, lon, name}
  landing: null,
  pick: 'launch',
  vehicles: [],
  vehicle: 'cargo_hopper',
  cargo: 250,
  sites: [],
  world: null,
  table: null, // bundled capability tables
  curve: null, // [{cargo_kg, max_range_km, ...}] for the current vehicle
  result: null,
  busy: false,
  jobs: [],
  view: { lon: -100, lat: 36, k: 7 }, // centre, pixels per degree of latitude
  hover: null,
};

// ------------------------------------------------------------------------- geodesy
const RAD = Math.PI / 180;

/** WGS-84 geodesic distance (m) and initial azimuth (deg), Vincenty (as the server). */
function geodesic(lat1, lon1, lat2, lon2) {
  const a = 6378137, f = 1 / 298.257223563, b = a * (1 - f);
  const L = (lon2 - lon1) * RAD;
  const U1 = Math.atan((1 - f) * Math.tan(lat1 * RAD)), U2 = Math.atan((1 - f) * Math.tan(lat2 * RAD));
  const sU1 = Math.sin(U1), cU1 = Math.cos(U1), sU2 = Math.sin(U2), cU2 = Math.cos(U2);
  let lam = L, sSig, cSig, sig, cA2, c2m, it = 0, lamP;
  do {
    const sl = Math.sin(lam), cl = Math.cos(lam);
    sSig = Math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl);
    if (sSig === 0) return { dist: 0, az: 0 };
    cSig = sU1 * sU2 + cU1 * cU2 * cl;
    sig = Math.atan2(sSig, cSig);
    const sA = (cU1 * cU2 * sl) / sSig;
    cA2 = 1 - sA * sA;
    c2m = cA2 !== 0 ? cSig - (2 * sU1 * sU2) / cA2 : 0;
    const C = (f / 16) * cA2 * (4 + f * (4 - 3 * cA2));
    lamP = lam;
    lam = L + (1 - C) * f * sA * (sig + C * sSig * (c2m + C * cSig * (-1 + 2 * c2m * c2m)));
  } while (Math.abs(lam - lamP) > 1e-12 && ++it < 200);
  if (it >= 200) return sphereDist(lat1, lon1, lat2, lon2); // nearly antipodal
  const u2 = (cA2 * (a * a - b * b)) / (b * b);
  const A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)));
  const B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)));
  const dSig = B * sSig * (c2m + (B / 4) * (cSig * (-1 + 2 * c2m * c2m) - (B / 6) * c2m * (-3 + 4 * sSig * sSig) * (-3 + 4 * c2m * c2m)));
  const sl = Math.sin(lam), cl = Math.cos(lam);
  const az = Math.atan2(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl) / RAD;
  return { dist: b * A * (sig - dSig), az: (az + 360) % 360 };
}

function sphereDist(lat1, lon1, lat2, lon2) {
  const p1 = lat1 * RAD, p2 = lat2 * RAD, dl = (lon2 - lon1) * RAD;
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  const y = Math.sin(dl) * Math.cos(p2), x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  return { dist: 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(1, h))), az: ((Math.atan2(y, x) / RAD) + 360) % 360 };
}

/** Point at distance d (m) and azimuth az (deg) from (lat, lon) on the mean sphere. */
function destination(lat, lon, az, d) {
  const R = 6371008.8, p1 = lat * RAD, th = az * RAD, c = d / R;
  const p2 = Math.asin(Math.sin(p1) * Math.cos(c) + Math.cos(p1) * Math.sin(c) * Math.cos(th));
  const l2 = lon * RAD + Math.atan2(Math.sin(th) * Math.sin(c) * Math.cos(p1), Math.cos(c) - Math.sin(p1) * Math.sin(p2));
  return [p2 / RAD, ((l2 / RAD + 540) % 360) - 180];
}

/** Great-circle path between two sites (spherical linear interpolation). */
function greatCircle(a, b, n = 96) {
  const v = (s) => [Math.cos(s.lat * RAD) * Math.cos(s.lon * RAD), Math.cos(s.lat * RAD) * Math.sin(s.lon * RAD), Math.sin(s.lat * RAD)];
  const p = v(a), q = v(b);
  const om = Math.acos(clamp(p[0] * q[0] + p[1] * q[1] + p[2] * q[2], -1, 1));
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const s0 = om < 1e-9 ? 1 - t : Math.sin((1 - t) * om) / Math.sin(om);
    const s1 = om < 1e-9 ? t : Math.sin(t * om) / Math.sin(om);
    const x = s0 * p[0] + s1 * q[0], y = s0 * p[1] + s1 * q[1], z = s0 * p[2] + s1 * q[2];
    out.push([Math.atan2(z, Math.hypot(x, y)) / RAD, Math.atan2(y, x) / RAD]);
  }
  return out;
}

// ------------------------------------------------------------------------- formatting
const nf = (x, d = 0) => (x == null || !Number.isFinite(x) ? '–' : x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));
const fmtTime = (s) => (s == null ? '–' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`);
const fmtLat = (x) => `${Math.abs(x).toFixed(3)}°${x >= 0 ? 'N' : 'S'}`;
const fmtLon = (x) => `${Math.abs(x).toFixed(3)}°${x >= 0 ? 'E' : 'W'}`;

let toastTimer = 0;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'show warn';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ''; }, 6000);
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

// ------------------------------------------------------------------------- map
const canvas = $('map');
const ctx = canvas.getContext('2d');
let W = 0, H = 0, DPR = 1, dirty = true;

const C = { // monochrome palette (style.css tokens)
  bg: '#0a0a0a', grat: '#161616', grat2: '#1c1c1c', coast: '#a3a3a3', border: '#6b6b6b', state: '#333333',
  site: '#6b6b6b', siteLbl: '#6b6b6b', t1: '#f2f2f2', t2: '#a3a3a3', t3: '#6b6b6b',
};

function resize() {
  const r = canvas.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = r.width; H = r.height;
  canvas.width = Math.round(W * DPR); canvas.height = Math.round(H * DPR);
  dirty = true;
  if (S.wantFit && W > 0 && H > 0) fit();
}

const kx = () => S.view.k * Math.cos(clamp(S.view.lat, -70, 70) * RAD);
function px(lat, lon) {
  let dl = lon - S.view.lon;
  dl = ((dl + 540) % 360) - 180; // nearest copy of the world
  return [W / 2 + dl * kx(), H / 2 - (lat - S.view.lat) * S.view.k];
}
function unpx(x, y) {
  const lon = S.view.lon + (x - W / 2) / kx();
  return { lat: clamp(S.view.lat - (y - H / 2) / S.view.k, -89.9, 89.9), lon: ((lon + 540) % 360) - 180 };
}

function strokeFlat(flat, wrapCut = 180) {
  // one polyline of [lon, lat, ...]; break where it jumps across the screen seam
  let prevX = null;
  for (let i = 0; i < flat.length; i += 2) {
    const [x, y] = px(flat[i + 1], flat[i]);
    if (prevX === null || Math.abs(x - prevX) > wrapCut * kx()) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    prevX = x;
  }
}

function strokeLatLon(pts) {
  let prevX = null;
  for (const [lat, lon] of pts) {
    const [x, y] = px(lat, lon);
    if (prevX === null || Math.abs(x - prevX) > 180 * kx()) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    prevX = x;
  }
}

function visibleBox() {
  const a = unpx(0, 0), b = unpx(W, H);
  return { latMax: a.lat, latMin: b.lat };
}

function draw() {
  dirty = false;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  const k = S.view.k;
  // graticule
  const step = k > 40 ? 1 : k > 16 ? 5 : k > 6 ? 10 : 30;
  ctx.lineWidth = 1;
  ctx.strokeStyle = C.grat;
  ctx.beginPath();
  const box = visibleBox();
  for (let lat = -90; lat <= 90; lat += step) {
    if (lat < box.latMin - step || lat > box.latMax + step) continue;
    const y = px(lat, S.view.lon)[1];
    ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(W, Math.round(y) + 0.5);
  }
  const lon0 = Math.floor((S.view.lon - W / 2 / kx()) / step) * step;
  for (let lon = lon0; lon <= S.view.lon + W / 2 / kx() + step; lon += step) {
    const x = W / 2 + (lon - S.view.lon) * kx();
    ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, H);
  }
  ctx.stroke();

  if (S.world) {
    if (k > 9) {
      ctx.strokeStyle = C.state; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); for (const f of S.world.states) strokeFlat(f); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.border; ctx.lineWidth = 1;
    ctx.beginPath(); for (const f of S.world.borders) strokeFlat(f); ctx.stroke();
    ctx.strokeStyle = C.coast; ctx.lineWidth = 1;
    ctx.beginPath(); for (const f of S.world.coast) strokeFlat(f); ctx.stroke();
  }

  // labels: selected sites and route first, then bundled sites where there is room
  const placed = [];
  const fits = (r) => r.x >= 0 && r.y >= 0 && r.x + r.w <= W && r.y + r.h <= H && !placed.some((q) => r.x < q.x + q.w && q.x < r.x + r.w && r.y < q.y + q.h && q.y < r.y + r.h);
  const labels = [];
  const want = (text, x, y, color, font, must) => {
    ctx.font = font;
    const tw = ctx.measureText(text).width;
    if (must && x + tw + 4 > W - 8) x = Math.max(8, x - tw - 26); // keep inside the map
    if (must) y = clamp(y, 10, H - 10);
    const r = { x: x - 4, y: y - 7, w: tw + 8, h: 14 };
    if (must || fits(r)) { placed.push(r); labels.push([text, x, y, color, font, must]); return true; }
    return false;
  };
  const mono5 = '500 10px "IBM Plex Mono", monospace', mono4 = '400 10px "IBM Plex Mono", monospace';
  const near = (s, t) => t && Math.abs(s.lat - t.lat) < 1e-3 && Math.abs(s.lon - t.lon) < 1e-3;

  // max-range ring around the launch site for the current cargo
  const maxR = currentMaxRange();
  let route = null;
  if (S.launch && S.landing) route = geodesic(S.launch.lat, S.launch.lon, S.landing.lat, S.landing.lon);
  if (S.launch && maxR > 0) {
    const ring = [];
    for (let az = 0; az <= 360; az += 3) ring.push(destination(S.launch.lat, S.launch.lon, az, maxR * 1000));
    ctx.strokeStyle = C.t2; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
    ctx.beginPath(); strokeLatLon(ring); ctx.stroke(); ctx.setLineDash([]);
    // label where the ring crosses the route direction (else due north)
    const at = destination(S.launch.lat, S.launch.lon, route ? route.az : 0, maxR * 1000);
    const [tx, ty] = px(at[0], at[1]);
    want(`MAX RANGE ${nf(maxR)} KM · ${nf(S.cargo)} KG`, tx + 8, ty + 14, C.t2, mono5, true);
  }
  // route
  if (route) {
    const gc = greatCircle(S.launch, S.landing);
    ctx.strokeStyle = C.t1; ctx.lineWidth = 1.5;
    ctx.setLineDash(S.result && !S.result.feasible ? [4, 3] : []);
    ctx.beginPath(); strokeLatLon(gc); ctx.stroke(); ctx.setLineDash([]);
    const mid = gc[Math.floor(gc.length / 2)];
    const [mx, my] = px(mid[0], mid[1]);
    want(`${nf(route.dist / 1000)} KM`, mx + 8, my + 12, C.t1, mono5, true);
  }
  for (const key of ['launch', 'landing']) {
    const s = S[key];
    if (!s) continue;
    const [x, y] = px(s.lat, s.lon);
    const name = s.name ? ` · ${s.name.toUpperCase()}` : '';
    want(`${key === 'launch' ? 'LAUNCH' : 'LANDING'}${name}`, x + 9, y - 10, C.t1, mono5, true);
  }

  // bundled sites (hidden under a selected site)
  for (const s of S.sites) {
    if (near(s, S.launch) || near(s, S.landing)) continue;
    const [x, y] = px(s.lat, s.lon);
    if (x < -50 || x > W + 50 || y < -20 || y > H + 20) continue;
    const hot = S.hover && S.hover.site === s;
    ctx.strokeStyle = hot ? C.t1 : C.site;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(x) - 2.5, Math.round(y) - 2.5, 5, 5);
    if (k > 10 || hot) want(s.name, x + 8, y, hot ? C.t1 : C.siteLbl, mono4, hot);
  }

  for (const key of ['launch', 'landing']) {
    const s = S[key];
    if (!s) continue;
    const [x, y] = px(s.lat, s.lon);
    ctx.save();
    ctx.translate(Math.round(x) + 0.5, Math.round(y) + 0.5);
    if (key === 'landing') ctx.rotate(Math.PI / 4);
    ctx.fillStyle = 'rgba(10,10,10,0.6)'; ctx.fillRect(-4, -4, 8, 8);
    ctx.strokeStyle = C.t1; ctx.lineWidth = 1; ctx.strokeRect(-4, -4, 8, 8);
    ctx.restore();
  }
  for (const [text, x, y, color, font, plate] of labels) label(text, x, y, color, font, plate);
}

function label(text, x, y, color, font = '500 10px "IBM Plex Mono", monospace', plate = true) {
  ctx.font = font;
  const w = ctx.measureText(text).width;
  if (plate) {
    ctx.fillStyle = 'rgba(10,10,10,0.72)';
    ctx.fillRect(x - 4, y - 7, w + 8, 14);
  }
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y);
}

function frame() {
  if (dirty) draw();
  requestAnimationFrame(frame);
}

// pan / zoom / pick
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, lon: S.view.lon, lat: S.view.lat, moved: false };
});
canvas.addEventListener('pointermove', (e) => {
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (drag.moved) {
      canvas.classList.add('panning');
      S.view.lon = ((drag.lon - dx / kx() + 540) % 360) - 180;
      S.view.lat = clamp(drag.lat + dy / S.view.k, -80, 80);
      dirty = true;
    }
  }
  const ll = unpx(x, y);
  let site = null;
  for (const s of S.sites) {
    const [sx, sy] = px(s.lat, s.lon);
    if (Math.abs(sx - x) < 6 && Math.abs(sy - y) < 6) { site = s; break; }
  }
  if ((S.hover?.site || null) !== site) dirty = true;
  S.hover = { ...ll, site };
  $('map-read').textContent = `${fmtLat(ll.lat)}  ${fmtLon(ll.lon)}${site ? `  ${site.name}` : ''}`;
});
canvas.addEventListener('pointerup', (e) => {
  canvas.classList.remove('panning');
  if (drag && !drag.moved) {
    const r = canvas.getBoundingClientRect();
    const ll = unpx(e.clientX - r.left, e.clientY - r.top);
    const site = S.hover?.site;
    setSite(S.pick, site ? { lat: site.lat, lon: site.lon, name: site.name } : { lat: +ll.lat.toFixed(4), lon: +ll.lon.toFixed(4), name: '' });
    if (S.pick === 'launch' && !S.landing) setPick('landing');
  }
  drag = null;
});
canvas.addEventListener('pointerleave', () => { $('map-read').textContent = ''; S.hover = null; dirty = true; });
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });

function zoomAt(x, y, f) {
  const before = unpx(x, y);
  S.view.k = clamp(S.view.k * f, 1.2, 400);
  const after = unpx(x, y);
  S.view.lon = ((S.view.lon + (before.lon - after.lon) + 540) % 360) - 180;
  S.view.lat = clamp(S.view.lat + (before.lat - after.lat), -80, 80);
  dirty = true;
}
$('zoom-in').onclick = () => zoomAt(W / 2, H / 2, 1.6);
$('zoom-out').onclick = () => zoomAt(W / 2, H / 2, 1 / 1.6);
$('zoom-fit').onclick = () => fit();

function fit() {
  if (!(W > 0 && H > 0)) { S.wantFit = true; return; } // before layout: fit on first resize
  S.wantFit = false;
  const pts = [S.launch, S.landing].filter(Boolean);
  if (!pts.length) return;
  if (pts.length === 1) { S.view.lat = pts[0].lat; S.view.lon = pts[0].lon; S.view.k = Math.max(S.view.k, 8); dirty = true; return; }
  const gc = greatCircle(pts[0], pts[1], 32);
  const lats = gc.map((p) => p[0]);
  // unwrap longitudes around the launch site
  const lons = gc.map((p) => pts[0].lon + (((p[1] - pts[0].lon + 540) % 360) - 180));
  const la0 = Math.min(...lats), la1 = Math.max(...lats), lo0 = Math.min(...lons), lo1 = Math.max(...lons);
  S.view.lat = (la0 + la1) / 2;
  S.view.lon = ((((lo0 + lo1) / 2) + 540) % 360) - 180;
  const c = Math.cos(clamp(S.view.lat, -70, 70) * RAD);
  S.view.k = clamp(Math.min((W * 0.6) / Math.max((lo1 - lo0) * c, 0.5), (H * 0.6) / Math.max(la1 - la0, 0.5)), 1.2, 200);
  dirty = true;
}

// ------------------------------------------------------------------------- form
function setPick(which) {
  S.pick = which;
  for (const b of document.querySelectorAll('#pick-seg button')) b.classList.toggle('on', b.dataset.pick === which);
}
for (const b of document.querySelectorAll('#pick-seg button')) b.onclick = () => setPick(b.dataset.pick);

function setSite(which, site, quiet = false) {
  S[which] = site;
  $(`${which}-name`).value = site?.name || '';
  $(`${which}-lat`).value = site ? site.lat.toFixed(4) : '';
  $(`${which}-lon`).value = site ? site.lon.toFixed(4) : '';
  dirty = true;
  if (!quiet) { syncURL(); scheduleFeas(); }
}

for (const which of ['launch', 'landing']) {
  $(`${which}-name`).addEventListener('change', (e) => {
    const q = e.target.value.trim().toLowerCase();
    const s = S.sites.find((x) => x.name.toLowerCase() === q) || S.sites.find((x) => x.name.toLowerCase().includes(q));
    if (s) { setSite(which, { lat: s.lat, lon: s.lon, name: s.name }); fit(); }
    else if (q) toast(`No bundled site matches "${e.target.value}". Enter latitude and longitude instead.`);
  });
  for (const f of ['lat', 'lon']) {
    $(`${which}-${f}`).addEventListener('change', () => {
      const lat = parseFloat($(`${which}-lat`).value), lon = parseFloat($(`${which}-lon`).value);
      if (Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
        setSite(which, { lat, lon, name: '' });
      } else if ($(`${which}-lat`).value && $(`${which}-lon`).value) toast('Latitude must be within ±90° and longitude within ±180°.');
    });
  }
}
$('swap').onclick = () => {
  const a = S.launch, b = S.landing;
  setSite('launch', b, true); setSite('landing', a);
};

function setCargo(v, from) {
  const vmax = vehicleInfo()?.cargo_max_kg ?? 450;
  S.cargo = clamp(Math.round(v), 0, 20000);
  if (from !== 'input') $('cargo').value = String(S.cargo);
  if (from !== 'slider') $('cargo-slider').value = String(Math.min(S.cargo, vmax));
  dirty = true;
  drawCurve();
  syncURL();
  scheduleFeas();
}
$('cargo').addEventListener('change', (e) => {
  const v = parseFloat(e.target.value);
  if (Number.isFinite(v) && v >= 0) setCargo(v, 'input'); else toast('Cargo must be a mass in kg, 0 or more.');
});
$('cargo-slider').addEventListener('input', (e) => setCargo(+e.target.value, 'slider'));
$('vehicle').addEventListener('change', async (e) => {
  S.vehicle = e.target.value;
  updateVehicle();
  await loadCurve();
  syncURL();
  scheduleFeas();
});

function vehicleInfo() { return S.vehicles.find((v) => v.name === S.vehicle); }
function updateVehicle() {
  const v = vehicleInfo();
  const facts = v && v.dry_mass_kg != null ? ` Dry ${nf(v.dry_mass_kg)} kg, propellant ${nf(v.propellant_kg)} kg, ${nf(v.thrust_kn)} kN, cargo up to ${nf(v.cargo_max_kg)} kg.` : '';
  $('vehicle-desc').textContent = v ? `${v.description}${facts}` : '';
  $('cargo-slider').max = String(v?.cargo_max_kg ?? 450);
}

function syncURL() {
  const p = new URLSearchParams();
  if (S.launch) p.set('from', `${S.launch.lat.toFixed(4)},${S.launch.lon.toFixed(4)}`);
  if (S.landing) p.set('to', `${S.landing.lat.toFixed(4)},${S.landing.lon.toFixed(4)}`);
  p.set('vehicle', S.vehicle);
  p.set('cargo', String(S.cargo));
  history.replaceState(null, '', `${location.pathname}?${p}`);
}

// ------------------------------------------------------------------------- capability
function envelope(curve) {
  const pts = [...curve].sort((a, b) => a.cargo_kg - b.cargo_kg).map((r) => ({ ...r }));
  for (let i = pts.length - 2; i >= 0; i--) pts[i].max_range_km = Math.max(pts[i].max_range_km, pts[i + 1].max_range_km);
  return pts;
}
function maxRangeAt(curve, cargo) {
  if (!curve?.length) return 0;
  const pts = envelope(curve);
  if (cargo <= pts[0].cargo_kg) return pts[0].max_range_km;
  for (let i = 1; i < pts.length; i++) {
    if (cargo <= pts[i].cargo_kg) {
      const a = pts[i - 1], b = pts[i], t = (cargo - a.cargo_kg) / (b.cargo_kg - a.cargo_kg);
      return a.max_range_km + t * (b.max_range_km - a.max_range_km);
    }
  }
  return null; // beyond the tabulated cargo range
}
function cargoForRange(curve, rangeKm) {
  const pts = envelope(curve);
  if (!pts.length || pts[0].max_range_km < rangeKm) return null;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (b.max_range_km < rangeKm && rangeKm <= a.max_range_km) return a.cargo_kg + ((b.cargo_kg - a.cargo_kg) * (a.max_range_km - rangeKm)) / (a.max_range_km - b.max_range_km);
  }
  return pts[pts.length - 1].cargo_kg;
}
const currentMaxRange = () => (S.curve ? maxRangeAt(S.curve, S.cargo) ?? 0 : 0);

async function loadCurve() {
  S.curve = null;
  const tab = S.table?.[S.vehicle];
  if (STATIC || (tab && tab.hash === vehicleInfo()?.hash)) {
    S.curve = tab?.curve || null;
  } else {
    try { S.curve = (await getJSON(`/api/planner/curve?vehicle=${encodeURIComponent(S.vehicle)}`)).curve; } catch (err) { toast(`Range curve: ${err.message}`); }
  }
  dirty = true;
  drawCurve();
}

function drawCurve() {
  const box = $('curve');
  if (!S.curve?.length) { box.replaceChildren(el('p', { class: 'empty' }, 'No capability data for this vehicle.')); return; }
  const w = 328, h = 150, l = 40, r = 8, t = 10, b = 26;
  const pts = envelope(S.curve);
  const cmax = Math.max(pts[pts.length - 1].cargo_kg, S.cargo);
  const route = routeKm();
  const rmax = Math.max(...pts.map((p) => p.max_range_km), route || 0) * 1.1;
  const X = (c) => l + (c / cmax) * (w - l - r), Y = (km) => t + (1 - km / rmax) * (h - t - b);
  const ticksY = niceTicks(0, rmax, 4), ticksX = niceTicks(0, cmax, 4);
  const parts = [`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Maximum range versus cargo mass">`];
  for (const v of ticksY) parts.push(`<line x1="${l}" x2="${w - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1c1c1c"/><text class="ax" x="${l - 6}" y="${Y(v) + 3}" text-anchor="end">${nf(v)}</text>`);
  for (const v of ticksX) parts.push(`<text class="ax" x="${X(v)}" y="${h - b + 12}" text-anchor="middle">${nf(v)}</text>`);
  parts.push(`<line x1="${l}" x2="${w - r}" y1="${h - b}" y2="${h - b}" stroke="#333"/>`);
  parts.push(`<text class="axl" x="${w - r}" y="${h - 3}" text-anchor="end">cargo, kg</text><text class="axl" x="${l - 6}" y="${t - 2}" text-anchor="end">km</text>`);
  parts.push(`<polyline fill="none" stroke="#f2f2f2" stroke-width="1.5" points="${pts.map((p) => `${X(p.cargo_kg).toFixed(1)},${Y(p.max_range_km).toFixed(1)}`).join(' ')}"/>`);
  for (const p of pts) parts.push(`<rect x="${X(p.cargo_kg) - 1.5}" y="${Y(p.max_range_km) - 1.5}" width="3" height="3" fill="#f2f2f2"/>`);
  if (route) parts.push(`<line x1="${l}" x2="${w - r}" y1="${Y(route)}" y2="${Y(route)}" stroke="#a3a3a3" stroke-dasharray="4 3"/><text class="ax" x="${w - r}" y="${Y(route) + 11}" text-anchor="end">route ${nf(route)} km</text>`);
  parts.push(`<line x1="${X(S.cargo)}" x2="${X(S.cargo)}" y1="${t}" y2="${h - b}" stroke="#6b6b6b"/>`);
  if (route) {
    const ok = S.result?.feasible;
    parts.push(`<rect x="${X(S.cargo) - 4}" y="${Y(route) - 4}" width="8" height="8" transform="rotate(45 ${X(S.cargo)} ${Y(route)})" fill="${ok ? '#f2f2f2' : '#0a0a0a'}" stroke="#f2f2f2"/>`);
  }
  parts.push('</svg>');
  box.innerHTML = parts.join('');
}

function niceTicks(a, b, n) {
  const span = b - a, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v);
  return out;
}

const routeKm = () => (S.launch && S.landing ? geodesic(S.launch.lat, S.launch.lon, S.landing.lat, S.landing.lon).dist / 1000 : null);

// ------------------------------------------------------------------------- feasibility
let feasTimer = 0, feasCtl = null;
function scheduleFeas() {
  clearTimeout(feasTimer);
  feasTimer = setTimeout(runFeas, 250);
}

async function runFeas() {
  if (!S.launch || !S.landing) { renderResult(); return; }
  if (STATIC) { S.result = staticFeas(); renderResult(); dirty = true; drawCurve(); return; }
  feasCtl?.abort();
  feasCtl = new AbortController();
  S.busy = true; renderResult();
  try {
    S.result = await getJSON('/api/planner/feasibility', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: feasCtl.signal,
      body: JSON.stringify({ launch: S.launch, landing: S.landing, vehicle: S.vehicle, cargo_kg: S.cargo }),
    });
  } catch (err) {
    if (err.name === 'AbortError') return;
    S.result = null; toast(`Feasibility: ${err.message}`);
  }
  S.busy = false;
  renderResult(); drawCurve(); renderSim(); dirty = true;
}

/** Demo mode: interpolate the precomputed (range, cargo) grid of the bundled table. */
function staticFeas() {
  const tab = S.table?.[S.vehicle];
  const route = routeKm();
  const base = { range_requested_km: route, cargo_kg: S.cargo, vehicle: S.vehicle, source: 'table' };
  if (!tab) return { ...base, feasible: false, reason: 'no precomputed table for this vehicle', result: null };
  const maxR = maxRangeAt(tab.curve, S.cargo);
  base.max_range_km = maxR;
  const cargos = tab.cargos_kg, ranges = tab.ranges_km;
  if (maxR == null) return { ...base, feasible: false, reason: `cargo above the tabulated ${nf(cargos[cargos.length - 1])} kg`, result: null, cargo_for_range_kg: cargoFloor(tab.curve, route) };
  if (route < ranges[0]) return { ...base, feasible: false, reason: `shorter than the tabulated ${nf(ranges[0])} km (run plume viz for short hops)`, result: null };
  if (route > maxR) {
    return { ...base, feasible: false, reason: 'beyond the max range for this cargo', result: null, cargo_for_range_kg: cargoFloor(tab.curve, route) };
  }
  const fi = interpIndex(cargos, S.cargo), fj = interpIndex(ranges, route);
  const val = (field) => bilinear(tab.grid[field], fi, fj);
  const res = {
    range_km: route,
    fuel_margin_kg: val('fuel_margin_kg'),
    apogee_km: val('apogee_km'),
    flight_time_s: val('flight_time_s'),
    peak_cargo_g: val('peak_cargo_g'),
    kick_deg: val('kick_deg'),
    propellant_at_meco_kg: val('propellant_at_meco_kg'),
    cargo_g_limit: tab.cargo_g_limit,
  };
  res.g_limit_exceeded = res.peak_cargo_g != null && res.peak_cargo_g > tab.cargo_g_limit;
  const ok = (res.fuel_margin_kg ?? -1) >= 0 && (res.peak_cargo_g ?? 99) <= tab.cargo_g_limit * (1 + G_TOL);
  return { ...base, feasible: ok, reason: ok ? 'ok' : 'at the edge of the envelope (interpolated)', result: res };
}
const cargoFloor = (curve, km) => { const c = cargoForRange(curve, km); return c == null ? null : Math.max(0, Math.floor(c)); };
function interpIndex(arr, x) {
  if (x <= arr[0]) return 0;
  for (let i = 1; i < arr.length; i++) if (x <= arr[i]) return i - 1 + (x - arr[i - 1]) / (arr[i] - arr[i - 1]);
  return arr.length - 1;
}
function bilinear(grid, fi, fj) {
  const i0 = Math.floor(fi), j0 = Math.floor(fj), ti = fi - i0, tj = fj - j0;
  let s = 0, wsum = 0;
  for (const [di, dj] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const v = grid[i0 + di]?.[j0 + dj];
    const w = (di ? ti : 1 - ti) * (dj ? tj : 1 - tj);
    if (v != null && w > 0) { s += v * w; wsum += w; }
  }
  return wsum > 0 ? s / wsum : null;
}

function renderResult() {
  const box = $('result');
  $('result-src').textContent = STATIC ? 'precomputed table' : S.result?.compute_s != null ? `3-DOF · ${S.result.compute_s.toFixed(1)} s` : '';
  if (!S.launch || !S.landing) { box.replaceChildren(el('p', { class: 'empty pad' }, 'Place a launch and a landing site: click the map or search the site list.')); return; }
  const busy = el('div', { class: `busy${S.busy ? '' : ' idle'}` });
  const R = S.result;
  if (!R) { box.replaceChildren(busy, el('p', { class: 'empty pad' }, S.busy ? 'Flying the 3-DOF mission...' : 'No result.')); return; }
  const route = R.geodesic_km ?? R.range_requested_km;
  const verdict = el('div', { class: `verdict${R.feasible ? ' ok' : ''}` }, el('b', {}, R.feasible ? 'FEASIBLE' : 'NOT FEASIBLE'), el('span', {}, R.reason === 'ok' ? `${nf(route)} km with ${nf(S.cargo)} kg` : R.reason));
  const kids = [busy, verdict];
  if (!R.feasible) {
    const c = R.cargo_for_range_kg;
    let txt;
    if (R.max_range_km != null && route > R.max_range_km) {
      txt = `Max range with ${nf(S.cargo)} kg: <em>${nf(R.max_range_km)} km</em> (route ${nf(route)} km). `;
      txt += c != null ? `The route is within reach with up to <em>${nf(c)} kg</em> of cargo.` : 'Out of reach for this vehicle even without cargo.';
    } else if (c != null) txt = `Within reach with up to <em>${nf(c)} kg</em> of cargo.`;
    if (txt) kids.push(el('div', { class: 'advice', html: txt }));
  }
  const r = R.result;
  const rows = [['Distance (geodesic)', `${nf(route, 1)}`, 'km'], ['Max range, this cargo', nf(R.max_range_km), 'km']];
  if (r) {
    rows.push(
      ['Propellant margin', `${nf(r.fuel_margin_kg)}`, 'kg'],
      ['Apogee', nf(r.apogee_km), 'km'],
      ['Flight time', fmtTime(r.flight_time_s), 'min:s'],
      ['Peak cargo load', nf(r.peak_cargo_g, 2), `g (limit ${nf(r.cargo_g_limit)})`],
    );
    if (r.meco_s != null) rows.push(['Engine cutoff', `T+${nf(r.meco_s)} s, ${nf(r.meco_speed_m_s)} m/s`, '']);
    if (r.kick_deg != null) rows.push(['Pitch kick', nf(r.kick_deg, 1), 'deg']);
    if (r.landing_burn_kg != null) rows.push(['Landing burn (reserve)', nf(r.landing_burn_kg), 'kg']);
  }
  const dl = el('dl', { class: 'kv' });
  for (const [k, v, u] of rows) dl.append(el('dt', {}, k), el('dd', {}, v, u ? el('small', {}, u) : null));
  kids.push(dl);
  if (r?.g_limit_exceeded && R.feasible) kids.push(el('p', { class: 'warn-line' }, `Peak cargo load ${nf(r.peak_cargo_g, 2)} g is slightly above the ${nf(r.cargo_g_limit)} g limit (in the unpowered descent; see the Monte Carlo results).`));
  if (STATIC) kids.push(el('p', { class: 'warn-line' }, 'Demo: values interpolated from a precomputed table. Run plume viz locally for the live planner and full simulations.'));
  else if (R.azimuth_deg != null) kids.push(el('p', { class: 'fine pad' }, `Initial azimuth ${nf(R.azimuth_deg, 1)}°. ${(R.notes || []).slice(1).join(' ')}`));
  box.replaceChildren(...kids);
}

// ------------------------------------------------------------------------- simulation jobs
const SIM = { flightFid: 'fast', relFid: 'fast', runs: 16 };
function seg(options, value, onpick) {
  const s = el('div', { class: 'seg', role: 'group' });
  for (const [v, label] of options) s.append(el('button', { class: v === value ? 'on' : '', onclick: () => onpick(v) }, label));
  return s;
}

function estimateMin(kind) {
  if (kind === 'flight') return SIM.flightFid === 'high' ? '5–8' : '2–3';
  const per = SIM.relFid === 'high' ? 5 : 1.3;
  const n = Math.round(2 + (SIM.runs * per) / 4 + (SIM.relFid === 'high' ? 6 : 2));
  return `about ${n}`;
}

function renderSim() {
  const box = $('sim');
  if (STATIC) {
    box.replaceChildren(el('div', { class: 'demo', html: 'Full 6-DOF flights and reliability estimates run on your machine: <code>uv sync --all-extras</code>, <code>uv run plume viz</code>, then open <code>/planner</code>. This page is a demo with precomputed feasibility.' }));
    return;
  }
  const ready = S.launch && S.landing && S.result;
  const feasible = ready && S.result.feasible;
  const kids = [];
  kids.push(el('div', { class: 'opts' }, el('span', { class: 'lbl' }, 'Full flight'), seg([['fast', 'Fast'], ['high', 'High']], SIM.flightFid, (v) => { SIM.flightFid = v; renderSim(); })));
  kids.push(el('div', { class: 'act' },
    el('span', { class: 'fine' }, `6-DOF flight with replay, ${estimateMin('flight')} min. Flat terrain at sea level.`),
    el('button', { class: 'act-btn', disabled: !feasible, onclick: () => submitJob('flight') }, 'Run flight')));
  kids.push(el('div', { class: 'opts' }, el('span', { class: 'lbl' }, 'Reliability'),
    seg([[16, '16 runs'], [32, '32 runs']], SIM.runs, (v) => { SIM.runs = v; renderSim(); }),
    seg([['fast', 'Fast'], ['high', 'High']], SIM.relFid, (v) => { SIM.relFid = v; renderSim(); })));
  const warn = SIM.relFid === 'high' ? ' High fidelity is the reference but slow.' : ' Fast fidelity under-predicts this vehicle (see README); treat as screening.';
  kids.push(el('div', { class: 'act' },
    el('span', { class: 'fine' }, `Screening Monte Carlo, ${estimateMin('rel')} min on 4 cores, plus a flight-safety export.${warn}`),
    el('button', { class: 'act-btn', disabled: !feasible, onclick: () => submitJob('reliability') }, 'Estimate')));
  if (ready && !feasible) kids.push(el('p', { class: 'fine' }, 'Simulations are enabled for feasible routes.'));
  for (const j of S.jobs) kids.push(jobCard(j));
  box.replaceChildren(...kids);
}

function jobCard(j) {
  const title = j.kind === 'flight' ? '6-DOF flight' : `Reliability · ${j.runs} runs`;
  const card = el('div', { class: 'job' },
    el('div', { class: 'job-head' }, el('b', {}, `${title} · ${j.fidelity}`), el('span', {}, `${j.state} · ${j.id}`)));
  if (j.state === 'running' || j.state === 'queued') {
    card.append(el('div', { class: 'bar' }, el('i', { style: { width: `${Math.round(100 * (j.progress || 0))}%` } })));
  }
  card.append(el('div', { class: 'msg' }, j.error || j.message || j.stage || ''));
  const f = j.result?.flight;
  if (f) {
    const dl = el('dl', { class: 'kv' });
    const rows = [['Outcome', f.reason], ['Landing error', `${nf(f.landing_error_m, 1)} m`], ['Propellant left', `${nf(f.fuel_remaining_kg)} kg`], ['Peak cargo load', `${nf(f.max_cargo_g, 2)} g`], ['Flight time', fmtTime(f.flight_time_s)]];
    for (const [k, v] of rows) dl.append(el('dt', {}, k), el('dd', {}, v));
    card.append(dl);
  }
  const rel = j.result?.reliability;
  if (rel) {
    const dl = el('dl', { class: 'kv' });
    const ci = rel.success_ci95 || [0, 0];
    const rows = [['Mission success', `${nf(100 * rel.success_probability, 1)} %`], ['95 % interval', `${nf(100 * ci[0], 1)}–${nf(100 * ci[1], 1)} %`], ['Vehicle recovered', `${nf(100 * rel.recovery_probability, 1)} %`], ['CEP50', `${nf(rel.cep50_m, 1)} m`]];
    for (const [k, v] of rows) dl.append(el('dt', {}, k), el('dd', {}, v));
    card.append(dl, el('div', { class: 'msg' }, rel.label));
  }
  const links = el('div', { class: 'links' });
  if (f?.replay_id) links.append(el('a', { href: `/?replay=${encodeURIComponent(f.replay_id)}&eng=1`, target: '_blank' }, 'Open replay'));
  for (const [file, label] of [['report.html', 'MC report'], ['safety.html', 'Safety summary'], ['safety.geojson', 'GeoJSON'], ['safety.kml', 'KML']]) {
    if (j.files?.includes(file)) links.append(el('a', { href: `/api/planner/jobs/${j.id}/files/${file}`, target: '_blank' }, label));
  }
  if (links.childNodes.length) card.append(links);
  return card;
}

async function submitJob(kind) {
  try {
    const body = { kind, launch: S.launch, landing: S.landing, vehicle: S.vehicle, cargo_kg: S.cargo, fidelity: kind === 'flight' ? SIM.flightFid : SIM.relFid, runs: SIM.runs };
    const job = await getJSON('/api/planner/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    S.jobs = [job, ...S.jobs.filter((x) => x.id !== job.id)];
    renderSim();
    pollJobs();
  } catch (err) { toast(`Could not start the job: ${err.message}`); }
}

let pollTimer = 0;
async function pollJobs() {
  clearTimeout(pollTimer);
  try { S.jobs = await getJSON('/api/planner/jobs'); } catch { /* server gone */ }
  renderSim();
  if (S.jobs.some((j) => j.state === 'running' || j.state === 'queued')) pollTimer = setTimeout(pollJobs, 2000);
}

// ------------------------------------------------------------------------- key
function renderKey() {
  const line = (dash, w = 1.5, c = C.t1) => `<svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke="${c}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ''}/></svg>`;
  const sq = (rot) => `<svg width="24" height="10"><rect x="8" y="1" width="8" height="8" fill="none" stroke="${C.t1}"${rot ? ' transform="rotate(45 12 5)"' : ''}/></svg>`;
  const site = `<svg width="24" height="10"><rect x="9.5" y="2.5" width="5" height="5" fill="none" stroke="${C.site}"/></svg>`;
  $('map-key').innerHTML = [
    [sq(false), 'Launch site'], [sq(true), 'Landing site'], [line(''), 'Route (great circle)'],
    [line('6 4', 1, C.t2), 'Max range, current cargo'], [site, 'Bundled spaceports and airports'],
  ].map(([s, t]) => `${s}<span>${t}</span>`).join('');
}

// ------------------------------------------------------------------------- boot
async function boot() {
  if (STATIC) { $('mode-tag').hidden = false; $('lnk-viewer').href = './'; }
  resize();
  new ResizeObserver(() => { resize(); }).observe(canvas.parentElement);
  renderKey();
  requestAnimationFrame(frame);
  const [world, sites, table, vehicles] = await Promise.all([
    getJSON(asset('world.json')).catch(() => null),
    getJSON(asset('sites.json')).catch(() => ({ sites: [] })),
    getJSON(asset('capability.json')).catch(() => ({ vehicles: {} })),
    STATIC ? Promise.resolve(null) : getJSON('/api/planner/vehicles').catch(() => null),
  ]);
  S.world = world;
  if (!world) toast('Map data (static/planner/world.json) did not load.');
  S.sites = sites.sites || [];
  S.table = table.vehicles || {};
  S.vehicles = vehicles || Object.values(S.table).map((t) => ({ name: t.vehicle, description: t.description, cargo_nominal_kg: t.cargo_nominal_kg, cargo_max_kg: t.cargo_max_kg, dry_mass_kg: t.dry_mass_kg, propellant_kg: t.propellant_kg, thrust_kn: t.thrust_kn, hash: t.hash }));
  $('site-list').replaceChildren(...S.sites.map((s) => el('option', { value: s.name }, `${s.kind} · ${s.country}`)));
  $('vehicle').replaceChildren(...S.vehicles.map((v) => el('option', { value: v.name }, v.name)));
  const want = params.get('vehicle');
  S.vehicle = S.vehicles.some((v) => v.name === want) ? want : S.vehicles[0]?.name || 'cargo_hopper';
  $('vehicle').value = S.vehicle;
  updateVehicle();
  const parse = (s) => { const [a, b] = (s || '').split(',').map(Number); return Number.isFinite(a) && Number.isFinite(b) ? { lat: a, lon: b, name: '' } : null; };
  const named = (ll) => { if (!ll) return null; const s = S.sites.find((x) => Math.abs(x.lat - ll.lat) < 1e-3 && Math.abs(x.lon - ll.lon) < 1e-3); return s ? { ...ll, name: s.name } : ll; };
  const ref = S.sites.slice(0, 2);
  const a = named(parse(params.get('from'))) || (params.has('from') ? null : ref[0] && { lat: ref[0].lat, lon: ref[0].lon, name: ref[0].name });
  const b = named(parse(params.get('to'))) || (params.has('to') ? null : ref[1] && { lat: ref[1].lat, lon: ref[1].lon, name: ref[1].name });
  setSite('launch', a, true);
  setSite('landing', b, true);
  if (a && !b) setPick('landing');
  S.cargo = params.has('cargo') ? Math.max(0, +params.get('cargo') || 0) : vehicleInfo()?.cargo_nominal_kg ?? 250;
  $('cargo').value = String(S.cargo);
  $('cargo-slider').value = String(S.cargo);
  await loadCurve();
  fit();
  renderSim();
  syncURL();
  runFeas();
  if (!STATIC) pollJobs();
}

boot();
