// Replay information panel (toggle I): title, provenance (sim / real, fidelity, physics models),
// vehicle, scene, outcome and Monte Carlo dispersion summary.  One block per replay in compare mode.

import { el } from './util.js';

const nice = (k) => String(k).replaceAll('_', ' ');
function fmt(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(3);
    return String(+v.toFixed(a < 10 ? 4 : a < 1000 ? 2 : 0));
  }
  if (Array.isArray(v)) return v.map(fmt).join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
function kv(rows) {
  const dl = el('dl', { class: 'kv' });
  for (const [k, v] of rows) {
    if (v === undefined) continue;
    dl.append(el('dt', { text: k, title: k }), el('dd', { text: typeof v === 'string' ? v : fmt(v), title: typeof v === 'string' ? v : fmt(v) }));
  }
  return dl;
}
const sec = (title, ...kids) => el('section', { class: 'info-sec' }, el('h3', { text: title }), ...kids);

export function renderInfo(root, replays, labels, onClose) {
  const head = el('header', { class: 'panel-head' }, el('span', { class: 'lbl', text: 'Replay information' }),
    el('button', { class: 'x', text: 'Close', title: 'Close (I)', onclick: onClose }));
  const blocks = [head];
  replays.forEach((r, i) => {
    const m = r.meta, V = m.vehicle || {}, sc = m.scene || {};
    const ab = replays.length > 1 ? el('div', { class: 'info-ab', text: `${String.fromCharCode(65 + i)} · ${labels[i]}` }) : null;
    blocks.push(sec('Replay', ab, el('p', { class: 'info-title', text: r.title }), el('p', { class: 'info-sub', text: r.id }),
      kv([
        ['Source', (m.source || 'sim') === 'real' ? 'Real flight data' : 'Simulation'],
        ['Fidelity', m.fidelity ?? (m.models?.fidelity) ?? undefined],
        ['Controller', m.controller],
        ['Seed', m.seed],
        ['Recorded', m.created ? String(m.created).replace('T', ' ').slice(0, 19) : undefined],
        ['Plume version', m.plume_version],
        ['Frames', r.n],
        ['Duration', `${(r.tEnd - r.t0).toFixed(1)} s`],
      ])));
    const o = m.outcome;
    if (o) {
      blocks.push(sec('Outcome',
        el('div', { class: `result ${o.success ? 'ok' : 'bad'}` }, el('b', { text: o.success ? 'SUCCESS' : 'FAILURE' }), el('span', { text: nice(o.reason || '') })),
        kv(Object.entries(o.metrics || {}).map(([k, v]) => [nice(k), v]))));
    }
    if (m.models && typeof m.models === 'object') {
      blocks.push(sec('Physics models', kv(Object.entries(m.models).map(([k, v]) => [nice(k), typeof v === 'string' ? v : fmt(v)]))));
    }
    const E = V.engine || {}, L = V.legs || {}, G = V.grid_fins;
    blocks.push(sec('Vehicle', kv([
      ['Name', V.name],
      ['Length', V.length !== undefined ? `${fmt(V.length)} m` : undefined],
      ['Diameter', V.diameter !== undefined ? `${fmt(V.diameter)} m` : undefined],
      ['Dry mass', V.dry_mass !== undefined ? `${fmt(V.dry_mass)} kg` : undefined],
      ['Propellant', V.prop_mass_initial !== undefined ? `${fmt(V.prop_mass_initial)} kg` : undefined],
      ['Cargo', V.cargo_mass ? `${fmt(V.cargo_mass)} kg` : undefined],
      ['Max thrust', E.thrust_max ? `${fmt(E.thrust_max / 1000)} kN` : undefined],
      ['Nozzle exit radius', E.nozzle_radius !== undefined ? `${fmt(E.nozzle_radius)} m` : undefined],
      ['Legs', L.count ? `${L.count} · span ${fmt(L.span)} m` : undefined],
      ['Grid fins', G ? `${G.count} · ${fmt(G.span)} × ${fmt(G.chord)} m · ±${fmt((G.max_deflection || 0) * 57.2958)}°` : undefined],
    ])));
    const org = sc.origin;
    blocks.push(sec('Scene', kv([
      ['Frame', sc.frame === 'wgs84' ? 'WGS84 (ellipsoid)' : sc.frame],
      ['Origin', org ? `${fmt(org.lat_deg)}°, ${fmt(org.lon_deg)}°` : undefined],
      ['Origin height', org && org.height !== undefined ? `${fmt(org.height)} m` : undefined],
      ['Earth radius', sc.earth_radius ? `${fmt(sc.earth_radius / 1000)} km` : undefined],
      ['Ground', sc.ground ? (sc.ground.terrain_id ? `terrain · ${sc.ground.terrain_id}` : sc.ground.type) : undefined],
      ['Pads', (sc.pads || []).map((p) => p.name).join(', ') || undefined],
      ['Target radius', sc.target?.radius !== undefined ? `${fmt(sc.target.radius)} m` : undefined],
    ])));
    const D = m.dispersion;
    if (D) {
      blocks.push(sec('Dispersion (Monte Carlo)', kv([
        ['Runs', D.runs],
        ['Probability', D.probability !== undefined ? `${fmt(100 * D.probability)} %` : undefined],
        ['Semi-axes', Array.isArray(D.semi_axes_m) ? `${D.semi_axes_m.map((x) => fmt(x)).join(' × ')} m` : undefined],
        ['Orientation', D.angle_rad !== undefined ? `${fmt(D.angle_rad * 57.2958)}° from east` : undefined],
        ['Points', Array.isArray(D.points) ? D.points.length : undefined],
      ])));
    }
  });
  root.replaceChildren(...blocks);
}
