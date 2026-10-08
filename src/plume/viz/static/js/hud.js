// In-viewport plate: source tag + title, mission clock and the key flight readouts, and the outcome
// once the replay reaches its end.  Everything else lives in the telemetry column.

import { el, fmtClock } from './util.js';

const CELLS = [
  { k: 'clock', label: 'Mission time', cls: 'clock' },
  { k: 'alt', label: 'Altitude' },
  { k: 'vz', label: 'Vert speed', need: 'vel' },
  { k: 'hs', label: 'Horiz speed', need: 'vel', cls: 'opt' },
  { k: 'thr', label: 'Throttle', need: 'throttle', cls: 'opt' },
  { k: 'phase', label: 'Phase', need: 'phase', cls: 'phase opt2' },
];

/** value + unit split so units can be styled */
export function splitDistance(m) {
  if (!Number.isFinite(m)) return ['—', ''];
  const a = Math.abs(m);
  if (a >= 10000) return [(m / 1000).toFixed(2), 'km'];
  if (a >= 100) return [m.toFixed(0), 'm'];
  return [m.toFixed(1), 'm'];
}
export function splitSpeed(v) {
  if (!Number.isFinite(v)) return ['—', ''];
  return Math.abs(v) >= 1000 ? [(v / 1000).toFixed(2), 'km/s'] : [v.toFixed(1), 'm/s'];
}

export class Hud {
  constructor(root, { compact = false } = {}) {
    this.root = root;
    this.cells = {};
    this.tag = el('span', { class: 'tag', text: 'SIM' });
    this.ttl = el('span', { class: 'ttl' });
    this.grid = el('div', { class: 'hud-grid' });
    this.badge = el('div', { class: 'hud-badge', hidden: true });
    for (const c of CELLS) {
      const v = el('b', { text: '—' });
      const cell = el('div', { class: `cell ${c.cls || ''}`, 'data-k': c.k }, el('label', { text: c.label }), v);
      this.cells[c.k] = { cell, v, spec: c, last: '' };
      this.grid.append(cell);
    }
    root.append(el('div', { class: 'hud-head' }, this.tag, this.ttl), this.grid, this.badge);
    root.classList.add('hud');
    if (compact) root.classList.add('compact');
    this.replay = null;
  }

  setReplay(replay, tag = null) {
    this.replay = replay;
    this.tag.textContent = tag || replay.source.toUpperCase();
    this.tag.classList.toggle('real', replay.source === 'real');
    this.ttl.textContent = replay.title;
    for (const c of Object.values(this.cells)) {
      const need = c.spec.need;
      c.cell.hidden = !!need && !(replay.live || replay.has(need));
    }
    this.badge.hidden = true;
    this.lastOutcomeKey = '';
  }

  _set(k, val, unit = '') {
    const c = this.cells[k];
    const key = `${val}|${unit}`;
    if (c.last === key) return;
    c.last = key;
    c.v.textContent = val;
    if (unit) c.v.append(el('small', { text: unit }));
  }

  update(s, t) {
    const r = this.replay;
    if (!r || !s) return;
    this._set('clock', fmtClock(t));
    this._set('alt', ...splitDistance(s.alt));
    this._set('vz', ...splitSpeed(s.vz));
    this._set('hs', ...splitSpeed(s.hspeed));
    if (s.throttle !== undefined) this._set('thr', `${Math.round(s.throttle * 100)}`, '%');
    if (s.phase !== undefined) this._set('phase', String(s.phase).replaceAll('_', ' ').toUpperCase());

    const o = r.outcome;
    const atEnd = r.n > 0 && t >= r.tEnd - 1e-6;
    if (o && atEnd) {
      const key = JSON.stringify(o);
      if (key !== this.lastOutcomeKey) {
        this.lastOutcomeKey = key;
        const metrics = Object.entries(o.metrics || {}).slice(0, 4);
        this.badge.className = `hud-badge ${o.success ? 'ok' : 'bad'}`;
        this.badge.replaceChildren(
          el('b', { text: o.success ? 'SUCCESS' : 'FAILURE' }),
          o.reason ? el('span', { text: String(o.reason).replaceAll('_', ' ') }) : null,
          metrics.length ? el('div', { class: 'm' }, metrics.flatMap(([k, v]) => [
            el('i', { text: k.replaceAll('_', ' ') }), el('span', { text: typeof v === 'number' ? String(+v.toFixed(2)) : String(v) }),
          ])) : null,
        );
      }
      this.badge.hidden = false;
    } else {
      this.badge.hidden = true;
    }
  }
}
