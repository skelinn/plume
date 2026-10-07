// Big-number HUD: T+, altitude, speeds, throttle, fuel, Mach, q, phase + outcome badge.

import { el, fmtClock, fmtDistance, fmtSpeed, fmtFixed } from './util.js';

const CELLS = [
  { k: 'alt', label: 'ALTITUDE', big: true, need: null },
  { k: 'vz', label: 'V-SPEED', big: true, need: 'vel' },
  { k: 'hs', label: 'H-SPEED', big: true, need: 'vel' },
  { k: 'thr', label: 'THROTTLE', need: 'throttle' },
  { k: 'fuel', label: 'FUEL', need: 'prop_mass' },
  { k: 'mach', label: 'MACH', need: 'mach' },
  { k: 'q', label: 'Q', need: 'q_dyn' },
  { k: 'g', label: 'LOAD', need: 'g_load' },
  { k: 'phase', label: 'PHASE', need: 'phase', wide: true },
];

export class Hud {
  constructor(root, { compact = false } = {}) {
    this.root = root;
    this.compact = compact;
    this.cells = {};
    this.clock = el('div', { class: 'hud-clock', text: 'T+00:00.0' });
    this.title = el('div', { class: 'hud-title' });
    this.grid = el('div', { class: 'hud-grid' });
    this.badge = el('div', { class: 'hud-badge', hidden: true });
    for (const c of CELLS) {
      const v = el('b', { text: '—' });
      const cell = el('div', { class: `cell${c.big ? ' big' : ''}${c.wide ? ' wide' : ''}`, 'data-k': c.k }, el('label', { text: c.label }), v);
      this.cells[c.k] = { cell, v, spec: c, last: '' };
      this.grid.append(cell);
    }
    root.append(el('div', { class: 'hud-head' }, this.title, this.clock), this.grid, this.badge);
    root.classList.add('hud');
    if (compact) root.classList.add('compact');
    this.replay = null;
  }

  setReplay(replay, tag = null) {
    this.replay = replay;
    this.title.replaceChildren(
      el('span', { class: `src src-${replay.source}`, text: tag || replay.source.toUpperCase() }),
      el('span', { class: 'ttl', text: replay.title }),
    );
    for (const c of Object.values(this.cells)) {
      const need = c.spec.need;
      c.cell.hidden = !!need && !(replay.live || replay.has(need));
    }
    this.prop0 = replay.propInitial();
    this.badge.hidden = true;
    this.lastOutcomeKey = '';
  }

  _set(k, text) {
    const c = this.cells[k];
    if (c.last !== text) { c.v.textContent = text; c.last = text; }
  }

  update(s, t) {
    const r = this.replay;
    if (!r || !s) return;
    const txt = fmtClock(t);
    if (this.clock.textContent !== txt) this.clock.textContent = txt;
    this._set('alt', fmtDistance(s.alt));
    this._set('vz', fmtSpeed(s.vz));
    this._set('hs', fmtSpeed(s.hspeed));
    if (s.throttle !== undefined) this._set('thr', `${Math.round(s.throttle * 100)}%`);
    if (s.prop_mass !== undefined) {
      this.prop0 = Math.max(this.prop0 || 0, s.prop_mass); // never report > 100 % (meta may be stale)
      this._set('fuel', this.prop0 > 0 ? `${Math.max(0, (100 * s.prop_mass) / this.prop0).toFixed(0)}%` : `${s.prop_mass.toFixed(0)} kg`);
    }
    if (s.mach !== undefined) this._set('mach', fmtFixed(s.mach, 2));
    if (s.q_dyn !== undefined) this._set('q', `${fmtFixed(s.q_dyn / 1000, 1)} kPa`);
    if (s.g_load !== undefined) this._set('g', `${fmtFixed(s.g_load, 2)} g`);
    if (s.phase !== undefined) this._set('phase', String(s.phase).replaceAll('_', ' ').toUpperCase());

    const o = r.outcome;
    const atEnd = r.n > 0 && t >= r.tEnd - 1e-6;
    if (o && atEnd) {
      const key = JSON.stringify(o);
      if (key !== this.lastOutcomeKey) {
        this.lastOutcomeKey = key;
        const metrics = Object.entries(o.metrics || {}).slice(0, 4).map(([k, v]) => `${k.replaceAll('_', ' ')} ${typeof v === 'number' ? +v.toFixed(2) : v}`);
        this.badge.className = `hud-badge ${o.success ? 'ok' : 'bad'}`;
        this.badge.replaceChildren(
          el('b', { text: o.success ? 'SUCCESS' : 'FAILED' }),
          el('span', { text: String(o.reason || '').replaceAll('_', ' ') }),
          metrics.length ? el('small', { text: metrics.join(' · ') }) : null,
        );
      }
      this.badge.hidden = false;
    } else {
      this.badge.hidden = true;
    }
  }
}
