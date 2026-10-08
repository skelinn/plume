// Telemetry column: a compact readout table (one value column per replay) and small monochrome
// uPlot charts.  Series are told apart by line style only: replay A solid white, replay B dashed
// grey.  The playhead is a 1 px rule; click/drag on a chart seeks.  With two replays the series
// are overlaid on a common time grid.

import uPlot from 'uplot';
import { bisect, el } from './util.js';
import { splitDistance, splitSpeed } from './hud.js';

const DEG = 180 / Math.PI;

const READOUTS = [
  { k: 'alt', label: 'Altitude', f: (s) => splitDistance(s.alt) },
  { k: 'speed', label: 'Speed', f: (s) => splitSpeed(s.speed) },
  { k: 'vz', label: 'Vert speed', need: 'vel', f: (s) => splitSpeed(s.vz) },
  { k: 'hs', label: 'Horiz speed', need: 'vel', f: (s) => splitSpeed(s.hspeed) },
  { k: 'thr', label: 'Throttle', need: 'throttle', f: (s) => [(100 * s.throttle).toFixed(0), '%'] },
  { k: 'prop', label: 'Propellant', need: 'prop_mass', f: (s) => [s.prop_mass.toFixed(0), 'kg'] },
  { k: 'mass', label: 'Mass', need: 'mass', f: (s) => [s.mass.toFixed(0), 'kg'] },
  { k: 'mach', label: 'Mach', need: 'mach', f: (s) => [s.mach.toFixed(2), ''] },
  { k: 'q', label: 'Dyn pressure', need: 'q_dyn', f: (s) => [(s.q_dyn / 1000).toFixed(2), 'kPa'] },
  { k: 'g', label: 'Load', need: 'g_load', f: (s) => [s.g_load.toFixed(2), 'g'] },
  { k: 'aoa', label: 'Angle of attack', need: 'aoa_total', f: (s) => [(s.aoa_total * DEG).toFixed(1), 'deg'] },
  { k: 'phase', label: 'Phase', need: 'phase', txt: true, f: (s) => [String(s.phase).replaceAll('_', ' '), ''] },
];

const CHARTS = [
  { key: 'alt', label: 'Altitude', unit: 'm', col: 'alt', digits: 0 },
  { key: 'speed', label: 'Speed', unit: 'm/s', col: 'speed', digits: 1 },
  { key: 'thr', label: 'Throttle', unit: '%', col: 'throttle', digits: 0, scale: 100 },
  { key: 'prop', label: 'Propellant', unit: 'kg', col: 'prop_mass', digits: 0 },
  { key: 'g', label: 'Load', unit: 'g', col: 'g_load', digits: 2 },
  { key: 'q', label: 'Dyn pressure', unit: 'kPa', col: 'q_dyn', digits: 2, scale: 1e-3 },
  { key: 'aoa', label: 'Angle of attack', unit: 'deg', col: 'aoa_total', digits: 1, scale: DEG },
];

const MAX_POINTS = 2400;
const GRID_N = 1400;
export const SERIES = [
  { stroke: '#f2f2f2', width: 1.25, dash: null, swatch: '' },
  { stroke: '#a3a3a3', width: 1.25, dash: [5, 3], swatch: '4 3' },
];

export function legendItems(labels) {
  return labels.map((l, i) => el('span', { class: 'lg', html: `<svg width="20" height="6" viewBox="0 0 20 6"><line x1="0" y1="3" x2="20" y2="3" stroke="${SERIES[i].stroke}" stroke-width="1.5" ${SERIES[i].swatch ? `stroke-dasharray="${SERIES[i].swatch}"` : ''}/></svg>` }, l));
}

function interpAt(T, c, t) {
  const n = T.length;
  if (!n || !c || t < T[0] - 1e-9 || t > T[n - 1] + 1e-9) return null;
  const i = Math.min(Math.max(bisect(T, t), 0), n - 1);
  if (i >= n - 1) return c[n - 1];
  const f = (t - T[i]) / (T[i + 1] - T[i] || 1);
  return c[i] + (c[i + 1] - c[i]) * f;
}

export class Telemetry {
  constructor(chartsRoot, readoutsRoot, { onSeek = () => {} } = {}) {
    this.root = chartsRoot;
    this.roRoot = readoutsRoot;
    this.onSeek = onSeek;
    this.charts = [];
    this.replays = [];
    this.labels = [];
    this.t = 0;
    this._lastDraw = 0;
  }

  setReplays(replays, labels) {
    this.dispose();
    this.replays = replays;
    this.labels = labels;
    this._buildReadouts();
    for (const spec of CHARTS) {
      if (replays.some((r) => r.live || r.column(spec.col))) this._addChart(spec);
    }
    this.refresh();
    this.setTime(this.t, true);
  }

  _buildReadouts() {
    const rs = this.replays;
    const table = el('table', { class: 'ro-table' });
    if (rs.length > 1) {
      table.append(el('tr', {}, el('th'), ...rs.map((_, i) => el('th', { text: String.fromCharCode(65 + i) })), el('th')));
    }
    this.ro = [];
    for (const R of READOUTS) {
      if (R.need && !rs.some((r) => r.live || r.has(R.need))) continue;
      const vals = rs.map(() => el('td', { class: `v${R.txt ? ' txt' : ''}`, text: '—' }));
      const unit = el('td', { class: 'u' });
      table.append(el('tr', {}, el('td', { class: 'k', text: R.label }), ...vals, unit));
      this.ro.push({ R, vals, unit, last: [] });
    }
    this.roRoot.replaceChildren(table);
  }

  _addChart(spec) {
    const valueEls = this.replays.map(() => el('b', { text: '—' }));
    const head = el('div', { class: 'chart-head' }, el('span', { class: 'chart-label', text: spec.label }),
      el('span', { class: 'chart-vals' }, valueEls.flatMap((v, i) => [i ? el('i', { text: '/' }) : null, v])),
      el('span', { class: 'chart-unit', text: spec.unit }));
    const body = el('div', { class: 'chart-body' });
    const wrap = el('div', { class: 'chart', 'data-key': spec.key }, head, body);
    this.root.append(wrap);
    const chart = { spec, wrap, body, valueEls, u: null, data: null };
    this.charts.push(chart);
    chart.ro = new ResizeObserver(() => {
      if (chart.u && body.clientWidth > 0) chart.u.setSize({ width: body.clientWidth, height: body.clientHeight });
      else if (!chart.u && body.clientWidth > 0 && chart.data) this._makePlot(chart);
    });
    chart.ro.observe(body);
  }

  _col(r, spec) {
    const c = r.column(spec.col);
    if (!c || !spec.scale) return c;
    r._scaled = r._scaled || new Map();
    const key = `${spec.col}|${spec.scale}`;
    const hit = r._scaled.get(key);
    if (hit && hit.src === c && hit.n === c.length) return hit.col;
    const col = c.map((x) => x * spec.scale);
    r._scaled.set(key, { src: c, n: c.length, col });
    return col;
  }

  _buildData(spec) {
    const rs = this.replays;
    const cols = rs.map((r) => this._col(r, spec));
    if (rs.length === 1) {
      const T = rs[0].frames.t, c = cols[0] || [];
      const stride = Math.max(1, Math.ceil(T.length / MAX_POINTS));
      const xs = [], ys = [];
      for (let i = 0; i < T.length; i += stride) { xs.push(T[i]); ys.push(c[i] ?? null); }
      if ((T.length - 1) % stride) { xs.push(T[T.length - 1]); ys.push(c[T.length - 1] ?? null); }
      return [xs, ys];
    }
    const t0 = Math.min(...rs.map((r) => r.t0)), t1 = Math.max(...rs.map((r) => r.tEnd));
    const xs = Array.from({ length: GRID_N }, (_, i) => t0 + ((t1 - t0) * i) / (GRID_N - 1));
    const ys = rs.map((r, k) => xs.map((x) => (cols[k] ? interpAt(r.frames.t, cols[k], x) : null)));
    return [xs, ...ys];
  }

  refresh() {
    for (const ch of this.charts) {
      const data = this._buildData(ch.spec);
      ch.data = data;
      const ok = data[0].length >= 2 && data[0][data[0].length - 1] > data[0][0];
      if (!ch.u) { if (ok && ch.body.clientWidth > 0) this._makePlot(ch); } else if (ok) ch.u.setData(data);
    }
  }

  _makePlot(ch) {
    const self = this;
    const axisFont = '10px "IBM Plex Mono", ui-monospace, monospace';
    const stroke = '#6b6b6b';
    const grid = { stroke: '#1c1c1c', width: 1 };
    const series = [{}];
    this.replays.forEach((r, i) => {
      const S = SERIES[i] || SERIES[1];
      series.push({ label: this.labels[i], stroke: S.stroke, width: S.width, dash: S.dash || undefined, points: { show: false }, spanGaps: false });
    });
    const w = Math.max(ch.body.clientWidth, 120), h = Math.max(ch.body.clientHeight, 48);
    const fmt = (v) => {
      const a = Math.abs(v);
      if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
      if (a >= 1e4) return `${+(v / 1e3).toFixed(0)}k`;
      if (a >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
      return `${+v.toFixed(a < 1 ? 2 : a < 10 ? 1 : 0)}`;
    };
    ch.u = new uPlot({
      width: w, height: h,
      padding: [4, 4, 0, 0],
      legend: { show: false },
      cursor: { show: false },
      select: { show: false },
      scales: { x: { time: false } },
      axes: [
        { stroke, font: axisFont, size: 16, ticks: { show: false }, grid, gap: 2, values: (u, vals) => vals.map((v) => `${+v.toFixed(0)}`) },
        { stroke, font: axisFont, size: 36, ticks: { show: false }, grid, gap: 4, values: (u, vals) => vals.map(fmt) },
      ],
      series,
      hooks: {
        draw: [
          (u) => {
            const { ctx } = u;
            const x = Math.round(u.valToPos(self.t, 'x', true)) + 0.5;
            const { top, height, left, width } = u.bbox;
            if (x < left || x > left + width) return;
            ctx.save();
            ctx.strokeStyle = 'rgba(242,242,242,0.7)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
            self.replays.forEach((r, i) => {
              const v = interpAt(r.frames.t, self._col(r, ch.spec), self.t);
              if (v == null) return;
              const y = Math.round(u.valToPos(v, 'y', true));
              ctx.fillStyle = i ? '#a3a3a3' : '#f2f2f2';
              ctx.fillRect(x - 2.5, y - 2.5, 5, 5);
            });
            ctx.restore();
          },
        ],
        ready: [
          (u) => {
            ch.ready = true;
            let down = false;
            const seek = (e) => {
              const rect = u.over.getBoundingClientRect();
              self.onSeek(u.posToVal(e.clientX - rect.left, 'x'));
            };
            u.over.style.cursor = 'col-resize';
            u.over.addEventListener('pointerdown', (e) => { down = true; u.over.setPointerCapture(e.pointerId); seek(e); });
            u.over.addEventListener('pointermove', (e) => { if (down) seek(e); });
            u.over.addEventListener('pointerup', () => { down = false; });
          },
        ],
      },
    }, ch.data, ch.body);
  }

  /** Move the playhead and update readouts (throttled). */
  setTime(t, force = false) {
    this.t = t;
    const now = performance.now();
    if (!force && now - this._lastDraw < 33) return;
    this._lastDraw = now;
    const samples = this.replays.map((r) => (r.n ? r.sample(t) : null));
    for (const row of this.ro || []) {
      let unit = '';
      row.vals.forEach((td, i) => {
        const s = samples[i];
        const R = row.R;
        let txt = '—';
        const r = this.replays[i];
        if (s && (!R.need || r.has(R.need) || (r.live && s[R.need] !== undefined))) {
          const [v, u] = R.f(s);
          txt = v; unit = unit || u;
        }
        if (td.textContent !== txt) td.textContent = txt;
      });
      if (row.unit.textContent !== unit) row.unit.textContent = unit;
    }
    for (const ch of this.charts) {
      this.replays.forEach((r, i) => {
        const v = interpAt(r.frames.t, this._col(r, ch.spec), t);
        const txt = v == null ? '—' : v.toFixed(ch.spec.digits);
        if (ch.valueEls[i].textContent !== txt) ch.valueEls[i].textContent = txt;
      });
      try { if (ch.ready) ch.u.redraw(false, false); } catch (e) { console.warn('chart redraw failed', e); }
    }
  }

  dispose() {
    for (const ch of this.charts) { ch.ro.disconnect(); ch.u?.destroy(); ch.wrap.remove(); }
    this.charts = [];
  }
}
