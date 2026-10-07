// uPlot telemetry charts: altitude, speed, propellant, g-load.
// The playhead is drawn by a hook; click/drag on a plot seeks.  With two replays the series
// are overlaid on a common time grid ("time-aligned on t").

import uPlot from 'uplot';
import { bisect, el } from './util.js';

const CHARTS = [
  { key: 'alt', label: 'ALTITUDE', unit: 'm', col: 'alt', digits: 0 },
  { key: 'speed', label: 'SPEED', unit: 'm/s', col: 'speed', digits: 1 },
  { key: 'prop', label: 'PROPELLANT', unit: 'kg', col: 'prop_mass', digits: 0 },
  { key: 'g', label: 'G-LOAD', unit: 'g', col: 'g_load', digits: 2 },
];

const MAX_POINTS = 3000;
const GRID_N = 1400;

export const SERIES_COLORS = { sim: '#35d6ff', real: '#ffb347', other: '#b69cff' };

export function seriesColors(replays) {
  if (replays.length < 2) return [SERIES_COLORS[replays[0]?.source] || SERIES_COLORS.sim];
  const [a, b] = replays;
  if (a.source !== b.source) return replays.map((r) => SERIES_COLORS[r.source] || SERIES_COLORS.other);
  return [SERIES_COLORS.sim, SERIES_COLORS.other];
}

function interpAt(T, c, t) {
  const n = T.length;
  if (!n || t < T[0] - 1e-9 || t > T[n - 1] + 1e-9) return null;
  const i = Math.min(Math.max(bisect(T, t), 0), n - 1);
  if (i >= n - 1) return c[n - 1];
  const f = (t - T[i]) / (T[i + 1] - T[i] || 1);
  return c[i] + (c[i + 1] - c[i]) * f;
}

export class Telemetry {
  constructor(root, { onSeek = () => {} } = {}) {
    this.root = root;
    this.onSeek = onSeek;
    this.charts = [];
    this.replays = [];
    this.colors = [];
    this.labels = [];
    this.t = 0;
    this._lastDraw = 0;
  }

  /** @param replays Replay[] (1 or 2) @param labels legend text per replay */
  setReplays(replays, labels) {
    this.dispose();
    this.replays = replays;
    this.labels = labels;
    this.colors = seriesColors(replays);
    for (const spec of CHARTS) {
      // include a chart if any replay has the column (live replays may not have received it yet)
      if (replays.some((r) => r.live || r.column(spec.col))) this._addChart(spec);
    }
    this.refresh();
    this.setTime(this.t, true);
  }

  _addChart(spec) {
    const valueEls = this.replays.map((_, i) => el('b', { style: { color: this.colors[i] }, text: '—' }));
    const head = el('div', { class: 'chart-head' }, el('span', { class: 'chart-label', text: spec.label }),
      el('span', { class: 'chart-vals' }, valueEls.flatMap((v, i) => [i ? el('i', { text: '/' }) : null, v])),
      el('span', { class: 'chart-unit', text: spec.unit }));
    const body = el('div', { class: 'chart-body' });
    const wrap = el('div', { class: 'chart', 'data-key': spec.key }, head, body);
    this.root.append(wrap);
    const chart = { spec, wrap, body, valueEls, u: null, data: null, series: [] };
    this.charts.push(chart);
    chart.ro = new ResizeObserver(() => {
      if (chart.u && body.clientWidth > 0) chart.u.setSize({ width: body.clientWidth, height: body.clientHeight });
    });
    chart.ro.observe(body);
  }

  _buildData(spec) {
    const rs = this.replays;
    const cols = rs.map((r) => r.column(spec.col));
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

  /** Rebuild all data (also used by live mode as frames arrive). */
  refresh() {
    for (const ch of this.charts) {
      const data = this._buildData(ch.spec);
      ch.data = data;
      if (!ch.u) {
        // live runs start empty; uPlot needs a non-degenerate x range
        if (data[0].length >= 2 && data[0][data[0].length - 1] > data[0][0]) this._makePlot(ch);
      } else if (data[0].length >= 2 && data[0][data[0].length - 1] > data[0][0]) {
        ch.u.setData(data);
      }
    }
  }

  _makePlot(ch) {
    const self = this;
    const axisFont = '10px "Segoe UI", system-ui, sans-serif';
    const stroke = '#6f7e96';
    const series = [{}];
    this.replays.forEach((r, i) => {
      const color = this.colors[i];
      series.push({
        label: this.labels[i], stroke: color, width: 1.6, points: { show: false },
        fill: i === 0 ? `${color}14` : undefined, spanGaps: false,
      });
    });
    const w = Math.max(ch.body.clientWidth, 200), h = Math.max(ch.body.clientHeight, 60);
    ch.u = new uPlot({
      width: w, height: h,
      padding: [8, 10, 0, 0],
      legend: { show: false },
      cursor: { show: false },
      select: { show: false },
      scales: { x: { time: false } },
      axes: [
        { stroke, font: axisFont, size: 20, ticks: { show: false }, grid: { stroke: 'rgba(255,255,255,0.05)', width: 1 }, gap: 2 },
        {
          stroke, font: axisFont, size: 44, ticks: { show: false }, grid: { stroke: 'rgba(255,255,255,0.05)', width: 1 }, gap: 2,
          values: (u, vals) => vals.map((v) => (Math.abs(v) >= 10000 ? `${+(v / 1000).toFixed(1)}k` : +v.toFixed(2))),
        },
      ],
      series,
      hooks: {
        draw: [
          (u) => {
            const { ctx } = u;
            const x = Math.round(u.valToPos(self.t, 'x', true));
            const { top, height, left, width } = u.bbox;
            if (x < left || x > left + width) return;
            ctx.save();
            ctx.strokeStyle = 'rgba(255,138,48,0.9)';
            ctx.lineWidth = Math.max(1, devicePixelRatio);
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
            self.replays.forEach((r, i) => {
              const v = interpAt(r.frames.t, r.column(ch.spec.col) || [], self.t);
              if (v == null) return;
              const y = u.valToPos(v, 'y', true);
              ctx.fillStyle = self.colors[i];
              ctx.beginPath(); ctx.arc(x, y, 3.2 * devicePixelRatio, 0, Math.PI * 2); ctx.fill();
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
            u.over.style.cursor = 'ew-resize';
            u.over.addEventListener('pointerdown', (e) => { down = true; u.over.setPointerCapture(e.pointerId); seek(e); });
            u.over.addEventListener('pointermove', (e) => { if (down) seek(e); });
            u.over.addEventListener('pointerup', () => { down = false; });
          },
        ],
      },
    }, ch.data, ch.body);
  }

  /** Move the playhead (throttled redraw). */
  setTime(t, force = false) {
    this.t = t;
    const now = performance.now();
    if (!force && now - this._lastDraw < 33) return;
    this._lastDraw = now;
    for (const ch of this.charts) {
      this.replays.forEach((r, i) => {
        const c = r.column(ch.spec.col);
        const v = c ? interpAt(r.frames.t, c, t) : null;
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
