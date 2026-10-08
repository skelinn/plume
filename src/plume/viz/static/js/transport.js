// Transport bar: play/pause, mission clock / duration and the latest event, a scrubber with event
// ticks (replay A above the track, B below), the playback-speed selector and the LIVE button.

import { el, fmtClock } from './util.js';

const MAJOR = new Set(['ignition', 'cutoff', 'touchdown', 'landing', 'crash', 'failure', 'staging', 'apogee']);

export class Transport {
  constructor(root, { onToggle, onSeek, onLive, speeds = [1], onSpeed }) {
    this.root = root;
    this.onSeek = onSeek;
    this.t0 = 0;
    this.t1 = 1;
    this.playBtn = el('button', { class: 'play', title: 'Play / pause (Space)', 'aria-label': 'Play or pause', onclick: onToggle });
    this.playBtn.innerHTML = '<svg viewBox="0 0 16 16" width="12" height="12"><path class="p-play" d="M3 1.5v13l11-6.5z"/><path class="p-pause" d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z"/></svg>';
    this.cur = el('b', { text: 'T+00:00.0' });
    this.dur = el('span', { text: '/ 00:00.0' });
    this.evt = el('em', { text: '' });
    this.readout = el('div', { class: 'readout' }, this.cur, this.dur, this.evt);
    this.fill = el('div', { class: 'fill' });
    this.knob = el('div', { class: 'knob' });
    this.markers = el('div', { class: 'markers' });
    this.track = el('div', { class: 'track' }, this.fill, this.markers, this.knob);
    this.scrub = el('div', { class: 'scrub', role: 'slider', 'aria-label': 'Timeline', tabindex: '0' }, this.track);
    this.speedSel = el('select', { 'aria-label': 'Playback speed' });
    for (const s of speeds) this.speedSel.append(el('option', { value: s, text: `${s}×` }));
    this.speedSel.addEventListener('change', () => onSpeed?.(+this.speedSel.value));
    this.speed = el('label', { class: 'speed' }, el('span', { class: 'lbl', text: 'Speed' }), this.speedSel);
    this.liveBtn = el('button', { class: 'live-btn', hidden: true, title: 'Follow live (L)', onclick: onLive, text: 'LIVE' });
    root.append(this.playBtn, this.readout, this.scrub, this.speed, this.liveBtn);

    let down = false;
    const seek = (e) => {
      const r = this.track.getBoundingClientRect();
      const f = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
      this.onSeek(this.t0 + f * (this.t1 - this.t0));
    };
    this.scrub.addEventListener('pointerdown', (e) => { down = true; this.scrub.setPointerCapture(e.pointerId); seek(e); });
    this.scrub.addEventListener('pointermove', (e) => { if (down) seek(e); });
    this.scrub.addEventListener('pointerup', () => { down = false; });
  }

  setSpeed(v) {
    if (![...this.speedSel.options].some((o) => +o.value === v)) this.speedSel.append(el('option', { value: v, text: `${v}×` }));
    this.speedSel.value = String(v);
  }

  setReplays(replays) {
    this.replays = replays;
    this.refreshRange();
    this.markers.replaceChildren();
    this.events = [];
    replays.forEach((r, k) => {
      for (const ev of r.events || []) {
        const m = el('i', { class: `ev ev-${ev.type} k${k}${MAJOR.has(ev.type) ? ' major' : ''}`, title: `${ev.label || ev.type}  ${fmtClock(ev.t)}` });
        m.dataset.t = ev.t;
        m.addEventListener('pointerdown', (e) => { e.stopPropagation(); this.onSeek(ev.t); });
        this.markers.append(m);
        if (k === 0) this.events.push(ev);
      }
    });
    this.events.sort((a, b) => a.t - b.t);
    this.layoutMarkers();
  }

  refreshRange() {
    const rs = this.replays || [];
    this.t0 = rs.length ? Math.min(...rs.map((r) => r.t0)) : 0;
    this.t1 = rs.length ? Math.max(...rs.map((r) => r.tEnd)) : 1;
    if (this.t1 <= this.t0) this.t1 = this.t0 + 1;
    const d = this.t1 - this.t0;
    const txt = `/ ${String(Math.floor(d / 60)).padStart(2, '0')}:${(d % 60).toFixed(1).padStart(4, '0')}`;
    if (this.dur.textContent !== txt) this.dur.textContent = txt;
  }

  layoutMarkers() {
    for (const m of this.markers.children) m.style.left = `${(100 * (+m.dataset.t - this.t0)) / (this.t1 - this.t0)}%`;
  }

  setLive(on, following) {
    this.liveBtn.hidden = !on;
    this.liveBtn.classList.toggle('on', !!following);
  }

  update(t, playing) {
    const f = Math.min(Math.max((t - this.t0) / (this.t1 - this.t0), 0), 1);
    this.fill.style.width = `${f * 100}%`;
    this.knob.style.left = `${f * 100}%`;
    const txt = fmtClock(t);
    if (this.cur.textContent !== txt) this.cur.textContent = txt;
    let last = '';
    for (const ev of this.events || []) { if (ev.t <= t + 1e-6) last = ev.label || ev.type; else break; }
    if (this.evt.textContent !== last) this.evt.textContent = last;
    this.root.classList.toggle('playing', playing);
    this.scrub.setAttribute('aria-valuenow', t.toFixed(1));
  }
}
