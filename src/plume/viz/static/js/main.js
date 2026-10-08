// Plume viewer entry point: UI wiring, playback clock, replay/compare/live orchestration.
//
// URL parameters:
//   replay=<id>|live      replay id from /api/replays ("live" = stream from /ws/live)
//   compare=<idA>,<idB>   side-by-side comparison
//   camera=chase|ground|top|free
//   speed=<x>  t=<seconds>  channel=<live channel>  trail=speed|phase  paused=1
//   capture=1&t0=&t1=&fps=   deterministic capture mode (see capture.js)

import { el, clamp } from './util.js';
import { Replay, fetchReplay } from './replay.js';
import { ViewManager, suggestPartner } from './compare.js';
import { Telemetry, seriesColors } from './telemetry.js';
import { Transport } from './transport.js';
import { LiveSession } from './live.js';
import { setupCapture } from './capture.js';
import { MODES } from './cameras.js';

const params = new URLSearchParams(location.search);
const CAPTURE = params.get('capture') === '1';
document.body.classList.toggle('capture', CAPTURE);
const $ = (id) => document.getElementById(id);
const SPEEDS = [0.25, 0.5, 1, 2, 5, 10, 20, 50];
const LIVE_ID = '__live__';

const S = {
  t: 0,
  playing: !CAPTURE && params.get('paused') !== '1',
  speed: 1,
  speedFromUrl: params.has('speed'),
  cam: MODES.includes(params.get('camera')) ? params.get('camera') : 'chase',
  trail: params.get('trail') === 'phase' ? 'phase' : 'speed',
  list: [],
  replays: [],
  ids: [],
  live: null, // {session, follow, status}
  token: 0,
};

const manager = new ViewManager($('viewports'), { capture: CAPTURE });
manager.camMode = S.cam;
const telemetry = new Telemetry($('charts'), { onSeek: (t) => seek(t) });
const transport = new Transport($('transport'), { onToggle: togglePlay, onSeek: (t) => seek(t), onLive: goLive });

// ------------------------------------------------------------------ small UI helpers
function showOverlay(html, { spinner = true } = {}) {
  const o = $('overlay');
  o.hidden = false;
  o.replaceChildren(el('div', { class: 'ov-card' }, spinner ? el('div', { class: 'spinner' }) : null, el('div', { class: 'ov-text', html })));
}
const hideOverlay = () => { $('overlay').hidden = true; };
let toastTimer = 0;
function toast(msg, kind = 'warn') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ''; }, 6000);
}

function fillSelect(sel, selected, includeLive) {
  sel.replaceChildren();
  const titles = new Map();
  for (const r of S.list) titles.set(r.title, (titles.get(r.title) || 0) + 1);
  for (const r of S.list) {
    const label = titles.get(r.title) > 1 ? `${r.title} — ${r.id}` : r.title;
    sel.append(el('option', { value: r.id, text: `${r.source === 'real' ? '◆ ' : ''}${label}` }));
  }
  if (includeLive) sel.append(el('option', { value: LIVE_ID, text: '● Live stream' }));
  if (selected) sel.value = selected;
}

function setCamera(mode) {
  S.cam = mode;
  manager.setCamera(mode);
  for (const b of document.querySelectorAll('#cam-seg button')) b.classList.toggle('on', b.dataset.cam === mode);
  syncUrl();
}

function syncUrl() {
  if (CAPTURE) return;
  const p = new URLSearchParams();
  if (S.live) p.set('replay', 'live');
  else if (S.ids.length > 1) p.set('compare', S.ids.join(','));
  else if (S.ids.length) p.set('replay', S.ids[0]);
  p.set('camera', S.cam);
  if (S.trail !== 'speed') p.set('trail', S.trail);
  if (S.live && params.get('channel')) p.set('channel', params.get('channel'));
  history.replaceState(null, '', `?${p}`);
}

function defaultSpeed(duration) {
  if (duration <= 90) return 1;
  return SPEEDS.find((s) => s >= duration / 60) || 50;
}

function setSpeed(v) {
  S.speed = v;
  const sel = $('sel-speed');
  if (![...sel.options].some((o) => +o.value === v)) {
    sel.append(el('option', { value: v, text: `${v}×` }));
  }
  sel.value = String(v);
}

// ------------------------------------------------------------------ opening replays
async function install(replays, ids, { keepTime = false } = {}) {
  const token = ++S.token;
  S.replays = replays;
  S.ids = ids;
  const labels = replays.length > 1
    ? replays.map((r) => (replays.every((x) => x.source === replays[0].source) ? r.title : r.source.toUpperCase()))
    : [replays[0].title];
  const warnings = await manager.show(replays, replays.length > 1 ? labels.map((l) => l.toUpperCase()) : null);
  if (token !== S.token) return;
  manager.setTrailMode(S.trail);
  telemetry.setReplays(replays, labels);
  renderLegend(replays, labels);
  transport.setReplays(replays);
  const tEnd = Math.max(...replays.map((r) => r.tEnd));
  const t0 = Math.min(...replays.map((r) => r.t0));
  if (!keepTime) S.t = params.has('t') && !S.tApplied ? +params.get('t') : t0;
  S.tApplied = true;
  if (!S.speedFromUrl) setSpeed(defaultSpeed(tEnd - t0));
  else setSpeed(+params.get('speed') || 1);
  S.speedFromUrl = false;
  $('btn-compare').classList.toggle('on', replays.length > 1);
  $('field-b').hidden = replays.length < 2;
  setCamera(S.cam);
  hideOverlay();
  for (const w of warnings) toast(w);
  syncUrl();
}

function renderLegend(replays, labels) {
  const colors = seriesColors(replays);
  $('legend').replaceChildren(...replays.map((r, i) => el('span', { class: 'lg' }, el('i', { style: { background: colors[i] } }), labels[i])));
  $('legend').hidden = replays.length < 2;
}

async function openIds(ids, opts) {
  stopLive();
  showOverlay(`Loading <b>${ids.join('</b> and <b>')}</b>…`);
  try {
    const replays = await Promise.all(ids.map(fetchReplay));
    $('sel-a').value = ids[0];
    if (ids[1]) $('sel-b').value = ids[1];
    await install(replays, ids, opts);
  } catch (e) {
    console.error(e);
    showOverlay(`Could not load replay: ${e.message}`, { spinner: false });
  }
}

// ------------------------------------------------------------------ live
function stopLive() {
  if (S.live) { S.live.session.close(); S.live = null; }
  transport.setLive(false);
}

async function startLive() {
  stopLive();
  const channel = params.get('channel') || 'default';
  const live = (S.live = { follow: true, status: 'connecting', lastRefresh: 0, channel });
  $('sel-a').value = LIVE_ID;
  transport.setLive(true, true);
  showOverlay(`Waiting for a simulation…<br><small>POST frames to <code>/api/live/${channel}</code> (see <code>plume.viz.live.LiveStreamer</code>)</small>`);
  live.session = new LiveSession(channel, {
    onStatus: (st) => {
      live.status = st;
      if (st === 'offline') showOverlay('Live connection lost — retrying…');
      else if (st === 'idle' && !S.replays.length) showOverlay(`Waiting for a simulation…<br><small>POST frames to <code>/api/live/${channel}</code> (see <code>plume.viz.live.LiveStreamer</code>)</small>`);
    },
    onMeta: async (meta) => {
      const r = new Replay({ meta, frames: { t: [], pos: [], quat: [] }, events: [] }, 'live');
      r.live = true;
      live.replay = r;
      S.playing = true;
      await install([r], [LIVE_ID]);
      $('sel-a').value = LIVE_ID;
      if (S.live === live) live.follow = true;
      hideOverlay();
    },
    onFrames: (frames) => {
      const r = live.replay;
      if (!r) return;
      r.append(frames);
      for (const v of manager.views) v.world?.syncTrail();
      transport.refreshRange();
      transport.layoutMarkers();
    },
    onEnd: (outcome) => {
      const r = live.replay;
      if (!r) return;
      r.meta.outcome = outcome || r.meta.outcome;
      telemetry.refresh();
      toast(outcome?.success ? `Live run finished: ${outcome.reason || 'success'}` : `Live run finished${outcome?.reason ? `: ${outcome.reason}` : ''}`, 'info');
    },
  });
  live.session.connect();
  syncUrl();
}

function goLive() {
  if (!S.live) return;
  S.live.follow = true;
  S.playing = true;
  if (S.live.replay) S.t = S.live.replay.tEnd;
}

// ------------------------------------------------------------------ playback
function totalRange() {
  if (!S.replays.length) return [0, 1];
  return [Math.min(...S.replays.map((r) => r.t0)), Math.max(...S.replays.map((r) => r.tEnd))];
}

function seek(t) {
  const [a, b] = totalRange();
  S.t = clamp(t, a, b);
  if (S.live) S.live.follow = false;
}

function togglePlay() {
  const [, end] = totalRange();
  if (!S.playing && S.t >= end - 1e-6 && !S.live) S.t = totalRange()[0];
  S.playing = !S.playing;
  if (S.live && S.playing) S.live.follow = S.t >= end - 0.5;
}

function stepFrame(dir, big) {
  const r = S.replays[0];
  if (!r || r.n < 2) return;
  const dt = big ? 1 : (r.tEnd - r.t0) / (r.n - 1);
  S.playing = false;
  seek(S.t + dir * dt);
}

let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!S.replays.length) return;
  const [t0, t1] = totalRange();
  if (S.playing) {
    S.t += dt * S.speed;
    if (S.live) {
      if (S.live.follow && t1 - S.t > 2) S.t = t1 - 0.3; // fell behind the stream: catch up
      if (S.t > t1) S.t = t1;
    } else if (S.t >= t1) {
      S.t = t1;
      S.playing = false;
    }
  }
  S.t = clamp(S.t, t0, Math.max(t1, t0));
  manager.render(S.t, now / 1000);
  telemetry.setTime(S.t);
  transport.update(S.t, S.playing);
  if (S.live) {
    transport.setLive(true, S.live.follow);
    if (now - S.live.lastRefresh > 500 && S.live.replay) { S.live.lastRefresh = now; telemetry.refresh(); }
  }
}

// ------------------------------------------------------------------ wiring
function initUi() {
  const speedSel = $('sel-speed');
  for (const s of SPEEDS) speedSel.append(el('option', { value: s, text: `${s}×` }));
  speedSel.value = '1';
  speedSel.addEventListener('change', () => { S.speed = +speedSel.value; });
  $('cam-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-cam]');
    if (b) setCamera(b.dataset.cam);
  });
  $('sel-a').addEventListener('change', () => {
    const id = $('sel-a').value;
    if (id === LIVE_ID) { params.set('channel', params.get('channel') || 'default'); startLive(); return; }
    openIds(S.ids.length > 1 ? [id, S.ids[1]] : [id]);
  });
  $('sel-b').addEventListener('change', () => openIds([S.ids[0], $('sel-b').value]));
  $('btn-compare').addEventListener('click', toggleCompare);
  $('btn-trail').addEventListener('click', toggleTrail);
  $('btn-tele').addEventListener('click', () => document.body.classList.toggle('tele-open'));
  $('btn-help').addEventListener('click', () => { $('help').hidden = !$('help').hidden; });
  $('help').addEventListener('click', () => { $('help').hidden = true; });
  document.addEventListener('keydown', onKey);
  $('btn-trail').textContent = `Trail: ${S.trail}`;
}

function toggleTrail() {
  S.trail = S.trail === 'speed' ? 'phase' : 'speed';
  $('btn-trail').textContent = `Trail: ${S.trail}`;
  manager.setTrailMode(S.trail);
  syncUrl();
}

function toggleCompare() {
  if (S.live || !S.ids.length) { toast('Comparison needs recorded replays.', 'info'); return; }
  if (S.ids.length > 1) { openIds([S.ids[0]], { keepTime: true }); return; }
  const a = S.list.find((r) => r.id === S.ids[0]);
  const b = a && suggestPartner(a, S.list);
  if (!b) { toast('No second replay available to compare.', 'info'); return; }
  openIds([S.ids[0], b.id], { keepTime: true });
}

function onKey(e) {
  if (e.target.closest?.('select, input, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); togglePlay(); }
  else if (k === 'ArrowRight') { e.preventDefault(); stepFrame(1, e.shiftKey); }
  else if (k === 'ArrowLeft') { e.preventDefault(); stepFrame(-1, e.shiftKey); }
  else if (k === '1') setCamera('chase');
  else if (k === '2') setCamera('ground');
  else if (k === '3') setCamera('top');
  else if (k === '4') setCamera('free');
  else if (k === 'c' || k === 'C') toggleCompare();
  else if (k === 'l' || k === 'L') { if (S.live) goLive(); else startLive(); }
  else if (k === 't' || k === 'T') toggleTrail();
  else if (k === 'Home') seek(totalRange()[0]);
  else if (k === 'End') seek(totalRange()[1]);
  else if (k === ']' || k === '[') {
    const i = SPEEDS.indexOf(S.speed);
    const j = clamp((i < 0 ? SPEEDS.findIndex((s) => s >= S.speed) : i) + (k === ']' ? 1 : -1), 0, SPEEDS.length - 1);
    setSpeed(SPEEDS[j]);
  } else if (k === 'h' || k === 'H') document.body.classList.toggle('chrome-hidden');
  else if (k === '?' || k === '/') $('help').hidden = !$('help').hidden;
  else if (k === 'Escape') $('help').hidden = true;
}

// ------------------------------------------------------------------ capture hooks
const app = {
  get replays() { return S.replays; },
  canvas: () => manager.primary?.canvas,
  renderAt: (t, anim) => { S.t = t; manager.render(t, anim); },
  async openForCapture() {
    // ?replay=<id> for one flight, or ?compare=<idA>,<idB> for side-by-side capture
    const cmp = params.get('compare');
    const ids = cmp ? cmp.split(',').slice(0, 2) : [params.get('replay')];
    if (!ids[0]) throw new Error('capture mode needs ?replay=<id> or ?compare=<a>,<b>');
    const replays = await Promise.all(ids.map((id) => fetchReplay(id)));
    manager.camMode = S.cam;
    await install(replays, ids);
    S.playing = false;
    hideOverlay();
  },
};

async function boot() {
  initUi();
  if (CAPTURE) {
    setupCapture(app, params);
    return;
  }
  setSpeed(1);
  try {
    S.list = await (await fetch('/api/replays')).json();
  } catch (e) {
    showOverlay(`Could not reach the Plume server: ${e.message}`, { spinner: false });
    return;
  }
  fillSelect($('sel-a'), null, true);
  fillSelect($('sel-b'), null, false);
  requestAnimationFrame(tick);
  let ids = [];
  if (params.get('compare')) ids = params.get('compare').split(',').filter(Boolean).slice(0, 2);
  else if (params.get('replay') && params.get('replay') !== 'live') ids = [params.get('replay')];
  if (params.get('replay') === 'live') { await startLive(); return; }
  if (!ids.length && S.list.length) ids = [S.list[0].id];
  if (!ids.length) {
    showOverlay('No replays found.<br><small>Record one with <code>plume</code> or start a live stream (press <b>L</b>).</small>', { spinner: false });
    return;
  }
  await openIds(ids);
}

// handy for debugging from the console / test scripts
window.plume = { S, manager, telemetry, transport };

boot();
