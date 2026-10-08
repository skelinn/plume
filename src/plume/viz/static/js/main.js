// Plume viewer entry point: UI wiring, playback clock, replay/compare/live orchestration.
//
// URL parameters:
//   replay=<id>|live      replay id from /api/replays ("live" = stream from /ws/live)
//   compare=<idA>,<idB>   side-by-side comparison
//   camera=chase|ground|top|free
//   speed=<x>  t=<seconds>  channel=<live channel>  trail=speed|phase  paused=1
//   eng=1 (engineering overlay)  quality=high|low  info=1
//   vehicle=<id>          multi-vehicle replays: the vehicle to follow (meta.vehicles[].id)
//   capture=1&t0=&t1=&fps=   deterministic capture mode (see capture.js)

import { el, clamp } from './util.js';
import { Replay, fetchReplay } from './replay.js';
import { ViewManager, suggestPartner } from './compare.js';
import { Telemetry, legendItems } from './telemetry.js';
import { renderInfo } from './info.js';
import { Transport } from './transport.js';
import { LiveSession } from './live.js';
import { setupCapture } from './capture.js';
import { MODES } from './cameras.js';
import { STATIC, api } from './api.js';

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
  eng: params.get('eng') === '1',
  info: params.get('info') === '1',
  quality: initialQuality(),
  vehicle: params.get('vehicle') || null,
};

function initialQuality() {
  const q = params.get('quality');
  if (q === 'high' || q === 'low') return q;
  try { const v = localStorage.getItem('plume.quality'); if (v === 'high' || v === 'low') return v; } catch { /* storage unavailable */ }
  return 'high';
}

const manager = new ViewManager($('viewports'), { capture: CAPTURE, quality: S.quality, eng: S.eng });
manager.camMode = S.cam;
const telemetry = new Telemetry($('charts'), $('readouts'), { onSeek: (t) => seek(t) });
const transport = new Transport($('transport'), {
  onToggle: togglePlay, onSeek: (t) => seek(t), onLive: goLive, speeds: SPEEDS, onSpeed: (v) => { S.speed = v; },
});

// ------------------------------------------------------------------ small UI helpers
function showOverlay(html, { spinner = true } = {}) {
  const o = $('overlay');
  o.hidden = false;
  o.replaceChildren(el('div', { class: 'ov-card' }, el('div', { class: 'ov-text', html }), spinner ? el('div', { class: 'progress' }) : null));
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
    sel.append(el('option', { value: r.id, text: `${label}${r.source === 'real' && !/real/i.test(label) ? ' (real)' : ''}` }));
  }
  if (includeLive) sel.append(el('option', { value: LIVE_ID, text: 'Live stream' }));
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
  if (S.eng) p.set('eng', '1');
  if (S.info) p.set('info', '1');
  if (S.vehicle && S.replays[0]?.multi && S.vehicle !== S.replays[0].vehicleId) p.set('vehicle', S.vehicle);
  if (S.live && params.get('channel')) p.set('channel', params.get('channel'));
  history.replaceState(null, '', `?${p}`);
}

function defaultSpeed(duration) {
  if (duration <= 90) return 1;
  return SPEEDS.find((s) => s >= duration / 60) || 50;
}

function setSpeed(v) {
  S.speed = v;
  transport.setSpeed(v);
}

function setEng(on) {
  S.eng = !!on;
  manager.setEng(S.eng);
  $('btn-eng').setAttribute('aria-pressed', String(S.eng));
  syncUrl();
}

function setInfo(on) {
  S.info = !!on && S.replays.length > 0;
  $('info').hidden = !S.info;
  $('btn-info').setAttribute('aria-pressed', String(S.info));
  if (S.info) renderInfo($('info'), S.replays, S.labels || [], () => setInfo(false));
  syncUrl();
}

function setQuality(q) {
  S.quality = q;
  manager.setQuality(q);
  for (const b of document.querySelectorAll('#quality-seg button')) b.classList.toggle('on', b.dataset.q === q);
  try { localStorage.setItem('plume.quality', q); } catch { /* storage unavailable */ }
}

// ------------------------------------------------------------------ multi-vehicle replays
/** The sub-replay of vehicle `id` in replay `r` (or `r` itself). */
function focusOf(r, id) {
  return r.vehicles?.find((v) => v.id === id)?.replay || r;
}

function buildVehicleSeg(replays) {
  const seg = $('veh-seg');
  const list = replays[0]?.multi ? replays[0].followable : [];
  seg.replaceChildren(...list.map((v, i) => el('button', { 'data-veh': v.id, title: `Follow ${v.name}${i < 9 ? ' (V cycles)' : ''}`, text: v.name })));
  $('veh-group').hidden = list.length < 2;
}

function setVehicle(id, { quiet = false } = {}) {
  const r0 = S.replays[0];
  if (!r0) return;
  const list = r0.multi ? r0.followable : [];
  if (!list.some((v) => v.id === id)) id = r0.vehicleId;
  S.vehicle = id;
  manager.setFocus(id);
  telemetry.setReplays(S.replays.map((r) => focusOf(r, id)), S.labels || []);
  for (const b of document.querySelectorAll('#veh-seg button')) b.classList.toggle('on', b.dataset.veh === id);
  if (!quiet) syncUrl();
}

function cycleVehicle() {
  const r0 = S.replays[0];
  if (!r0?.multi) return;
  const list = r0.followable;
  const i = list.findIndex((v) => v.id === S.vehicle);
  setVehicle(list[(i + 1) % list.length].id);
}

// ------------------------------------------------------------------ opening replays
async function install(replays, ids, { keepTime = false } = {}) {
  const token = ++S.token;
  S.replays = replays;
  S.ids = ids;
  const labels = replays.length > 1
    ? replays.map((r) => (replays.every((x) => x.source === replays[0].source) ? r.title : r.source.toUpperCase()))
    : [replays[0].title];
  S.labels = labels;
  const warnings = await manager.show(replays, replays.length > 1 ? labels.map((l, i) => `${String.fromCharCode(65 + i)}  ${l.toUpperCase()}`) : null);
  if (token !== S.token) return;
  manager.setTrailMode(S.trail);
  telemetry.setReplays(replays, labels);
  buildVehicleSeg(replays);
  if (replays[0]?.multi) setVehicle(S.vehicle || replays[0].vehicleId, { quiet: true });
  renderLegend(replays, labels);
  transport.setReplays(replays);
  const tEnd = Math.max(...replays.map((r) => r.tEnd));
  const t0 = Math.min(...replays.map((r) => r.t0));
  if (!keepTime) S.t = params.has('t') && !S.tApplied ? +params.get('t') : t0;
  S.tApplied = true;
  if (!S.speedFromUrl) setSpeed(defaultSpeed(tEnd - t0));
  else setSpeed(+params.get('speed') || 1);
  S.speedFromUrl = false;
  $('btn-compare').setAttribute('aria-pressed', String(replays.length > 1));
  $('field-b').hidden = replays.length < 2;
  setCamera(S.cam);
  setInfo(S.info);
  hideOverlay();
  for (const w of warnings) toast(w);
  syncUrl();
}

function renderLegend(replays, labels) {
  $('legend').replaceChildren(...legendItems(labels.map((l, i) => String.fromCharCode(65 + i))));
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
  if (STATIC) {
    showOverlay('Live streaming needs a local viewer.<br><small>Run <code>uv run plume viz</code> and stream with <code>--live</code>.</small>', { spinner: false });
    return;
  }
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
  transport.setSpeed(1);
  $('cam-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-cam]');
    if (b) setCamera(b.dataset.cam);
  });
  $('veh-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-veh]');
    if (b) setVehicle(b.dataset.veh);
  });
  $('sel-a').addEventListener('change', () => {
    const id = $('sel-a').value;
    if (id === LIVE_ID) { params.set('channel', params.get('channel') || 'default'); startLive(); return; }
    openIds(S.ids.length > 1 ? [id, S.ids[1]] : [id]);
  });
  $('sel-b').addEventListener('change', () => openIds([S.ids[0], $('sel-b').value]));
  $('btn-compare').addEventListener('click', toggleCompare);
  $('btn-trail').addEventListener('click', toggleTrail);
  $('btn-eng').addEventListener('click', () => setEng(!S.eng));
  $('btn-info').addEventListener('click', () => setInfo(!S.info));
  $('quality-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-q]');
    if (b) setQuality(b.dataset.q);
  });
  $('btn-tele').addEventListener('click', toggleTele);
  $('btn-help').addEventListener('click', () => { $('help').hidden = !$('help').hidden; });
  $('help').addEventListener('click', (e) => { if (e.target === $('help') || e.target.closest('[data-close]')) $('help').hidden = true; });
  document.addEventListener('keydown', onKey);
  $('btn-trail').textContent = `Trail ${S.trail}`;
  $('btn-eng').setAttribute('aria-pressed', String(S.eng));
  for (const b of document.querySelectorAll('#quality-seg button')) b.classList.toggle('on', b.dataset.q === S.quality);
}

function toggleTele() {
  const on = document.body.classList.toggle('tele-open');
  $('btn-tele').setAttribute('aria-pressed', String(on));
}

function toggleTrail() {
  S.trail = S.trail === 'speed' ? 'phase' : 'speed';
  $('btn-trail').textContent = `Trail ${S.trail}`;
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
  else if (k === 'v' || k === 'V') cycleVehicle();
  else if (k === 'l' || k === 'L') { if (S.live) goLive(); else startLive(); }
  else if (k === 't' || k === 'T') toggleTrail();
  else if (k === 'e' || k === 'E') setEng(!S.eng);
  else if (k === 'i' || k === 'I') setInfo(!S.info);
  else if (k === 'q' || k === 'Q') setQuality(S.quality === 'high' ? 'low' : 'high');
  else if (k === 'Home') seek(totalRange()[0]);
  else if (k === 'End') seek(totalRange()[1]);
  else if (k === ']' || k === '[') {
    const i = SPEEDS.indexOf(S.speed);
    const j = clamp((i < 0 ? SPEEDS.findIndex((s) => s >= S.speed) : i) + (k === ']' ? 1 : -1), 0, SPEEDS.length - 1);
    setSpeed(SPEEDS[j]);
  } else if (k === 'h' || k === 'H') document.body.classList.toggle('chrome-hidden');
  else if (k === '?' || k === '/') $('help').hidden = !$('help').hidden;
  else if (k === 'Escape') { $('help').hidden = true; if (S.info) setInfo(false); document.body.classList.remove('tele-open'); }
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
    S.list = await (await fetch(api.replays())).json();
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
/**
 * Debug / screenshot helper: free camera looking at a point of the vehicle.  `target` and `cam` are
 * in vehicle mesh space (metres from the hull base; +y = vehicle axis), e.g. closeup([0, 11, 0], [2, 10, 2]).
 */
function closeup(target, cam, fov = 50) {
  setCamera('free');
  const v = manager.primary;
  if (!v?.world) return;
  const w = v.world, q = w.quatW, half = w.L / 2;
  const T = w.axis.clone().set(target[0], target[1] - half, target[2]).applyQuaternion(q);
  const C = w.axis.clone().set(cam[0], cam[1] - half, cam[2]).applyQuaternion(q);
  v.rig.persp.position.copy(C);
  v.rig.controls.target.copy(T);
  v.rig._freeInit = true;
  v.rig.freeFov = fov;
}

window.plume = { S, manager, telemetry, transport, setEng, setInfo, setQuality, seek, closeup, setCamera, setVehicle };

boot();
