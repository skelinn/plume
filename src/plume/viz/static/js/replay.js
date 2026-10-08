// Replay data model: columnar frames, interpolation, derived quantities.

import * as THREE from 'three';
import { bisect, clamp } from './util.js';

const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();

export class Replay {
  /** @param data {meta, frames, events}  @param id replay id (string) */
  constructor(data, id = '') {
    this.id = id;
    this.meta = data.meta || {};
    this.meta.vehicle = this.meta.vehicle || { name: 'vehicle', length: 5, diameter: 0.5 };
    this.meta.scene = this.meta.scene || { frame: 'flat', ground: { type: 'plane' } };
    this.events = data.events || [];
    this.frames = {};
    this.live = false;
    this.version = 0; // bumped on append
    const f = data.frames || {};
    for (const k of Object.keys(f)) this.frames[k] = f[k];
    for (const k of ['t', 'pos', 'quat']) this.frames[k] = this.frames[k] || [];
    const sc = this.meta.scene;
    this.spherical = (sc.frame === 'spherical' || sc.frame === 'wgs84') && sc.earth_radius > 0;
    this.R = this.spherical ? sc.earth_radius : 0;
    this._derivedVel = !this.frames.vel;
    if (this._derivedVel) this.frames.vel = [];
    this._fillVel(0);
    this._dn = 0;
    this._derive();
  }

  /**
   * Derived per-frame columns used by the 3-D models (computed incrementally, so live runs work):
   *   _soot  accumulated "dirty" burn time (s): throttle-weighted, mostly from burns flown into the
   *          vehicle's own plume (retro-propulsion) or near the ground
   *   _glow  nozzle interior heat: follows throttle up instantly, cools with a 2 s time constant
   *   _fins  grid-fin deployment, ramped over ~1.6 s from the recorded 0/1 `fins_out`
   *   _legs  leg deployment, ramped from `legs_out` (if the replay has it)
   */
  _derive() {
    const F = this.frames, n = this.n;
    if (this._dn >= n) return;
    const thr = F.throttle, thrust = F.thrust, tmax = this.meta.vehicle?.engine?.thrust_max || 0;
    for (const k of ['_soot', '_glow', '_fins', '_legs']) F[k] = F[k] || [];
    const q = new THREE.Quaternion(), ax = new THREE.Vector3();
    for (let i = this._dn; i < n; i++) {
      const dt = i ? Math.max(F.t[i] - F.t[i - 1], 0) : 0;
      let th = thr && thr.length > i ? thr[i] : (thrust && tmax > 0 && thrust.length > i ? thrust[i] / tmax : 0);
      th = Math.max(0, Math.min(th || 0, 1.2));
      // body axis (world, ENU) vs velocity: retro burns sit in their own exhaust
      const qq = F.quat[i];
      q.set(qq[1], qq[2], qq[3], qq[0]);
      ax.set(0, 0, 1).applyQuaternion(q);
      const v = F.vel[i] || [0, 0, 0];
      const sp = Math.hypot(v[0], v[1], v[2]);
      const retro = sp > 3 ? Math.max(0, -(ax.x * v[0] + ax.y * v[1] + ax.z * v[2]) / sp) : 0;
      const alt = F.alt && F.alt.length > i ? F.alt[i] : 1e3;
      const near = Math.exp(-Math.max(alt, 0) / 25);
      const prevS = i ? F._soot[i - 1] : 0;
      F._soot[i] = prevS + dt * th * (0.06 + 0.94 * Math.max(retro, near));
      const prevG = i ? F._glow[i - 1] : 0;
      F._glow[i] = Math.max(th, prevG * Math.exp(-dt / 2.0));
      const ramp = (src, dst, T) => {
        const target = src && src.length > i ? (src[i] > 0.5 ? 1 : 0) : 1;
        const prev = i ? dst[i - 1] : target;
        const step = dt / T;
        dst[i] = target > prev ? Math.min(target, prev + step) : Math.max(target, prev - step);
      };
      ramp(F.fins_out, F._fins, 1.6);
      ramp(F.legs_out, F._legs, 2.0);
    }
    this._dn = n;
  }

  /** First time a string column takes a value (e.g. phase 'descent'), or null. */
  firstTime(col, value) {
    const c = this.frames[col];
    if (!c) return null;
    for (let i = 0; i < c.length; i++) if (c[i] === value) return this.frames.t[i];
    return null;
  }

  get n() { return this.frames.t.length; }
  get t0() { return this.n ? this.frames.t[0] : 0; }
  get tEnd() { return this.n ? this.frames.t[this.n - 1] : 0; }
  get source() { return this.meta.source || 'sim'; }
  get title() { return this.meta.title || this.id; }
  get vehicle() { return this.meta.vehicle; }
  get outcome() { return this.meta.outcome || null; }
  has(name) { return this.n > 0 && !!this.frames[name] && this.frames[name].length === this.n; }

  /** Append columnar frames (live mode). */
  append(frames) {
    const before = this.n;
    for (const k of Object.keys(frames)) {
      if (k === 'vel' && this._derivedVel) continue;
      const dst = (this.frames[k] = this.frames[k] || []);
      const src = frames[k];
      for (let i = 0; i < src.length; i++) dst.push(src[i]);
    }
    this._fillVel(before);
    this._derive();
    this.version++;
  }

  _fillVel(from) {
    if (!this._derivedVel) return;
    const { t, pos, vel } = this.frames;
    for (let i = from; i < t.length; i++) {
      if (i === 0) { vel[i] = [0, 0, 0]; continue; }
      const dt = t[i] - t[i - 1] || 1e-9;
      vel[i] = [0, 1, 2].map((k) => (pos[i][k] - pos[i - 1][k]) / dt);
      if (i === 1) vel[0] = vel[1].slice();
    }
  }

  /** Unit ENU "up" at an ENU position. */
  upEnu(p) {
    if (!this.spherical) return [0, 0, 1];
    const x = p[0], y = p[1], z = p[2] + this.R;
    const r = Math.hypot(x, y, z) || 1;
    return [x / r, y / r, z / r];
  }

  /** Altitude proxy when the `alt` column is absent. */
  fallbackAlt(p) {
    const legs = this.vehicle.legs?.height || 0;
    if (!this.spherical) return p[2] - legs;
    return Math.hypot(p[0], p[1], p[2] + this.R) - this.R - legs;
  }

  /**
   * Interpolated state at time t (clamped to the recording).
   * Vectors are lerped, quaternions slerped, strings (phase) are stepped.
   */
  sample(t) {
    const n = this.n;
    if (!n) return null;
    const T = this.frames.t;
    let i0 = bisect(T, t), f = 0;
    if (i0 < 0) i0 = 0;
    let i1 = i0;
    if (i0 < n - 1) {
      i1 = i0 + 1;
      f = clamp((t - T[i0]) / (T[i1] - T[i0] || 1), 0, 1);
    }
    const s = { t: clamp(t, T[0], T[n - 1]), i: i0, f };
    for (const k of Object.keys(this.frames)) {
      const col = this.frames[k];
      if (col.length !== n || k === 't') continue;
      const a = col[i0], b = col[i1];
      if (k === 'quat') {
        _qa.set(a[1], a[2], a[3], a[0]);
        _qb.set(b[1], b[2], b[3], b[0]);
        _qa.slerp(_qb, f);
        s.quat = [_qa.w, _qa.x, _qa.y, _qa.z];
      } else if (typeof a === 'string') {
        s[k] = f < 1 ? a : b;
      } else if (Array.isArray(a)) {
        s[k] = a.map((v, j) => v + (b[j] - v) * f);
      } else if (typeof a === 'number') {
        s[k] = a + (b - a) * f;
      }
    }
    if (s.alt === undefined) s.alt = this.fallbackAlt(s.pos);
    const v = s.vel || [0, 0, 0];
    s.speed = Math.hypot(v[0], v[1], v[2]);
    const up = this.upEnu(s.pos);
    s.vz = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
    s.hspeed = Math.sqrt(Math.max(s.speed * s.speed - s.vz * s.vz, 0));
    return s;
  }

  /** A numeric column by name ('speed' and 'alt' are derived when needed), or null. Cached per version. */
  column(name) {
    const n = this.n;
    if (!n) return null;
    this._cache = this._cache || new Map();
    const hit = this._cache.get(name);
    if (hit && hit.version === this.version && hit.n === n) return hit.col;
    const F = this.frames;
    let col = null;
    if (name === 'speed') col = F.vel.map((v) => Math.hypot(v[0], v[1], v[2]));
    else if (name === 'alt') col = F.alt && F.alt.length === n ? F.alt : F.pos.map((p) => this.fallbackAlt(p));
    else {
      const c = F[name];
      col = c && c.length === n && typeof c[0] === 'number' ? c : null;
    }
    this._cache.set(name, { version: this.version, n, col });
    return col;
  }

  /** Initial propellant mass (for fuel %). */
  propInitial() {
    const v = this.vehicle.prop_mass_initial;
    if (v > 0) return v;
    const c = this.frames.prop_mass;
    if (!c || !c.length) return 0;
    let m = 0;
    for (const x of c) if (x > m) m = x;
    return m;
  }

  /** Maximum speed over the recording (for trail colouring). */
  maxSpeed() {
    const c = this.column('speed');
    if (!c) return 1;
    let m = 1;
    for (const x of c) if (x > m) m = x;
    return m;
  }
}

export async function fetchReplay(id) {
  const url = '/api/replays/' + id.split('/').map(encodeURIComponent).join('/');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`replay ${id}: HTTP ${r.status}`);
  const text = await r.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    // Python's json writes NaN / Infinity, which is not JSON: read those as null
    data = JSON.parse(text.replace(/([:[,]\s*)-?(?:NaN|Infinity)(?=\s*[,\]}])/g, '$1null'));
    console.warn(`replay ${id}: non-finite numbers (NaN/Infinity) read as null`);
  }
  return new Replay(data, id);
}
