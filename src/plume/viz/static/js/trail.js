// Trajectory trail: screen-space-width ribbons built from independent segments.
//
//  * `main` ribbon  - path up to the current time, coloured by speed or phase.
//  * `pred` ribbon  - the whole recorded path, faint ("predicted"/future).
//
// Points are stored in float64 W-space and written to chunks of CAPSEG segments, each with its
// own float64 anchor (floating origin, see coords.js).  Every segment is a separate quad
// (start+/-, end+/-) so the vertex shader can clip each one against the near plane on its own;
// that matters because the chase camera sits right next to the trail.

import * as THREE from 'three';
import { bisect, clamp } from './util.js';

const CAPSEG = 256;

const VERT = /* glsl */ `
  attribute vec3 aOther; attribute float aSide; attribute vec3 aColor;
  uniform vec2 uRes; uniform float uNear; uniform float uWidth;
  varying vec3 vCol;
  vec3 toNear(vec3 a, vec3 b) {            // a at/behind the near plane, b in front
    float t = (-uNear - a.z) / (b.z - a.z);
    return a + (b - a) * t;
  }
  void main() {
    vCol = aColor;
    vec3 a = (modelViewMatrix * vec4(position, 1.0)).xyz;
    vec3 b = (modelViewMatrix * vec4(aOther, 1.0)).xyz;
    bool af = a.z < -uNear, bf = b.z < -uNear;
    if (!af && !bf) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }   // whole segment behind camera
    if (!af) a = toNear(a, b);
    if (!bf) b = toNear(b, a);
    vec4 ca = projectionMatrix * vec4(a, 1.0), cb = projectionMatrix * vec4(b, 1.0);
    vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * uRes * 0.5;
    float l = length(d);
    d = l > 1e-5 ? d / l : vec2(1.0, 0.0);
    vec2 nrm = vec2(-d.y, d.x);
    gl_Position = ca;
    // sideways by half the width, plus a half-width cap past the end so neighbouring quads overlap
    gl_Position.xy += (nrm * aSide - d) * (uWidth / uRes) * ca.w;
  }`;

const FRAG = /* glsl */ `
  uniform float uOpacity; uniform vec3 uFlat; uniform float uUseFlat;
  varying vec3 vCol;
  void main() { gl_FragColor = vec4(mix(vCol, uFlat, uUseFlat), uOpacity); }`;

const PHASE_COLORS = [[0.2, 0.85, 1.0], [1.0, 0.55, 0.15], [0.55, 0.9, 0.4], [0.85, 0.5, 1.0], [1.0, 0.85, 0.3], [0.4, 0.6, 1.0]];

function speedColor(f, out) {
  // cyan -> pale yellow -> orange
  const a = [0.16, 0.8, 1.0], b = [1.0, 0.93, 0.55], c = [1.0, 0.42, 0.1];
  const [p, q, t] = f < 0.5 ? [a, b, f * 2] : [b, c, (f - 0.5) * 2];
  out[0] = p[0] + (q[0] - p[0]) * t; out[1] = p[1] + (q[1] - p[1]) * t; out[2] = p[2] + (q[2] - p[2]) * t;
  return out;
}

function makeChunkGeometry(shared) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', shared.pos);
  g.setAttribute('aOther', shared.other);
  g.setAttribute('aSide', shared.side);
  g.setAttribute('aColor', shared.col);
  g.setIndex(shared.index);
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e12); // positions are anchor-relative; skip culling
  return g;
}

export class Trail {
  constructor({ maxSpeed = 100 } = {}) {
    this.maxSpeed = maxSpeed;
    this.colorBy = 'speed';
    this.group = new THREE.Group();
    this.pts = [];
    this.times = [];
    this.speeds = [];
    this.phases = [];
    this.phaseIds = new Map();
    this.chunks = [];
    this.nVisible = 0;
    this.uniformsMain = {
      uRes: { value: new THREE.Vector2(1, 1) }, uNear: { value: 0.1 }, uWidth: { value: 2.6 },
      uOpacity: { value: 0.95 }, uFlat: { value: new THREE.Vector3(1, 1, 1) }, uUseFlat: { value: 0 },
    };
    this.uniformsPred = {
      uRes: this.uniformsMain.uRes, uNear: this.uniformsMain.uNear, uWidth: { value: 1.4 },
      uOpacity: { value: 0.2 }, uFlat: { value: new THREE.Vector3(0.6, 0.78, 1.0) }, uUseFlat: { value: 1 },
    };
    const mk = (u) => new THREE.ShaderMaterial({
      uniforms: u, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    this.matMain = mk(this.uniformsMain);
    this.matPred = mk(this.uniformsPred);
    this.head = this._newChunk(new THREE.Vector3(), 1, true);
    this.group.add(this.head.main);
  }

  get length() { return this.pts.length; }

  _newChunk(anchor, capSeg = CAPSEG, headOnly = false) {
    const nv = capSeg * 4;
    const shared = {
      pos: new THREE.BufferAttribute(new Float32Array(nv * 3), 3),
      other: new THREE.BufferAttribute(new Float32Array(nv * 3), 3),
      side: new THREE.BufferAttribute(new Float32Array(nv), 1),
      col: new THREE.BufferAttribute(new Float32Array(nv * 3), 3),
      index: new THREE.BufferAttribute(new Uint16Array(capSeg * 6), 1),
    };
    for (let s = 0; s < capSeg; s++) {
      const vb = s * 4;
      shared.side.array.set([-1, 1, -1, 1], vb);
      shared.index.array.set([vb, vb + 1, vb + 2, vb + 1, vb + 3, vb + 2], s * 6);
    }
    shared.side.needsUpdate = shared.index.needsUpdate = true;
    const gMain = makeChunkGeometry(shared);
    gMain.setDrawRange(0, 0);
    const main = new THREE.Mesh(gMain, this.matMain);
    main.frustumCulled = false;
    main.renderOrder = 5;
    const chunk = { anchor: anchor.clone(), shared, main, nSeg: 0 };
    if (!headOnly) {
      const gPred = makeChunkGeometry(shared);
      gPred.setDrawRange(0, 0);
      chunk.pred = new THREE.Mesh(gPred, this.matPred);
      chunk.pred.frustumCulled = false;
      chunk.pred.renderOrder = 4;
      this.group.add(main, chunk.pred);
    }
    return chunk;
  }

  _color(i, out) {
    if (this.colorBy === 'phase') {
      const ph = this.phases[i];
      if (ph === undefined || ph === null) return speedColor(0, out);
      if (!this.phaseIds.has(ph)) this.phaseIds.set(ph, this.phaseIds.size);
      const c = PHASE_COLORS[this.phaseIds.get(ph) % PHASE_COLORS.length];
      out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
      return out;
    }
    return speedColor(clamp(this.speeds[i] / this.maxSpeed, 0, 1), out);
  }

  /** Append one point (W-space). */
  addPoint(posW, t, speed = 0, phase = null) {
    const v = this.pts.length;
    this.pts.push(posW.clone());
    this.times.push(t);
    this.speeds.push(speed);
    this.phases.push(phase);
    if (v === 0) return;
    this._writeSegment(v - 1);
  }

  _writeSegment(k) { // segment between points k and k+1
    const ci = Math.floor(k / CAPSEG), s = k - ci * CAPSEG;
    let ch = this.chunks[ci];
    if (!ch) { ch = this._newChunk(this.pts[k]); this.chunks[ci] = ch; }
    const A = this.pts[k], B = this.pts[k + 1];
    const { pos, other, col } = ch.shared;
    const ax = A.x - ch.anchor.x, ay = A.y - ch.anchor.y, az = A.z - ch.anchor.z;
    const bx = B.x - ch.anchor.x, by = B.y - ch.anchor.y, bz = B.z - ch.anchor.z;
    const ca = this._color(k, [0, 0, 0]), cb = this._color(k + 1, [0, 0, 0]);
    const vb = s * 4;
    const P = [[ax, ay, az], [ax, ay, az], [bx, by, bz], [bx, by, bz]];
    const O = [[bx, by, bz], [bx, by, bz], [ax, ay, az], [ax, ay, az]];
    const C = [ca, ca, cb, cb];
    for (let q = 0; q < 4; q++) {
      pos.array.set(P[q], (vb + q) * 3);
      other.array.set(O[q], (vb + q) * 3);
      col.array.set(C[q], (vb + q) * 3);
    }
    pos.needsUpdate = other.needsUpdate = col.needsUpdate = true;
    ch.nSeg = Math.max(ch.nSeg, s + 1);
    ch.pred.geometry.setDrawRange(0, ch.nSeg * 6);
  }

  setColorBy(mode) {
    this.colorBy = mode;
    for (let k = 0; k + 1 < this.pts.length; k++) this._writeSegment(k);
  }

  /** Show the path up to time t; the open end is joined to the live rocket position. */
  setTime(t, headW) {
    const nv = bisect(this.times, t) + 1; // points with time <= t
    this.nVisible = nv;
    const nSeg = Math.max(nv - 1, 0);
    this.chunks.forEach((ch, ci) => {
      const vis = clamp(nSeg - ci * CAPSEG, 0, ch.nSeg);
      ch.main.geometry.setDrawRange(0, vis * 6);
    });
    const h = this.head;
    if (nv >= 1 && headW) {
      const a = this.pts[nv - 1];
      h.anchor.copy(a);
      const d = headW.clone().sub(a);
      const { pos, other, col } = h.shared;
      pos.array.set([0, 0, 0, 0, 0, 0, d.x, d.y, d.z, d.x, d.y, d.z]);
      other.array.set([d.x, d.y, d.z, d.x, d.y, d.z, 0, 0, 0, 0, 0, 0]);
      const c = this._color(nv - 1, [0, 0, 0]);
      for (let q = 0; q < 4; q++) col.array.set(c, q * 3);
      pos.needsUpdate = other.needsUpdate = col.needsUpdate = true;
      h.main.geometry.setDrawRange(0, 6);
    } else {
      h.main.geometry.setDrawRange(0, 0);
    }
  }

  layout(origin) {
    for (const ch of this.chunks) {
      ch.main.position.copy(ch.anchor).sub(origin);
      ch.pred.position.copy(ch.main.position);
    }
    this.head.main.position.copy(this.head.anchor).sub(origin);
  }

  setCameraInfo(near, w, h, pixelRatio = 1) {
    this.uniformsMain.uNear.value = near;
    this.uniformsMain.uRes.value.set(w, h);
    this.uniformsMain.uWidth.value = 2.6 * pixelRatio;
    this.uniformsPred.uWidth.value = 1.4 * pixelRatio;
  }

  dispose() {
    for (const ch of [...this.chunks, this.head]) {
      ch.main.geometry.dispose();
      ch.pred?.geometry.dispose();
    }
    this.matMain.dispose();
    this.matPred.dispose();
  }
}
