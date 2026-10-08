// Engineering overlay (toggle E): drawn after tone mapping, on top of the scene, in thin lines with
// small monospaced labels and a restrained palette (desaturated hues + line style, with a legend).
//
//  * vectors from the (estimated) CG, log-scaled: ground velocity, air-relative velocity, wind,
//    thrust, aerodynamic force, gravity  (forces are scaled by acceleration so vehicles of any size
//    read the same);  body axes triad;  local horizon ring and up;
//  * guidance: predicted impact point (`impact`) with a line to the target, target tolerance ring,
//    and the remaining recorded path;
//  * Monte Carlo dispersion (`meta.dispersion`): ellipse at the stated probability + scatter points;
//  * readouts: alpha, beta, total AoA, q, Mach, load, |v_air|, wind, fin deflections;
//  * north arrow, and a scale bar in the top-down view.
// Every column is optional; missing data simply hides the corresponding element.

import * as THREE from 'three';
import { enuToW, Y_UP } from './coords.js';
import { el } from './util.js';

const C = {
  vel: 0xf2f2f2, vair: 0x8db4d4, wind: 0x8db4d4, thrust: 0xe3a46a, aero: 0x92c49a, grav: 0xb5a1da,
  axX: 0xd08a84, axY: 0x92c49a, axZ: 0x8db4d4, dim: 0xa3a3a3, disp: 0xdcc47c,
};
const VECS = [
  { k: 'vel', name: 'Velocity (ground)', short: 'V', kind: 'v', dash: false },
  { k: 'vair', name: 'Air-relative velocity', short: 'VAIR', kind: 'v', dash: false },
  { k: 'wind', name: 'Wind', short: 'WIND', kind: 'v', dash: true },
  { k: 'thrust', name: 'Thrust', short: 'T', kind: 'f', dash: false },
  { k: 'aero', name: 'Aerodynamic force', short: 'AERO', kind: 'f', dash: false },
  { k: 'grav', name: 'Gravity', short: 'W', kind: 'f', dash: false },
];
const DEG = 180 / Math.PI;

function hex(c) { return `#${c.toString(16).padStart(6, '0')}`; }
function swatch(color, dash) {
  return `<svg width="24" height="8" viewBox="0 0 24 8"><line x1="0" y1="4" x2="18" y2="4" stroke="${hex(color)}" stroke-width="1.5" ${dash ? 'stroke-dasharray="3 2"' : ''}/><path d="M18 1 L24 4 L18 7 Z" fill="${hex(color)}"/></svg>`;
}
const fmtN = (f) => (Math.abs(f) >= 1e6 ? `${(f / 1e6).toFixed(2)} MN` : Math.abs(f) >= 1e3 ? `${(f / 1e3).toFixed(1)} kN` : `${f.toFixed(0)} N`);
const fmtV = (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} km/s` : `${v.toFixed(1)} m/s`);
const fmtD = (m) => (Math.abs(m) >= 10000 ? `${(m / 1000).toFixed(1)} km` : Math.abs(m) >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`);

// ------------------------------------------------------------------ screen-space lines
// WebGL lines are 1 px; these are quads expanded in screen space (constant pixel width), clipped
// against the near plane per segment, with optional dashes measured along the line (metres).
const FAT_VERT = /* glsl */ `
attribute vec3 aOther; attribute float aSide; attribute float aDist;
uniform vec2 uRes; uniform float uWidth; uniform float uNear;
varying float vDist;
vec3 toNear(vec3 a, vec3 b) { float t = (-uNear - a.z) / (b.z - a.z); return a + (b - a) * t; }
void main() {
  vDist = aDist;
  vec3 a = (modelViewMatrix * vec4(position, 1.0)).xyz;
  vec3 b = (modelViewMatrix * vec4(aOther, 1.0)).xyz;
  bool af = a.z < -uNear, bf = b.z < -uNear;
  if (!af && !bf) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  if (!af) a = toNear(a, b);
  if (!bf) b = toNear(b, a);
  vec4 ca = projectionMatrix * vec4(a, 1.0), cb = projectionMatrix * vec4(b, 1.0);
  vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * uRes;
  float l = length(d);
  d = l > 1e-5 ? d / l : vec2(1.0, 0.0);
  gl_Position = ca;
  gl_Position.xy += vec2(-d.y, d.x) * aSide * uWidth / uRes * ca.w;
}`;
const FAT_FRAG = /* glsl */ `
uniform vec3 uColor; uniform float uOpacity; uniform float uDash;
varying float vDist;
void main() {
  if (uDash > 0.0 && mod(vDist, uDash) > uDash * 0.58) discard;
  gl_FragColor = vec4(uColor, uOpacity);
}`;

const FAT_SHARED = { uRes: { value: new THREE.Vector2(1, 1) }, uNear: { value: 0.1 } };

class FatLine {
  constructor(scene, color, maxPts, { width = 1.5, dash = false, opacity = 1, loop = false } = {}) {
    this.max = maxPts;
    this.loop = loop;
    this.dash = dash;
    const nSeg = loop ? maxPts : maxPts - 1;
    const nv = nSeg * 4;
    const g = new THREE.BufferGeometry();
    this.pos = new THREE.BufferAttribute(new Float32Array(nv * 3), 3);
    this.oth = new THREE.BufferAttribute(new Float32Array(nv * 3), 3);
    this.dist = new THREE.BufferAttribute(new Float32Array(nv), 1);
    const side = new Float32Array(nv), idx = new Uint32Array(nSeg * 6);
    for (let s = 0; s < nSeg; s++) {
      side.set([-1, 1, -1, 1], s * 4);
      const v = s * 4;
      idx.set([v, v + 1, v + 2, v + 1, v + 3, v + 2], s * 6);
    }
    g.setAttribute('position', this.pos);
    g.setAttribute('aOther', this.oth);
    g.setAttribute('aDist', this.dist);
    g.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e12);
    this.u = {
      ...FAT_SHARED, uWidth: { value: width }, uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity }, uDash: { value: 0 },
    };
    this.mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: FAT_VERT, fragmentShader: FAT_FRAG, transparent: true,
      depthTest: false, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 99;
    this.geo = g;
    scene.add(this.mesh);
  }

  get visible() { return this.mesh.visible; }
  set visible(v) { this.mesh.visible = v; }

  /** @param pts Vector3[] (render space), @param dashLen dash period in metres (0 = solid) */
  set(pts, dashLen = 0) {
    const n = Math.min(pts.length, this.max);
    const nSeg = this.loop ? n : n - 1;
    let acc = 0;
    for (let s = 0; s < nSeg; s++) {
      const A = pts[s], B = pts[(s + 1) % n];
      const L = A.distanceTo(B);
      const v = s * 4;
      for (let q = 0; q < 4; q++) {
        const P = q < 2 ? A : B, O = q < 2 ? B : A;
        this.pos.setXYZ(v + q, P.x, P.y, P.z);
        this.oth.setXYZ(v + q, O.x, O.y, O.z);
        this.dist.setX(v + q, q < 2 ? acc : acc + L);
      }
      acc += L;
    }
    this.pos.needsUpdate = this.oth.needsUpdate = this.dist.needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(nSeg, 0) * 6);
    this.u.uDash.value = this.dash ? dashLen || acc / 12 : 0;
  }

  dispose() { this.geo.dispose(); this.mesh.material.dispose(); }
}

class Arrow {
  constructor(scene, color, dash, width = 2) {
    this.line = new FatLine(scene, color, 2, { width, dash });
    this.head = new THREE.Mesh(new THREE.ConeGeometry(1, 2.6, 12).translate(0, -1.3, 0),
      new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false, toneMapped: false, transparent: true }));
    this.head.renderOrder = 101;
    this.head.frustumCulled = false;
    scene.add(this.head);
  }

  set(a, b, headSize) {
    const d = b.clone().sub(a);
    const len = d.length();
    const hs = Math.min(headSize, len * 0.3);
    this.line.set([a, b.clone().addScaledVector(d, -hs * 0.8 / Math.max(len, 1e-6))], len / 9);
    this.head.position.copy(b);
    this.head.quaternion.setFromUnitVectors(Y_UP, d.normalize());
    this.head.scale.set(hs * 0.36, hs, hs * 0.36);
  }

  visible(v) { this.line.visible = v; this.head.visible = v; }
  dispose() { this.line.dispose(); this.head.geometry.dispose(); this.head.material.dispose(); }
}

export class EngOverlay {
  constructor(world, root) {
    this.world = world;
    this.root = root;
    this.enabled = false;
    this.scene = new THREE.Scene();
    const r = world.replay;
    this.has = (k) => r.has(k);
    this.arrows = {};
    for (const v of VECS) this.arrows[v.k] = new Arrow(this.scene, C[v.k], v.dash);
    this.axes = [C.axX, C.axY, C.axZ].map((c) => new Arrow(this.scene, c, false));
    this.horizon = new FatLine(this.scene, C.dim, 72, { dash: true, opacity: 0.8, loop: true, width: 1.2 });
    this.upLine = new Arrow(this.scene, C.dim, true);
    this.path = new FatLine(this.scene, 0xf2f2f2, 900, { opacity: 0.6, width: 1.5 });
    this.impactRing = new FatLine(this.scene, 0xf2f2f2, 48, { loop: true, width: 1.5 });
    this.impactCrossA = new FatLine(this.scene, 0xf2f2f2, 2, { width: 1.5 });
    this.impactCrossB = new FatLine(this.scene, 0xf2f2f2, 2, { width: 1.5 });
    this.toTarget = new FatLine(this.scene, 0xf2f2f2, 2, { dash: true, opacity: 0.85, width: 1.5 });
    this.targetRing = new FatLine(this.scene, C.dim, 96, { dash: true, loop: true, opacity: 0.9, width: 1.5 });
    this.cgMark = [0, 1, 2].map(() => new FatLine(this.scene, 0xf2f2f2, 2, { width: 1.5 }));
    this._buildDispersion();
    this._buildHtml();
    this.setEnabled(false);
  }

  // ------------------------------------------------------------------ dispersion (static, W-space)
  _buildDispersion() {
    const D = this.world.replay.meta.dispersion;
    this.disp = null;
    if (!D || !Array.isArray(D.semi_axes_m) || D.semi_axes_m.length < 2) return;
    const F = this.world.frame;
    const T = () => this.world.terrain;
    // Schema: centre and points are local east/north offsets (m) at the target (as written by
    // plume.analysis.montecarlo.dispersion_meta).  Without a target they are taken as ENU positions.
    const tgt = this.world.replay.meta.scene?.target?.pos;
    const base = tgt ? F.toMap(enuToW(tgt)) : { u: 0, v: 0, h: 0 };
    const c0 = D.center || [0, 0, 0];
    const cm = { u: base.u + c0[0], v: base.v + c0[1], h: base.h };
    const drape = (u, v, h0) => {
      const h = T()?.heightAt(u, v);
      return F.surface(u, v, (h ?? h0) + 0.6, new THREE.Vector3());
    };
    const [a, b] = D.semi_axes_m, th = D.angle_rad || 0;
    const ring = [];
    for (let i = 0; i < 128; i++) {
      const t = (i / 128) * Math.PI * 2;
      const x = a * Math.cos(t) * Math.cos(th) - b * Math.sin(t) * Math.sin(th);
      const y = a * Math.cos(t) * Math.sin(th) + b * Math.sin(t) * Math.cos(th);
      ring.push({ u: cm.u + x, v: cm.v + y });
    }
    const pts = (D.points || []).map((p) => ({ u: base.u + p[0], v: base.v + p[1], h: base.h }));
    // CEP: median miss about the target from the points, else from the ellipse
    let cep = null;
    if (pts.length >= 5) {
      const d = pts.map((p) => Math.hypot(p.u - base.u, p.v - base.v)).sort((x, y) => x - y);
      cep = d[Math.floor(d.length / 2)];
    } else if (D.probability > 0 && D.probability < 1) {
      const k = Math.sqrt(-2 * Math.log(1 - D.probability));
      cep = 0.5887 * (a / k + b / k);
    }
    const cW = F.surface(cm.u, cm.v, cm.h, new THREE.Vector3());
    this.disp = { D, cW, cm, ring, pts, drape, cep, ready: false };
    this.dispLine = new FatLine(this.scene, C.disp, 128, { loop: true, width: 1.5 });
    const fillGeo = new THREE.BufferGeometry();
    fillGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(129 * 3), 3));
    const idx = [];
    for (let i = 0; i < 128; i++) idx.push(128, i, (i + 1) % 128);
    fillGeo.setIndex(idx);
    this.dispFill = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({ color: C.disp, transparent: true, opacity: 0.12, depthTest: false, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
    this.dispFill.frustumCulled = false;
    this.dispFill.renderOrder = 98;
    this.scene.add(this.dispFill);
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(pts.length, 1) * 3), 3));
    this.dispPts = new THREE.Points(pg, new THREE.PointsMaterial({ color: C.disp, size: 3, sizeAttenuation: false, depthTest: false, depthWrite: false, toneMapped: false, transparent: true, opacity: 0.9 }));
    this.dispPts.frustumCulled = false;
    this.dispPts.renderOrder = 99;
    this.scene.add(this.dispPts);
  }

  _dispW() {
    // W-space ellipse/points, draped once the terrain is in
    const d = this.disp;
    if (d.ready && d.terrainSeen === !!this.world.terrain?.layers.length) return;
    d.ringW = d.ring.map((p) => d.drape(p.u, p.v, d.cm.h));
    d.ptsW = d.pts.map((p) => d.drape(p.u, p.v, p.h));
    d.cDr = d.drape(d.cm.u, d.cm.v, d.cm.h);
    d.ready = true;
    d.terrainSeen = !!this.world.terrain?.layers.length;
  }

  // ------------------------------------------------------------------ HTML
  _buildHtml() {
    const lay = this.world.labelLayer;
    this.labels = {};
    if (lay) {
      for (const v of [...VECS, { k: 'impact' }, { k: 'cg' }, { k: 'disp' }, { k: 'xb' }, { k: 'yb' }, { k: 'zb' }]) {
        const e = el('div', { class: 'eng-label', hidden: true });
        lay.append(e);
        this.labels[v.k] = e;
      }
    }
    if (!this.root) return;
    this.rows = {};
    const rows = el('div', { class: 'rows' });
    const addRow = (k, label) => {
      const b = el('b', { text: '—' });
      rows.append(el('i', { text: label }), b);
      this.rows[k] = b;
    };
    addRow('alpha', 'α  angle of attack');
    addRow('beta', 'β  sideslip');
    addRow('aoa', 'Total AoA');
    addRow('q', 'Dynamic pressure');
    addRow('mach', 'Mach');
    addRow('g', 'Load');
    addRow('vair', '|V air|');
    addRow('wind', 'Wind');
    addRow('fins', 'Fins δ');
    addRow('impact', 'Impact miss');
    const key = el('div', { class: 'key' });
    this.keyVals = {};
    for (const v of VECS) {
      const em = el('em', { text: '' });
      const sw = el('div', { html: swatch(C[v.k], v.dash) });
      key.append(sw, el('span', { text: v.name }), em);
      this.keyVals[v.k] = { em, sw, span: sw.nextSibling };
    }
    this.dispKey = el('div', { class: 'key', hidden: true });
    this.panel = el('div', { class: 'eng-panel' },
      el('div', { class: 'ph' }, el('span', { class: 'lbl', text: 'Engineering' }), el('span', { class: 'lbl', text: 'CG est. · log scale' })),
      rows, key, this.dispKey);
    this.north = el('div', { class: 'northarrow', html: '<svg width="40" height="52" viewBox="0 0 40 52"><text x="20" y="11" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="#f2f2f2">N</text><g class="na"><path d="M20 18 L26 42 L20 37 L14 42 Z" fill="none" stroke="#f2f2f2" stroke-width="1"/><path d="M20 18 L20 37 L14 42 Z" fill="#f2f2f2"/></g></svg>' });
    this.scale = el('div', { class: 'scalebar' }, el('span', { text: '' }), el('div', { class: 'bar' }));
    this.root.append(this.panel, this.north, this.scale);
    if (this.disp) {
      const d = this.disp.D;
      const parts = [];
      if (d.runs) parts.push(`${d.runs} runs`);
      if (d.probability) parts.push(`P${Math.round(d.probability * 100)} ellipse ${fmtD(2 * d.semi_axes_m[0])} × ${fmtD(2 * d.semi_axes_m[1])}`);
      if (this.disp.cep != null) parts.push(`CEP ${fmtD(this.disp.cep)}`);
      this.dispKey.hidden = false;
      this.dispKey.append(el('div', { html: `<svg width="24" height="10" viewBox="0 0 24 10"><ellipse cx="12" cy="5" rx="10" ry="4" fill="${hex(C.disp)}" fill-opacity="0.15" stroke="${hex(C.disp)}"/></svg>` }),
        el('span', { text: 'Monte Carlo dispersion' }), el('em', { text: '' }),
        el('div'), el('span', { text: parts.join(' · '), style: { gridColumn: '2 / 4', color: '#a3a3a3', fontFamily: 'var(--mono)', fontSize: '10px' } }));
    }
  }

  setEnabled(on) {
    this.enabled = !!on;
    if (this.panel) { this.panel.hidden = !on; this.north.hidden = !on; this.scale.hidden = true; }
    for (const e of Object.values(this.labels)) e.hidden = true;
    const U = this.world.groundU;
    if (U) U.uEng.value = on ? 1 : 0;
  }

  // ------------------------------------------------------------------ per frame
  layout(origin, cam) {
    if (!this.enabled) return;
    const W = this.world, s = W.sample, st = W.state;
    if (!s || !st) return;
    const L = W.L;
    const up = st.basis.up;
    const cgW = W.baseW.clone().addScaledVector(st.axis, W.rocket.cgZ);
    const cg = cgW.clone().sub(origin);
    this._cgW = cgW;
    const camDist = Math.max(cam.camW.distanceTo(cgW), 1);
    const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
    // CG marker: three short crossed segments
    const ck = Math.min(0.08 * L, camDist * 0.02);
    [V3(1, 0, 0), V3(0, 1, 0), V3(0, 0, 1)].forEach((a, i) => this.cgMark[i].set([cg.clone().addScaledVector(a, -ck), cg.clone().addScaledVector(a, ck)]));

    const mass = s.mass ?? ((W.vehicle.dry_mass || 0) + (s.prop_mass ?? W.vehicle.prop_mass_initial ?? 0) + (W.vehicle.cargo_mass || 0));
    const g0 = W.frame.spherical ? 9.80665 * (W.frame.R / (W.frame.R + Math.max(s.alt, 0))) ** 2 : 9.80665;
    const scaleV = Math.max(0.42 * L, camDist * 0.075);
    const scaleF = Math.max(0.7 * L, camDist * 0.11);
    const head = Math.max(0.06 * L, camDist * 0.016);
    const vecW = (enu) => (enu ? enuToW(enu, new THREE.Vector3()) : null);
    const data = {
      vel: vecW(s.vel),
      vair: this.has('v_air') ? vecW(s.v_air) : null,
      wind: this.has('wind') ? vecW(s.wind) : null,
      thrust: this.has('f_thrust') ? vecW(s.f_thrust) : null,
      aero: this.has('f_aero') ? vecW(s.f_aero) : null,
      grav: mass > 0 ? up.clone().multiplyScalar(-mass * g0) : null,
    };
    this._vecInfo = {};
    for (const v of VECS) {
      const A = this.arrows[v.k];
      const d = data[v.k];
      const mag = d ? d.length() : 0;
      const tiny = v.kind === 'v' ? mag < 0.05 : mag < 1e-3 * mass;
      if (!d || tiny) { A.visible(false); this._vecInfo[v.k] = null; continue; }
      const len = v.kind === 'v' ? scaleV * Math.log10(1 + mag) : scaleF * Math.log10(1 + mag / Math.max(mass, 1e-6));
      const tip = cg.clone().addScaledVector(d.clone().normalize(), Math.max(len, head * 1.5));
      A.visible(true);
      A.set(cg, tip, head);
      this._vecInfo[v.k] = { tip, mag, kind: v.kind };
    }
    // body axes triad (body x, body y, body z = vehicle axis)
    const q = W.quatW;
    const bx = V3(1, 0, 0).applyQuaternion(q), by = V3(0, 0, -1).applyQuaternion(q), bz = st.axis.clone();
    const aL = Math.max(0.3 * L, camDist * 0.05);
    [bx, by, bz].forEach((d, i) => this.axes[i].set(cg, cg.clone().addScaledVector(d, aL), head * 0.6));
    this._axTips = [bx, by, bz].map((d) => cgW.clone().addScaledVector(d, aL * 1.08));
    // local horizon ring + up
    const e1 = st.basis.east, e2 = st.basis.north;
    const hr = Math.max(0.55 * L, camDist * 0.08);
    const ring = [];
    for (let i = 0; i < 72; i++) {
      const a = (i / 72) * Math.PI * 2;
      ring.push(cg.clone().addScaledVector(e1, Math.cos(a) * hr).addScaledVector(e2, Math.sin(a) * hr));
    }
    this.horizon.set(ring, hr * 0.14);
    this.upLine.set(cg, cg.clone().addScaledVector(up, hr * 0.8), head * 0.5);

    // remaining recorded path
    const r = W.replay;
    const i0 = Math.max(0, s.i);
    const stride = Math.max(1, Math.ceil((r.n - i0) / 880));
    const path = [W.baseW.clone().sub(origin)];
    for (let i = i0 + 1; i < r.n && path.length < 900; i += stride) path.push(enuToW(r.frames.pos[i], new THREE.Vector3()).sub(origin));
    if (path.length < 2) path.push(path[0].clone());
    this.path.set(path);

    // target tolerance ring + predicted impact
    const F = W.frame;
    const ringPts = (centerW, R, nSeg, lift = 0.4) => {
      const m = F.toMap(centerW), out = [];
      for (let i = 0; i < nSeg; i++) {
        const a = (i / nSeg) * Math.PI * 2;
        const u = m.u + R * Math.cos(a), v = m.v + R * Math.sin(a);
        const h = W.terrain?.heightAt(u, v) ?? m.h;
        out.push(F.surface(u, v, h + lift, new THREE.Vector3()).sub(origin));
      }
      return out;
    };
    if (W.targetW) {
      this.targetRing.visible = true;
      this.targetRing.set(ringPts(W.targetW, W.targetR, 96), W.targetR * 0.2);
    } else this.targetRing.visible = false;
    const hasImp = this.has('impact') && s.impact;
    for (const o of [this.impactRing, this.impactCrossA, this.impactCrossB, this.toTarget]) o.visible = !!hasImp;
    this._impactW = null;
    this._miss = null;
    if (hasImp) {
      const iw = enuToW(s.impact, new THREE.Vector3());
      this._impactW = iw;
      const R = Math.max(W.targetR * 0.25, camDist * 0.012, 2);
      this.impactRing.set(ringPts(iw, R, 48));
      const m = F.toMap(iw);
      const h0 = W.terrain?.heightAt(m.u, m.v) ?? m.h;
      const P = (du, dv) => F.surface(m.u + du, m.v + dv, h0 + 0.4, new THREE.Vector3()).sub(origin);
      this.impactCrossA.set([P(-1.6 * R, 0), P(1.6 * R, 0)]);
      this.impactCrossB.set([P(0, -1.6 * R), P(0, 1.6 * R)]);
      if (W.targetW) {
        const c0 = P(0, 0), c1 = W.targetW.clone().sub(origin);
        this.toTarget.set([c0, c1], Math.max(c0.distanceTo(c1) / 24, 0.3));
        const tm = F.toMap(W.targetW);
        this._miss = Math.hypot(m.u - tm.u, m.v - tm.v);
      } else this.toTarget.visible = false;
    }

    // dispersion
    if (this.disp) {
      this._dispW();
      const d = this.disp;
      const fp = this.dispFill.geometry.attributes.position;
      const pts = d.ringW.map((p) => p.clone().sub(origin));
      this.dispLine.set(pts);
      pts.forEach((p, i) => fp.setXYZ(i, p.x, p.y, p.z));
      fp.setXYZ(128, d.cDr.x - origin.x, d.cDr.y - origin.y, d.cDr.z - origin.z);
      fp.needsUpdate = true;
      const pp2 = this.dispPts.geometry.attributes.position;
      d.ptsW.forEach((p, i) => pp2.setXYZ(i, p.x - origin.x, p.y - origin.y, p.z - origin.z));
      pp2.needsUpdate = true;
      this.dispPts.geometry.setDrawRange(0, d.ptsW.length);
    }
    this._readouts(s, data, mass);
  }

  _readouts(s, data, mass) {
    if (!this.rows) return;
    const set = (k, txt) => { const e = this.rows[k]; if (e.textContent !== txt) e.textContent = txt; };
    const deg = (x) => (Number.isFinite(x) ? `${(x * DEG).toFixed(1)}°` : '—');
    set('alpha', s.alpha !== undefined ? deg(s.alpha) : '—');
    set('beta', s.beta !== undefined ? deg(s.beta) : '—');
    set('aoa', s.aoa_total !== undefined ? deg(s.aoa_total) : '—');
    set('q', s.q_dyn !== undefined ? `${(s.q_dyn / 1000).toFixed(2)} kPa` : '—');
    set('mach', s.mach !== undefined ? s.mach.toFixed(2) : '—');
    set('g', s.g_load !== undefined ? `${s.g_load.toFixed(2)} g` : '—');
    set('vair', data.vair ? fmtV(data.vair.length()) : '—');
    set('wind', data.wind ? fmtV(data.wind.length()) : '—');
    set('fins', s.fins ? `${s.fins.map((x) => String(Math.round(x * DEG) || 0)).join(' / ')}°` : '—');
    set('impact', this._miss != null ? fmtD(this._miss) : '—');
    for (const v of VECS) {
      const info = this._vecInfo?.[v.k];
      const kv = this.keyVals[v.k];
      const txt = info ? (info.kind === 'v' ? fmtV(info.mag) : fmtN(info.mag)) : 'n/a';
      if (kv.em.textContent !== txt) kv.em.textContent = txt;
      kv.sw.style.opacity = info ? '1' : '0.3';
    }
    void mass;
  }

  updateLabels(cam, origin, width, height) {
    if (!this.enabled) return;
    const { camera } = cam;
    const v = new THREE.Vector3();
    const placed = [];
    const place = (e, posR, text, dx = 6) => {
      if (!e) return false;
      if (!posR) { e.hidden = true; return false; }
      v.copy(posR).project(camera);
      const ok = v.z < 1 && v.z > -1 && Math.abs(v.x) < 1.02 && Math.abs(v.y) < 1.02;
      e.hidden = !ok;
      if (ok) {
        if (text !== undefined && e.textContent !== text) e.textContent = text;
        placed.push({ e, x: ((v.x + 1) / 2) * width + dx, y: ((1 - v.y) / 2) * height - 8, w: 8 + 6.2 * e.textContent.length });
      }
      return ok;
    };
    const cgS = this._cgW ? this._cgW.clone().sub(origin).project(camera) : null;
    for (const vv of VECS) {
      const info = this._vecInfo?.[vv.k];
      let tip = info?.tip;
      if (tip && cgS) {
        const ts = tip.clone().project(camera);
        if (Math.hypot((ts.x - cgS.x) * width, (ts.y - cgS.y) * height) < 28) tip = null; // foreshortened: no label
      }
      place(this.labels[vv.k], tip, info ? `${vv.short} ${info.kind === 'v' ? fmtV(info.mag) : fmtN(info.mag)}` : '');
    }
    ['xb', 'yb', 'zb'].forEach((k, i) => place(this.labels[k], this._axTips ? this._axTips[i].clone().sub(origin) : null, k));
    place(this.labels.cg, this._cgW ? this._cgW.clone().sub(origin) : null, 'CG', -30);
    place(this.labels.impact, this._impactW ? this._impactW.clone().sub(origin) : null, this._miss != null ? `IMPACT  ${fmtD(this._miss)} to target` : 'IMPACT');
    place(this.labels.disp, this.disp?.cDr ? this.disp.cDr.clone().sub(origin) : null, this.disp ? `DISPERSION P${Math.round((this.disp.D.probability || 0) * 100)}` : '');
    // de-clutter: push overlapping labels apart vertically (greedy, top to bottom)
    placed.sort((a, b) => a.y - b.y);
    for (let i = 0; i < placed.length; i++) {
      for (let j = 0; j < i; j++) {
        const A = placed[j], B = placed[i];
        if (Math.abs(A.x - B.x) < Math.max(A.w, B.w) && B.y - A.y < 15) B.y = A.y + 15;
      }
      placed[i].e.style.transform = `translate(${placed[i].x.toFixed(1)}px, ${placed[i].y.toFixed(1)}px)`;
    }
    // north arrow: screen direction of local north at the vehicle
    if (this.north && this._cgW) {
      const st = this.world.state;
      const a = this._cgW.clone().sub(origin), b = a.clone().addScaledVector(st.basis.north, Math.max(cam.camW.distanceTo(this._cgW) * 0.05, 1));
      const pa = a.project(camera), pb = b.project(camera);
      const ang = Math.atan2(-(pb.y - pa.y) * height, (pb.x - pa.x) * width) + Math.PI / 2;
      const g = this.north.querySelector('.na');
      if (g && Number.isFinite(ang)) g.setAttribute('transform', `rotate(${(ang * DEG).toFixed(1)} 20 31)`);
    }
    // scale bar in the top-down (orthographic) view
    if (this.scale) {
      const ortho = camera.isOrthographicCamera;
      this.scale.hidden = !ortho;
      if (ortho) {
        const mpp = (camera.top - camera.bottom) / height;
        const target = mpp * width * 0.18;
        const p10 = Math.pow(10, Math.floor(Math.log10(target)));
        const nice = [1, 2, 5, 10].map((m) => m * p10).reduce((best, x) => (Math.abs(x - target) < Math.abs(best - target) ? x : best), p10);
        this.scale.firstChild.textContent = fmtD(nice);
        this.scale.lastChild.style.width = `${(nice / mpp).toFixed(0)}px`;
      }
    }
  }

  render(renderer, camera) {
    if (!this.enabled) return;
    renderer.getDrawingBufferSize(FAT_SHARED.uRes.value);
    FAT_SHARED.uNear.value = camera.isOrthographicCamera ? -1e9 : camera.near;
    const pr = renderer.getPixelRatio();
    this.scene.traverse((o) => { if (o.material?.uniforms?.uWidth) o.material.uniforms.uWidth.value = (o.userData.w ??= o.material.uniforms.uWidth.value) * pr; });
    renderer.render(this.scene, camera);
  }

  dispose() {
    this.scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    for (const e of Object.values(this.labels)) e.remove();
    this.panel?.remove(); this.north?.remove(); this.scale?.remove();
  }
}
