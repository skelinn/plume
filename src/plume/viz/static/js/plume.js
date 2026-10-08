// Engine exhaust, rendered as an emissive volume ray-marched inside a bounding cylinder.
//
// The emission field is physically motivated:
//   * a bright, near-white supersonic core with shock diamonds / Mach disks whose spacing scales with
//     the exit diameter, strongest near sea level where the jet is close to ambient pressure;
//   * an orange/yellow turbulent sheath (afterburning fuel-rich gas mixing with air);
//   * with altitude the jet is under-expanded: it balloons into a wide, translucent bell that is
//     much larger in near-vacuum, the diamonds vanish and the gas glows faintly (plus scattered
//     sunlight), while the core near the nozzle stays bright;
//   * brightness and length scale with throttle; the jet spreads radially where it meets the ground.
// The volume is additive HDR (bloom does the rest) and depends only on uniforms, so capture mode is
// deterministic.  `Dust` is the debris/dust cloud kicked up when the plume reaches the ground.

import * as THREE from 'three';
import { GLSL_NOISE } from './shaders.js';
import { clamp, smoothstep } from './util.js';

let _noise3d = null;
function noise3D() {
  if (_noise3d) return _noise3d;
  const N = 32, data = new Uint8Array(N * N * N);
  const h = (x, y, z) => {
    let k = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
    k = Math.imul(k ^ (k >>> 13), 1274126177) >>> 0;
    return ((k ^ (k >>> 16)) >>> 0) / 4294967295;
  };
  // tiling value noise: lattice period P cells over the N texels
  const vn = (x, y, z, P, seed) => {
    const X = (x * P) / N, Y = (y * P) / N, Z = (z * P) / N;
    const xi = Math.floor(X), yi = Math.floor(Y), zi = Math.floor(Z);
    const s = (t) => t * t * (3 - 2 * t);
    const u = s(X - xi), v = s(Y - yi), w = s(Z - zi);
    const L = (a, b, c) => h(((a % P) + P) % P + seed, ((b % P) + P) % P, ((c % P) + P) % P);
    const c000 = L(xi, yi, zi), c100 = L(xi + 1, yi, zi), c010 = L(xi, yi + 1, zi), c110 = L(xi + 1, yi + 1, zi);
    const c001 = L(xi, yi, zi + 1), c101 = L(xi + 1, yi, zi + 1), c011 = L(xi, yi + 1, zi + 1), c111 = L(xi + 1, yi + 1, zi + 1);
    const a = c000 + (c100 - c000) * u, b = c010 + (c110 - c010) * u, c = c001 + (c101 - c001) * u, d = c011 + (c111 - c011) * u;
    const e = a + (b - a) * v, g = c + (d - c) * v;
    return e + (g - e) * w;
  };
  let i = 0;
  for (let z = 0; z < N; z++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) data[i++] = Math.min(255, (vn(x, y, z, 8, 0) * 0.65 + vn(x, y, z, 16, 101) * 0.35) * 255);
  const t = new THREE.Data3DTexture(data, N, N, N);
  t.format = THREE.RedFormat;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  _noise3d = t;
  return t;
}

const VERT = /* glsl */ `
uniform vec3 uScale;
varying vec3 vPos;
void main() {
  vPos = position * uScale;      // plume-local metres
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp sampler3D;
uniform sampler3D uNoise;
uniform vec3 uCamL;          // camera position in plume-local space
uniform float uRe, uLen, uVac, uTh, uT, uSolid, uBoundR, uBoundL, uBright;
uniform vec4 uGround;        // plane: dot(xyz, p) + w = height above ground (local units)
uniform vec3 uSunL;          // sun direction (local)
uniform vec3 uSunCol;
uniform int uSteps;
varying vec3 vPos;

float nz(vec3 p) { return texture(uNoise, p).r; }

vec3 emit(vec3 p, out float dens) {
  dens = 0.0;
  float x = -p.y;
  if (x < 0.0) return vec3(0.0);
  float hg = dot(uGround.xyz, p) + uGround.w;
  if (hg < 0.0) return vec3(0.0);
  float rho = length(p.xz);
  float xr = x / uRe;
  // jet boundary: slow spread at sea level, rapid under-expanded ballooning in vacuum
  float Rj = uRe * (1.0 + mix(0.055, 0.2, uVac) * xr + uVac * 2.2 * sqrt(xr));
  float splash = exp(-hg / (1.4 * uRe));
  Rj *= 1.0 + 2.2 * splash;
  float r = rho / Rj;
  if (r > 2.2) return vec3(0.0);
  float s = x / uLen;
  float fall = exp(-2.2 * s * s) * (1.0 - smoothstep(0.75, 1.1, s));
  // turbulent sheath, advected downstream
  vec3 q = p / uRe;
  float adv = uT * mix(6.0, 2.5, uVac);
  float n = nz(vec3(q.x * 0.11, q.y * 0.05 + adv * 0.05, q.z * 0.11)) * 0.65
          + nz(vec3(q.x * 0.31 + 0.3, q.y * 0.16 + adv * 0.13, q.z * 0.31)) * 0.35;
  float edge = exp(-r * r * mix(2.6, 1.5, uVac));
  float sheath = edge * fall * (0.35 + 1.3 * n * n);
  // supersonic core
  float Lc = uLen * mix(0.42, 0.22, uVac);
  float Rc = uRe * mix(0.82, 0.9, uVac) * (1.0 - 0.6 * smoothstep(0.0, Lc, x));
  float coreFall = 1.0 - smoothstep(0.1 * Lc, Lc, x);
  float rc = rho / Rc;
  float core = exp(-rc * rc * 2.6) * coreFall;
  // shock diamonds: luminous recompression regions repeating every ~1.6 exit diameters... fading out
  float dia = 0.0;
  if (uSolid < 0.5) {
    float cell = 1.55 * uRe * 2.0 * (1.0 + 2.0 * uVac);
    float ph = x / cell + 0.35;
    float f = fract(ph), idx = floor(ph);
    float width = Rc * 0.95 * clamp(min(f / 0.22, (0.95 - f) / 0.6), 0.0, 1.0);
    float body = 1.0 - smoothstep(width * 0.75, width * 1.05 + 1e-4, rho);
    float disk = exp(-pow((f - 0.2) / 0.035, 2.0)) * (1.0 - smoothstep(0.55 * Rc, 0.75 * Rc, rho));
    dia = (body * 0.8 + disk * 1.6) * exp(-idx * 0.42) * step(idx, 7.0) * (1.0 - uVac) * coreFall;
    core *= 0.45 + 0.55 * clamp(body + disk, 0.0, 1.0) * (1.0 - uVac) + 0.55 * uVac;
  }
  vec3 cCore = mix(vec3(1.0, 0.9, 0.75), vec3(1.0, 0.82, 0.55), uSolid);
  vec3 cSheath = mix(mix(vec3(1.0, 0.5, 0.14), vec3(0.95, 0.27, 0.06), smoothstep(0.1, 0.8, s)), vec3(1.0, 0.62, 0.25), uSolid);
  vec3 cVac = vec3(0.85, 0.52, 0.3);
  float kS = mix(1.7, 0.1, uVac) * mix(1.0, 1.4, uSolid);
  vec3 col = cSheath * sheath * kS * 0.5
           + cVac * edge * fall * uVac * 0.03 * (0.6 + 0.8 * n)
           + cCore * core * mix(9.0, 6.0, uVac) * mix(1.0, 1.6, uSolid)
           + vec3(1.0, 0.93, 0.8) * dia * 16.0;
  // sun scattered by the exhaust gas (shows the vacuum plume as a pale, sunlit cloud)
  dens = edge * fall;
  col += uSunCol * dens * uVac * 0.012 * (0.5 + n);
  // splash: hot gas fanning out along the ground
  col += cSheath * splash * edge * fall * 0.8;
  return col;
}

void main() {
  vec3 ro = uCamL;
  vec3 rd = normalize(vPos - ro);
  // ray vs. bounding cylinder (radius uBoundR about y, y in [-uBoundL, 0])
  float a = dot(rd.xz, rd.xz), b = dot(ro.xz, rd.xz), c = dot(ro.xz, ro.xz) - uBoundR * uBoundR;
  float disc = b * b - a * c;
  if (disc <= 0.0 || a < 1e-9) discard;
  float sq = sqrt(disc);
  float t0 = (-b - sq) / a, t1 = (-b + sq) / a;
  float ya = (0.0 - ro.y) / rd.y, yb = (-uBoundL - ro.y) / rd.y;
  float tymin = min(ya, yb), tymax = max(ya, yb);
  t0 = max(max(t0, tymin), 0.0); t1 = min(t1, tymax);
  bool inside = c < 0.0 && ro.y < 0.0 && ro.y > -uBoundL;
  if (inside == gl_FrontFacing) discard;      // outside: front faces only; inside: back faces only
  if (t1 <= t0) discard;
  int N = uSteps;
  float ds = (t1 - t0) / float(N);
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  vec3 acc = vec3(0.0);
  float dens;
  for (int i = 0; i < 64; i++) {
    if (i >= N) break;
    vec3 p = ro + rd * (t0 + (float(i) + jitter) * ds);
    acc += emit(p, dens);
  }
  acc *= ds / uRe * uTh * uBright;
  gl_FragColor = vec4(acc, 1.0);   // additive blending = (SRC_ALPHA, ONE)
}`;

export class Plume {
  /**
   * @param re nozzle exit radius (m)
   * @param opts {solid: bool, quality: 'high'|'low'}
   */
  constructor(re, { solid = false, quality = 'high' } = {}) {
    this.re = re;
    this.solid = solid;
    this.group = new THREE.Group();
    this.length = 0;
    this.intensity = 0;
    this.vac = 0;
    this.u = {
      uNoise: { value: noise3D() },
      uCamL: { value: new THREE.Vector3() },
      uRe: { value: re }, uLen: { value: 10 * re }, uVac: { value: 0 }, uTh: { value: 0 }, uT: { value: 0 },
      uSolid: { value: solid ? 1 : 0 }, uBoundR: { value: 3 * re }, uBoundL: { value: 12 * re }, uBright: { value: 1 },
      uGround: { value: new THREE.Vector4(0, 0, 0, 1e9) },
      uSunL: { value: new THREE.Vector3(0, 1, 0) }, uSunCol: { value: new THREE.Vector3(1, 1, 1) },
      uSteps: { value: quality === 'low' ? 20 : 40 },
      uScale: { value: new THREE.Vector3(1, 1, 1) },
    };
    this.mat = new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: VERT, fragmentShader: FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    // unit cylinder, top at y=0, bottom at y=-1 (scaled to the bounds each frame)
    this.mesh = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 32, 1, false).translate(0, -0.5, 0), this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 20;
    const inv = new THREE.Matrix4(), tmp = new THREE.Vector3();
    this.mesh.onBeforeRender = (renderer, scene, camera) => {
      // camera in plume-local metres (the group carries no scale; the mesh is scaled to the bounds)
      inv.copy(this.group.matrixWorld).invert();
      this.u.uCamL.value.copy(camera.getWorldPosition(tmp)).applyMatrix4(inv);
    };
    this.group.add(this.mesh);
  }

  setQuality(q) { this.u.uSteps.value = q === 'low' ? 20 : 40; }

  /**
   * @param throttle 0..1, @param alt altitude (m, ambient pressure proxy)
   * @param groundL ground plane in plume-local coordinates {n: Vector3, d} or null
   */
  update(throttle, alt, time, groundL = null, sunL = null, sunCol = null) {
    const th = clamp(throttle || 0, 0, 1.2);
    const on = th > 0.01;
    this.mesh.visible = on;
    this.intensity = on ? th : 0;
    if (!on) { this.length = 0; return; }
    const re = this.re;
    // 0 at sea level -> 1 in near-vacuum (pressure ~ exp(-h / 7.5 km))
    // ambient pressure ratio ln(p0/p) = h / 7.5 km; the jet starts ballooning above ~6 km and is a
    // fully under-expanded vacuum plume by ~40 km
    const vac = clamp((Math.max(alt || 0, 0) / 7500 - 0.8) / 4.5, 0, 1) ** 1.2;
    this.vac = vac;
    const len = this.solid
      ? re * (14 + 10 * th)
      : re * (11 + 17 * Math.pow(th, 0.85)) * (1 + 2.2 * vac);
    this.length = len;
    const u = this.u;
    u.uLen.value = len;
    u.uVac.value = vac;
    u.uTh.value = 0.35 + 0.65 * th;
    u.uT.value = time;
    u.uBright.value = 1;
    const xr = len / re;
    let R = re * (1 + (0.055 + 0.145 * vac) * xr + vac * 2.2 * Math.sqrt(xr)) * 1.6;
    let Lb = len * 1.1;
    if (groundL) {
      u.uGround.value.set(groundL.n.x, groundL.n.y, groundL.n.z, groundL.d);
      // height of the exit plane above the ground ~ d; if the ground cuts the plume, add the splash width
      const hExit = groundL.d;
      if (hExit < len) { R = Math.max(R, re * 4.5); Lb = Math.min(Lb, Math.max(hExit, 0) / Math.max(Math.abs(groundL.n.y), 0.2) + re * 3); }
    } else u.uGround.value.set(0, 0, 0, 1e9);
    u.uBoundR.value = R;
    u.uBoundL.value = Math.max(Lb, re);
    this.mesh.scale.set(R, Math.max(Lb, re), R);
    u.uScale.value.copy(this.mesh.scale);
    if (sunL) u.uSunL.value.copy(sunL);
    if (sunCol) u.uSunCol.value.set(sunCol.r, sunCol.g, sunCol.b);
  }

  dispose() { this.mesh.geometry.dispose(); this.mat.dispose(); }
}

// ------------------------------------------------------------------ ground dust / debris
const DUST_VERT = /* glsl */ `
attribute vec4 aSeed;   // angle, speed, phase, size
uniform float uT, uR, uI;
uniform vec3 uE1, uE2, uUp;
varying float vA; varying vec2 vUv; varying float vAge; varying float vSeed;
void main() {
  float rate = 0.35 + 0.25 * aSeed.y;
  float age = fract(uT * rate + aSeed.z);
  float ang = aSeed.x;
  float dist = uR * (0.25 + age * (0.9 + 1.4 * aSeed.y));
  float hgt = uR * (0.05 + 0.32 * pow(age, 1.3) * (0.6 + aSeed.w));
  vec3 c = (uE1 * cos(ang) + uE2 * sin(ang)) * dist + uUp * hgt;
  float size = uR * (0.18 + 0.75 * age) * (0.6 + 0.6 * aSeed.w);
  vec4 mv = modelViewMatrix * vec4(c, 1.0);
  mv.xy += (uv - 0.5) * size * 2.0;
  gl_Position = projectionMatrix * mv;
  vUv = uv; vAge = age; vSeed = aSeed.z;
  vA = uI * pow(1.0 - age, 1.6) * smoothstep(0.0, 0.08, age);
}`;

const DUST_FRAG = /* glsl */ `
${GLSL_NOISE}
uniform vec3 uCol; uniform vec3 uLit; uniform vec3 uGlow;
varying float vA; varying vec2 vUv; varying float vAge; varying float vSeed;
void main() {
  vec2 q = vUv - 0.5;
  float d = length(q) * 2.0;
  float n = vnoise(vUv * 5.0 + vSeed * 31.0) * 0.6 + vnoise(vUv * 11.0 + vSeed * 17.0) * 0.4;
  float a = (1.0 - smoothstep(0.35, 1.0, d + (n - 0.5) * 0.5)) * vA;
  if (a < 0.003) discard;
  vec3 col = uCol * uLit * (0.75 + 0.35 * (0.5 - q.y)) + uGlow * (1.0 - vAge) * 0.6;
  gl_FragColor = vec4(col, a * 0.42);
}`;

export class Dust {
  constructor({ count = 56 } = {}) {
    const geo = new THREE.InstancedBufferGeometry();
    const base = new THREE.PlaneGeometry(1, 1);
    geo.index = base.index;
    geo.attributes.position = base.attributes.position;
    geo.attributes.uv = base.attributes.uv;
    const seeds = new Float32Array(count * 4);
    let s = 12345;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < count; i++) {
      seeds[i * 4] = (i / count) * Math.PI * 2 + rnd() * 0.4;
      seeds[i * 4 + 1] = rnd();
      seeds[i * 4 + 2] = rnd();
      seeds[i * 4 + 3] = rnd();
    }
    geo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seeds, 4));
    geo.instanceCount = count;
    this.u = {
      uT: { value: 0 }, uR: { value: 5 }, uI: { value: 0 },
      uE1: { value: new THREE.Vector3(1, 0, 0) }, uE2: { value: new THREE.Vector3(0, 0, 1) }, uUp: { value: new THREE.Vector3(0, 1, 0) },
      uCol: { value: new THREE.Vector3(0.42, 0.37, 0.31) }, uLit: { value: new THREE.Vector3(1, 1, 1) }, uGlow: { value: new THREE.Vector3(0, 0, 0) },
    };
    this.mesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: DUST_VERT, fragmentShader: DUST_FRAG, transparent: true, depthWrite: false,
    }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 18;  // after terrain + vehicle
    this.mesh.visible = false;
  }

  /**
   * @param pos render-space impingement point, @param up local up, @param intensity 0..1
   * @param radius cloud scale (m), @param lit sun+sky light colour, @param glow plume glow colour
   */
  update(pos, up, intensity, radius, time, lit, glow) {
    this.mesh.visible = intensity > 0.01;
    if (!this.mesh.visible) return;
    this.mesh.position.copy(pos);
    const u = this.u;
    u.uT.value = time;
    u.uR.value = radius;
    u.uI.value = clamp(intensity, 0, 1);
    u.uUp.value.copy(up);
    const e1 = new THREE.Vector3(1, 0, 0);
    if (Math.abs(up.x) > 0.9) e1.set(0, 0, 1);
    e1.addScaledVector(up, -e1.dot(up)).normalize();
    u.uE1.value.copy(e1);
    u.uE2.value.crossVectors(up, e1).normalize();
    u.uLit.value.set(lit.r, lit.g, lit.b);
    u.uGlow.value.set(glow.r, glow.g, glow.b);
  }

  dispose() { this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
}

// ------------------------------------------------------------------ solid-motor smoke trail
const SMOKE_VERT = /* glsl */ `
attribute vec3 aCenter; attribute vec2 aInfo;   // (birth time, seed)
uniform float uT; uniform float uScale;
varying vec2 vUv; varying float vA; varying float vSeed;
void main() {
  float age = uT - aInfo.x;
  vSeed = aInfo.y;
  if (age < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float size = uScale * (0.6 + 4.0 * sqrt(age) + 0.25 * age);
  vec4 mv = modelViewMatrix * vec4(aCenter + vec3(0.0, 0.05 * age * uScale, 0.0), 1.0);
  mv.xy += (uv - 0.5) * size;
  gl_Position = projectionMatrix * mv;
  vUv = uv;
  vA = 0.55 * exp(-age / 25.0) * smoothstep(0.0, 0.15, age + 0.02);
}`;
const SMOKE_FRAG = /* glsl */ `
${GLSL_NOISE}
uniform vec3 uLit;
varying vec2 vUv; varying float vA; varying float vSeed;
void main() {
  vec2 q = vUv - 0.5;
  float n = vnoise(vUv * 4.0 + vSeed * 13.0) * 0.6 + vnoise(vUv * 9.0 + vSeed * 7.0) * 0.4;
  float a = (1.0 - smoothstep(0.2, 0.5, length(q) + (n - 0.5) * 0.25)) * vA;
  if (a < 0.004) discard;
  gl_FragColor = vec4(uLit * (0.78 + 0.22 * n), a);
}`;

/** White-grey smoke left along the path while a solid motor burns (deterministic in replay time). */
export class SmokeTrail {
  /** @param pts [{W: Vector3 (float64 W-space), t: birth time}] */
  constructor(pts, scale = 0.25) {
    this.pts = pts;
    const n = Math.max(pts.length, 1);
    const geo = new THREE.InstancedBufferGeometry();
    const base = new THREE.PlaneGeometry(1, 1);
    geo.index = base.index;
    geo.attributes.position = base.attributes.position;
    geo.attributes.uv = base.attributes.uv;
    this.center = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    const info = new Float32Array(n * 2);
    pts.forEach((p, i) => { info[i * 2] = p.t; info[i * 2 + 1] = (i * 0.618) % 1; });
    geo.setAttribute('aCenter', this.center);
    geo.setAttribute('aInfo', new THREE.InstancedBufferAttribute(info, 2));
    geo.instanceCount = pts.length;
    this.u = { uT: { value: 0 }, uScale: { value: scale }, uLit: { value: new THREE.Vector3(1, 1, 1) } };
    this.mesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
      uniforms: this.u, vertexShader: SMOKE_VERT, fragmentShader: SMOKE_FRAG, transparent: true, depthWrite: false,
    }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 17;
    this.mesh.visible = pts.length > 0;
  }

  update(t, origin, lit) {
    if (!this.pts.length) return;
    this.u.uT.value = t;
    this.u.uLit.value.set(lit.r, lit.g, lit.b);
    const a = this.center.array;
    this.pts.forEach((p, i) => { a[i * 3] = p.W.x - origin.x; a[i * 3 + 1] = p.W.y - origin.y; a[i * 3 + 2] = p.W.z - origin.z; });
    this.center.needsUpdate = true;
  }

  dispose() { this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
}

export { smoothstep };
