// Terrain (heightmaps from /api/terrain) and the flat ground plane, both shaded with a natural
// palette on MeshStandardMaterial (so they get the sun, soft shadows and sky lighting):
//   scrub / dry grass / soil patches by multi-scale noise, rock on steep slopes and high ground,
//   snow on gentle slopes at altitude, compacted gravel aprons and old scorch around pads,
//   distance-faded fine noise and a derivative-based bump for close-up detail.
// The engineering view adds a metric grid and elevation contours (uEng).
//
// Heightmaps are cut into chunks of CHUNK cells, each with its own float64 anchor; vertex positions
// are stored in float32 relative to the anchor and the mesh is placed at (anchor - origin) every
// frame, keeping float32 error sub-centimetre near the camera (see coords.js).

import * as THREE from 'three';
import { GLSL_NOISE } from './shaders.js';

const CHUNK = 64;
const MAX_PADS = 6;
const fetchCache = new Map();

/** Fetch metadata + float32 heights; results are cached per (id, maxDim). */
export function fetchTerrain(id, maxDim = 1024) {
  const key = `${id}|${maxDim}`;
  if (!fetchCache.has(key)) {
    const base = '/api/terrain/' + id.split('/').map(encodeURIComponent).join('/');
    const p = (async () => {
      const mr = await fetch(`${base}?max_dim=${maxDim}`);
      if (!mr.ok) throw new Error(`terrain ${id}: HTTP ${mr.status}`);
      const meta = await mr.json();
      const hr = await fetch(`${base}/heights?max_dim=${maxDim}`);
      if (!hr.ok) throw new Error(`terrain ${id} heights: HTTP ${hr.status}`);
      const heights = new Float32Array(await hr.arrayBuffer());
      if (heights.length !== meta.nx * meta.ny) throw new Error(`terrain ${id}: bad heights size`);
      return { meta, heights };
    })();
    p.catch(() => fetchCache.delete(key));
    fetchCache.set(key, p);
  }
  return fetchCache.get(key);
}

function niceStep(x) {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-6))));
  const m = x / e;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e;
}

/** Shared uniforms for all ground materials of one world. */
export function makeGroundUniforms() {
  return {
    uZMin: { value: 0 }, uZMax: { value: 1 }, uContour: { value: 10 }, uEng: { value: 0 },
    uPadN: { value: 0 },
    uPadP: { value: Array.from({ length: MAX_PADS }, () => new THREE.Vector3()) }, // (u, v, radius)
    uPlaneC: { value: new THREE.Vector2() },
    uBounds: { value: new THREE.Vector4(-1e9, -1e9, 1e9, 1e9) },
    uMapRef: { value: new THREE.Vector2() },   // 1024 m multiple near the camera (see vMapF)   // base terrain extent (map u0, v0, u1, v1)
    uUpW: { value: new THREE.Vector3(0, 1, 0) },
  };
}

const PARS = /* glsl */ `
uniform float uZMin; uniform float uZMax; uniform float uContour; uniform float uEng;
uniform int uPadN; uniform vec3 uPadP[${MAX_PADS}]; uniform vec4 uBounds;
varying float vElev; varying vec2 vMap; varying vec2 vMapF; varying vec3 vNW;
${GLSL_NOISE}
// Fine detail must not be evaluated on raw map coordinates: at 700 km float32 only resolves ~6 cm.
// vMapF = map - uMapRef (uMapRef: a 1024 m multiple near the camera) is small and exact, and every
// fine octave is periodic over 1024 m, so moving the reference never changes the pattern.
float vnoiseP(vec2 p, float N, float seed) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  vec2 i0 = mod(i, N) + seed, i1 = mod(i + 1.0, N) + seed;
  return mix(mix(hash21(i0), hash21(vec2(i1.x, i0.y)), f.x), mix(hash21(vec2(i0.x, i1.y)), hash21(i1), f.x), f.y);
}
float fineN(float s, float seed) { return vnoiseP(vMapF / s, 1024.0 / s, seed); }
float gAmt;
`;

const ALBEDO = /* glsl */ `
{
  vec2 m = vMap;
  float e = clamp((vElev - uZMin) / max(uZMax - uZMin, 1.0), 0.0, 1.0);
  vec3 N = normalize(vNW);
  float slope = 1.0 - clamp(dot(N, normalize(vAtmW - uPlanetC)), 0.0, 1.0);   // vAtmW, uPlanetC: atmosphere patch
  // anti-aliased detail: fade octaves finer than ~2 px
  float px = max(length(fwidth(vMapF)), 1e-4);
  float n0 = fbm2(m / 2600.0);
  float n1 = fbm2(mat2(0.8, 0.6, -0.6, 0.8) * m / 420.0 + 7.3);
  float n2 = fbm2(mat2(0.6, -0.8, 0.8, 0.6) * m / 55.0 + 3.1);
  float n3 = fineN(8.0, 1.7) * (1.0 - smoothstep(0.5, 2.0, px));
  float n4 = fineN(1.0, 9.2) * (1.0 - smoothstep(0.08, 0.4, px));
  vec3 scrub = vec3(0.060, 0.072, 0.036);
  vec3 grass = vec3(0.135, 0.120, 0.066);
  vec3 soil = vec3(0.190, 0.142, 0.095);
  vec3 sand = vec3(0.300, 0.250, 0.180);
  vec3 rock = vec3(0.160, 0.150, 0.140);
  vec3 snow = vec3(0.78, 0.80, 0.84);
  vec3 c = mix(scrub, grass, smoothstep(0.3, 0.7, n0 * 0.6 + n1 * 0.4));
  c = mix(c, soil, smoothstep(0.58, 0.78, n1 * 0.7 + n2 * 0.3) * 0.85);
  c = mix(c, sand, smoothstep(0.7, 0.9, n0) * smoothstep(0.4, 0.0, e) * 0.5);
  float rocky = smoothstep(0.18, 0.42, slope + (n2 - 0.5) * 0.15) + smoothstep(0.7, 0.95, e) * 0.5;
  c = mix(c, rock * (0.85 + 0.3 * vnoise(vec2(m.x / 40.0, vElev / 6.0))), clamp(rocky, 0.0, 1.0));
  float sn = smoothstep(3200.0, 3800.0, vElev + (n1 - 0.5) * 500.0) * (1.0 - smoothstep(0.25, 0.45, slope));
  c = mix(c, snow, sn);
  c *= 0.82 + 0.22 * n2 + 0.16 * (n3 - 0.5) + 0.12 * (n4 - 0.5);
  // close-up texture: soil mottling, grass tufts and pebbles, only where they resolve
  float f1 = 1.0 - smoothstep(0.02, 0.25, px), f2 = 1.0 - smoothstep(0.006, 0.06, px);
  if (f1 > 0.0) {
    float m1 = fineN(0.5, 2.3), m2 = fineN(0.125, 8.1);
    float tuft = smoothstep(0.62, 0.8, fineN(0.25, 4.4) * 0.7 + m2 * 0.3);
    c *= 1.0 + f1 * ((m1 - 0.5) * 0.35 + (m2 - 0.5) * 0.25);
    c = mix(c, vec3(0.05, 0.06, 0.028), tuft * f1 * 0.55 * (1.0 - clamp(rocky, 0.0, 1.0)));
    // pebbles: jittered round stones on a 9 cm lattice
    vec2 pg = vMapF / 0.0625, pc = mod(floor(pg), 16384.0), pf = fract(pg);
    vec2 pj = vec2(hash21(pc + 2.7), hash21(pc + 5.9)) * 0.6 + 0.2;
    float peb = step(0.9, hash21(pc)) * (1.0 - smoothstep(0.12, 0.2, length(pf - pj))) * f2;
    c = mix(c, vec3(0.22, 0.2, 0.17) * (0.75 + 0.5 * hash21(pc + 1.3)), peb * 0.55);
  }
  // fade to the planet's mean tone at the edge of the base map (the sky shader draws the planet beyond)
  float ex = min(min(m.x - uBounds.x, uBounds.z - m.x), min(m.y - uBounds.y, uBounds.w - m.y));
  float span = min(uBounds.z - uBounds.x, uBounds.w - uBounds.y);
  c = mix(c, vec3(0.092, 0.09, 0.054), (1.0 - smoothstep(0.0, 0.06 * span, ex)) * 0.85);
  // pads: compacted gravel apron, old scorch
  for (int i = 0; i < ${MAX_PADS}; i++) {
    if (i >= uPadN) break;
    vec3 P = uPadP[i];
    float r = length(m - P.xy) / max(abs(P.z), 0.5);
    // the slab mesh replaces the ground inside its footprint (it is seated where the vehicle stood)
    if (P.z > 0.0 && r < 0.995) discard;
    float ap = 1.0 - smoothstep(1.6, 2.6 + 0.6 * n2, r);
    c = mix(c, vec3(0.24, 0.215, 0.185) * (0.9 + 0.2 * n3), ap * 0.9);
    float sc = (1.0 - smoothstep(0.9, 2.2 + 0.8 * n1, r)) * (0.6 + 0.4 * n2);
    c = mix(c, vec3(0.04, 0.037, 0.034), sc * 0.55);
  }
  // engineering view: map grid + contours
  if (uEng > 0.5) {
    float g = gridLine(m, 100.0) * 0.18 + gridLine(m, 1000.0) * 0.28 + gridLine(m, 10000.0) * 0.36;
    float cl = uZMax - uZMin > 2.0 ? contourLine(vElev, uContour) * 0.22 + contourLine(vElev, uContour * 5.0) * 0.3 : 0.0;
    c = mix(c, vec3(0.62), clamp(g + cl, 0.0, 0.6));
  }
  diffuseColor.rgb = c;
  gAmt = rocky;
}
`;

const BUMP = /* glsl */ `
{
  float px = max(length(fwidth(vMapF)), 1e-4);
  float h = (fineN(4.0, 0.0) * 0.6 + fineN(1.0, 5.0) * 0.25) * (1.0 - smoothstep(0.05, 0.6, px)) * 0.18
          + fbm2(vMap / 28.0) * 1.4 * (1.0 - smoothstep(0.5, 6.0, px));
  vec3 sx = dFdx(-vViewPosition), sy = dFdy(-vViewPosition);
  vec3 R1 = cross(sy, normal), R2 = cross(normal, sx);
  float det = dot(sx, R1);
  vec3 grad = sign(det) * (dFdx(h) * R1 + dFdy(h) * R2);
  normal = normalize(abs(det) * normal - grad * 0.6);
}
`;

/**
 * Ground material.
 * @param U makeGroundUniforms()
 * @param atmosphere Atmosphere (aerial perspective)
 * @param plane true for the flat plane (map coords from position + uPlaneC)
 */
export function groundMaterial(U, atmosphere, { plane = false, detail = false } = {}) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.94, metalness: 0 });
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, U);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        ${plane ? 'uniform vec2 uPlaneC;' : 'attribute float aElev; attribute vec2 aMap;'}
        uniform vec2 uMapRef;
        varying float vElev; varying vec2 vMap; varying vec2 vMapF; varying vec3 vNW;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        ${plane
    ? 'vElev = 0.0; vMap = vec2(position.x, -position.z) + uPlaneC; vMapF = vec2(position.x, -position.z) + (uPlaneC - uMapRef);'
    : 'vElev = aElev; vMap = aMap; vMapF = aMap - uMapRef;'}
        vNW = normal;`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform vec3 uUpW;\n${PARS}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${ALBEDO}`)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(0.95, 0.82, gAmt);')
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${BUMP}`);
  };
  m.customProgramCacheKey = () => `ground|${plane}`;
  if (detail) {
    // detail tiles overwrite the coarse base's depth where they dip below it (see addLayer)
    m.depthFunc = THREE.AlwaysDepth;
  }
  atmosphere.patch(m);
  return m;
}

export class Terrain {
  constructor(frame, atmosphere, U) {
    this.frame = frame;
    this.atm = atmosphere;
    this.U = U;
    this.group = new THREE.Group();
    this.items = [];
    this.layers = [];
    this._zSet = false;
    this.zMin = Infinity; this.zMax = -Infinity;
    this.matBase = groundMaterial(U, atmosphere);
    this.matDetail = groundMaterial(U, atmosphere, { detail: true });
  }

  /** Terrain height (m, including the detail-tile lift) at map coords, or null outside all layers. */
  heightAt(u, v) {
    for (let k = this.layers.length - 1; k >= 0; k--) {
      const { meta, heights, lift } = this.layers[k];
      const { nx, ny } = meta;
      const fx = ((u - meta.x_min) / (meta.x_max - meta.x_min)) * (nx - 1);
      const fy = ((v - meta.y_min) / (meta.y_max - meta.y_min)) * (ny - 1);
      if (fx < 0 || fy < 0 || fx > nx - 1 || fy > ny - 1) continue;
      const i = Math.min(Math.floor(fx), nx - 2), j = Math.min(Math.floor(fy), ny - 2);
      const tx = fx - i, ty = fy - j;
      const h = heights[j * nx + i] * (1 - tx) * (1 - ty) + heights[j * nx + i + 1] * tx * (1 - ty)
        + heights[(j + 1) * nx + i] * (1 - tx) * ty + heights[(j + 1) * nx + i + 1] * tx * ty;
      return h + lift;
    }
    return null;
  }

  addLayer({ meta, heights }, { detail = false, lift = 0 } = {}) {
    const { nx, ny } = meta;
    this.layers.push({ meta, heights, lift });
    const dx = (meta.x_max - meta.x_min) / (nx - 1);
    const dy = (meta.y_max - meta.y_min) / (ny - 1);
    const frame = this.frame;
    if (!this._zSet) {
      this.U.uZMin.value = meta.z_min;
      this.U.uZMax.value = Math.max(meta.z_max, meta.z_min + 1);
      this.U.uContour.value = niceStep((meta.z_max - meta.z_min) / 14 || 1);
      this.U.uBounds.value.set(meta.x_min, meta.y_min, meta.x_max, meta.y_max);
      this._zSet = true;
    }
    const P = new Float64Array(nx * ny * 3);
    const tmp = new THREE.Vector3();
    for (let j = 0; j < ny; j++) {
      const v = meta.y_min + j * dy;
      for (let i = 0; i < nx; i++) {
        const u = meta.x_min + i * dx;
        frame.surface(u, v, heights[j * nx + i] + lift, tmp);
        const k = (j * nx + i) * 3;
        P[k] = tmp.x; P[k + 1] = tmp.y; P[k + 2] = tmp.z;
      }
    }
    const nrm = [0, 0, 0];
    const normalAt = (i, j) => {
      const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, nx - 1);
      const j0 = Math.max(j - 1, 0), j1 = Math.min(j + 1, ny - 1);
      const a = (j * nx + i0) * 3, b = (j * nx + i1) * 3, c = (j0 * nx + i) * 3, d = (j1 * nx + i) * 3;
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[d] - P[c], vy = P[d + 1] - P[c + 1], vz = P[d + 2] - P[c + 2];
      const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
      const l = Math.hypot(x, y, z) || 1;
      nrm[0] = x / l; nrm[1] = y / l; nrm[2] = z / l;
    };
    const material = detail ? this.matDetail : this.matBase;
    for (let cj = 0; cj < ny - 1; cj += CHUNK) {
      for (let ci = 0; ci < nx - 1; ci += CHUNK) {
        const i1 = Math.min(ci + CHUNK, nx - 1), j1 = Math.min(cj + CHUNK, ny - 1);
        const w = i1 - ci + 1, h = j1 - cj + 1;
        const mi = (ci + i1) >> 1, mj = (cj + j1) >> 1;
        const ak = (mj * nx + mi) * 3;
        const anchor = new THREE.Vector3(P[ak], P[ak + 1], P[ak + 2]);
        const pos = new Float32Array(w * h * 3), nor = new Float32Array(w * h * 3);
        const elev = new Float32Array(w * h), map = new Float32Array(w * h * 2);
        let o = 0;
        for (let j = cj; j <= j1; j++) {
          for (let i = ci; i <= i1; i++, o++) {
            const k = (j * nx + i) * 3;
            pos[o * 3] = P[k] - anchor.x; pos[o * 3 + 1] = P[k + 1] - anchor.y; pos[o * 3 + 2] = P[k + 2] - anchor.z;
            normalAt(i, j);
            nor[o * 3] = nrm[0]; nor[o * 3 + 1] = nrm[1]; nor[o * 3 + 2] = nrm[2];
            elev[o] = heights[j * nx + i];
            map[o * 2] = meta.x_min + i * dx; map[o * 2 + 1] = meta.y_min + j * dy;
          }
        }
        const idx = new Uint16Array((w - 1) * (h - 1) * 6);
        let q = 0;
        for (let j = 0; j < h - 1; j++) {
          for (let i = 0; i < w - 1; i++) {
            const a = j * w + i, b = a + 1, c = a + w, d = c + 1;
            idx[q++] = a; idx[q++] = b; idx[q++] = c;
            idx[q++] = b; idx[q++] = d; idx[q++] = c;
          }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
        geo.setAttribute('aElev', new THREE.BufferAttribute(elev, 1));
        geo.setAttribute('aMap', new THREE.BufferAttribute(map, 2));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        geo.computeBoundingSphere();
        const mesh = new THREE.Mesh(geo, material);
        mesh.renderOrder = detail ? 2 : 1;
        mesh.receiveShadow = true;
        this.group.add(mesh);
        this.items.push({ mesh, anchor });
      }
    }
    this.zMin = Math.min(this.zMin, meta.z_min);
    this.zMax = Math.max(this.zMax, meta.z_max);
  }

  layout(origin) {
    for (const { mesh, anchor } of this.items) mesh.position.copy(anchor).sub(origin);
  }

  dispose() {
    for (const { mesh } of this.items) mesh.geometry.dispose();
    this.matBase.dispose(); this.matDetail.dispose();
    this.items = [];
    this.group.clear();
  }
}

/** Large flat ground; recentred on the camera in 1 km steps (map coords stay pinned to the world). */
export class GroundPlane {
  constructor(atmosphere, U) {
    this.U = U;
    const geo = new THREE.PlaneGeometry(120000, 120000, 1, 1).rotateX(-Math.PI / 2);
    this.mat = groundMaterial(U, atmosphere, { plane: true });
    this.mat.polygonOffset = true;
    this.mat.polygonOffsetFactor = 2;
    this.mat.polygonOffsetUnits = 2;
    this.mesh = new THREE.Mesh(geo, this.mat);
    this.mesh.renderOrder = 1;
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true;
  }

  layout(camW, origin) {
    const cx = Math.round(camW.x / 1000) * 1000, cz = Math.round(camW.z / 1000) * 1000;
    this.mesh.position.set(cx - origin.x, -origin.y, cz - origin.z);
    this.U.uPlaneC.value.set(cx, -cz);
  }

  dispose() { this.mesh.geometry.dispose(); this.mat.dispose(); }
}
