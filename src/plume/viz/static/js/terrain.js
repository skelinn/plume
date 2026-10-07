// Terrain (heightmaps from /api/terrain) and the infinite-looking ground plane.
//
// Heightmaps are cut into chunks of CHUNK cells.  Each chunk has its own float64 anchor
// (a vertex near its centre); vertex positions are stored in float32 *relative to that anchor*
// and the mesh is placed at (anchor - origin) every frame.  Together with the camera-near
// origin this keeps float32 error at the sub-centimetre level where it matters, even for
// 800 km maps on a spherical Earth.

import * as THREE from 'three';
import { GLSL_COMMON } from './shaders.js';

const CHUNK = 64;

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

const VERT = /* glsl */ `
  attribute float aElev;
  attribute vec2 aMap;
  varying vec3 vN; varying float vElev; varying vec2 vMap; varying vec3 vView; varying vec3 vRel;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vView = -mv.xyz;
    vN = normal;                       // meshes are never rotated, object normal == render-space normal
    vElev = aElev; vMap = aMap;
    vRel = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * mv;
  }`;

const FRAG = /* glsl */ `
  ${GLSL_COMMON}
  uniform float uZMin; uniform float uZMax; uniform float uContour; uniform float uDetail;
  varying vec3 vN; varying float vElev; varying vec2 vMap; varying vec3 vView; varying vec3 vRel;
  vec3 ramp(float e) {
    vec3 c0 = vec3(0.075, 0.165, 0.175);
    vec3 c1 = vec3(0.165, 0.265, 0.185);
    vec3 c2 = vec3(0.36, 0.335, 0.235);
    vec3 c3 = vec3(0.50, 0.46, 0.41);
    vec3 c4 = vec3(0.86, 0.89, 0.93);
    vec3 c = mix(c0, c1, smoothstep(0.0, 0.25, e));
    c = mix(c, c2, smoothstep(0.25, 0.5, e));
    c = mix(c, c3, smoothstep(0.5, 0.75, e));
    c = mix(c, c4, smoothstep(0.78, 1.0, e));
    return c;
  }
  void main() {
    vec3 N = normalize(vN);
    float e = clamp((vElev - uZMin) / max(uZMax - uZMin, 1.0), 0.0, 1.0);
    vec3 albedo = ramp(e);
    albedo = mix(albedo, albedo * vec3(0.8, 0.75, 0.72), smoothstep(0.1, 0.45, 1.0 - N.y));   // steep = rock
    float diff = max(dot(N, uSunDir), 0.0);
    vec3 col = albedo * (0.24 + 0.95 * diff);
    // map-space grid (100 m / 1 km / 10 km / 100 km) and elevation contours
    float g = gridLine(vMap, 100.0) * 0.10 + gridLine(vMap, 1000.0) * 0.18
            + gridLine(vMap, 10000.0) * 0.28 + gridLine(vMap, 100000.0) * 0.4;
    col = mix(col, vec3(0.55, 0.78, 0.95), clamp(g, 0.0, 0.5));
    float cl = contourLine(vElev, uContour) * 0.16 + contourLine(vElev, uContour * 5.0) * 0.16;
    col = mix(col, vec3(1.0, 0.9, 0.7), cl);
    col += plumeGlow(vRel, albedo);
    col = applyFog(col, length(vView));
    gl_FragColor = vec4(col, 1.0);
  }`;

export class Terrain {
  constructor(frame, env) {
    this.frame = frame;
    this.env = env;
    this.group = new THREE.Group();
    this.items = [];
    this.zUniforms = { uZMin: { value: 0 }, uZMax: { value: 1 } };
    this.zMin = Infinity;
    this.zMax = -Infinity;
    this._zSet = false;
    this.layers = [];
  }

  /** Terrain height (m, including the detail-tile lift) at map coords, or null outside all layers. */
  heightAt(u, v) {
    for (let k = this.layers.length - 1; k >= 0; k--) { // detail tiles were added last
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

  /**
   * Add a heightmap layer.
   * @param data {meta, heights} from fetchTerrain
   * @param opts {detail: draw above the base layer, lift: metres}
   */
  addLayer({ meta, heights }, { detail = false, lift = 0 } = {}) {
    const { nx, ny } = meta;
    this.layers.push({ meta, heights, lift });
    const dx = (meta.x_max - meta.x_min) / (nx - 1);
    const dy = (meta.y_max - meta.y_min) / (ny - 1);
    const frame = this.frame;
    if (!this._zSet) { // the base layer defines the colour ramp; detail tiles reuse it
      this.zUniforms.uZMin.value = meta.z_min;
      this.zUniforms.uZMax.value = Math.max(meta.z_max, meta.z_min + 1);
      this._zSet = true;
    }

    // Full-grid positions in float64 W-space.
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

    const contour = niceStep((meta.z_max - meta.z_min) / 14 || 1);
    const material = new THREE.ShaderMaterial({
      uniforms: { ...this.env, ...this.zUniforms, uContour: { value: contour }, uDetail: { value: detail ? 1 : 0 } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.DoubleSide,
      // Detail tiles are drawn after the base (renderOrder) with an ALWAYS depth function: they
      // overwrite the base's depth even where they dip metres below the coarse mesh, which no
      // polygon offset can fix robustly at every distance.  (depthTest:false would also disable
      // the depth *write* in WebGL.)  Everything else (rocket, pads) has a higher renderOrder, so
      // it still tests against the tile's depth.
      depthFunc: detail ? THREE.AlwaysDepth : THREE.LessEqualDepth,
    });

    const nrm = [0, 0, 0];
    const normalAt = (i, j) => {
      const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, nx - 1);
      const j0 = Math.max(j - 1, 0), j1 = Math.min(j + 1, ny - 1);
      const a = (j * nx + i0) * 3, b = (j * nx + i1) * 3, c = (j0 * nx + i) * 3, d = (j1 * nx + i) * 3;
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[d] - P[c], vy = P[d + 1] - P[c + 1], vz = P[d + 2] - P[c + 2];
      // east x north = up
      let x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
      const l = Math.hypot(x, y, z) || 1;
      nrm[0] = x / l; nrm[1] = y / l; nrm[2] = z / l;
    };

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
        // Triangles wound CCW seen from above (east = +x, north = -z in three space).
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
    for (const { mesh } of this.items) { mesh.geometry.dispose(); mesh.material.dispose(); }
    this.items = [];
    this.group.clear();
  }
}

/** Large flat ground with a multi-scale grid; recentred on the camera in 1 km steps. */
export class GroundPlane {
  constructor(env) {
    const geo = new THREE.PlaneGeometry(80000, 80000).rotateX(-Math.PI / 2);
    const mat = new THREE.ShaderMaterial({
      uniforms: { ...env },
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 2,
      vertexShader: /* glsl */ `
        varying vec2 vLocal; varying vec3 vView; varying vec3 vRel;
        void main() {
          vLocal = position.xz;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vView = -mv.xyz;
          vRel = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        ${GLSL_COMMON}
        varying vec2 vLocal; varying vec3 vView; varying vec3 vRel;
        void main() {
          vec3 albedo = vec3(0.075, 0.098, 0.135);
          float n = vnoise(vLocal * 0.35) * 0.5 + vnoise(vLocal * 1.7) * 0.25;
          albedo *= 0.9 + 0.25 * n;
          float g = gridLine(vLocal, 1.0) * 0.07 + gridLine(vLocal, 10.0) * 0.14
                  + gridLine(vLocal, 100.0) * 0.22 + gridLine(vLocal, 1000.0) * 0.34;
          vec3 col = albedo * (0.9 + 0.5 * max(uSunDir.y, 0.0));
          col = mix(col, vec3(0.42, 0.66, 0.9), clamp(g, 0.0, 0.7));
          col += plumeGlow(vRel, albedo);
          col = applyFog(col, length(vView));
          // the plane is finite (80 km): melt its edge into the haze so it never shows a seam
          float edge = smoothstep(0.55, 0.95, max(abs(vLocal.x), abs(vLocal.y)) / 40000.0);
          col = mix(col, uFogColor, edge);
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.renderOrder = 1;
    this.mesh.frustumCulled = false;
  }

  layout(camW, origin) {
    // multiples of 1 km keep the grid pinned to the world while the plane follows the camera
    const cx = Math.round(camW.x / 1000) * 1000, cz = Math.round(camW.z / 1000) * 1000;
    this.mesh.position.set(cx - origin.x, -origin.y, cz - origin.z);
  }

  dispose() { this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
}
