// Drone ship and ocean for ship-landing replays (meta.ship, ground type "ocean").
//
// * DroneShip: a dark-grey landing barge built from meta.ship (deck L x W at `freeboard` above the
//   waterline, hull of beam `hull_beam` and draught `draft`, wing extensions, bulwarks, stern
//   equipment), with a procedural steel-deck texture (plate seams, wear, blast scorch) and the
//   painted landing circle at `target_offset`.  Posed every frame from the recorded deck pose
//   (frames.deck_pos / deck_quat: the ship origin is the waterline amidships, ENU, +x = bow).
// * Ocean: the sea surface synthesised from the same wave components the simulator used
//   (meta.ship.sea.waves = [amplitude, omega, direction, phase], linear deep-water waves, ENU):
//   eta(e, n, t) = sum a cos(omega t - k (e cos th + n sin th) + phase), k = omega^2 / g,
//   so the rendered swell matches the recorded heave, pitch and roll.  Vertices are displaced near
//   the camera; the normals are evaluated per pixel from the analytic gradient.

import * as THREE from 'three';

const G = 9.80665;
const MAX_WAVES = 48;

// ------------------------------------------------------------------ small JS noise
function hash(x, y, s = 0) {
  let h = (x * 374761393 + y * 668265263 + s * 1442695041) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}
function vnoise(x, y, seed) {
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = hash(xi, yi, seed), b = hash(xi + 1, yi, seed), c = hash(xi, yi + 1, seed), d = hash(xi + 1, yi + 1, seed);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
function fbm(x, y, seed, oct = 4) {
  let s = 0, a = 0.5, f = 1, n = 0;
  for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f, seed + i * 7); n += a; a *= 0.5; f *= 2; }
  return s / n;
}

/** Steel deck: colour + roughness canvases (u along the ship, v across, 1 px = 1/ppm m). */
function deckTextures(sh) {
  const L = sh.length, W = sh.deck_width, R = sh.target_radius || 10;
  const ppm = Math.min(24, 2048 / L);
  const w = Math.round(L * ppm), h = Math.round(W * ppm);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  const img = g.createImageData(w, h);
  const rough = document.createElement('canvas');
  rough.width = w; rough.height = h;
  const gr = rough.getContext('2d');
  const rimg = gr.createImageData(w, h);
  const [tx, ty] = sh.target_offset || [0, 0];
  const plateL = 12.0, plateW = 2.4;
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const k = (j * w + i) * 4;
      const x = i / ppm - L / 2, y = W / 2 - j / ppm;         // ship frame: x fwd, y port
      const sx = Math.abs((((x / plateL) % 1) + 1.5) % 1 - 0.5) * plateL;
      const sy = Math.abs((((y / plateW) % 1) + 1.5) % 1 - 0.5) * plateW;
      const seam = Math.max(Math.exp(-(sx * sx) / 0.0025), Math.exp(-(sy * sy) / 0.0025));
      const plate = hash(Math.floor(x / plateL), Math.floor(y / plateW), 3) - 0.5;
      const n1 = fbm(x / 6, y / 6, 11), n2 = fbm(x * 1.7, y * 1.7, 12, 3);
      const rr = Math.hypot(x - tx, y - ty) / R;
      const streak = vnoise(Math.atan2(y - ty, x - tx) * 7 + 40, rr * 2.5, 31);
      const scorch = Math.min(Math.exp(-((rr / 0.55) ** 2)) * (0.6 + 0.4 * streak) + 0.3 * Math.exp(-((rr / 1.1) ** 4)) * streak, 1);
      const rust = Math.max(0, fbm(x / 3, y / 3, 41) - 0.62) * 2.2;
      let v = 0.30 + (n1 - 0.5) * 0.08 + (n2 - 0.5) * 0.05 + plate * 0.04;
      v *= 1 - 0.55 * scorch;
      v *= 1 - 0.35 * seam;
      img.data[k] = Math.min(255, 255 * (v + 0.10 * rust));
      img.data[k + 1] = Math.min(255, 255 * (v * 1.0 + 0.04 * rust));
      img.data[k + 2] = Math.min(255, 255 * (v * 1.04));
      img.data[k + 3] = 255;
      const ro = 0.55 + (n2 - 0.5) * 0.2 + 0.25 * rust - 0.15 * scorch;
      rimg.data[k] = rimg.data[k + 1] = rimg.data[k + 2] = Math.min(255, 255 * ro);
      rimg.data[k + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  gr.putImageData(rimg, 0, 0);
  const P = (x, y) => [(x + L / 2) * ppm, (W / 2 - y) * ppm];
  // landing circle, cross and the edge safety line (weathered white paint)
  g.strokeStyle = 'rgba(226, 226, 220, 0.85)';
  g.fillStyle = 'rgba(226, 226, 220, 0.82)';
  const [cx, cy] = P(tx, ty);
  g.lineWidth = 0.6 * ppm;
  g.beginPath(); g.arc(cx, cy, R * ppm, 0, 2 * Math.PI); g.stroke();
  g.lineWidth = 0.25 * ppm;
  g.beginPath(); g.arc(cx, cy, 0.45 * R * ppm, 0, 2 * Math.PI); g.stroke();
  const arm = 0.8 * R * ppm, bar = 0.5 * ppm;
  g.fillRect(cx - arm, cy - bar / 2, 2 * arm, bar);
  g.fillRect(cx - bar / 2, cy - arm, bar, 2 * arm);
  g.lineWidth = 0.3 * ppm;
  g.strokeRect(1.2 * ppm, 1.2 * ppm, w - 2.4 * ppm, h - 2.4 * ppm);
  // yellow/black hazard band at the bow and stern ends
  for (const end of [-1, 1]) {
    const x0 = end < 0 ? 0 : w - 1.0 * ppm;
    for (let y = 0; y < h; y += 1.2 * ppm) {
      g.fillStyle = (Math.round(y / (1.2 * ppm)) % 2) ? 'rgba(200, 160, 40, 0.8)' : 'rgba(25, 25, 25, 0.85)';
      g.fillRect(x0, y, 1.0 * ppm, 1.2 * ppm);
    }
  }
  // wear: speckle the paint
  const wimg = g.getImageData(0, 0, w, h);
  for (let i = 0; i < w * h; i++) {
    const k = i * 4;
    if (wimg.data[k] > img.data[k] + 40 && hash(i % w, (i / w) | 0, 77) < 0.22) {
      wimg.data[k] = img.data[k]; wimg.data[k + 1] = img.data[k + 1]; wimg.data[k + 2] = img.data[k + 2];
    }
  }
  g.putImageData(wimg, 0, 0);
  const map = new THREE.CanvasTexture(c);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  const roughnessMap = new THREE.CanvasTexture(rough);
  return { map, roughnessMap };
}

/** ship frame (x fwd, y port, z up) -> mesh space (three: x, z up -> y, y -> -z) */
const S = (x, y, z) => new THREE.Vector3(x, z, -y);

function box(lx, ly, lz, cx, cy, cz) {
  const geo = new THREE.BoxGeometry(lx, lz, ly);
  const c = S(cx, cy, cz);
  geo.translate(c.x, c.y, c.z);
  return geo;
}

export class DroneShip {
  constructor(sh, atmosphere) {
    this.meta = sh;
    this.group = new THREE.Group();
    this.group.name = 'drone-ship';
    const L = sh.length, W = sh.deck_width, B = sh.hull_beam || W, fb = sh.freeboard, T = sh.draft || 4;
    const tex = deckTextures(sh);
    const deckMat = new THREE.MeshStandardMaterial({ map: tex.map, roughnessMap: tex.roughnessMap, roughness: 1, metalness: 0.35 });
    const hullMat = new THREE.MeshStandardMaterial({ color: 0x434a52, roughness: 0.65, metalness: 0.15 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x1d2024, roughness: 0.75, metalness: 0.3 });
    const rustMat = new THREE.MeshStandardMaterial({ color: 0x3b2d26, roughness: 0.9, metalness: 0.1 });
    const equipMat = new THREE.MeshStandardMaterial({ color: 0x5d6166, roughness: 0.6, metalness: 0.4 });
    this._mats = [deckMat, hullMat, darkMat, rustMat, equipMat];
    for (const m of this._mats) atmosphere.patch(m);
    this._geos = [];
    const add = (geo, mat, { cast = true, recv = true } = {}) => {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = cast; m.receiveShadow = recv;
      this.group.add(m);
      this._geos.push(geo);
      return m;
    };
    // deck plate (textured top), hull below it, a rust band at the waterline
    const deckTh = 0.6;
    const deckGeo = new THREE.PlaneGeometry(L, W).rotateX(-Math.PI / 2);
    deckGeo.translate(0, fb + 0.005, 0);
    add(deckGeo, deckMat, { cast: false });
    add(box(L, W, deckTh, 0, 0, fb - deckTh / 2), darkMat, { cast: false });
    add(box(L * 0.995, B, fb + T - deckTh, 0, 0, (fb - deckTh - T) / 2), hullMat, { cast: false });
    add(box(L * 0.997, B * 1.002, 0.9, 0, 0, 0.1), rustMat, { cast: false, recv: false });
    // wing supports under the deck overhang
    const wing = (W - B) / 2;
    if (wing > 0.5) {
      for (let x = -L / 2 + 6; x <= L / 2 - 6; x += 12) {
        for (const s of [-1, 1]) add(box(0.5, wing, 1.6, x, s * (B / 2 + wing / 2), fb - deckTh - 0.8), darkMat, { cast: false });
      }
    }
    // low bulwarks along the sides and equipment at the stern (thruster units, generators)
    for (const s of [-1, 1]) add(box(L * 0.9, 0.3, 1.1, 0, s * (W / 2 - 0.3), fb + 0.55), equipMat);
    for (const s of [-1, 1]) {
      add(box(6.0, 2.5, 2.6, -L / 2 + 4.5, s * (W / 2 - 4), fb + 1.3), equipMat);
      add(box(3.0, 2.5, 3.4, -L / 2 + 9.5, s * (W / 2 - 4), fb + 1.7), darkMat);
    }
    this.group.traverse((o) => { if (o.isMesh) o.renderOrder = 4; });
    this.W = new THREE.Vector3();
    this.q = new THREE.Quaternion();
  }

  dispose() {
    for (const g of this._geos) g.dispose();
    for (const m of this._mats) { m.map?.dispose(); m.roughnessMap?.dispose(); m.dispose(); }
  }
}

// ------------------------------------------------------------------ ocean
const OCEAN_GLSL = `
uniform float uOceanT;
uniform float uOceanAmp;
uniform vec3 uOceanOrigin;
uniform vec3 uOceanCam;
uniform vec4 uWaves[${MAX_WAVES}];
uniform int uWaveN;
varying vec3 vOceanW;
float oceanEta(vec2 en) {
  float s = 0.0;
  for (int i = 0; i < ${MAX_WAVES}; i++) {
    if (i >= uWaveN) break;
    vec4 w = uWaves[i];
    float k = w.y * w.y / ${G.toFixed(5)};
    s += w.x * cos(w.y * uOceanT - k * (en.x * cos(w.z) + en.y * sin(w.z)) + w.w);
  }
  return s;
}
vec2 oceanGrad(vec2 en) {
  vec2 g = vec2(0.0);
  for (int i = 0; i < ${MAX_WAVES}; i++) {
    if (i >= uWaveN) break;
    vec4 w = uWaves[i];
    float k = w.y * w.y / ${G.toFixed(5)};
    float sn = sin(w.y * uOceanT - k * (en.x * cos(w.z) + en.y * sin(w.z)) + w.w);
    g += w.x * k * sn * vec2(cos(w.z), sin(w.z));
  }
  return g;
}`;

export class Ocean {
  /** @param sea meta.ship.sea  @param atmosphere Atmosphere */
  constructor(sea, atmosphere) {
    const waves = (sea?.waves || []).slice(0, MAX_WAVES);
    this.U = {
      uOceanT: { value: 0 },
      uOceanAmp: { value: 1 },
      uOceanOrigin: { value: new THREE.Vector3() },
      uOceanCam: { value: new THREE.Vector3() },
      uWaves: { value: Array.from({ length: MAX_WAVES }, (_, i) => new THREE.Vector4(...(waves[i] || [0, 1, 0, 0]))) },
      uWaveN: { value: waves.length },
    };
    this.group = new THREE.Group();
    this.group.name = 'ocean';
    // near patch (displaced, 4 km, 8 m cells) and a far skirt (flat geometry, same shading)
    const near = new THREE.PlaneGeometry(4000, 4000, 500, 500).rotateX(-Math.PI / 2);
    const far = new THREE.PlaneGeometry(160000, 160000, 1, 1).rotateX(-Math.PI / 2);
    far.translate(0, -0.6, 0);
    this.matNear = this._material(true, atmosphere);
    this.matFar = this._material(false, atmosphere);
    this.near = new THREE.Mesh(near, this.matNear);
    this.far = new THREE.Mesh(far, this.matFar);
    for (const m of [this.near, this.far]) { m.frustumCulled = false; m.receiveShadow = true; m.renderOrder = 1; this.group.add(m); }
    this._c = new THREE.Vector3();
  }

  _material(displace, atmosphere) {
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x0a1820, roughness: 0.12, metalness: 0.0, ior: 1.333, specularIntensity: 1.0,
      clearcoat: 0.0, envMapIntensity: 1.0,
    });
    const U = this.U;
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, U);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${OCEAN_GLSL}`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>
          vec3 wpos = (modelMatrix * vec4(transformed, 1.0)).xyz + uOceanOrigin;
          ${displace ? `float fade = 1.0 - smoothstep(600.0, 1900.0, length(wpos.xz - uOceanCam.xz));
          transformed.y += uOceanAmp * fade * oceanEta(vec2(wpos.x, -wpos.z));` : ''}
          vOceanW = wpos;`);
      const begin = THREE.ShaderChunk.normal_fragment_begin.replace('normalize( vNormal )', 'oceanNormalV()');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${OCEAN_GLSL}
          vec3 oceanNormalV() {
            vec2 gr = uOceanAmp * oceanGrad(vec2(vOceanW.x, -vOceanW.z));
            // fade the slope detail with distance (no aliasing at grazing angles)
            float d = length(vOceanW.xz - uOceanCam.xz);
            gr *= 1.0 / (1.0 + d / 3000.0);
            vec3 nW = normalize(vec3(-gr.x, 1.0, gr.y));
            return normalize((viewMatrix * vec4(nW, 0.0)).xyz);
          }`)
        .replace('#include <normal_fragment_begin>', begin);
    };
    mat.customProgramCacheKey = () => `ocean|${displace}`;
    atmosphere.patch(mat);
    return mat;
  }

  /** @param t replay time (s)  @param camW camera (W)  @param origin floating origin (W) */
  layout(t, camW, origin) {
    this.U.uOceanT.value = t;
    this.U.uOceanCam.value.copy(camW);
    this.U.uOceanOrigin.value.copy(origin);
    // snap the near patch to the camera on a 8 m grid (vertex positions stay put on the swell)
    const cx = Math.round(camW.x / 8) * 8, cz = Math.round(camW.z / 8) * 8;
    this.near.position.set(cx - origin.x, -origin.y, cz - origin.z);
    this.far.position.set(cx - origin.x, -origin.y, cz - origin.z);
  }

  /** Elevation at ENU (e, n) and time t (for placing objects on the water). */
  eta(e, n, t) {
    let s = 0;
    const W = this.U.uWaves.value;
    for (let i = 0; i < this.U.uWaveN.value; i++) {
      const w = W[i], k = (w.y * w.y) / G;
      s += w.x * Math.cos(w.y * t - k * (e * Math.cos(w.z) + n * Math.sin(w.z)) + w.w);
    }
    return s;
  }

  dispose() {
    this.near.geometry.dispose(); this.far.geometry.dispose();
    this.matNear.dispose(); this.matFar.dispose();
  }
}
