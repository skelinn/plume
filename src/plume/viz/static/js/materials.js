// Procedural textures and physically based materials for the vehicle models and pads.
// Everything is generated at load time from seeded noise (no image assets), and cached.

import * as THREE from 'three';
import { GLSL_NOISE } from './shaders.js';

// ------------------------------------------------------------------ JS noise
function hash2(x, y, s = 0) {
  let h = (x * 374761393 + y * 668265263 + s * 1442695041) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}
/** periodic value noise (period px cells) */
function vnoise(x, y, period, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const fx = x - xi, fy = y - yi;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const m = (v) => ((v % period) + period) % period;
  const a = hash2(m(xi), m(yi), seed), b = hash2(m(xi + 1), m(yi), seed);
  const c = hash2(m(xi), m(yi + 1), seed), d = hash2(m(xi + 1), m(yi + 1), seed);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
function fbm(x, y, period, oct = 4, seed = 0) {
  let s = 0, a = 0.5, f = 1, n = 0;
  for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f, period * f, seed + i * 7); n += a; a *= 0.5; f *= 2; }
  return s / n;
}

/** Height field (Float32Array w*h) -> tangent-space normal map. */
function normalTexture(hgt, w, h, strength, wrap = true) {
  const data = new Uint8Array(w * h * 4);
  const at = (x, y) => {
    if (wrap) { x = (x + w) % w; y = (y + h) % h; } else { x = Math.min(Math.max(x, 0), w - 1); y = Math.min(Math.max(y, 0), h - 1); }
    return hgt[y * w + x];
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const l = Math.hypot(dx, dy, 1);
      const k = (y * w + x) * 4;
      data[k] = ((-dx / l) * 0.5 + 0.5) * 255;
      // canvas/texture rows run top->bottom while v runs bottom->top: flip the y slope
      data[k + 1] = ((dy / l) * 0.5 + 0.5) * 255;
      data[k + 2] = ((1 / l) * 0.5 + 0.5) * 255;
      data[k + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  t.wrapS = t.wrapT = wrap ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.anisotropy = 8;
  t.needsUpdate = true;
  return t;
}

function grayTexture(vals, w, h, srgb = false, wrap = true) {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, vals[i] * 255));
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v; data[i * 4 + 3] = 255;
  }
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  t.wrapS = t.wrapT = wrap ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

const cache = new Map();
const cached = (key, fn) => { if (!cache.has(key)) cache.set(key, fn()); return cache.get(key); };

// ------------------------------------------------------------------ texture sets
/**
 * Painted aluminium-lithium barrel panel, one tile = a quarter of the circumference x one barrel
 * section: a longitudinal seam on the left edge, a circumferential weld at the bottom edge.
 */
export function hullPaintTextures() {
  return cached('hull', () => {
    const W = 256, H = 512;
    const hgt = new Float32Array(W * H), rough = new Float32Array(W * H), alb = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const n = fbm(x / 64, y / 64, 4 * 1, 3, 3);
        const fine = hash2(x, y, 9);
        // circumferential weld bead at the tile's bottom edge, longitudinal seam on the left
        const dy = Math.min(y, H - y), dx = Math.min(x, W - x);
        const weld = Math.exp(-(dy * dy) / 6) + 0.6 * Math.exp(-(dx * dx) / 3);
        // stringer "oil canning": gentle undulation between seams
        const canning = Math.sin((x / W) * Math.PI * 6) * 0.15;
        hgt[i] = weld * 1.0 + canning * 0.4 + (n - 0.5) * 0.2;
        alb[i] = 0.93 + (n - 0.5) * 0.05 - weld * 0.12 + (fine - 0.5) * 0.015;
        rough[i] = 0.40 + (n - 0.5) * 0.12 + weld * 0.25;
      }
    }
    return {
      map: grayTexture(alb, W, H, true),
      roughnessMap: grayTexture(rough, W, H),
      normalMap: normalTexture(hgt, W, H, 1.6),
    };
  });
}

/** 2x2 twill carbon-fibre weave (tile = 4 x 4 tows). */
export function carbonTextures() {
  return cached('carbon', () => {
    const S = 128;
    const hgt = new Float32Array(S * S), alb = new Float32Array(S * S), rough = new Float32Array(S * S);
    const tow = S / 4;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const cx = Math.floor(x / tow), cy = Math.floor(y / tow);
        const over = ((cx + cy) & 3) < 2; // twill
        const fx = (x % tow) / tow, fy = (y % tow) / tow;
        const bulge = over ? Math.sin(fy * Math.PI) : Math.sin(fx * Math.PI);
        const fib = over ? Math.sin(fx * Math.PI * 9) : Math.sin(fy * Math.PI * 9);
        hgt[i] = bulge * 0.8 + fib * 0.05;
        alb[i] = (over ? 0.17 : 0.13) + bulge * 0.04;
        rough[i] = 0.36 + (over ? 0.0 : 0.08) + (1 - bulge) * 0.1;
      }
    }
    return { map: grayTexture(alb, S, S, true), roughnessMap: grayTexture(rough, S, S), normalMap: normalTexture(hgt, S, S, 0.9) };
  });
}

/** Regeneratively-cooled tube wall: 8 tubes per tile across u, constant along v. */
export function tubeTextures() {
  return cached('tubes', () => {
    const W = 256, H = 64, n = 8;
    const hgt = new Float32Array(W * H), alb = new Float32Array(W * H), rough = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const f = (x / W) * n;
        const k = Math.floor(f), fr = f - k;
        const prof = Math.sqrt(Math.max(1 - (2 * fr - 1) ** 2, 0));
        hgt[i] = prof * 3.0;
        const tint = hash2(k, 0, 5);
        alb[i] = 0.55 + 0.25 * tint + 0.2 * prof;
        rough[i] = 0.32 + 0.25 * (1 - prof) + 0.1 * vnoise(x / 16, y / 8, 16, 2);
      }
    }
    return { map: grayTexture(alb, W, H, true), roughnessMap: grayTexture(rough, W, H), normalMap: normalTexture(hgt, W, H, 1.0) };
  });
}

/** Low-contrast brushed streaks for bare metal (roughness modulation). */
export function brushedTexture() {
  return cached('brushed', () => {
    const W = 256, H = 256;
    const r = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = vnoise(x / 1.5, y / 48, 256 / 1.5, 4) * 0.6 + vnoise(x / 5, y / 90, 256 / 5, 8) * 0.4;
        r[y * W + x] = 0.78 + 0.32 * s;
      }
    }
    return grayTexture(r, W, H);
  });
}

/** Concrete landing pad (top face, planar uv over the disc). */
export function padTextures(name, radius) {
  return cached(`pad|${name}|${radius}`, () => {
    const S = 1024;
    const c = document.createElement('canvas');
    c.width = c.height = S;
    const g = c.getContext('2d');
    const img = g.createImageData(S, S);
    const hgt = new Float32Array(S * S), rough = new Float32Array(S * S);
    const mpp = (2 * radius) / S;                 // metres per pixel
    const slab = 4.0;                              // saw-cut joint spacing (m)
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const wx = (x - S / 2) * mpp, wy = (y - S / 2) * mpp;
        const rr = Math.hypot(wx, wy) / radius;
        const n1 = fbm(x / 96, y / 96, 1024, 4, 11), n2 = fbm(x / 12, y / 12, 1024, 3, 12);
        const grain = hash2(x, y, 13);
        // slab joints
        const jx = Math.abs(((wx / slab) % 1 + 1.5) % 1 - 0.5) * slab, jy = Math.abs(((wy / slab) % 1 + 1.5) % 1 - 0.5) * slab;
        const joint = Math.max(Math.exp(-(jx * jx) / (0.0009 + mpp * mpp)), Math.exp(-(jy * jy) / (0.0009 + mpp * mpp)));
        // per-slab tone
        const slabTone = hash2(Math.floor(wx / slab), Math.floor(wy / slab), 21) - 0.5;
        // blast scorch: radial soot around the centre with streaks
        const ang = Math.atan2(wy, wx);
        const streak = vnoise(ang * 9 + 50, rr * 2, 1e6, 31);
        const scorch = Math.exp(-((rr / 0.42) ** 2)) * (0.55 + 0.45 * streak) + 0.25 * Math.exp(-((rr / 0.7) ** 4)) * streak;
        let v = 0.56 + (n1 - 0.5) * 0.16 + (n2 - 0.5) * 0.07 + (grain - 0.5) * 0.05 + slabTone * 0.05;
        v *= 1 - 0.78 * Math.min(scorch, 1);
        v *= 1 - 0.45 * joint;
        const warm = 1 - 0.25 * Math.min(scorch, 1);
        img.data[i * 4] = Math.min(255, v * 255 * 1.0);
        img.data[i * 4 + 1] = Math.min(255, v * 255 * 0.985 * warm + v * 255 * 0.985 * (1 - warm) * 0.92);
        img.data[i * 4 + 2] = Math.min(255, v * 255 * 0.95 * (1 - 0.1 * Math.min(scorch, 1)));
        img.data[i * 4 + 3] = 255;
        hgt[i] = -joint * 1.5 + (n2 - 0.5) * 0.4 + (grain - 0.5) * 0.15;
        rough[i] = 0.86 + (n2 - 0.5) * 0.12 - 0.15 * Math.min(scorch, 1);
      }
    }
    g.putImageData(img, 0, 0);
    // markings (weathered paint): outer ring, cross, name
    const px = (m) => (m / (2 * radius)) * S;
    g.globalCompositeOperation = 'source-over';
    g.strokeStyle = 'rgba(232, 232, 226, 0.88)';
    g.lineWidth = px(Math.max(0.35, radius * 0.03));
    g.beginPath(); g.arc(S / 2, S / 2, px(radius * 0.86), 0, Math.PI * 2); g.stroke();
    const arm = px(radius * 0.32), bar = px(Math.max(0.5, radius * 0.06));
    g.fillStyle = 'rgba(232, 232, 226, 0.82)';
    g.save(); g.translate(S / 2, S / 2); g.rotate(Math.PI / 4);
    g.fillRect(-arm, -bar / 2, 2 * arm, bar); g.fillRect(-bar / 2, -arm, bar, 2 * arm);
    g.restore();
    g.fillStyle = 'rgba(214, 170, 52, 0.85)';     // safety-yellow tick marks on the ring
    for (let k = 0; k < 8; k++) {
      g.save(); g.translate(S / 2, S / 2); g.rotate((k * Math.PI) / 4 + Math.PI / 8);
      g.fillRect(px(radius * 0.86) - px(0.4), -px(0.12), px(0.8), px(0.24));
      g.restore();
    }
    g.fillStyle = 'rgba(232, 232, 226, 0.8)';
    g.font = `600 ${px(Math.max(0.8, radius * 0.075))}px "IBM Plex Sans", "Segoe UI", Arial, sans-serif`;
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(String(name || 'PAD').toUpperCase().slice(0, 12), S / 2, S / 2 + px(radius * 0.66));
    // weather the paint: knock random speckles out of the markings
    const wimg = g.getImageData(0, 0, S, S);
    for (let i = 0; i < S * S; i++) {
      const k = i * 4;
      if (wimg.data[k] > img.data[k] + 30 && hash2(i % S, (i / S) | 0, 77) < 0.18) {
        wimg.data[k] = img.data[k]; wimg.data[k + 1] = img.data[k + 1]; wimg.data[k + 2] = img.data[k + 2];
      }
    }
    g.putImageData(wimg, 0, 0);
    const map = new THREE.CanvasTexture(c);
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 8;
    return { map, roughnessMap: grayTexture(rough, S, S, false, false), normalMap: normalTexture(hgt, S, S, 0.9, false) };
  });
}

// ------------------------------------------------------------------ text decals
export function textTexture(lines, { w = 1024, h = 256, font = '600 180px "IBM Plex Sans", Arial, sans-serif', color = '#141414', align = 'center' } = {}) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.clearRect(0, 0, w, h);
  g.fillStyle = color;
  g.font = font;
  g.textAlign = align;
  g.textBaseline = 'middle';
  const ls = Array.isArray(lines) ? lines : [lines];
  ls.forEach((l, i) => g.fillText(l, align === 'center' ? w / 2 : 8, ((i + 0.5) * h) / ls.length));
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// ------------------------------------------------------------------ soot
/**
 * Shared soot uniforms: uSoot (0..1 amount), uSootTop (m above the hull base), uRocketInv (inverse of
 * the rocket group's world matrix, so every part evaluates soot in the vehicle's mesh space).
 */
export function makeSootUniforms() {
  return { uSoot: { value: 0 }, uSootTop: { value: 1 }, uRocketInv: { value: new THREE.Matrix4() }, uSootR: { value: 1 } };
}

export function patchSoot(material, U, { strength = 1 } = {}) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader, r) => {
    prev?.call(material, shader, r);
    Object.assign(shader.uniforms, U);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform mat4 uRocketInv;\nvarying vec3 vSootP;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSootP = (uRocketInv * modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform float uSoot; uniform float uSootTop; uniform float uSootR;\nvarying vec3 vSootP;\n${GLSL_NOISE}\nfloat sootAmt;`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        {
          float ang = atan(vSootP.z, vSootP.x);
          float y = vSootP.y;
          float blotch = fbm2(vec2(ang * 1.6, y * 0.45));
          float streak = vnoise(vec2(ang * 4.5, y * 0.22));
          float top = uSootTop * (0.7 + 0.6 * blotch);
          float cover = 1.0 - smoothstep(top * 0.3, top, y);
          sootAmt = clamp(uSoot * ${strength.toFixed(2)} * cover * (0.35 + 0.5 * blotch + 0.35 * streak), 0.0, 0.9);
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.03, 0.027, 0.024), sootAmt);
        }`)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.9, sootAmt);')
      .replace('#include <lights_physical_fragment>', '#include <lights_physical_fragment>\n#ifdef USE_CLEARCOAT\nmaterial.clearcoat *= 1.0 - sootAmt;\n#endif');
  };
  const prevKey = material.customProgramCacheKey?.bind(material);
  material.customProgramCacheKey = () => `soot${strength}|${prevKey ? prevKey() : ''}`;
  return material;
}

// ------------------------------------------------------------------ material factories
export function hullPaint({ repeat = [4, 6] } = {}) {
  const t = hullPaintTextures();
  const m = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, map: t.map, roughnessMap: t.roughnessMap, normalMap: t.normalMap, normalScale: new THREE.Vector2(0.35, 0.35),
    roughness: 1.0, metalness: 0.0, clearcoat: 0.25, clearcoatRoughness: 0.35,
  });
  m.userData.repeat = repeat;
  return m;
}

/** Clone a texture-set material with its own uv repeat (texture data stays shared). */
export function withRepeat(material, rx, ry) {
  const m = material.clone();
  for (const k of ['map', 'roughnessMap', 'normalMap', 'metalnessMap']) {
    if (m[k]) { m[k] = m[k].clone(); m[k].repeat.set(rx, ry); m[k].needsUpdate = true; }
  }
  m.onBeforeCompile = material.onBeforeCompile;
  m.customProgramCacheKey = material.customProgramCacheKey;
  return m;
}

export function carbon({ repeat = 40 } = {}) {
  const t = carbonTextures();
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x2a2a2a, map: t.map, roughnessMap: t.roughnessMap, normalMap: t.normalMap, normalScale: new THREE.Vector2(0.4, 0.4),
    roughness: 1.0, metalness: 0.0, clearcoat: 0.55, clearcoatRoughness: 0.28,
  });
  for (const k of ['map', 'roughnessMap', 'normalMap']) m[k] = m[k].clone(), m[k].repeat.set(repeat, repeat), m[k].needsUpdate = true;
  return m;
}

export function titanium() {
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x837d75, metalness: 1.0, roughness: 0.55, roughnessMap: brushedTexture(), vertexColors: true,
    anisotropy: 0.55, anisotropyRotation: 0.0,
  });
  return m;
}

export const aluminium = () => new THREE.MeshStandardMaterial({ color: 0xc8cacc, metalness: 1.0, roughness: 0.38, roughnessMap: brushedTexture() });
export const chrome = () => new THREE.MeshStandardMaterial({ color: 0xe6e8ea, metalness: 1.0, roughness: 0.12 });
export const darkMetal = () => new THREE.MeshStandardMaterial({ color: 0x3a3936, metalness: 0.85, roughness: 0.5, roughnessMap: brushedTexture() });
export const blackPaint = () => new THREE.MeshPhysicalMaterial({ color: 0x1b1b1c, metalness: 0.0, roughness: 0.55, clearcoat: 0.2, clearcoatRoughness: 0.5 });
export const heatShield = () => new THREE.MeshStandardMaterial({ color: 0x2c2a27, metalness: 0.0, roughness: 0.95 });
export const blanket = () => new THREE.MeshStandardMaterial({ color: 0x8c8678, metalness: 0.3, roughness: 0.75 });
