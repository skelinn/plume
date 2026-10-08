// Physically based sky and aerial perspective.
//
//  * A small sky-radiance LUT (512 x 256, HDR) is ray-marched with single Rayleigh + Mie
//    scattering around a spherical planet (Nishita-style) for the current camera altitude.  It is
//    re-rendered only when the camera altitude or local "up" changes noticeably.
//  * The sky dome samples the LUT per pixel and adds what needs full resolution: the sun disk, the
//    planet surface below the horizon (procedural albedo, lit and attenuated), and stars, which
//    fade in above ~45 km as the sky goes black.
//  * The same dome, rendered without sun/stars into a PMREM cube, is the scene's image-based
//    lighting, so metal and paint reflect the actual sky / ground at the vehicle's altitude.
//  * Lit materials (MeshStandard/Physical) are patched (onBeforeCompile) to apply an analytic
//    aerial-perspective term (exponential atmosphere optical depth + single scattering), which is
//    the altitude-dependent "fog": thick blue-grey haze near the ground, nothing in space.

import * as THREE from 'three';
import { GLSL_NOISE } from './shaders.js';
import { clamp } from './util.js';

export const PLANET_R = 6371e3;
const ATM_TOP = 100e3;
// scattering coefficients (1/m) and scale heights (m)
const BR = [5.802e-6, 13.558e-6, 33.1e-6];
const BM_S = 9.0e-6, BM_E = 10.0e-6;
const HR = 8000, HM = 1200;
export const SUN_E = 4.2; // exo-atmospheric sun irradiance in scene units

const GLSL_CONST = /* glsl */ `
const float PR = ${PLANET_R.toFixed(1)};
const float PA = ${(PLANET_R + ATM_TOP).toFixed(1)};
const vec3 BR = vec3(${BR.map((x) => x.toExponential(4)).join(', ')});
const float BM_S = ${BM_S.toExponential(4)};
const float BM_E = ${BM_E.toExponential(4)};
const float HR = ${HR.toFixed(1)};
const float HM = ${HM.toFixed(1)};
const float PI_ = 3.14159265;
float phaseR(float mu) { return 3.0 / (16.0 * PI_) * (1.0 + mu * mu); }
float phaseM(float mu) {
  const float g = 0.78;
  float d = 1.0 + g * g - 2.0 * g * mu;
  return (1.0 - g * g) / (4.0 * PI_ * d * sqrt(d));
}
// optical depth of an exponential layer along a straight path from height h0 to h1 of length d
float odExp(float h0, float h1, float d, float H) {
  float dh = h1 - h0;
  float a = exp(-h0 / H);
  if (abs(dh) < 0.01 * H) return d * exp(-0.5 * (h0 + h1) / H);
  return d * (a - exp(-h1 / H)) / (dh / H);
}
// analytic single-scattering aerial perspective between heights hc (eye) and hp (point), distance d
vec3 sunTrans(float h, float muS);
vec3 aerialH(vec3 col, float hc, float hp, float d, float mu, float muS, float sunE) {
  float oR = odExp(hc, hp, d, HR), oM = odExp(hc, hp, d, HM);
  vec3 tau = BR * oR + BM_E * oM;
  vec3 T = exp(-tau);
  vec3 sT = sunTrans(0.5 * (hc + hp), muS);
  vec3 sc = BR * oR * phaseR(mu) + BM_S * oM * phaseM(mu) + (BR * oR + BM_S * oM) * 0.035;
  return col * T + sunE * sT * sc * (1.0 - T) / max(tau, vec3(1e-6));
}
// sun transmittance from height h for a sun at cos-zenith muS (Chapman-ish, flat approx near zenith)
vec3 sunTrans(float h, float muS) {
  float m = 1.0 / max(muS + 0.15 * pow(max(93.885 - degrees(acos(clamp(muS, -1.0, 1.0))), 0.01), -1.253), 0.02);
  return exp(-(BR * HR * exp(-h / HR) + BM_E * HM * exp(-h / HM)) * m);
}
`;

/** Uniforms + aerial perspective, shared by the dome, terrain, plume and every patched material. */
export const GLSL_ATMOS = /* glsl */ `
uniform vec3 uSunDir;
uniform vec3 uPlanetC;     // planet centre, render space
uniform float uSunE;
uniform float uHaze;       // multiplier on the aerial-perspective thickness (1 = physical)
uniform float uPlanetR;    // radius of the scene's planet (spherical / wgs84 scenes)
${GLSL_CONST}
float altitudeOf(vec3 p) { return length(p - uPlanetC) - uPlanetR; }
vec3 aerial(vec3 col, vec3 p) {
  vec3 c = cameraPosition;
  vec3 v = p - c;
  float d = length(v);
  if (d < 1.0) return col;
  float hc = max(altitudeOf(c), 0.0), hp = max(altitudeOf(p), 0.0);
  vec3 up = normalize(c - uPlanetC);
  return aerialH(col, hc, hp, d * uHaze, dot(v / d, uSunDir), dot(up, uSunDir), uSunE);
}`;

const LUT_W = 512, LUT_H = 256;

const LUT_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uUp; uniform vec3 uSkyX; uniform vec3 uSkyZ; uniform vec3 uSun;
uniform float uCamH; uniform float uDip; uniform float uSunE; uniform int uSteps;
varying vec2 vUv;
${GLSL_CONST}
// ray / sphere (radius R, centred at the planet centre) from a point at height h above PR.
// b = dot(o, d) with |o| = PR + h. Returns (t0, t1) or (-1, -1).
vec2 hitSphere(float h, float b, float R) {
  float c = (PR + h - R) * (PR + h + R);
  float disc = b * b - c;
  if (disc < 0.0) return vec2(-1.0);
  float s = sqrt(disc);
  return vec2(-b - s, -b + s);
}
void main() {
  float s = (vUv.y - 0.5) * 2.0;
  float ep = sign(s) * s * s * 1.5707963;
  float e = clamp(ep - uDip, -1.5707963, 1.5707963);
  float az = (vUv.x - 0.5) * 6.2831853;
  vec3 d = cos(e) * (cos(az) * uSkyX + sin(az) * uSkyZ) + sin(e) * uUp;
  float h0 = uCamH;
  float b = (PR + h0) * dot(uUp, d);
  vec2 ta = hitSphere(h0, b, PA);
  if (ta.y <= 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float t0 = max(ta.x, 0.0), t1 = ta.y;
  vec2 tg = hitSphere(h0, b, PR);
  if (tg.x > 0.0) t1 = tg.x;
  int N = uSteps;
  float ds = (t1 - t0) / float(N);
  vec3 o = uUp * (PR + h0);
  vec3 sumR = vec3(0.0), sumM = vec3(0.0);
  float oR = 0.0, oM = 0.0;
  for (int i = 0; i < 48; i++) {
    if (i >= N) break;
    float t = t0 + (float(i) + 0.5) * ds;
    vec3 p = o + d * t;
    float r = length(p);
    float h = r - PR;
    float dr = exp(-h / HR) * ds, dm = exp(-h / HM) * ds;
    oR += 0.5 * dr; oM += 0.5 * dm;
    vec3 n = p / r;
    float bl = r * dot(n, uSun);
    vec2 tgl = hitSphere(h, bl, PR);
    if (tgl.x <= 0.0) {
      vec2 tl = hitSphere(h, bl, PA);
      float lds = tl.y / 8.0, lR = 0.0, lM = 0.0;
      for (int j = 0; j < 8; j++) {
        float hl = length(p + uSun * (float(j) + 0.5) * lds) - PR;
        lR += exp(-hl / HR) * lds; lM += exp(-hl / HM) * lds;
      }
      vec3 att = exp(-(BR * (oR + lR) + BM_E * (oM + lM)));
      sumR += att * dr; sumM += att * dm;
    }
    oR += 0.5 * dr; oM += 0.5 * dm;
  }
  float mu = dot(d, uSun);
  vec3 L = uSunE * (sumR * BR * phaseR(mu) + sumM * BM_S * phaseM(mu));
  // crude multiple-scattering lift so the sky keeps a little blue in the shadowed parts
  L += uSunE * (sumR * BR + sumM * BM_S) * 0.02;
  vec3 T = exp(-(BR * oR + BM_E * oM));
  gl_FragColor = vec4(L, dot(T, vec3(0.3333)));
}`;

const DOME_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_Position.z = gl_Position.w * 0.99995;   // no depth: always behind everything
}`;

const DOME_FRAG = /* glsl */ `
uniform sampler2D uLut;
uniform vec3 uUp; uniform vec3 uSkyX; uniform vec3 uSkyZ;
uniform float uCamH; uniform float uDip; uniform float uEnv; uniform float uStars;
uniform vec3 uSunDirS; uniform float uSunE2; uniform float uSunE;
varying vec3 vDir;
${GLSL_CONST}
${GLSL_NOISE}
vec2 lutUv(vec3 d) {
  float e = asin(clamp(dot(d, uUp), -1.0, 1.0)) + uDip;
  float az = atan(dot(d, uSkyZ), dot(d, uSkyX));
  float v = 0.5 + 0.5 * sign(e) * sqrt(min(abs(e) / 1.5707963, 1.0));
  return vec2(az / 6.2831853 + 0.5, v);
}
vec3 groundAlbedo(vec3 n) {
  vec3 q = n * (PR / 18000.0);
  float a = fbm3(q * 0.35), b = fbm3(q * 2.1 + 3.1), c = vnoise3(q * 9.0);
  vec3 scrub = vec3(0.065, 0.074, 0.04), grass = vec3(0.13, 0.116, 0.066), soil = vec3(0.17, 0.13, 0.088);
  vec3 col = mix(scrub, grass, smoothstep(0.35, 0.7, a));
  col = mix(col, soil, smoothstep(0.6, 0.85, b) * 0.6);
  return col * (0.88 + 0.24 * c);
}
void main() {
  vec3 d = normalize(vDir);
  vec4 lut = texture2D(uLut, lutUv(d));
  vec3 col = lut.rgb;
  float T = lut.a;
  // does the ray hit the planet?
  float b = (PR + uCamH) * dot(uUp, d);
  float c = uCamH * (2.0 * PR + uCamH);
  float disc = b * b - c;
  bool ground = disc > 0.0 && b < 0.0;
  if (ground) {
    // planet surface beyond the meshes: lit and hazed exactly like the terrain (analytic model)
    float t = -b - sqrt(disc);
    vec3 n = normalize(uUp * (PR + uCamH) + d * t);
    float muS = dot(n, uSunDirS);
    vec3 E = uSunE * sunTrans(uCamH, dot(uUp, uSunDirS)) * max(muS, 0.0) + vec3(0.30, 0.38, 0.52) * uSunE * 0.05;
    vec3 alb = uEnv > 0.5 ? vec3(0.1, 0.095, 0.06) : groundAlbedo(n);
    col = aerialH(alb / 3.14159 * E, uCamH, 0.0, t, dot(d, uSunDirS), dot(uUp, uSunDirS), uSunE);
  } else if (uEnv < 0.5) {
    // sun disk (limb darkened); radiance clamped so bloom stays well-behaved
    float cs = dot(d, uSunDirS);
    float rs = 0.0062;                       // angular radius (rad), ~1.3x the real sun for visibility
    float x = sqrt(max(1.0 - cs * cs, 0.0)) / rs;
    if (cs > 0.0 && x < 1.0) {
      float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x * x));
      col += uSunE2 * sunTrans(uCamH, dot(uUp, uSunDirS)) * limb;
    }
    // stars, visible once the sky is dark
    if (uStars > 0.0) {
      vec3 p = d * 220.0;
      vec3 ip = floor(p);
      float r = hash31(ip);
      vec3 jit = vec3(hash31(ip + 1.7), hash31(ip + 5.1), hash31(ip + 9.3)) - 0.5;
      float w = max(fwidth(p.x) + fwidth(p.y), 1e-4);
      float dist = length(fract(p) - 0.5 - jit * 0.6);
      float star = step(0.985, r) * (1.0 - smoothstep(0.0, max(0.9 * w, 0.06), dist));
      float mag = pow(hash31(ip + 3.3), 6.0);
      vec3 tint = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.86, 0.7), hash31(ip + 7.7));
      col += tint * star * (0.15 + 3.0 * mag) * uStars * T;
    }
  }
  gl_FragColor = vec4(max(col, vec3(0.0)), 1.0);
}`;

export class Atmosphere {
  /**
   * @param renderer THREE.WebGLRenderer
   * @param frame coords.Frame
   * @param sunDir W-space unit vector to the sun
   */
  constructor(renderer, frame, sunDir, { quality = 'high' } = {}) {
    this.renderer = renderer;
    this.frame = frame;
    this.quality = quality;
    this.sunDir = sunDir.clone();
    // uniforms for lit materials (aerial perspective)
    this.uniforms = {
      uSunDir: { value: this.sunDir.clone() },
      uPlanetC: { value: new THREE.Vector3(0, -PLANET_R, 0) },
      uSunE: { value: SUN_E },
      uHaze: { value: 1.0 },
      uPlanetR: { value: frame.spherical ? frame.R : PLANET_R },
    };
    this.lut = new THREE.WebGLRenderTarget(LUT_W, LUT_H, {
      type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.RepeatWrapping, wrapT: THREE.ClampToEdgeWrapping,
      generateMipmaps: false,
    });
    this.lut.texture.wrapS = THREE.RepeatWrapping;
    this.skyU = {
      uUp: { value: new THREE.Vector3(0, 1, 0) },
      uSkyX: { value: new THREE.Vector3(1, 0, 0) },
      uSkyZ: { value: new THREE.Vector3(0, 0, 1) },
      uSun: { value: this.sunDir.clone() },
      uSunDirS: { value: this.sunDir.clone() },
      uCamH: { value: 0 },
      uDip: { value: 0 },
      uSunE: { value: SUN_E },
      uSunE2: { value: 160 },
      uSteps: { value: quality === 'low' ? 14 : 28 },
      uLut: { value: this.lut.texture },
      uStars: { value: 0 },
    };
    this.lutMat = new THREE.ShaderMaterial({
      uniforms: this.skyU,
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: LUT_FRAG,
      depthTest: false, depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.lutMat);
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const domeMat = (env) => new THREE.ShaderMaterial({
      uniforms: { ...this.skyU, uEnv: { value: env ? 1 : 0 } },
      vertexShader: DOME_VERT, fragmentShader: DOME_FRAG,
      side: THREE.BackSide, depthWrite: false, depthTest: false,
    });
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), domeMat(false));
    this.dome.renderOrder = -1000;
    this.dome.frustumCulled = false;
    this.envDome = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), domeMat(true));
    this.envDome.frustumCulled = false;
    this.envScene = new THREE.Scene();
    this.envScene.add(this.envDome);
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.envRT = null;
    this._key = null;
    this._lastEnv = -1e9;
    this.sunColor = new THREE.Color(1, 1, 1);
  }

  objects() { return [this.dome]; }

  get envMap() { return this.envRT ? this.envRT.texture : null; }

  /** Patch a built-in lit material so its output gets aerial perspective. */
  patch(material) {
    const U = this.uniforms;
    const prev = material.onBeforeCompile;
    material.onBeforeCompile = (shader, r) => {
      prev?.call(material, shader, r);
      Object.assign(shader.uniforms, U);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vAtmW;')
        .replace('#include <fog_vertex>', `#include <fog_vertex>
          vec4 atmW = vec4(transformed, 1.0);
          #ifdef USE_INSTANCING
            atmW = instanceMatrix * atmW;
          #endif
          vAtmW = (modelMatrix * atmW).xyz;`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\nvarying vec3 vAtmW;\n${GLSL_ATMOS}`)
        .replace('#include <fog_fragment>', '#include <fog_fragment>\ngl_FragColor.rgb = aerial(gl_FragColor.rgb, vAtmW);');
    };
    const prevKey = material.customProgramCacheKey?.bind(material);
    material.customProgramCacheKey = () => `atm|${prevKey ? prevKey() : ''}`;
    material.needsUpdate = true;
    return material;
  }

  /** Colour * intensity of the sun as seen from altitude h (for the directional light). */
  sunAt(h, up) {
    const mu = up.dot(this.sunDir);
    const m = 1 / Math.max(mu + 0.15 * Math.pow(Math.max(93.885 - (Math.acos(clamp(mu, -1, 1)) * 180) / Math.PI, 0.01), -1.253), 0.02);
    const k = (i) => Math.exp(-(BR[i] * HR * Math.exp(-h / HR) + BM_E * HM * Math.exp(-h / HM)) * m);
    this.sunColor.setRGB(k(0), k(1), k(2));
    return this.sunColor;
  }

  /**
   * Per frame: camera context -> uniforms; regenerate the LUT / environment when needed.
   * @param camW camera W position, @param origin render origin, @param camera three camera
   */
  update(camW, origin, camera, { capture = false, now = 0 } = {}) {
    const F = this.frame;
    const up = F.spherical ? F.up(camW) : new THREE.Vector3(0, 1, 0);
    const h = Math.max(F.altitude(camW), 0);
    // planet centre in render space: the real one (spherical) or a sphere tangent at the datum
    if (F.spherical) this.uniforms.uPlanetC.value.copy(F.centerW).sub(origin);
    else this.uniforms.uPlanetC.value.set(-origin.x, -PLANET_R - origin.y, -origin.z);
    this.uniforms.uSunDir.value.copy(this.sunDir);
    this.dome.position.copy(camera.position);
    this.dome.scale.setScalar(Math.max(camera.near * 4, 1) * 2);
    this.skyU.uStars.value = clamp((h - 55000) / 30000, 0, 1);

    // LUT / environment refresh
    const key = `${Math.round(Math.log1p(h / 40) * 14)}|${up.x.toFixed(3)},${up.y.toFixed(3)},${up.z.toFixed(3)}`;
    if (key !== this._key && (capture || now - this._lastEnv > 0.25 || this._key === null)) {
      this._key = key;
      this._lastEnv = now;
      this._renderLut(h, up);
    }
  }

  _renderLut(h, up) {
    const U = this.skyU;
    U.uUp.value.copy(up);
    const sx = this.sunDir.clone().addScaledVector(up, -this.sunDir.dot(up));
    if (sx.lengthSq() < 1e-8) sx.set(1, 0, 0).addScaledVector(up, -up.x);
    sx.normalize();
    U.uSkyX.value.copy(sx);
    U.uSkyZ.value.crossVectors(sx, up).normalize();
    U.uSun.value.copy(this.sunDir);
    U.uSunDirS.value.copy(this.sunDir);
    U.uCamH.value = h;
    U.uDip.value = Math.acos(clamp(PLANET_R / (PLANET_R + h), -1, 1));
    const r = this.renderer;
    const prevRT = r.getRenderTarget();
    const prevAuto = r.autoClear;
    r.autoClear = true;
    r.setRenderTarget(this.lut);
    r.render(this.quadScene, this.quadCam);
    r.setRenderTarget(prevRT);
    // environment (no sun disk, no stars: the directional light carries the sun)
    const old = this.envRT;
    this.envRT = this.pmrem.fromScene(this.envScene, 0, 0.1, 10);
    r.autoClear = prevAuto;
    old?.dispose();
    this.onEnv?.(this.envRT.texture);
  }

  dispose() {
    this.lut.dispose();
    this.envRT?.dispose();
    this.pmrem.dispose();
    this.lutMat.dispose();
    this.dome.geometry.dispose(); this.dome.material.dispose();
    this.envDome.geometry.dispose(); this.envDome.material.dispose();
    this.quad.geometry.dispose();
  }
}
