// HDR post-processing: the scene renders into a linear half-float target (MSAA on "high"), then
// a small bloom chain (soft-threshold prefilter, 13-tap downsamples, tent upsamples) and a final
// composite pass (exposure, ACES filmic tone mapping, sRGB, dither) draw to the canvas.
// Engine plumes, the sun and hot metal are much brighter than 1.0 and bloom naturally.

import * as THREE from 'three';

const VERT = 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';

const PREFILTER = /* glsl */ `
uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uThreshold; uniform float uKnee;
varying vec2 vUv;
vec3 tap(vec2 o) { return texture2D(tSrc, vUv + o * uTexel).rgb; }
void main() {
  vec3 c = (tap(vec2(-1.0, -1.0)) + tap(vec2(1.0, -1.0)) + tap(vec2(-1.0, 1.0)) + tap(vec2(1.0, 1.0))) * 0.25;
  c = min(c, vec3(4000.0));
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  rq = rq * rq / (4.0 * uKnee + 1e-5);
  float w = max(rq, br - uThreshold) / max(br, 1e-5);
  gl_FragColor = vec4(c * w, 1.0);
}`;

const DOWN = /* glsl */ `
uniform sampler2D tSrc; uniform vec2 uTexel;
varying vec2 vUv;
vec3 tap(vec2 o) { return texture2D(tSrc, vUv + o * uTexel).rgb; }
void main() {
  vec3 a = tap(vec2(-2.0, -2.0)), b = tap(vec2(0.0, -2.0)), c = tap(vec2(2.0, -2.0));
  vec3 d = tap(vec2(-1.0, -1.0)), e = tap(vec2(1.0, -1.0));
  vec3 f = tap(vec2(-2.0, 0.0)), g = tap(vec2(0.0, 0.0)), h = tap(vec2(2.0, 0.0));
  vec3 i = tap(vec2(-1.0, 1.0)), j = tap(vec2(1.0, 1.0));
  vec3 k = tap(vec2(-2.0, 2.0)), l = tap(vec2(0.0, 2.0)), m = tap(vec2(2.0, 2.0));
  vec3 o = (d + e + i + j) * 0.125 + (a + b + f + g) * 0.03125 + (b + c + g + h) * 0.03125
         + (f + g + k + l) * 0.03125 + (g + h + l + m) * 0.03125;
  gl_FragColor = vec4(o, 1.0);
}`;

const UP = /* glsl */ `
uniform sampler2D tSrc; uniform sampler2D tBase; uniform vec2 uTexel; uniform float uRadius;
varying vec2 vUv;
vec3 tap(vec2 o) { return texture2D(tSrc, vUv + o * uTexel * uRadius).rgb; }
void main() {
  vec3 s = tap(vec2(-1.0, -1.0)) + tap(vec2(1.0, -1.0)) + tap(vec2(-1.0, 1.0)) + tap(vec2(1.0, 1.0))
         + 2.0 * (tap(vec2(0.0, -1.0)) + tap(vec2(0.0, 1.0)) + tap(vec2(-1.0, 0.0)) + tap(vec2(1.0, 0.0)))
         + 4.0 * tap(vec2(0.0));
  gl_FragColor = vec4(texture2D(tBase, vUv).rgb + s / 16.0, 1.0);
}`;

const COMPOSITE = /* glsl */ `
uniform sampler2D tScene; uniform sampler2D tBloom;
uniform float uExposure; uniform float uBloom; uniform float uUseBloom; uniform float uVignette;
varying vec2 vUv;
// ACES filmic (Stephen Hill's fit, as in three.js)
vec3 RRTAndODTFit(vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}
vec3 aces(vec3 color) {
  const mat3 ACESInputMat = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
  const mat3 ACESOutputMat = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
  color = ACESInputMat * color;
  color = RRTAndODTFit(color);
  color = ACESOutputMat * color;
  return clamp(color, 0.0, 1.0);
}
vec3 toSRGB(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec3 c = texture2D(tScene, vUv).rgb;
  if (uUseBloom > 0.5) c += texture2D(tBloom, vUv).rgb * uBloom;
  c = max(c, vec3(0.0));
  vec2 q = vUv - 0.5;
  c *= 1.0 - uVignette * dot(q, q);
  c = aces(c * uExposure / 0.6);
  c = toSRGB(c);
  c += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
  gl_FragColor = vec4(c, 1.0);
}`;

export class PostFX {
  constructor(renderer, { quality = 'high' } = {}) {
    this.renderer = renderer;
    this.quality = quality;
    this.exposure = 1.0;
    this.bloomStrength = 0.06;
    this.scene = new THREE.Scene();
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
    const mk = (frag, uniforms) => new THREE.ShaderMaterial({
      uniforms, vertexShader: VERT, fragmentShader: frag, depthTest: false, depthWrite: false, toneMapped: false,
    });
    this.mPre = mk(PREFILTER, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: 1.6 }, uKnee: { value: 0.8 } });
    this.mDown = mk(DOWN, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });
    this.mUp = mk(UP, { tSrc: { value: null }, tBase: { value: null }, uTexel: { value: new THREE.Vector2() }, uRadius: { value: 1.0 } });
    this.mComp = mk(COMPOSITE, {
      tScene: { value: null }, tBloom: { value: null }, uExposure: { value: 1 }, uBloom: { value: 0.06 },
      uUseBloom: { value: 1 }, uVignette: { value: 0.35 },
    });
    this.hdr = null;
    this.mips = [];
    this.ups = [];
    this.w = 0; this.h = 0;
  }

  setQuality(q) {
    if (q === this.quality) return;
    this.quality = q;
    const { w, h } = this;
    this.w = this.h = 0;
    this.setSize(w, h);
  }

  setSize(w, h) {
    w = Math.max(1, Math.floor(w)); h = Math.max(1, Math.floor(h));
    if (w === this.w && h === this.h && this.hdr) return;
    this.w = w; this.h = h;
    this.hdr?.dispose();
    for (const t of [...this.mips, ...this.ups]) t.dispose();
    const opts = { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.hdr = new THREE.WebGLRenderTarget(w, h, { ...opts, depthBuffer: true, samples: this.quality === 'high' ? 4 : 0 });
    this.mips = []; this.ups = [];
    let mw = w, mh = h;
    const n = this.quality === 'high' ? 6 : 0;
    for (let i = 0; i < n; i++) {
      mw = Math.max(1, Math.floor(mw / 2)); mh = Math.max(1, Math.floor(mh / 2));
      this.mips.push(new THREE.WebGLRenderTarget(mw, mh, opts));
      if (i < n - 1) this.ups.push(new THREE.WebGLRenderTarget(mw, mh, opts));
    }
  }

  _pass(mat, target) {
    this.quad.material = mat;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.cam);
  }

  /** Render `scene` with `camera` through the HDR chain to the canvas; then `after()` draws overlays. */
  render(scene, camera, after) {
    const r = this.renderer;
    r.setRenderTarget(this.hdr);
    r.clear(true, true, false);
    r.render(scene, camera);
    let bloom = null;
    if (this.mips.length) {
      const m = this.mips;
      this.mPre.uniforms.tSrc.value = this.hdr.texture;
      this.mPre.uniforms.uTexel.value.set(1 / this.w, 1 / this.h);
      this._pass(this.mPre, m[0]);
      for (let i = 1; i < m.length; i++) {
        this.mDown.uniforms.tSrc.value = m[i - 1].texture;
        this.mDown.uniforms.uTexel.value.set(1 / m[i - 1].width, 1 / m[i - 1].height);
        this._pass(this.mDown, m[i]);
      }
      // upsample: ups[i] = mips[i] + blur(up(i+1))
      let src = m[m.length - 1];
      for (let i = m.length - 2; i >= 0; i--) {
        this.mUp.uniforms.tSrc.value = src.texture;
        this.mUp.uniforms.tBase.value = m[i].texture;
        this.mUp.uniforms.uTexel.value.set(1 / src.width, 1 / src.height);
        this._pass(this.mUp, this.ups[i]);
        src = this.ups[i];
      }
      bloom = src.texture;
    }
    const u = this.mComp.uniforms;
    u.tScene.value = this.hdr.texture;
    u.tBloom.value = bloom;
    u.uUseBloom.value = bloom ? 1 : 0;
    u.uExposure.value = this.exposure;
    u.uBloom.value = this.bloomStrength;
    this._pass(this.mComp, null);
    if (after) {
      const ac = r.autoClear;
      r.autoClear = false;
      after();
      r.autoClear = ac;
    }
  }

  dispose() {
    this.hdr?.dispose();
    for (const t of [...this.mips, ...this.ups]) t.dispose();
    for (const m of [this.mPre, this.mDown, this.mUp, this.mComp]) m.dispose();
    this.quad.geometry.dispose();
  }
}
