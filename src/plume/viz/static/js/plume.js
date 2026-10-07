// Engine plume: two nested additive cones (hot core + outer flame) driven by shader noise,
// plus a glow sprite at the nozzle exit.  Length/brightness follow throttle; the plume
// widens and shortens with altitude (vacuum expansion) using `alt` as a pressure proxy.

import * as THREE from 'three';
import { GLSL_COMMON } from './shaders.js';
import { smoothstep } from './util.js';

const VERT = /* glsl */ `
  uniform float uLen; uniform float uR; uniform float uBulge; uniform float uTip; uniform float uScale;
  varying float vS; varying float vAng; varying vec3 vN; varying vec3 vVw;
  void main() {
    float s = clamp(0.5 - position.y, 0.0, 1.0);                 // 0 at nozzle exit .. 1 at tip
    vec2 dir = normalize(position.xz);
    // radius profile: expands in vacuum (bulge) then necks down to the tip
    float prof = (1.0 + uBulge * sin(3.14159265 * pow(s, 0.55))) * mix(1.0, uTip, pow(s, 1.1));
    vec3 p = vec3(dir.x, 0.0, dir.y) * uR * prof * uScale;
    p.y = -s * uLen;
    vS = s;
    vAng = atan(dir.y, dir.x);
    vN = normalize(normalMatrix * vec3(dir.x, 0.0, dir.y));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vVw = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }`;

const FRAG = /* glsl */ `
  ${GLSL_COMMON}
  uniform float uLen; uniform float uR; uniform float uBright; uniform float uEdge; uniform float uFall;
  uniform float uCore; uniform float uDiamond; uniform float uSeed;
  uniform vec3 uColA; uniform vec3 uColB;
  varying float vS; varying float vAng; varying vec3 vN; varying vec3 vVw;
  void main() {
    vec3 N = normalize(vN), V = normalize(vVw);
    float fres = abs(dot(N, V));
    float edge = pow(clamp(fres, 0.0, 1.0), uEdge);
    float t = uTime + uSeed;
    float n1 = vnoise(vec2(vS * 7.0 - t * 16.0, vAng * 2.5));
    float n2 = vnoise(vec2(vS * 17.0 - t * 31.0, vAng * 4.0 + 7.0));
    float flick = 0.62 + 0.75 * (n1 * 0.65 + n2 * 0.35);
    float fall = pow(1.0 - vS, uFall);
    // shock diamonds in the core when the exhaust is near ambient pressure
    float dia = uCore * uDiamond * pow(0.5 + 0.5 * cos(vS * uLen / (uR * 1.9) * 6.2832 - t * 2.0), 4.0) * (1.0 - vS) * 0.8;
    float I = (fall * flick + dia) * edge * uBright;
    vec3 col = mix(uColA, uColB, smoothstep(0.0, 0.9, vS));
    gl_FragColor = vec4(col, max(I, 0.0));
  }`;

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
let _glow = null;
export const getGlowTexture = () => (_glow ||= glowTexture());

export class Plume {
  /** @param nozzleRadius exit radius of the bell (m) */
  constructor(nozzleRadius, env) {
    this.r = nozzleRadius;
    this.group = new THREE.Group();
    this.length = 0;
    this.intensity = 0;
    const shared = { uTime: env.uTime };
    const mk = (opts) => {
      const u = {
        ...shared,
        uSunDir: env.uSunDir, uFogColor: env.uFogColor, uFogDensity: env.uFogDensity,
        uLightPos: env.uLightPos, uLightI: env.uLightI, uLightR: env.uLightR, uLightCol: env.uLightCol,
        uLen: { value: 1 }, uR: { value: this.r }, uBulge: { value: 0.1 }, uTip: { value: 0.3 },
        uScale: { value: opts.scale }, uBright: { value: 1 }, uEdge: { value: opts.edge },
        uFall: { value: opts.fall }, uCore: { value: opts.core }, uDiamond: { value: 1 },
        uSeed: { value: opts.seed },
        uColA: { value: new THREE.Vector3(...opts.a) }, uColB: { value: new THREE.Vector3(...opts.b) },
      };
      const m = new THREE.ShaderMaterial({
        uniforms: u, vertexShader: VERT, fragmentShader: FRAG,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 36, 24, true), m);
      mesh.frustumCulled = false;
      mesh.renderOrder = 10;
      return { mesh, u };
    };
    this.outer = mk({ scale: 1.0, edge: 1.3, fall: 1.1, core: 0, seed: 0, a: [1.0, 0.6, 0.2], b: [0.95, 0.22, 0.05] });
    this.core = mk({ scale: 0.46, edge: 0.7, fall: 1.4, core: 1, seed: 3.7, a: [1.0, 0.96, 0.88], b: [1.0, 0.72, 0.32] });
    this.group.add(this.outer.mesh, this.core.mesh);

    this.glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: getGlowTexture(), color: 0xffa24a, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    }));
    this.glow.renderOrder = 11;
    this.group.add(this.glow);
  }

  /** @param throttle 0..1, @param alt metres above ground (pressure proxy) */
  update(throttle, alt) {
    const th = Math.max(0, Math.min(1, throttle || 0));
    this.group.visible = th > 0.015;
    this.intensity = th;
    if (!this.group.visible) { this.length = 0; return; }
    // 0 at sea level -> 1 in vacuum; wide, short and dim above ~20 km
    const vac = smoothstep(8000, 80000, alt || 0);
    const len = this.r * 2 * (2.4 + 8.0 * Math.pow(th, 0.8)) * (1 + 0.1 * vac);
    this.length = len;
    const bulge = 0.12 + 2.3 * vac;
    const tip = 0.28 + 0.5 * vac;
    for (const [part, bright] of [[this.outer, (0.5 + 0.5 * th) * (1 - 0.42 * vac)], [this.core, 0.9 + 0.7 * th]]) {
      const u = part.u;
      u.uLen.value = len * (part === this.core ? 0.72 : 1);
      u.uBulge.value = bulge * (part === this.core ? 0.55 : 1);
      u.uTip.value = tip;
      u.uBright.value = bright;
      u.uDiamond.value = (1 - vac) * (0.3 + 0.7 * th);
    }
    const gs = this.r * (3.6 + 5 * th) * (1 + 0.8 * vac);
    this.glow.scale.setScalar(gs);
    this.glow.material.opacity = 0.35 + 0.5 * th;
  }
}
