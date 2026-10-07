// Sky dome (gradient + stars + sun) and, for spherical scenes, a dark Earth backdrop sphere.
// Both are drawn first with depth writes off, so terrain/rocket always render on top.

import * as THREE from 'three';
import { GLSL_COMMON } from './shaders.js';
import { clamp, smoothstep } from './util.js';

export const HORIZON = [0.30, 0.40, 0.53];
export const ZENITH = [0.045, 0.09, 0.2];

export class Sky {
  constructor(env, frame) {
    this.frame = frame;
    this.env = env;
    this.uniforms = {
      ...env,
      uUp: { value: new THREE.Vector3(0, 1, 0) },
      uSpace: { value: 0 },
      uHorizonC: { value: 0 },
      uHorizon: { value: new THREE.Vector3(...HORIZON) },
      uZenith: { value: new THREE.Vector3(...ZENITH) },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() {
          vDir = position;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          // With near/far ~ 1e-7 the float32 projection puts distant points at ndc.z >= 1.0, which the
          // far plane then clips (a ragged black hole in the sky).  The dome has no depth anyway.
          gl_Position.z = gl_Position.w * 0.9999;
        }`,
      fragmentShader: /* glsl */ `
        ${GLSL_COMMON}
        uniform vec3 uUp; uniform float uSpace; uniform float uHorizonC;
        uniform vec3 uHorizon; uniform vec3 uZenith;
        varying vec3 vDir;
        void main() {
          vec3 d = normalize(vDir);
          float c = dot(d, uUp) - uHorizonC;            // angle-ish above the (dipped) horizon
          float atm = 1.0 - uSpace;
          vec3 sky = mix(uHorizon, uZenith, pow(clamp(c, 0.0, 1.0), 0.42));
          vec3 below = uHorizon * mix(1.0, 0.5, clamp(-c * 5.0, 0.0, 1.0));
          vec3 col = c >= 0.0 ? sky : below;
          col *= mix(0.04, 1.0, atm);
          // stars (fade in with altitude)
          vec3 p = d * 140.0;
          vec3 ip = floor(p);
          float r = hash31(ip);
          vec3 jitter = vec3(hash31(ip + 1.7), hash31(ip + 5.1), hash31(ip + 9.3)) - 0.5;
          float star = step(0.986, r) * smoothstep(0.42, 0.0, length(fract(p) - 0.5 - jitter * 0.6));
          float vis = clamp(uSpace * 1.4, 0.0, 1.0) * (c > -0.2 ? 1.0 : 0.6);
          col += vec3(0.85, 0.9, 1.0) * star * (0.35 + 0.65 * hash31(ip + 3.3)) * vis;
          // thin atmosphere limb glow seen from orbit
          float l1 = c / 0.018, l2 = (c + 0.012) / 0.05;   // (pow() of a negative base is undefined in GLSL)
          float limb = exp(-l1 * l1) * uSpace * 0.9 + exp(-l2 * l2) * uSpace * 0.25;
          col += vec3(0.25, 0.5, 1.0) * limb * 0.55;
          // sun
          float sd = max(dot(d, uSunDir), 0.0);
          col += vec3(1.0, 0.93, 0.8) * (pow(sd, 900.0) * 3.0 + pow(sd, 24.0) * 0.18) * (0.35 + 0.65 * atm + 0.5 * uSpace);
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
    this.dome.renderOrder = -1000;
    this.dome.frustumCulled = false;

    this.earth = null;
    if (frame.spherical) {
      const emat = new THREE.ShaderMaterial({
        uniforms: { ...env },
        depthWrite: false,
        depthTest: false,
        vertexShader: /* glsl */ `
          varying vec3 vN; varying vec3 vView;
          void main() {
            vN = normalize(position);
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            vView = -mv.xyz;
            gl_Position = projectionMatrix * mv;
            gl_Position.z = min(gl_Position.z, gl_Position.w * 0.9999);   // same far-plane guard as the dome
          }`,
        fragmentShader: /* glsl */ `
          ${GLSL_COMMON}
          varying vec3 vN; varying vec3 vView;
          void main() {
            vec3 N = normalize(vN);
            vec3 V = normalize(vView);
            float diff = clamp(dot(N, uSunDir) * 0.8 + 0.35, 0.0, 1.0);
            vec3 base = vec3(0.075, 0.12, 0.115);
            vec3 col = base * (0.3 + 1.0 * diff);
            float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
            col += vec3(0.12, 0.26, 0.55) * rim * (0.2 + 0.8 * diff);
            col = applyFog(col, length(vView));
            gl_FragColor = vec4(col, 1.0);
          }`,
      });
      this.earth = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 64), emat);
      this.earth.scale.setScalar(frame.R - 30);
      this.earth.renderOrder = -900;
      this.earth.frustumCulled = false;
    }
  }

  objects() { return this.earth ? [this.dome, this.earth] : [this.dome]; }

  /**
   * @param camW camera position (W-space), @param origin render origin, @param camera three camera
   * @param up local up at the camera, @param alt camera altitude above the datum
   */
  update(camW, origin, camera, up, alt, far) {
    this.dome.position.copy(camW).sub(origin);
    this.dome.scale.setScalar(Math.min(far * 0.5, 1e5)); // only the direction matters (z is pinned in the shader)
    this.uniforms.uUp.value.copy(up);
    const space = smoothstep(25000, 130000, alt);
    this.uniforms.uSpace.value = space;
    // the geometric horizon dips below the local horizontal as we climb
    const R = this.frame.spherical ? this.frame.R : 0;
    this.uniforms.uHorizonC.value = R ? -Math.sqrt(Math.max(1 - (R / (R + Math.max(alt, 0))) ** 2, 0)) : 0;
    if (this.earth) this.earth.position.copy(this.frame.centerW).sub(origin);
    // fog thins with altitude
    const dens = 2.0e-5 * Math.exp(-Math.max(alt, 0) / 9000) + 2e-8;
    this.env.uFogDensity.value = clamp(dens, 0, 1);
  }
}
