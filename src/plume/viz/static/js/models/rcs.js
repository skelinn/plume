// Cold-gas RCS puffs.  Thruster layout follows the physics (plume.physics.propulsion.ring_layout):
// pods at body azimuth 2 pi k / n, each with a +/- tangential thruster pair.  The replay only has
// the normalised torque command, so each thruster's duty is estimated from how well its own torque
// (about an estimated CG) lines up with the command; the puff leaves the nozzle opposite to the
// thrust, i.e. in the physically correct direction.

import * as THREE from 'three';
import { Y } from './geom.js';

const VERT = /* glsl */ `
varying float vS; varying vec3 vN; varying vec3 vV;
void main() {
  vS = clamp(-position.y, 0.0, 1.0);        // 0 at the nozzle -> 1 at the end (cone points down -y)
  vN = normalize(normalMatrix * normal);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = /* glsl */ `
uniform float uI; uniform vec3 uLit; uniform float uSeed;
varying float vS; varying vec3 vN; varying vec3 vV;
float h(float x) { return fract(sin(x * 91.7 + uSeed) * 43758.5); }
void main() {
  float f = abs(dot(normalize(vN), normalize(vV)));
  // soft, wispy: thin at the rim (grazing), fading downstream, broken up along the length
  float wisp = 0.75 + 0.25 * sin(vS * 19.0 + uSeed * 3.0) * sin(vS * 7.0 + uSeed);
  float a = uI * pow(1.0 - vS, 2.2) * pow(f, 2.4) * wisp;
  gl_FragColor = vec4(uLit * 1.1, clamp(a * 0.5, 0.0, 0.45));
}`;

export class RcsPuffs {
  /**
   * @param pods hull.rcsPods [{th, pos (mesh), tan (mesh), offset}]
   * @param V vehicle meta, @param cgZ estimated CG height (m)
   */
  constructor(pods, V, cgZ) {
    this.group = new THREE.Group();
    this.items = [];
    const D = V.diameter;
    const len = Math.max(0.65 * D, 0.25);
    const geo = new THREE.CylinderGeometry(len * 0.5, len * 0.025, 1, 20, 8, true).translate(0, -0.5, 0);
    this.geo = geo;
    const R = D / 2;
    for (const p of pods) {
      for (const sgn of [1, -1]) {
        // thrust direction (body): sgn * tangent t = (-sin th, cos th, 0); exhaust is opposite
        const tb = new THREE.Vector3(-Math.sin(p.th), Math.cos(p.th), 0).multiplyScalar(sgn);
        const pb = new THREE.Vector3(R * Math.cos(p.th), R * Math.sin(p.th), V.rcs_z - cgZ);
        const torque = new THREE.Vector3().crossVectors(pb, tb).normalize();
        const u = { uI: { value: 0 }, uLit: { value: new THREE.Vector3(1, 1, 1) }, uSeed: { value: this.items.length * 1.7 } };
        const m = new THREE.Mesh(geo, new THREE.ShaderMaterial({
          uniforms: u, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
        }));
        // exhaust direction in mesh space = -(sgn * tan_mesh); the cone's -y axis points along it
        const ex = p.tan.clone().multiplyScalar(-sgn);
        m.quaternion.setFromUnitVectors(Y.clone().negate(), ex);
        m.position.copy(p.pos).addScaledVector(ex, p.offset);
        m.scale.set(1, len, 1);
        m.renderOrder = 15;
        m.visible = false;
        m.frustumCulled = false;
        this.group.add(m);
        this.items.push({ m, u, torque, len });
      }
    }
  }

  /** @param rc normalised torque command [x, y, z] (body), @param lit light colour (THREE.Color) */
  update(rc, lit) {
    const c = new THREE.Vector3(rc?.[0] || 0, rc?.[1] || 0, rc?.[2] || 0);
    const mag = Math.min(c.length(), 1);
    if (mag > 1e-4) c.normalize();
    for (const it of this.items) {
      const duty = mag > 0.02 ? Math.max(0, it.torque.dot(c)) * Math.min(1, mag * 1.6) : 0;
      it.m.visible = duty > 0.04;
      it.u.uI.value = Math.min(1, 0.25 + duty);
      it.m.scale.set(0.6 + 0.6 * duty, it.len * (0.5 + 0.7 * duty), 0.6 + 0.6 * duty);
      it.u.uLit.value.set(lit.r, lit.g, lit.b);
    }
  }

  dispose() { this.geo.dispose(); for (const it of this.items) it.m.material.dispose(); }
}
