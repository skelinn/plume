// Payload-fairing half: a thin shell (cylinder + tangent-ogive nose) covering body +x, so two
// halves - the second drawn rotated 180 deg about the vehicle axis - close into a full fairing.
// Painted outside, bare composite inside, with a dark separation rail along both edges.

import * as THREE from 'three';
import * as M from '../materials.js';
import { ogiveRadius } from './geom.js';

export const isFairing = (V) => V?.kind === 'fairing_half';

export class FairingHalf {
  /** @param V meta.vehicle {length, diameter, nose_length} */
  constructor(V) {
    const L = V.length, r = (V.diameter / 2) * 1.012;      // just outside the stage below
    const ln = Math.min(V.nose_length ?? 0.6 * L, L);
    const lc = L - ln;
    this.group = new THREE.Group();
    this.group.name = 'fairing';
    this.materials = [];
    const pts = [new THREE.Vector2(r, 0), new THREE.Vector2(r, lc)];
    const N = 36;
    for (let i = 1; i <= N; i++) {
      const x = (i / N) * ln;
      pts.push(new THREE.Vector2(Math.max(ogiveRadius(r, ln, x), i === N ? 0 : 0.002), lc + x));
    }
    // LatheGeometry places phi = 0 on +z and phi = pi/2 on +x: span body +x (mesh +x)
    const geo = new THREE.LatheGeometry(pts, 48, 0, Math.PI);
    const outside = M.withRepeat(M.hullPaint(), 2, 2);
    outside.side = THREE.FrontSide;
    const inside = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.8, metalness: 0.1, side: THREE.BackSide });
    this.materials.push(outside, inside);
    this.group.add(new THREE.Mesh(geo, outside), new THREE.Mesh(geo, inside));
    // separation rails along the two edges (mesh +z and -z)
    const rail = new THREE.MeshStandardMaterial({ color: 0x3a3936, roughness: 0.5, metalness: 0.8 });
    this.materials.push(rail);
    for (const z of [1, -1]) {
      const g = new THREE.BoxGeometry(0.02 * r + 0.01, lc, 0.03 * r + 0.01).translate(0, lc / 2, z * r);
      this.group.add(new THREE.Mesh(g, rail));
    }
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  }

  dispose() {
    this.group.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
    for (const m of this.materials) { m.map?.dispose?.(); m.dispose(); }
  }
}
