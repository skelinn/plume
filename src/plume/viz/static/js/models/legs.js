// Landing legs: carbon-fibre A-frame (two tapered beams from a hull hinge to the footpad), a
// telescoping strut (outer cylinder, chrome piston with the crush-core stroke, end clevises) from
// higher up the hull to mid-leg, and a wide footpad on a ball joint.
//
// Local leg frame: +x radial, +y up, +z tangential; legs sit at azimuth pi/4 + 2 pi k / n
// (between the RCS pods, as in the physics model).  Stowing rotates the leg about its hinge
// (tangential axis) so it lies along the hull with the footpad up; the strut re-solves its length.

import * as THREE from 'three';
import { mergeGeometries, pnu, place, tube, Y } from './geom.js';

const LEG_PHASE = Math.PI / 4;

export class Legs {
  /**
   * @param legs meta.vehicle.legs {count, span, height, attach_z}
   * @param r hull radius, @param L vehicle length
   * @param mats {carbon, metal, chrome, pad, dark}
   */
  constructor(legs, r, L, mats) {
    this.group = new THREE.Group();
    this.group.name = 'legs';
    this.items = [];
    const n = legs.count | 0;
    const S = legs.span ?? 2 * r, H = legs.height ?? r, A = legs.attach_z ?? 0.15 * L;
    const hingeX = r + 0.05;
    const F = new THREE.Vector2(S, -H);                     // foot (contact point) in (x, y)
    const padR = THREE.MathUtils.clamp(0.11 * S + 0.05, 0.14, 0.5);
    const padH = Math.max(0.07, padR * 0.3);
    const apex = new THREE.Vector3(S, -H + padH + padR * 0.35, 0);
    const hinge = new THREE.Vector3(hingeX, A, 0);
    const legLen = apex.distanceTo(hinge);
    const w = Math.max(0.2 * r, 0.12);                      // A-frame half-width at the hull
    const beamR = Math.max(0.013 * legLen, 0.025);
    this.stowAngle = Math.PI - Math.atan2(S - hingeX, A + H) - 0.06;  // fold up against the hull
    this.A = A; this.hingeX = hingeX;

    // main A-frame (in the leg's pivot frame: origin at the hinge axis)
    const h0 = new THREE.Vector3(0, 0, 0);
    const ap = apex.clone().sub(hinge);
    const geos = [
      tube(new THREE.Vector3(0, 0, w), ap, beamR * 1.15, beamR * 0.75, 12),
      tube(new THREE.Vector3(0, 0, -w), ap, beamR * 1.15, beamR * 0.75, 12),
      // cross brace
      tube(new THREE.Vector3(0, 0, w).lerp(ap, 0.34), new THREE.Vector3(0, 0, -w).lerp(ap, 0.34), beamR * 0.55, beamR * 0.55, 8),
    ];
    const frameGeo = mergeGeometries(geos.map(pnu));
    // fittings: hinge barrel and the foot fitting
    const fitGeo = mergeGeometries([
      place(new THREE.CylinderGeometry(beamR * 1.25, beamR * 1.25, 2 * w + beamR * 3, 16), { rot: [Math.PI / 2, 0, 0] }),
      place(new THREE.SphereGeometry(beamR * 1.6, 16, 10), { pos: [ap.x, ap.y, 0] }),
      tube(new THREE.Vector3(ap.x, ap.y, 0), new THREE.Vector3(ap.x, ap.y - padR * 0.35, 0), beamR * 1.0, beamR * 1.3, 12),
    ].map(pnu));
    // footpad: bevelled disc + crush-core ring
    const padGeo = mergeGeometries([
      place(new THREE.CylinderGeometry(padR * 0.9, padR, padH * 0.55, 32), { pos: [ap.x, ap.y - padR * 0.35 - padH * 0.72, 0] }),
      place(new THREE.CylinderGeometry(padR * 0.55, padR * 0.85, padH * 0.45, 32), { pos: [ap.x, ap.y - padR * 0.35 - padH * 0.22, 0] }),
    ].map(pnu));
    // hull-side brackets (fixed)
    const B = new THREE.Vector3(hingeX, A + 0.42 * (A + H), 0);  // strut attach on the hull
    const brGeo = mergeGeometries([
      place(new THREE.BoxGeometry(0.1, 0.24, 2 * w + 0.12), { pos: [r + 0.02, A, 0] }),
      place(new THREE.BoxGeometry(0.1, 0.2, 0.16), { pos: [r + 0.02, B.y, 0] }),
    ].map(pnu));
    this.strutRo = Math.max(0.012 * legLen, 0.03);
    this.Bpt = B;
    this.Mfrac = 0.58;

    for (let k = 0; k < n; k++) {
      const phi = LEG_PHASE + (2 * Math.PI * k) / n;
      const az = new THREE.Group();
      az.rotation.y = phi;                                    // local +x -> body azimuth phi
      const brackets = new THREE.Mesh(brGeo, mats.metal);
      az.add(brackets);
      const pivot = new THREE.Group();
      pivot.position.copy(hinge);
      pivot.add(new THREE.Mesh(frameGeo, mats.carbon), new THREE.Mesh(fitGeo, mats.metal), new THREE.Mesh(padGeo, mats.pad));
      az.add(pivot);
      // strut: outer barrel from B, chrome piston from the leg point M (re-solved every update)
      const outer = new THREE.Mesh(new THREE.CylinderGeometry(this.strutRo, this.strutRo, 1, 14), mats.carbon);
      const collar = new THREE.Mesh(new THREE.CylinderGeometry(this.strutRo * 1.25, this.strutRo * 1.25, 1, 14), mats.metal);
      const piston = new THREE.Mesh(new THREE.CylinderGeometry(this.strutRo * 0.62, this.strutRo * 0.62, 1, 14), mats.chrome);
      const endA = new THREE.Mesh(new THREE.SphereGeometry(this.strutRo * 1.35, 12, 8), mats.metal);
      const endB = endA.clone();
      az.add(outer, collar, piston, endA, endB);
      this.group.add(az);
      this.items.push({ az, pivot, outer, collar, piston, endA, endB, ap });
    }
    this.legLen = legLen;
    this._geos = [frameGeo, fitGeo, padGeo, brGeo];
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.update(1);
  }

  /** @param out 0 = stowed against the hull .. 1 = deployed */
  update(out) {
    const o = THREE.MathUtils.clamp(out ?? 1, 0, 1);
    const e = o * o * (3 - 2 * o);
    const ang = (1 - e) * this.stowAngle;
    const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), d = new THREE.Vector3();
    for (const it of this.items) {
      it.pivot.rotation.z = ang;
      // strut endpoints in the azimuth frame
      tmpA.copy(this.Bpt);
      tmpB.copy(it.ap).multiplyScalar(this.Mfrac).applyAxisAngle(new THREE.Vector3(0, 0, 1), ang).add(it.pivot.position);
      d.subVectors(tmpB, tmpA);
      const len = d.length();
      const q = new THREE.Quaternion().setFromUnitVectors(Y, d.clone().normalize());
      const barrel = Math.min(len * 0.62, Math.max(len - 0.15, 0.1));
      const place1 = (m, from, l) => {
        m.scale.set(1, Math.max(l, 1e-3), 1);
        m.quaternion.copy(q);
        m.position.copy(tmpA).addScaledVector(d, (from + l / 2) / len);
      };
      place1(it.outer, 0, barrel);
      place1(it.collar, barrel - 0.04, 0.08);
      place1(it.piston, barrel - 0.05, len - barrel + 0.05);
      it.endA.position.copy(tmpA);
      it.endB.position.copy(tmpB);
    }
  }

  dispose() { for (const g of this._geos) g.dispose(); this.group.traverse((o) => { if (o.isMesh && !this._geos.includes(o.geometry)) o.geometry.dispose(); }); }
}
