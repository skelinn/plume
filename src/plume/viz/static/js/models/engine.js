// Liquid engine: gimbal block, powerhead/injector dome, chamber, regeneratively-cooled bell (cooling-
// tube normal map, hat bands, lip), turbopump with turbine exhaust duct, feed lines, two gimbal
// actuators to the fixed thrust structure, and the base heat shield with its flexible boot.
// The bell interior glows (emissive, HDR) near the throat while the engine fires.

import * as THREE from 'three';
import { tubeTextures } from '../materials.js';
import { lathe, mergeGeometries, pnu, place, tube, Y } from './geom.js';

function bezier(p0, p1, p2, t) {
  const a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, c = t * t;
  return [a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1]];
}

let _glowTex = null;
function glowRamp() {
  if (_glowTex) return _glowTex;
  const c = document.createElement('canvas');
  c.width = 4; c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 256);   // canvas y=0 is uv v=1 (flipY) = the throat
  grad.addColorStop(0.0, '#fff0d8');
  grad.addColorStop(0.07, '#ffb060');
  grad.addColorStop(0.2, '#a04010');
  grad.addColorStop(0.45, '#200800');
  grad.addColorStop(1.0, '#000000');
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  _glowTex = new THREE.CanvasTexture(c);
  _glowTex.colorSpace = THREE.SRGBColorSpace;
  return _glowTex;
}

export class Engine {
  /**
   * @param E meta.vehicle.engine {nozzle_radius, gimbal_z}
   * @param r hull radius
   * @param mats {dark, metal, shield, blanket}
   */
  constructor(E, r, mats) {
    const re = E?.nozzle_radius ?? 0.3 * r;
    const gz = E?.gimbal_z ?? 1.5 * re;
    this.re = re;
    this.group = new THREE.Group();       // fixed parts (mesh space)
    this.group.name = 'engine';
    this.gimbal = new THREE.Group();      // moving parts, origin at the gimbal pivot
    this.gimbal.position.set(0, gz, 0);
    this.group.add(this.gimbal);

    const Le = Math.max(gz + 1.6 * re, 2.6 * re);            // pivot -> exit plane
    const rt = 0.4 * re, rc = 0.56 * re;
    const Lb = Math.min(1.9 * re, Le - 1.0 * re);
    const yInj = -0.32 * re, yCc = -(Le - Lb - 0.34 * re), yT = -(Le - Lb), yE = -Le;
    this.exitY = yE;
    this.length = Le;

    // ---- bell profile (Rao-ish: 30 deg initial, 8 deg exit), throat -> exit
    const th0 = (30 * Math.PI) / 180, the = (8 * Math.PI) / 180;
    const P0 = [rt, yT], P2 = [re, yE];
    // control point = intersection of the two tangents
    const t0 = [Math.sin(th0), -Math.cos(th0)], t2 = [Math.sin(the), -Math.cos(the)];
    const det = t0[0] * -t2[1] - t0[1] * -t2[0];
    const s = ((P2[0] - P0[0]) * -t2[1] - (P2[1] - P0[1]) * -t2[0]) / det;
    const P1 = [P0[0] + t0[0] * s, P0[1] + t0[1] * s];
    const bell = [];
    for (let i = 0; i <= 28; i++) bell.push(bezier(P0, P1, P2, i / 28));
    this.bellProfile = bell;
    // chamber + convergent section, injector -> throat
    const conv = [];
    for (let i = 0; i <= 10; i++) {
      const u = i / 10;
      const rr = rc + (rt - rc) * (0.5 - 0.5 * Math.cos(Math.PI * u));
      conv.push([rr, yCc + (yT - yCc) * u]);
    }
    const outerPts = [[0.28 * re, 0.0], [0.42 * re, -0.06 * re], [0.5 * re, yInj * 0.6], [rc, yInj], [rc, yCc], ...conv.slice(1), ...bell.slice(1)];
    const tubesT = tubeTextures();
    const nTubes = Math.round((2 * Math.PI * re) / 0.03 / 8) * 8;
    const mk = (t) => { const c = t.clone(); c.repeat.set(nTubes / 8, 1); c.needsUpdate = true; return c; };
    this.outerMat = new THREE.MeshStandardMaterial({
      color: 0x5e5248, metalness: 0.85, roughness: 1.0,
      map: mk(tubesT.map), roughnessMap: mk(tubesT.roughnessMap), normalMap: mk(tubesT.normalMap), normalScale: new THREE.Vector2(0.9, 0.9),
    });
    // lathe profiles run bottom -> top so three's normals face outwards
    const outer = new THREE.Mesh(lathe(outerPts.slice().reverse(), 72), this.outerMat);
    this.gimbal.add(outer);
    // interior (BackSide), slightly inset, glowing near the throat
    const innerPts = bell.map(([x, y]) => [x * 0.985, y]);
    this.innerMat = new THREE.MeshStandardMaterial({
      color: 0x2a2420, metalness: 0.6, roughness: 0.6, side: THREE.BackSide,
      emissive: 0xffffff, emissiveMap: glowRamp(), emissiveIntensity: 0,
    });
    // lathe v runs exit (0) -> throat (1); the ramp is brightest at v = 1
    this.inner = new THREE.Mesh(lathe(innerPts.slice().reverse(), 72), this.innerMat);
    this.gimbal.add(this.inner);

    // ---- bands, lip, gimbal block, powerhead, turbopump, ducts, feed lines
    const at = (f) => bezier(P0, P1, P2, f);
    const ring = (rad, y, tube_, seg = 72) => place(new THREE.TorusGeometry(rad, tube_, 8, seg), { pos: [0, y, 0], rot: [Math.PI / 2, 0, 0] });
    const [b1r, b1y] = at(0.42), [b2r, b2y] = at(0.75);
    const hw = [
      ring(re * 1.004, yE + 0.004 * re, 0.022 * re),
      ring(b1r + 0.012 * re, b1y, 0.014 * re),
      ring(b2r + 0.012 * re, b2y, 0.014 * re),
      ring(rt * 1.2, yT, 0.05 * re, 48),
      place(new THREE.BoxGeometry(0.5 * re, 0.22 * re, 0.5 * re), { pos: [0, 0.02 * re, 0] }),
      place(new THREE.CylinderGeometry(0.2 * re, 0.2 * re, 0.6 * re, 24), { pos: [rc + 0.24 * re, yInj - 0.2 * re, 0], rot: [Math.PI / 2, 0, 0] }),
      place(new THREE.CylinderGeometry(0.13 * re, 0.16 * re, 0.22 * re, 20), { pos: [rc + 0.24 * re, yInj - 0.2 * re, 0.38 * re], rot: [Math.PI / 2, 0, 0] }),
      tube(new THREE.Vector3(rc + 0.3 * re, yInj - 0.32 * re, 0), new THREE.Vector3(re * 1.12, yE + 0.32 * re, 0.0), 0.07 * re, 0.09 * re, 14),
      tube(new THREE.Vector3(0.3 * re, 0.9 * re, 0.22 * re), new THREE.Vector3(rc + 0.15 * re, yInj - 0.1 * re, 0.22 * re), 0.07 * re, 0.07 * re, 12),
      tube(new THREE.Vector3(-0.3 * re, 0.9 * re, -0.18 * re), new THREE.Vector3(-0.35 * re, yInj * 0.5, -0.2 * re), 0.08 * re, 0.08 * re, 12),
    ];
    this.hwGeo = mergeGeometries(hw.map(pnu));
    this.gimbal.add(new THREE.Mesh(this.hwGeo, mats.clevis || mats.metal));

    // ---- fixed: thrust structure, heat shield + boot
    const yBase = 0.0;
    // bell radius where it crosses the base plane (gimbal 0)
    let rb0 = re;
    const prof = [[rc, yCc + gz], ...conv.map(([x, y]) => [x, y + gz]), ...bell.map(([x, y]) => [x, y + gz])];
    for (let i = 1; i < prof.length; i++) {
      const [ra, ya] = prof[i - 1], [rb, yb] = prof[i];
      if ((ya - yBase) * (yb - yBase) <= 0) { rb0 = ra + ((rb - ra) * (yBase - ya)) / (yb - ya || 1); break; }
    }
    if (yE + gz > yBase) rb0 = 0.5 * re;
    const hole = Math.min(rb0 + 0.16 * re, r * 0.92);
    const shield = new THREE.Mesh(place(new THREE.RingGeometry(hole, r * 0.995, 64, 1), { pos: [0, yBase + 0.003, 0], rot: [Math.PI / 2, 0, 0] }), mats.shield);
    const boot = new THREE.Mesh(new THREE.CylinderGeometry(hole, rb0 + 0.01 * re, 0.06 * re, 48, 1, true).translate(0, yBase - 0.02 * re, 0), mats.blanket);
    const ts = new THREE.Mesh(new THREE.CylinderGeometry(0.32 * re, r * 0.7, Math.max(gz + 0.25 * re - 0.05, 0.05), 24, 1, true).translate(0, (gz + 0.25 * re + 0.05) / 2, 0), mats.dark);
    this.group.add(shield, boot, ts);

    // ---- gimbal actuators (re-solved each update)
    this.act = [];
    this.actGeo = new THREE.CylinderGeometry(1, 1, 1, 12);
    for (const ang of [0, Math.PI / 2]) {
      const fixed = new THREE.Vector3(Math.cos(ang) * (rc + 0.5 * re), gz + 0.2 * re, -Math.sin(ang) * (rc + 0.5 * re));
      const moving = new THREE.Vector3(Math.cos(ang) * (rc + 0.02 * re), yCc + 0.1 * re, -Math.sin(ang) * (rc + 0.02 * re));
      const body = new THREE.Mesh(this.actGeo, mats.dark);
      const rod = new THREE.Mesh(this.actGeo, mats.chrome || mats.metal);
      this.group.add(body, rod);
      this.act.push({ fixed, moving, body, rod });
    }
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.inner.castShadow = false;
    this.setGimbal(0, 0);
  }

  /** Thrust direction in body = (cos a sin b, -sin a, cos a cos b)  <=>  R = Ry(b) Rx(a). */
  setGimbal(a, b) {
    const qx = new THREE.Quaternion(Math.sin(a / 2), 0, 0, Math.cos(a / 2));
    const qy = new THREE.Quaternion(0, Math.sin(b / 2), 0, Math.cos(b / 2));
    const qb = new THREE.Quaternion().multiplyQuaternions(qy, qx);
    // body quaternion -> mesh space: same remap as ENU->three, (x, y, z) -> (x, z, -y)
    this.gimbal.quaternion.set(qb.x, qb.z, -qb.y, qb.w);
    this.gimbal.updateMatrix();
    const m = new THREE.Vector3(), d = new THREE.Vector3();
    for (const a_ of this.act) {
      m.copy(a_.moving).applyMatrix4(this.gimbal.matrix);
      d.subVectors(m, a_.fixed);
      const len = d.length();
      const q = new THREE.Quaternion().setFromUnitVectors(Y, d.clone().normalize());
      const rb = 0.055 * this.re;
      a_.body.scale.set(rb, len * 0.6, rb);
      a_.body.quaternion.copy(q);
      a_.body.position.copy(a_.fixed).addScaledVector(d, 0.3);
      a_.rod.scale.set(rb * 0.5, len * 0.45, rb * 0.5);
      a_.rod.quaternion.copy(q);
      a_.rod.position.copy(a_.fixed).addScaledVector(d, 0.775);
    }
  }

  /** @param glow 0..1 hot-interior intensity */
  setGlow(glow) {
    this.innerMat.emissiveIntensity = 7.0 * glow;
  }

  dispose() {
    this.group.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
    this.outerMat.dispose(); this.innerMat.dispose();
  }
}
