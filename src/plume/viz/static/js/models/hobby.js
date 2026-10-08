// High-power hobby rocket: glossy fibreglass airframe, painted ogive nose cone, three swept G10 fins
// with bevelled edges, rail buttons, motor retainer and graphite nozzle; a recovery parachute
// (orange/white gores, shroud lines, shock cord) that inflates after apogee; and the launch rail
// (tripod stand, 1515 rail, blast deflector) used for its pad.

import * as THREE from 'three';
import { lathe, mergeGeometries, ogiveRadius, place, pnu, polar } from './geom.js';

export function isHobby(V) {
  return /hobby/i.test(V.name || '') || (!(V.legs?.count > 0) && !V.grid_fins && (V.length || 0) < 4);
}

export class HobbyAirframe {
  constructor(V) {
    const L = V.length, D = V.diameter, r = D / 2;
    const ln = Math.min(V.nose_length ?? 0.2 * L, 0.45 * L);
    const lc = L - ln;
    this.group = new THREE.Group();
    this.group.name = 'hobby';
    this.materials = [];
    const add = (m) => { this.materials.push(m); return m; };
    const body = add(new THREE.MeshPhysicalMaterial({ color: 0xe9e6dc, roughness: 0.3, metalness: 0, clearcoat: 0.8, clearcoatRoughness: 0.12 }));
    const nose = add(new THREE.MeshPhysicalMaterial({ color: 0xb4321c, roughness: 0.28, metalness: 0, clearcoat: 0.9, clearcoatRoughness: 0.1 }));
    const fin = add(new THREE.MeshPhysicalMaterial({ color: 0x161616, roughness: 0.4, metalness: 0, clearcoat: 0.6, clearcoatRoughness: 0.2 }));
    const metal = add(new THREE.MeshStandardMaterial({ color: 0x2a2b2e, roughness: 0.35, metalness: 0.9 }));
    const graphite = add(new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.7, metalness: 0.2 }));
    const tape = add(new THREE.MeshPhysicalMaterial({ color: 0x141414, roughness: 0.5, clearcoat: 0.4 }));
    this.r = r; this.L = L;

    // airframe tube with a coupler band (where it separates for recovery)
    this.group.add(new THREE.Mesh(new THREE.CylinderGeometry(r, r, lc, 48, 1, true).translate(0, lc / 2, 0), body));
    const yCoup = 0.62 * lc;
    this.group.add(new THREE.Mesh(new THREE.CylinderGeometry(r * 1.004, r * 1.004, 0.012 * L, 48, 1, true).translate(0, yCoup, 0), tape));
    this.group.add(new THREE.Mesh(new THREE.CylinderGeometry(r * 1.004, r * 1.004, 0.008 * L, 48, 1, true).translate(0, lc - 0.02 * L, 0), tape));
    // nose (ogive) with a small metal tip
    const pts = [];
    for (let i = 0; i <= 36; i++) { const x = (i / 36) * ln; pts.push([Math.max(ogiveRadius(r, ln, x), 0.0003), lc + x]); }
    pts[36][0] = 0;
    this.group.add(new THREE.Mesh(lathe(pts, 48), nose));
    // three swept fins: root chord ~ 2.4 D, tip ~ 1 D, span ~ 1.25 D, sweep ~ 1.5 D, 3 mm thick, bevelled
    const cr = Math.max(2.4 * D, 0.08), ct = Math.max(0.95 * D, 0.03), sp = Math.max(1.25 * D, 0.05), sw = 1.5 * D;
    const th = Math.max(0.045 * D, 0.0025);
    const yr = 0.01 * L;
    const shape = new THREE.Shape();
    shape.moveTo(0, yr + cr); shape.lineTo(sp, yr + cr - sw); shape.lineTo(sp, yr + cr - sw - ct); shape.lineTo(0, yr); shape.closePath();
    const fg = new THREE.ExtrudeGeometry(shape, { depth: th * 0.4, bevelEnabled: true, bevelThickness: th * 0.3, bevelSize: th * 0.6, bevelSegments: 2, curveSegments: 1 });
    fg.translate(0, 0, -th * 0.2);
    const fins = [];
    for (let k = 0; k < 3; k++) {
      const phi = (k * 2 * Math.PI) / 3;
      const g = pnu(fg.clone());
      g.translate(r * 0.98, 0, 0);
      g.rotateY(phi);
      fins.push(g);
    }
    fg.dispose();
    this.group.add(new THREE.Mesh(mergeGeometries(fins), fin));
    // rail buttons between fins
    const bphi = Math.PI / 3;
    const radial = new THREE.Vector3(Math.cos(bphi), 0, -Math.sin(bphi));
    const qBtn = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), radial);
    const bmesh = mergeGeometries([0.12 * L, 0.55 * L].map((y) => {
      const g = new THREE.CylinderGeometry(0.0085, 0.0065, 0.012, 16).translate(0, 0.006, 0);
      g.applyQuaternion(qBtn);
      const at = polar(bphi, r, y);
      g.translate(at.x, at.y, at.z);
      return pnu(g);
    }));
    this.group.add(new THREE.Mesh(bmesh, metal));
    this.railPhi = bphi;
    // motor retainer + nozzle
    this.group.add(new THREE.Mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.62, 0.025, 32).translate(0, -0.0125, 0), metal));
    const noz = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.32, r * 0.42, 0.02, 24, 1, true).translate(0, -0.035, 0), graphite);
    noz.material.side = THREE.DoubleSide;
    this.group.add(noz);
    this.exitY = -0.045;
    this.nozzleR = Math.max(V.engine?.nozzle_radius ?? 0.4 * r, 0.004);
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.attachY = yCoup;
  }

  dispose() {
    this.group.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
    for (const m of this.materials) m.dispose();
  }
}

/** Parachute in world orientation (not a child of the rocket). */
export class Parachute {
  constructor(D = 0.9) {
    this.group = new THREE.Group();
    this.group.name = 'parachute';
    const Rc = D / 2;
    this.Rc = Rc;
    const gores = 12;
    // canopy: flattened hemisphere, apex vent, scalloped skirt
    const prof = [];
    for (let i = 0; i <= 18; i++) {
      const t = 0.12 + (i / 18) * (Math.PI / 2 - 0.12 + 0.18);
      prof.push([Rc * Math.sin(t), Rc * 0.62 * Math.cos(t)]);
    }
    const g = lathe(prof.reverse(), gores * 8);
    const pos = g.attributes.position, uv = g.attributes.uv;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const u = uv.getX(i), v = uv.getY(i);
      const gi = Math.floor(u * gores + 1e-4) % gores;
      const c = gi % 2 ? [0.86, 0.84, 0.79] : [0.86, 0.26, 0.07];
      col.set(c, i * 3);
      // scallop between gores near the skirt (v ~ 0)
      const sc = (1 - Math.cos(2 * Math.PI * u * gores)) * 0.5;
      const k = 1 - 0.07 * sc * (1 - v);
      pos.setX(i, pos.getX(i) * k); pos.setZ(i, pos.getZ(i) * k);
      pos.setY(i, pos.getY(i) + 0.05 * Rc * sc * (1 - v));
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    this.canopyMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, side: THREE.DoubleSide });
    this.canopy = new THREE.Mesh(g, this.canopyMat);
    this.canopy.castShadow = true;
    this.inflate = new THREE.Group();
    this.inflate.add(this.canopy);
    this.group.add(this.inflate);
    // shroud lines to the confluence point, then the shock cord to the rocket (updated per frame)
    this.lineLen = 1.15 * D;
    this.rim = [];
    for (let k = 0; k < gores; k++) {
      const a = (k / gores) * Math.PI * 2;
      this.rim.push(new THREE.Vector3(Rc * Math.cos(a) * 0.98, -0.1 * Rc, Rc * Math.sin(a) * 0.98));
    }
    this.lineGeo = new THREE.BufferGeometry();
    this.lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((gores + 1) * 2 * 3), 3));
    this.lines = new THREE.LineSegments(this.lineGeo, new THREE.LineBasicMaterial({ color: 0x8a8a84 }));
    this.lines.frustumCulled = false;
    this.group.add(this.lines);
    this.group.visible = false;
  }

  /**
   * @param pos render-space canopy anchor, @param axis unit (canopy points along +axis)
   * @param attach render-space point on the rocket, @param f inflation 0..1
   */
  update(pos, axis, attach, f) {
    this.group.visible = f > 0;
    if (!this.group.visible) return;
    this.group.position.copy(pos);
    this.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
    const e = f * f * (3 - 2 * f);
    this.inflate.scale.set(0.12 + 0.88 * e, 1.8 - 0.8 * e, 0.12 + 0.88 * e);
    const conf = new THREE.Vector3(0, -this.lineLen, 0);
    const a = this.lineGeo.attributes.position;
    this.rim.forEach((p, k) => {
      a.setXYZ(2 * k, p.x * this.inflate.scale.x, p.y * this.inflate.scale.y, p.z * this.inflate.scale.z);
      a.setXYZ(2 * k + 1, conf.x, conf.y, conf.z);
    });
    // shock cord in canopy-local coordinates
    this.group.updateMatrixWorld(true);
    const loc = attach.clone().applyMatrix4(this.group.matrixWorld.clone().invert());
    a.setXYZ(2 * this.rim.length, conf.x, conf.y, conf.z);
    a.setXYZ(2 * this.rim.length + 1, loc.x, loc.y, loc.z);
    a.needsUpdate = true;
  }

  dispose() { this.canopy.geometry.dispose(); this.canopyMat.dispose(); this.lineGeo.dispose(); this.lines.material.dispose(); }
}

/** Launch rail stand for small rockets: rail along +y at the rocket's rail-button side. */
export function launchRail(height, offset, phi) {
  // built in the rocket's launch frame: +y along the rail, rocket base at the origin
  const g = new THREE.Group();
  g.name = 'launch-rail';
  const alu = new THREE.MeshStandardMaterial({ color: 0xb9bcc0, metalness: 1, roughness: 0.35 });
  const steel = new THREE.MeshStandardMaterial({ color: 0x55575a, metalness: 0.9, roughness: 0.55 });
  const black = new THREE.MeshStandardMaterial({ color: 0x1c1c1c, metalness: 0.2, roughness: 0.6 });
  const p = polar(phi, offset + 0.015, 0);
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.03, height, 0.03).translate(p.x, height / 2, p.z), alu);
  const pi = polar(phi, offset + 0.001, 0);
  const slot = new THREE.Mesh(new THREE.BoxGeometry(0.009, height, 0.009).translate(pi.x, height / 2, pi.z), black);
  const plate = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.01, 0.3).translate(0, -0.004, 0), steel);
  const ph = polar(phi, offset + 0.05, 0);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.035, 0.32, 16).translate(ph.x, 0.14, ph.z), steel);
  g.add(rail, slot, plate, hub);
  for (let k = 0; k < 3; k++) {
    const a = phi + (k * 2 * Math.PI) / 3 + Math.PI / 3;
    const top = new THREE.Vector3(ph.x, 0.26, ph.z);
    const foot = new THREE.Vector3(ph.x + 0.7 * Math.cos(a), -0.01, ph.z - 0.7 * Math.sin(a));
    const d = foot.clone().sub(top);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, d.length(), 10), steel);
    m.position.copy(top).addScaledVector(d, 0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
    g.add(m);
  }
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}
