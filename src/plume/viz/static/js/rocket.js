// Procedural rocket built from meta.vehicle.
//
// The mesh is modelled in "mesh space" = body frame after the ENU->three remap, i.e. the vehicle
// axis (body +z, towards the nose) is mesh +y and body +y maps to mesh -z.  The hull base
// centre is the group origin, matching the replay's `pos`.  The group is then oriented with the
// converted replay quaternion (coords.quatEnuToW).

import * as THREE from 'three';
import { Plume, getGlowTexture } from './plume.js';

const Y = new THREE.Vector3(0, 1, 0);

/** Thin cylinder from a to b (mesh space). */
function strut(a, b, radius, material) {
  const d = new THREE.Vector3().subVectors(b, a);
  const len = d.length();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 10), material);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(Y, d.normalize());
  return m;
}

/** y-axis aligned frustum between heights y0..y1. */
function frustum(r0, r1, y0, y1, material, seg = 40) {
  const g = new THREE.CylinderGeometry(r1, r0, y1 - y0, seg, 1).translate(0, (y0 + y1) / 2, 0);
  return new THREE.Mesh(g, material);
}

/** Tangent ogive nose from radius r at y0 up to a tip at y0 + ln. */
function ogive(r, y0, ln, material) {
  const rho = (ln * ln + r * r) / (2 * r);
  const pts = [];
  const N = 20;
  for (let i = 0; i <= N; i++) {
    const x = (i / N) * ln;                                   // distance from the tip
    const rad = Math.sqrt(Math.max(rho * rho - (ln - x) * (ln - x), 0)) + r - rho;
    pts.push(new THREE.Vector2(Math.max(rad, 0), y0 + ln - x));
  }
  pts.reverse();                                              // base -> tip
  pts[pts.length - 1].x = 0;
  return new THREE.Mesh(new THREE.LatheGeometry(pts, 48), material);
}

export class Rocket {
  constructor(vehicle, env) {
    const V = vehicle;
    const L = V.length, D = V.diameter, r = D / 2;
    const ln = Math.min(V.nose_length ?? 0.15 * L, 0.45 * L);
    const lc = L - ln;
    this.L = L; this.D = D;
    this.group = new THREE.Group();
    this.group.name = 'rocket';
    this.thrustMax = V.engine?.thrust_max || 0;
    const root = this.group;

    const white = new THREE.MeshStandardMaterial({ color: 0xe4e8ee, roughness: 0.42, metalness: 0.15 });
    const grey = new THREE.MeshStandardMaterial({ color: 0xaab2bf, roughness: 0.5, metalness: 0.3 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x1d2128, roughness: 0.6, metalness: 0.4 });
    const accent = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness: 0.5, metalness: 0.1, emissive: 0x401800, emissiveIntensity: 0.35 });
    const legMat = new THREE.MeshStandardMaterial({ color: 0x59616e, roughness: 0.5, metalness: 0.55 });
    this.materials = [white, grey, dark, accent, legMat];

    // --- hull -----------------------------------------------------------------------------
    const skirtTop = Math.min(0.1 * lc, 0.6 * D + 0.01);
    root.add(frustum(r * 0.94, r, 0, skirtTop, dark));
    root.add(frustum(r, r, skirtTop, lc, white));
    root.add(frustum(r * 1.004, r * 1.004, lc * 0.9, lc * 0.935, dark));            // interstage band
    root.add(frustum(r * 1.004, r * 1.004, lc * 0.54, lc * 0.565, accent));          // accent stripe
    root.add(frustum(r * 1.004, r * 1.004, lc * 0.28, lc * 0.295, grey));            // weld ring
    root.add(ogive(r, lc, ln, grey));
    // raceway: a dark line down one side so roll is visible
    const rw = new THREE.Mesh(new THREE.BoxGeometry(0.07 * D, lc * 0.78, 0.04 * D), dark);
    rw.position.set(r * 1.01, lc * 0.5, 0);
    root.add(rw);

    // --- legs ------------------------------------------------------------------------------
    const legs = V.legs || {};
    const nLegs = legs.count | 0;
    if (nLegs > 0) {
      const span = legs.span ?? 2 * D, height = legs.height ?? D, attach = legs.attach_z ?? 0.15 * L;
      const padR = Math.max(0.09 * span + 0.03, 0.06);
      for (let k = 0; k < nLegs; k++) {
        const phi = ((k + 0.5) * 2 * Math.PI) / nLegs;
        const dir = new THREE.Vector3(Math.cos(phi), 0, Math.sin(phi));
        const A = dir.clone().multiplyScalar(r).setY(attach);
        const F = dir.clone().multiplyScalar(span).setY(-height);
        const B = dir.clone().multiplyScalar(r).setY(attach + 0.22 * lc);
        const M = A.clone().lerp(F, 0.68);
        const sr = Math.max(0.018 * span, 0.012);
        root.add(strut(A, F, sr, legMat));
        root.add(strut(B, M, sr * 0.7, legMat));
        const pad = new THREE.Mesh(new THREE.CylinderGeometry(padR * 0.85, padR, padR * 0.22, 20), grey);
        pad.position.copy(F).setY(-height + padR * 0.11);
        root.add(pad);
      }
    } else {
      // finned hobby rocket
      const c = 0.16 * lc, h = Math.max(0.9 * D, 0.05);
      const shape = new THREE.Shape();
      shape.moveTo(r * 0.98, c); shape.lineTo(r * 0.98, 0.0); shape.lineTo(r + h, -0.02 * lc); shape.lineTo(r + h * 0.85, c * 0.45); shape.closePath();
      const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.12 * D, bevelEnabled: false }).translate(0, 0, -0.06 * D);
      for (let k = 0; k < 4; k++) {
        const fin = new THREE.Mesh(geo, accent);
        fin.rotation.y = (k * Math.PI) / 2;
        root.add(fin);
      }
    }

    // --- grid fins (lattice panels hinged at the hull; fold up when stowed) ------------------
    this.gridFins = [];
    const GF = V.grid_fins;
    if (GF && GF.count > 0) {
      const span = GF.span, w = GF.chord, t = Math.max(0.3 * w, 0.03), bar = Math.max(0.03 * w, 0.01);
      const finMat = new THREE.MeshStandardMaterial({ color: 0xb4bcc8, roughness: 0.35, metalness: 0.7 });
      this.materials.push(finMat);
      const box = (sx, sy, sz, x, y, z, ry = 0) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), finMat);
        m.position.set(x, y, z);
        m.rotation.y = ry;
        return m;
      };
      for (let k = 0; k < GF.count; k++) {
        const phi = (2 * Math.PI * k) / GF.count;
        const hinge = new THREE.Group();                      // local +x radial, +y axial, z tangential
        hinge.position.set(r * Math.cos(phi), GF.z, -r * Math.sin(phi));
        hinge.rotation.y = phi;
        const fold = new THREE.Group();                       // rotates about z to stow along the hull
        const panel = new THREE.Group();                      // rotates about x (radial) = deflection
        // outer frame
        panel.add(box(span, t, bar, span / 2, 0, -w / 2), box(span, t, bar, span / 2, 0, w / 2));
        panel.add(box(bar, t, w, 0.02, 0, 0), box(bar, t, w, span, 0, 0));
        // diagonal lattice
        const nd = 5;
        const diag = Math.hypot(span, w) / nd;
        for (let i = 1; i < nd * 2; i++) {
          const f = i / (nd * 2);
          for (const sgn of [1, -1]) {
            const cx = f * span, len = Math.min(diag * 1.6, Math.hypot(span, w) * 0.5);
            const m = box(len * 0.6, t * 0.9, bar * 0.6, cx, 0, 0, sgn * Math.atan2(w, span));
            m.scale.x = Math.min(1, (Math.min(cx, span - cx) * 2.2) / (len * 0.6) + 0.15);
            panel.add(m);
          }
        }
        fold.add(panel);
        hinge.add(fold);
        root.add(hinge);
        this.gridFins.push({ fold, panel });
      }
    }

    // --- engine bell (gimbals about the pivot) ---------------------------------------------
    const re = V.engine?.nozzle_radius ?? 0.3 * r;
    const bellLen = Math.max(2.1 * re, 0.02);
    const pivotY = V.engine?.gimbal_z ?? 0.5 * bellLen;
    this.bellLen = bellLen;
    this.bellMat = new THREE.MeshStandardMaterial({
      color: 0x2b2f38, roughness: 0.45, metalness: 0.75, side: THREE.DoubleSide, emissive: 0x000000,
    });
    this.materials.push(this.bellMat);
    const rt = 0.38 * re;
    const prof = [new THREE.Vector2(rt * 1.5, rt * 0.9)];
    for (let i = 0; i <= 16; i++) {
      const t = i / 16;
      prof.push(new THREE.Vector2(rt + (re - rt) * (1 - Math.pow(1 - t, 1.8)), -t * bellLen));
    }
    this.bell = new THREE.Group();
    this.bell.position.set(0, pivotY, 0);
    this.bell.add(new THREE.Mesh(new THREE.LatheGeometry(prof, 40), this.bellMat));
    root.add(this.bell);

    this.plume = new Plume(re, env);
    this.plume.group.position.set(0, -bellLen, 0);
    this.bell.add(this.plume.group);

    // draw after the terrain (renderOrder 1-2) so detail tiles never overdraw the vehicle
    root.traverse((o) => { if (o.isMesh && !o.renderOrder) o.renderOrder = 5; });

    // --- RCS puffs --------------------------------------------------------------------------
    this.rcsZ = V.rcs_z ?? 0.9 * L;
    this.rcs = [];
    const puffSize = Math.max(0.5 * D, 0.1);
    const sides = [[1, 0], [-1, 0], [0, 1], [0, -1]];                 // body (x, y) outward directions
    for (const [bx, by] of sides) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: getGlowTexture(), color: 0xcfe8ff, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0,
      }));
      s.position.set(bx * (r + puffSize * 0.35), this.rcsZ, -by * (r + puffSize * 0.35));  // body y -> mesh -z
      s.scale.setScalar(puffSize);
      s.visible = false;
      s.renderOrder = 12;
      root.add(s);
      this.rcs.push({ sprite: s, bx, by, size: puffSize });
    }
  }

  /** Apply gimbal / throttle / RCS from a replay sample; returns the throttle used (0..1). */
  update(s) {
    // Thrust direction in body = (cos a sin b, -sin a, cos a cos b)  <=>  R = Ry(b) * Rx(a) applied to +z.
    const [a, b] = s.gimbal || [0, 0];
    const qb = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0));
    const qx = new THREE.Quaternion(Math.sin(a / 2), 0, 0, Math.cos(a / 2));
    const qy = new THREE.Quaternion(0, Math.sin(b / 2), 0, Math.cos(b / 2));
    qb.multiplyQuaternions(qy, qx);
    // body quaternion -> mesh space: same remap as ENU->three, (x, y, z) -> (x, z, -y)
    this.bell.quaternion.set(qb.x, qb.z, -qb.y, qb.w);

    let th = s.throttle;
    if (th === undefined) th = this.thrustMax > 0 && s.thrust !== undefined ? s.thrust / this.thrustMax : 0;
    this.plume.update(th, s.alt);
    const e = Math.min(th, 1);
    this.bellMat.emissive.setRGB(0.55 * e, 0.14 * e, 0.02 * e);

    if (this.gridFins.length) {
      const out = s.fins_out === undefined ? 1 : s.fins_out;
      const defl = s.fins || [];
      this.gridFins.forEach((g, i) => {
        g.fold.rotation.z = (1 - out) * (Math.PI / 2);        // stowed: folded up against the hull
        g.panel.rotation.x = defl[i] || 0;
      });
    }

    const rc = s.rcs || [0, 0, 0];
    const roll = Math.abs(rc[2]) * 0.5;
    // visual convention: +x torque fires the +y side jet, +y torque the -x side jet, etc.
    const drive = [Math.max(-rc[1], 0), Math.max(rc[1], 0), Math.max(rc[0], 0), Math.max(-rc[0], 0)];
    this.rcs.forEach((p, i) => {
      const v = Math.min(drive[i] + roll, 1);
      p.sprite.visible = v > 0.05;
      p.sprite.material.opacity = 0.35 + 0.65 * v;
      p.sprite.scale.setScalar(p.size * (0.5 + v));
    });
    return th;
  }

  dispose() {
    this.group.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    for (const m of this.materials) m.dispose();
  }
}
