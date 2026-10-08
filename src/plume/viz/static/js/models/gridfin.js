// Grid fins: a thick outer frame filled with a dense lattice of thin plates at +/-45 deg forming
// diamond cells, `depth` deep along the flow, with a root shaft, hinge clevis and an actuator
// fairing on the hull.  The whole panel (frame + lattice + shaft) is one merged geometry shared by
// all fins, with per-vertex heat tint on the windward (lower) face and tip.
//
// Local fin frame (before azimuth rotation): +x radial (span), +y along the vehicle axis (the flow
// passes through the cells along y), +z tangential (chord).
// Kinematics: hinge (on the hull, at the fin azimuth) -> fold (about the tangential z axis:
// 0 = deployed, +90 deg = stowed flat against the hull pointing at the nose) -> pitch (about the
// radial x axis: deflection) -> panel.

import * as THREE from 'three';
import { mergeGeometries, place, pnu, polar, roundedBox } from './geom.js';

function clipConvex(p0, dir, poly) {
  // Cyrus-Beck clip of the infinite line p0 + t dir against a CCW convex polygon [[x, z], ...]
  let t0 = -Infinity, t1 = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const ex = b[0] - a[0], ez = b[1] - a[1];
    const nx = ez, nz = -ex;                                // outward normal for a CCW polygon in (x, z)
    const den = nx * dir[0] + nz * dir[1];
    const num = nx * (p0[0] - a[0]) + nz * (p0[1] - a[1]);
    if (Math.abs(den) < 1e-12) { if (num > 0) return null; continue; }
    const t = -num / den;
    if (den > 0) t1 = Math.min(t1, t); else t0 = Math.max(t0, t);
    if (t0 > t1) return null;
  }
  return [t0, t1];
}

function offsetPolygon(poly, d) {
  // inward offset of a CCW convex polygon by d
  const n = poly.length, lines = [];
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    const ex = b[0] - a[0], ez = b[1] - a[1], l = Math.hypot(ex, ez);
    const nx = -ez / l, nz = ex / l;                        // inward normal
    lines.push({ p: [a[0] + nx * d, a[1] + nz * d], u: [ex / l, ez / l] });
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    const L1 = lines[(i + n - 1) % n], L2 = lines[i];
    const det = L1.u[0] * -L2.u[1] - L1.u[1] * -L2.u[0];
    const dx = L2.p[0] - L1.p[0], dz = L2.p[1] - L1.p[1];
    const t = (dx * -L2.u[1] - dz * -L2.u[0]) / det;
    out.push([L1.p[0] + L1.u[0] * t, L1.p[1] + L1.u[1] * t]);
  }
  return out;
}

/** In-plane bar from (ax, az) to (bx, bz), width w (in plane), height h (along y), centred at y0. */
function bar(ax, az, bx, bz, w, h, y0 = 0) {
  const len = Math.hypot(bx - ax, bz - az);
  const g = new THREE.BoxGeometry(len, h, w);
  return place(g, { pos: [(ax + bx) / 2, y0, (az + bz) / 2], rot: [0, Math.atan2(-(bz - az), bx - ax), 0] });
}

/** Panel geometry (frame + lattice + root plate + shaft) in the local fin frame. */
export function gridFinGeometry({ span, chord, depth, x0 }) {
  const c = chord, d = depth;
  const x1 = x0 + span;
  const chx = 0.16 * span, chz = 0.2 * c;                 // chamfered tip corners
  const outline = [[x0, c / 2], [x0, -c / 2], [x1 - chx, -c / 2], [x1, -c / 2 + chz], [x1, c / 2 - chz], [x1 - chx, c / 2]];
  // make it CCW in (x, z): signed area
  let area = 0;
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i + 1) % outline.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  if (area < 0) outline.reverse();
  const fw = Math.max(0.045 * span, 0.012);                // frame bar width
  const tp = Math.max(0.011 * span, 0.0035);               // lattice plate thickness
  const geos = [];
  // frame: one bar per outline edge, set inwards
  const inner = offsetPolygon(outline, fw);
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i + 1) % outline.length];
    const ia = inner[i], ib = inner[(i + 1) % outline.length];
    const isRoot = Math.abs(a[0] - x0) < 1e-6 && Math.abs(b[0] - x0) < 1e-6;
    const ma = [(a[0] + ia[0]) / 2, (a[1] + ia[1]) / 2], mb = [(b[0] + ib[0]) / 2, (b[1] + ib[1]) / 2];
    // extend the bars a little along their direction so the corners close
    const ex = mb[0] - ma[0], ez = mb[1] - ma[1], l = Math.hypot(ex, ez);
    const k = (fw * 0.5) / l;
    geos.push(bar(ma[0] - ex * k, ma[1] - ez * k, mb[0] + ex * k, mb[1] + ez * k, isRoot ? fw * 1.6 : fw, d * (isRoot ? 1.18 : 1.06)));
  }
  // lattice: plates at +/-45 deg, ~9 diamond cells across the span
  const nCells = Math.max(6, Math.round(9 * Math.min(span / 0.55, 1.6)));
  const pitch = span / nCells / Math.SQRT2;                // perpendicular spacing between parallel plates
  for (const s of [1, -1]) {
    const dir = [Math.SQRT1_2, s * Math.SQRT1_2];
    const nrm = [-dir[1], dir[0]];
    const proj = inner.map((p) => p[0] * nrm[0] + p[1] * nrm[1]);
    const lo = Math.min(...proj), hi = Math.max(...proj);
    const cx = (x0 + x1) / 2;
    const c0 = cx * nrm[0];
    for (let k = Math.ceil((lo - c0) / pitch); c0 + k * pitch <= hi; k++) {
      const off = c0 + k * pitch;
      const p0 = [nrm[0] * off, nrm[1] * off];
      const tt = clipConvex(p0, dir, inner);
      if (!tt || tt[1] - tt[0] < tp * 2) continue;
      const e0 = tt[0] - fw * 0.3, e1 = tt[1] + fw * 0.3;
      geos.push(bar(p0[0] + dir[0] * e0, p0[1] + dir[1] * e0, p0[0] + dir[0] * e1, p0[1] + dir[1] * e1, tp, d * 0.98));
    }
  }
  // root plate + shaft back to the hinge (the deflection axis is the local x axis)
  geos.push(place(new THREE.BoxGeometry(fw * 1.2, d * 1.25, c * 0.42), { pos: [x0 - fw * 0.2, 0, 0] }));
  const shaft = new THREE.CylinderGeometry(c * 0.07, c * 0.07, x0 + fw, 16);
  geos.push(place(shaft, { pos: [(x0 + fw) / 2 - fw * 0.5, 0, 0], rot: [0, 0, Math.PI / 2] }));
  const geo = mergeGeometries(geos.map(pnu));
  // heat tint: windward (lower, -y) face and the tip run darker/bronzed
  const pos = geo.attributes.position, col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    const low = THREE.MathUtils.smoothstep(-y, -0.1 * d, 0.5 * d);
    const tip = THREE.MathUtils.smoothstep(x, x0 + 0.6 * span, x1);
    const heat = Math.min(1, 0.8 * low + 0.4 * tip * low + 0.2 * tip);
    const n = Math.sin(x * 37.0 + y * 91.0) * 0.5 + 0.5;          // cheap per-vertex mottling
    // straw -> bronze -> dark grey-blue oxide
    col[i * 3] = (1.0 - 0.38 * heat) * (0.9 + 0.1 * n);
    col[i * 3 + 1] = (1.0 - 0.48 * heat) * (0.9 + 0.1 * n);
    col[i * 3 + 2] = (1.0 - 0.52 * heat + 0.1 * heat * heat) * (0.9 + 0.1 * n);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return geo;
}

export class GridFins {
  /**
   * @param GF meta.vehicle.grid_fins
   * @param hullR hull radius, @param rAt(y) hull radius at height y (nose taper)
   * @param mats {ti: titanium (vertex colours), dark: black fairings, metal: fittings}
   */
  constructor(GF, hullR, mats) {
    this.group = new THREE.Group();
    this.group.name = 'grid-fins';
    this.fins = [];
    this.maxDefl = GF.max_deflection || 0.35;
    const span = GF.span || 0.5, chord = GF.chord || 0.4;
    const depth = GF.depth || Math.max(0.22 * chord, 0.06);
    const zf = GF.z;
    const gap = 0.012;
    const hingeR = hullR + depth / 2 + gap;
    const x0 = Math.max(0.035, 0.07 * span);
    this.geo = gridFinGeometry({ span, chord, depth, x0 });
    // hinge clevis (fixed to the hull): base plate + two lugs either side of the shaft
    const clevis = mergeGeometries([
      place(roundedBox(chord * 0.5, depth * 1.9, hingeR - hullR + 0.02, 0.01), { pos: [-(hingeR - hullR) / 2 + 0.005, 0, 0], rot: [0, Math.PI / 2, 0] }),
      place(new THREE.BoxGeometry(x0 * 0.9, depth * 0.9, chord * 0.06), { pos: [x0 * 0.2, 0, chord * 0.16] }),
      place(new THREE.BoxGeometry(x0 * 0.9, depth * 0.9, chord * 0.06), { pos: [x0 * 0.2, 0, -chord * 0.16] }),
      place(new THREE.CylinderGeometry(depth * 0.18, depth * 0.18, chord * 0.42, 16), { rot: [Math.PI / 2, 0, 0] }),
    ].map(pnu));
    // actuator fairing below the hinge: a flattened half-capsule on the hull
    const fairLen = Math.max(0.42 * span, 0.12);
    const fair = new THREE.CapsuleGeometry(chord * 0.15, fairLen, 6, 16);
    place(fair, { pos: [0, -fairLen / 2 - depth * 0.7, 0], scale: [0.42, 1, 1] });
    this.clevisGeo = clevis;
    this.fairGeo = fair;
    for (let k = 0; k < GF.count; k++) {
      const phi = (2 * Math.PI * k) / GF.count;          // physics: fin k at azimuth 2 pi k / n
      const hinge = new THREE.Group();
      polar(phi, hingeR, zf, hinge.position);
      hinge.rotation.y = phi;
      const cl = new THREE.Mesh(clevis, mats.clevis || mats.metal);
      hinge.add(cl);
      const fm = new THREE.Mesh(fair, mats.dark);
      fm.position.x = -(hingeR - hullR);
      hinge.add(fm);
      const fold = new THREE.Group();
      const pitch = new THREE.Group();
      const panel = new THREE.Mesh(this.geo, mats.ti);
      pitch.add(panel);
      fold.add(pitch);
      hinge.add(fold);
      this.group.add(hinge);
      this.fins.push({ fold, pitch, phi });
    }
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  }

  /** @param out 0 stowed .. 1 deployed, @param defl per-fin deflection (rad) */
  update(out, defl) {
    const o = THREE.MathUtils.clamp(out, 0, 1);
    const e = o * o * (3 - 2 * o);
    this.fins.forEach((f, i) => {
      f.fold.rotation.z = (1 - e) * (Math.PI / 2);
      f.pitch.rotation.x = (defl?.[i] || 0) * e;
    });
  }

  dispose() { this.geo.dispose(); this.clevisGeo.dispose(); this.fairGeo.dispose(); }
}
