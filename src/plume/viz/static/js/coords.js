// Coordinate conventions
// ----------------------
// Replays are ENU (x east, y north, z up). three.js is y-up, so everything is remapped as
//     (x, y, z)_enu  ->  (x, z, -y)_three          (a proper rotation, det = +1)
// A proper rotation P maps a rotation quaternion q = (w; v) to (w; P v), so quaternions
// [w, x, y, z]_enu become (x, z, -y, w)_three.  The rocket mesh is modelled directly in
// "body frame after the same remap" (vehicle axis = +y, body y = -z), so applying the
// converted quaternion to it is correct.
//
// "W-space" below means three-axis-oriented *world* coordinates held in float64 (THREE.Vector3
// stores JS doubles).  Distances reach ~800 km, so nothing is ever handed to the GPU in W-space:
// every frame each object's position is (W - origin), subtracted in double, where `origin` is a
// point near the camera (see cameras.js).

import * as THREE from 'three';

export const Y_UP = new THREE.Vector3(0, 1, 0);

export function enuToW(p, out = new THREE.Vector3()) {
  return out.set(p[0], p[2], -p[1]);
}

/** [w,x,y,z] ENU body->world  ->  THREE.Quaternion in W-space. */
export function quatEnuToW(q, out = new THREE.Quaternion()) {
  return out.set(q[1], q[3], -q[2], q[0]);
}

/**
 * Scene geometry model: flat Earth, or a sphere of radius R whose tangent frame at the launch
 * site is the ENU frame (centre at (0,0,-R) ENU = (0,-R,0) W).
 */
export class Frame {
  constructor(scene = {}) {
    this.spherical = scene.frame === 'spherical' && scene.earth_radius > 0;
    this.R = this.spherical ? scene.earth_radius : Infinity;
    this.centerW = new THREE.Vector3(0, this.spherical ? -this.R : 0, 0);
  }

  /** Map-coordinate terrain point (u east, v north arc length, height h) -> W-space. */
  surface(u, v, h, out = new THREE.Vector3()) {
    if (!this.spherical) return out.set(u, h, -v);
    const s = Math.hypot(u, v);
    let nx = 0, ny = 0, nz = 1;
    if (s > 1e-9) {
      const th = s / this.R, st = Math.sin(th);
      nx = (st * u) / s; ny = (st * v) / s; nz = Math.cos(th);
    }
    const r = this.R + h;
    // ENU: c + r*n with c = (0,0,-R); then remap to W
    const ex = r * nx, ey = r * ny, ez = -this.R + r * nz;
    return out.set(ex, ez, -ey);
  }

  /** Inverse of surface(): W-space point -> map coordinates {u, v, h}. */
  toMap(p) {
    if (!this.spherical) return { u: p.x, v: -p.z, h: p.y };
    const d = p.clone().sub(this.centerW);
    const r = d.length();
    const nx = d.x / r, ny = -d.z / r, nz = d.y / r; // W -> ENU
    const sinT = Math.sqrt(Math.max(1 - nz * nz, 0));
    const s = this.R * Math.acos(Math.min(Math.max(nz, -1), 1));
    return { u: sinT < 1e-12 ? 0 : (s * nx) / sinT, v: sinT < 1e-12 ? 0 : (s * ny) / sinT, h: r - this.R };
  }

  /** Local "up" at a W-space point. */
  up(p, out = new THREE.Vector3()) {
    if (!this.spherical) return out.set(0, 1, 0);
    return out.copy(p).sub(this.centerW).normalize();
  }

  /** Height above the datum (sea-level sphere / z=0 plane). */
  altitude(p) {
    return this.spherical ? p.distanceTo(this.centerW) - this.R : p.y;
  }

  /** Orthonormal local tangent basis at p: {up, east, north}. */
  basis(p) {
    const up = this.up(p);
    const east = new THREE.Vector3(1, 0, 0);
    east.addScaledVector(up, -east.dot(up));
    if (east.lengthSq() < 1e-12) east.set(0, 0, 1);
    east.normalize();
    const north = new THREE.Vector3().crossVectors(up, east).normalize(); // up x east = north
    return { up, east, north };
  }
}
