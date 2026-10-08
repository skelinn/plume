// Geometry helpers for the procedural vehicle models.
//
// Mesh space = body frame after the ENU->three remap: the vehicle axis (body +z, towards the nose)
// is +y, body +x is +x and body +y is -z.  The hull base centre is the origin.

import * as THREE from 'three';

export const Y = new THREE.Vector3(0, 1, 0);

/** Merge (indexed or not) geometries with identical attribute sets into one indexed geometry. */
export function mergeGeometries(geos) {
  const list = geos;
  const names = Object.keys(list[0].attributes);
  const out = new THREE.BufferGeometry();
  let vcount = 0, icount = 0;
  for (const g of list) { vcount += g.attributes.position.count; icount += g.index ? g.index.count : g.attributes.position.count; }
  for (const n of names) {
    const size = list[0].attributes[n].itemSize;
    const arr = new Float32Array(vcount * size);
    let o = 0;
    for (const g of list) { const a = g.attributes[n]; arr.set(a.array.subarray(0, a.count * size), o); o += a.count * size; }
    out.setAttribute(n, new THREE.BufferAttribute(arr, size));
  }
  const idx = vcount > 65535 ? new Uint32Array(icount) : new Uint16Array(icount);
  let io = 0, base = 0;
  for (const g of list) {
    const n = g.attributes.position.count;
    if (g.index) for (let i = 0; i < g.index.count; i++) idx[io++] = g.index.array[i] + base;
    else for (let i = 0; i < n; i++) idx[io++] = i + base;
    base += n;
  }
  out.setIndex(new THREE.BufferAttribute(idx, 1));
  out.computeBoundingSphere();
  for (const g of geos) g.dispose();
  return out;
}

/** Bake a transform into a geometry (returns it). */
export function place(geo, { pos = [0, 0, 0], rot = [0, 0, 0], scale = [1, 1, 1], quat = null } = {}) {
  const m = new THREE.Matrix4().compose(
    new THREE.Vector3(...pos),
    quat || new THREE.Quaternion().setFromEuler(new THREE.Euler(...rot)),
    new THREE.Vector3(...scale),
  );
  geo.applyMatrix4(m);
  return geo;
}

/** Cylinder geometry from a to b (Vector3), radii ra at a and rb at b. */
export function tube(a, b, ra, rb = ra, seg = 12, open = false) {
  const d = new THREE.Vector3().subVectors(b, a);
  const len = d.length();
  const g = new THREE.CylinderGeometry(rb, ra, len, seg, 1, open);
  const q = new THREE.Quaternion().setFromUnitVectors(Y, d.clone().normalize());
  g.applyQuaternion(q);
  g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
  return g;
}

/** Mesh-space position of a point at body azimuth phi (rad), radius r, height y. */
export function polar(phi, r, y, out = new THREE.Vector3()) {
  return out.set(r * Math.cos(phi), y, -r * Math.sin(phi));
}

/** y-axis aligned frustum between heights y0..y1 with uv.v in metres (for tiled textures). */
export function frustum(r0, r1, y0, y1, seg = 64, open = true) {
  return new THREE.CylinderGeometry(r1, r0, y1 - y0, seg, 1, open).translate(0, (y0 + y1) / 2, 0);
}

/** Lathe from [r, y] points, uv.v = normalised arc length. */
export function lathe(pts, seg = 64) {
  return new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), seg);
}

/** Tangent ogive radius at distance x from the base (r base radius, ln nose length). */
export function ogiveRadius(r, ln, x) {
  const rho = (ln * ln + r * r) / (2 * r);
  return Math.max(Math.sqrt(Math.max(rho * rho - x * x, 0)) + r - rho, 0);
}

/** Rounded box (pill-like block) - a BoxGeometry with bevelled edges via ExtrudeGeometry. */
export function roundedBox(w, h, d, r = Math.min(w, h, d) * 0.2, segs = 3) {
  const s = new THREE.Shape();
  const x = -w / 2, y = -h / 2;
  r = Math.min(r, w / 2, h / 2);
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  const bev = Math.min(r * 0.6, d * 0.25);
  const g = new THREE.ExtrudeGeometry(s, { depth: d - 2 * bev, bevelEnabled: true, bevelThickness: bev, bevelSize: bev * 0.8, bevelSegments: segs, curveSegments: segs * 2 });
  g.translate(0, 0, -(d - 2 * bev) / 2);
  return g;
}

/** Strip non-shared attributes so geometries can be merged (keeps position/normal/uv). */
export function pnu(geo) {
  for (const k of Object.keys(geo.attributes)) if (!['position', 'normal', 'uv'].includes(k)) geo.deleteAttribute(k);
  if (!geo.attributes.uv) geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count * 2), 2));
  return geo;
}

export function shadowed(obj) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return obj;
}
