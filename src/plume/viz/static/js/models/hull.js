// Hull: white painted barrel with weld seams / panel lines (tiled procedural maps), a carbon-black
// upper band where the grid fins mount, tangent-ogive nose with a metal tip, raceway with segmented
// covers, RCS pods with tangential nozzle pairs, and generic decals (vertical "PLUME", serial,
// mission patch).  Soot build-up is applied by the shared soot patch (materials.js).

import * as THREE from 'three';
import * as M from '../materials.js';
import { frustum, lathe, mergeGeometries, ogiveRadius, place, pnu, polar, roundedBox } from './geom.js';

const BARREL = 1.6; // m per circumferential weld

function verticalTextTexture(text, { color = '#151515' } = {}) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 1536;
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.save();
  g.translate(c.width / 2, c.height / 2);
  g.rotate(Math.PI / 2);
  g.fillStyle = color;
  g.font = '600 210px "IBM Plex Sans", "Segoe UI", Arial, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  // letter-spaced
  const chars = text.split('');
  const adv = 168;
  chars.forEach((ch, i) => g.fillText(ch, (i - (chars.length - 1) / 2) * adv, 6));
  g.restore();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function patchTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.clearRect(0, 0, 256, 256);
  g.fillStyle = '#151515';
  g.beginPath(); g.arc(128, 128, 120, 0, Math.PI * 2); g.fill();
  g.strokeStyle = '#e8e8e4'; g.lineWidth = 6;
  g.beginPath(); g.arc(128, 128, 104, 0, Math.PI * 2); g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  // company mark (Plume Rocketry cubes), drawn once the image has loaded
  const img = new Image();
  img.onload = () => {
    const h = 132, w = (h * img.width) / img.height;
    g.drawImage(img, 128 - w / 2, 128 - h / 2, w, h);
    t.needsUpdate = true;
  };
  img.src = new URL('../../img/mark.png', import.meta.url).href;
  return t;
}

export class Hull {
  /**
   * @param V meta.vehicle
   * @param soot soot uniforms (materials.makeSootUniforms)
   */
  constructor(V, soot) {
    const L = V.length, D = V.diameter, r = D / 2;
    const ln = Math.min(V.nose_length ?? 0.15 * L, 0.45 * L);
    const lc = L - ln;
    this.r = r; this.L = L; this.lc = lc; this.ln = ln;
    this.group = new THREE.Group();
    this.group.name = 'hull';
    this.materials = [];
    const add = (m) => { this.materials.push(m); return m; };
    const circ = 2 * Math.PI * r;

    const GF = V.grid_fins && V.grid_fins.count > 0 ? V.grid_fins : null;
    const bandY = GF ? Math.max(GF.z - Math.max(0.9 * (GF.span || 0.5) + 0.25, 0.8), 0.55 * lc) : lc;
    this.bandY = bandY;

    // ---- barrel (painted) + carbon band
    const paint = M.hullPaint();
    const barrelMat = add(M.patchSoot(M.withRepeat(paint, 4, bandY / BARREL), soot));
    const barrel = new THREE.Mesh(frustum(r, r, 0, bandY, 96), barrelMat);
    this.group.add(barrel);
    if (GF) {
      const cm = add(M.patchSoot(M.carbon({ repeat: 1 }), soot, { strength: 0.6 }));
      for (const k of ['map', 'roughnessMap', 'normalMap']) cm[k].repeat.set(circ / 0.05, (lc - bandY) / 0.05);
      this.group.add(new THREE.Mesh(frustum(r * 1.003, r * 1.003, bandY, lc, 96), cm));
      this.bandMat = cm;
      // thin metallic closeout rings at both band edges
      const ringMat = add(M.aluminium());
      const rg = mergeGeometries([
        place(new THREE.TorusGeometry(r * 1.004, 0.006 * D + 0.002, 6, 96), { pos: [0, bandY, 0], rot: [Math.PI / 2, 0, 0] }),
        place(new THREE.TorusGeometry(r * 1.004, 0.006 * D + 0.002, 6, 96), { pos: [0, lc, 0], rot: [Math.PI / 2, 0, 0] }),
      ].map(pnu));
      this.group.add(new THREE.Mesh(rg, ringMat));
    }

    // ---- nose (tangent ogive) + tip
    const pts = [];
    const N = 40;
    for (let i = 0; i <= N; i++) {
      const x = (i / N) * ln;
      pts.push([Math.max(ogiveRadius(r, ln, x), 0.0005), lc + x]);
    }
    pts[N][0] = 0.0;
    const noseMat = add(M.patchSoot(M.withRepeat(paint, 4, 1), soot, { strength: 0.25 }));
    this.group.add(new THREE.Mesh(lathe(pts, 96), noseMat));
    const tipR = ogiveRadius(r, ln, ln * 0.97);
    const tip = new THREE.Mesh(new THREE.SphereGeometry(Math.max(tipR * 1.6, 0.012), 16, 10), add(M.aluminium()));
    tip.position.y = lc + ln * 0.975;
    this.group.add(tip);

    // ---- aft lip
    const lipMat = add(M.darkMetal());
    this.group.add(new THREE.Mesh(place(new THREE.TorusGeometry(r * 1.002, 0.008 * D + 0.002, 6, 96), { pos: [0, 0.01, 0], rot: [Math.PI / 2, 0, 0] }), lipMat));

    // ---- raceway (segmented covers), between a grid fin and a leg
    const rwPhi = (20 * Math.PI) / 180;
    const rwMat = add(M.patchSoot(new THREE.MeshPhysicalMaterial({ color: 0xe9e9e6, roughness: 0.5, metalness: 0, clearcoat: 0.2 }), soot));
    const rw = [];
    const segL = 2.4, y0 = 0.25, y1 = (GF ? bandY : lc) - 0.15;
    for (let y = y0; y < y1 - 0.2; y += segL) {
      const h = Math.min(segL - 0.012, y1 - y);
      const g = roundedBox(0.07 * D, h, 0.035 * D, 0.006 * D + 0.002, 2);
      // roundedBox: x tangential, y axial, z radial -> rotate so z points outwards at rwPhi
      place(g, { pos: [0, y + h / 2, 0] });
      const p = polar(rwPhi, r + 0.012 * D, 0);
      g.applyMatrix4(new THREE.Matrix4().makeRotationY(rwPhi + Math.PI / 2));
      g.translate(p.x, 0, p.z);
      rw.push(pnu(g));
    }
    if (rw.length) this.group.add(new THREE.Mesh(mergeGeometries(rw), rwMat));

    // ---- RCS pods (physics: pods at 2 pi k / n, +/- tangential thrusters)
    this.rcsPods = [];
    if (V.rcs_z !== undefined && V.rcs_z !== null && L > 3) {
      const zr = V.rcs_z;
      const rAt = zr > lc ? ogiveRadius(r, ln, zr - lc) : r;
      const podMat = add(new THREE.MeshPhysicalMaterial({ color: 0xdedcd6, roughness: 0.55, metalness: 0.1, clearcoat: 0.15 }));
      const nozMat = add(M.darkMetal());
      const pw = 0.13 * D, ph = 0.2 * D, pd = 0.05 * D;
      const nozL = 0.035 * D, nozR = 0.016 * D;
      const nPods = 4;
      // low blister: a capsule flattened against the hull (x tangential, y axial, z radial)
      const podGeo = pnu(place(new THREE.CapsuleGeometry(pw / 2, ph - pw, 6, 20), { scale: [1, 1, (2 * pd) / pw] }));
      const nozGeo = mergeGeometries([
        place(new THREE.CylinderGeometry(nozR, nozR * 0.55, nozL, 12, 1, true), { pos: [pw / 2 + nozL / 2 - 0.004, 0, pd * 0.05], rot: [0, 0, Math.PI / 2] }),
        place(new THREE.CylinderGeometry(nozR, nozR * 0.55, nozL, 12, 1, true), { pos: [-pw / 2 - nozL / 2 + 0.004, 0, pd * 0.05], rot: [0, 0, -Math.PI / 2] }),
      ].map(pnu));
      nozMat.side = THREE.DoubleSide;
      for (let k = 0; k < nPods; k++) {
        const th = (2 * Math.PI * k) / nPods;
        const g = new THREE.Group();
        polar(th, rAt - pd * 0.15, zr, g.position);
        g.rotation.y = th + Math.PI / 2;                  // local z -> radial outwards, local x -> -tangent
        g.add(new THREE.Mesh(podGeo, podMat), new THREE.Mesh(nozGeo, nozMat));
        this.group.add(g);
        // thruster exits (mesh space) and their exhaust directions (opposite to thrust)
        const tan = new THREE.Vector3(-Math.sin(th), 0, -Math.cos(th));   // body (-sin, cos, 0) -> mesh
        const c = g.position.clone();
        this.rcsPods.push({ th, pos: c, tan, offset: pw / 2 + nozL });
      }
    }

    // ---- decals
    if (L > 3) {
      const decalMat = (map, opacity = 1) => add(new THREE.MeshStandardMaterial({
        map, transparent: true, opacity, roughness: 0.45, metalness: 0, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
      }));
      const wordH = Math.min(0.42 * bandY, 3.4 * D);         // along the hull
      const wordW = wordH / 6;                                // canvas aspect 256 x 1536
      const yMid = Math.min(0.62 * bandY, bandY - wordH / 2 - 0.4);
      const wPhi = (200 * Math.PI) / 180;
      const arc = wordW / r;
      const gw = new THREE.CylinderGeometry(r * 1.0015, r * 1.0015, wordH, 32, 1, true, wPhi + Math.PI / 2 - arc / 2, arc);
      gw.translate(0, yMid, 0);
      this.group.add(new THREE.Mesh(gw, decalMat(verticalTextTexture('PLUME'))));
      // serial near the base, mission patch below the band, opposite side
      const sPhi = (200 * Math.PI) / 180;
      const sh = 0.09 * D, sw = sh * 4;
      const gs = new THREE.CylinderGeometry(r * 1.0015, r * 1.0015, sh, 16, 1, true, sPhi + Math.PI / 2 - sw / r / 2, sw / r);
      gs.translate(0, Math.max(0.12 * L, 0.6), 0);
      this.group.add(new THREE.Mesh(gs, decalMat(M.textTexture('PL-0001', { w: 1024, h: 256, font: '500 170px "IBM Plex Mono", Consolas, monospace' }), 0.9)));
      const pPhi = (20 * Math.PI) / 180 + Math.PI;
      const pd2 = 0.36 * D;
      const gp = new THREE.CylinderGeometry(r * 1.0015, r * 1.0015, pd2, 16, 1, true, pPhi + Math.PI / 2 - pd2 / r / 2, pd2 / r);
      gp.translate(0, bandY - pd2 * 0.9, 0);
      this.group.add(new THREE.Mesh(gp, decalMat(patchTexture())));
    }
    this.group.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  }

  dispose() {
    this.group.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
    for (const m of this.materials) { m.map?.dispose?.(); m.dispose(); }
  }
}
