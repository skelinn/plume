// One 3-D world per replay: sky, ground/terrain, pads, target, rocket, plume light, trail.
//
// Per frame:  world.update(t)            -> samples the replay, poses the rocket, reports
//                                           the camera context (centre, local basis, heading ...)
//             rig.update(ctx)            -> picks the floating origin + camera
//             world.layout(origin, cam)  -> places every object at (W - origin) in float32

import * as THREE from 'three';
import { Frame, enuToW, quatEnuToW, Y_UP } from './coords.js';
import { makeEnvUniforms } from './shaders.js';
import { Sky } from './sky.js';
import { Terrain, GroundPlane, fetchTerrain } from './terrain.js';
import { Rocket } from './rocket.js';
import { Trail } from './trail.js';
import { clamp, smoothstep } from './util.js';

const MAX_TRAIL_POINTS = 14000;

function padTexture(name) {
  const S = 512, c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  g.fillStyle = '#6a6f78';
  g.fillRect(0, 0, S, S);
  for (let i = 0; i < 2500; i++) { // concrete speckle
    g.fillStyle = `rgba(${Math.random() < 0.5 ? '255,255,255' : '0,0,0'},${Math.random() * 0.06})`;
    g.fillRect(Math.random() * S, Math.random() * S, 2 + Math.random() * 6, 2 + Math.random() * 6);
  }
  g.strokeStyle = '#e8edf3'; g.lineWidth = S * 0.03;
  g.beginPath(); g.arc(S / 2, S / 2, S * 0.46, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = '#f2b134'; g.lineWidth = S * 0.012;
  g.beginPath(); g.arc(S / 2, S / 2, S * 0.40, 0, Math.PI * 2); g.stroke();
  g.fillStyle = '#e8edf3';
  g.font = `bold ${S * 0.46}px "Segoe UI", Arial, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('H', S / 2, S / 2 + S * 0.02);
  g.font = `600 ${S * 0.06}px "Segoe UI", Arial, sans-serif`;
  g.fillText((name || '').slice(0, 10).toUpperCase(), S / 2, S * 0.83);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export class World {
  /**
   * @param replay Replay
   * @param opts {labelLayer: HTMLElement, capture: bool}
   */
  constructor(replay, { labelLayer = null, capture = false } = {}) {
    this.replay = replay;
    this.capture = capture;
    this.labelLayer = labelLayer;
    const meta = replay.meta, sc = meta.scene;
    this.frame = new Frame(sc);
    this.scene = new THREE.Scene();
    this.env = makeEnvUniforms();
    this.vehicle = meta.vehicle;
    this.L = meta.vehicle.length || 5;
    this.D = meta.vehicle.diameter || 0.5;

    // fixed sun, defined in the launch-site tangent frame (el 32 deg, from the south-west)
    const b0 = this.frame.basis(new THREE.Vector3(0, 0, 0));
    const el = (32 * Math.PI) / 180, az = (225 * Math.PI) / 180;
    this.sunDir = new THREE.Vector3()
      .addScaledVector(b0.up, Math.sin(el))
      .addScaledVector(b0.east, Math.cos(el) * Math.cos(az))
      .addScaledVector(b0.north, Math.cos(el) * Math.sin(az))
      .normalize();
    this.env.uSunDir.value.copy(this.sunDir);

    this.sky = new Sky(this.env, this.frame);
    this.scene.add(...this.sky.objects());

    this.hemi = new THREE.HemisphereLight(0x9db8e8, 0x2a3040, 1.15);
    this.sun = new THREE.DirectionalLight(0xfff0dc, 2.8);
    this.plumeLight = new THREE.PointLight(0xff8a3a, 0, 0, 2);
    this.scene.add(this.hemi, this.sun, this.sun.target, this.plumeLight);

    this.rocket = new Rocket(meta.vehicle, this.env);
    this.scene.add(this.rocket.group);

    this.trail = new Trail({ maxSpeed: Math.max(replay.maxSpeed(), 1) });
    this.scene.add(this.trail.group);

    this.terrain = null;
    this.plane = null;
    this.padObjs = [];
    this.targetObj = null;
    this.labels = [];
    this.locator = null;

    this.baseW = new THREE.Vector3();
    this.quatW = new THREE.Quaternion();
    this.centerW = new THREE.Vector3();
    this.state = null;
    this._trailAdded = 0;
    this._trailStride = Math.max(1, Math.ceil(replay.n / MAX_TRAIL_POINTS));
    this.warnings = [];
    this._buildGround();
    this._buildMarkers();
    this.syncTrail();
    // headings are computed once for the whole flight
    this._globalHeading = null;
  }

  // ------------------------------------------------------------------ construction
  _buildGround() {
    const g = this.replay.meta.scene.ground || { type: 'plane' };
    if (g.type === 'plane') {
      this.plane = new GroundPlane(this.env);
      this.scene.add(this.plane.mesh);
    } else if (g.type === 'terrain' && g.terrain_id) {
      this.terrain = new Terrain(this.frame, this.env);
      this.scene.add(this.terrain.group);
    }
  }

  /** Fetch + build heightmaps (async); resolves when ready. Failures degrade to a plane. */
  async load() {
    const g = this.replay.meta.scene.ground || {};
    if (!this.terrain) return;
    try {
      const [base, ...details] = await Promise.all([
        fetchTerrain(g.terrain_id, 1024),
        ...(g.detail_terrain_ids || []).map((id) => fetchTerrain(id, 1024).catch((e) => { this.warnings.push(String(e)); return null; })),
      ]);
      this.terrain.addLayer(base, { detail: false });
      for (const d of details) if (d) this.terrain.addLayer(d, { detail: true, lift: 0.2 });
      this._seatMarkers();
    } catch (e) {
      this.warnings.push(String(e));
      this.scene.remove(this.terrain.group);
      this.terrain = null;
      this.plane = new GroundPlane(this.env);
      this.scene.add(this.plane.mesh);
    }
  }

  /**
   * Pads/target are placed from the replay positions, but the (coarse, lifted) terrain mesh can
   * sit a metre or two above them.  Seat pads on the highest terrain point of their footprint
   * (the disc grows a skirt below), and drape the target rings over the surface.
   */
  _seatMarkers() {
    const F = this.frame, T = this.terrain;
    const probe = (W, R) => {
      const m = F.toMap(W);
      let lo = Infinity, hi = -Infinity;
      const take = (u, v) => {
        const h = T.heightAt(u, v);
        if (h != null) { lo = Math.min(lo, h); hi = Math.max(hi, h); }
      };
      take(m.u, m.v);
      for (const f of [0.5, 1])
        for (let k = 0; k < 12; k++) take(m.u + f * R * Math.cos((k * Math.PI) / 6), m.v + f * R * Math.sin((k * Math.PI) / 6));
      return { m, lo, hi };
    };
    for (const p of this.padObjs) {
      const { m, lo, hi } = probe(p.W, p.pad.radius);
      if (hi === -Infinity) continue;
      const top = Math.max(m.h, hi), bottom = Math.min(m.h, lo);
      F.surface(m.u, m.v, top, p.W);
      p.up = F.up(p.W);
      const hgt = top - bottom + 0.3;
      p.mesh.geometry.dispose();
      p.mesh.geometry = new THREE.CylinderGeometry(p.pad.radius, p.pad.radius, hgt, 72);
      p.offset = -hgt / 2 + 0.08;
    }
    const tg = this.targetObj;
    if (tg) {
      const m = F.toMap(tg.W);
      const h0 = T.heightAt(m.u, m.v);
      if (h0 == null) return;
      F.surface(m.u, m.v, Math.max(m.h, h0), tg.W);
      tg.up = F.up(tg.W);
      const tmp = new THREE.Vector3();
      const draped = (r0, r1, nr, seg) => {
        const pos = [], idx = [];
        for (let ir = 0; ir <= nr; ir++) {
          const r = r0 + ((r1 - r0) * ir) / nr;
          for (let k = 0; k <= seg; k++) {
            const a = (2 * Math.PI * k) / seg, u = m.u + r * Math.cos(a), v = m.v + r * Math.sin(a);
            F.surface(u, v, (T.heightAt(u, v) ?? m.h) + 0.12, tmp).sub(tg.W);
            pos.push(tmp.x, tmp.y, tmp.z);
          }
        }
        for (let ir = 0; ir < nr; ir++)
          for (let k = 0; k < seg; k++) {
            const a = ir * (seg + 1) + k, b = a + 1, c = a + seg + 1, d = c + 1;
            idx.push(a, c, b, b, c, d);
          }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setIndex(idx);
        return g;
      };
      const R = tg.R;
      tg.ring.geometry.dispose(); tg.inner.geometry.dispose(); tg.glow.geometry.dispose();
      tg.ring.geometry = draped(R * 0.9, R, 2, 160);
      tg.inner.geometry = draped(R * 0.28, R * 0.3, 1, 96);
      tg.glow.geometry = draped(0, R * 0.9, 8, 96);
      for (const o of [tg.ring, tg.inner, tg.glow]) o.rotation.set(0, 0, 0);
      tg.draped = true;
    }
  }

  _buildMarkers() {
    const sc = this.replay.meta.scene;
    const ringMat = (color, opacity) => new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false,
      side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6,
    });
    for (const pad of sc.pads || []) {
      const W = enuToW(pad.pos);
      const tex = padTexture(pad.name);
      const mat = new THREE.MeshStandardMaterial({
        map: tex, roughness: 0.92, metalness: 0, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
      });
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(pad.radius, pad.radius, 0.16, 72), mat);
      mesh.renderOrder = 5;
      this.scene.add(mesh);
      this.padObjs.push({ mesh, W, up: this.frame.up(W), pad });
      const tg = sc.target && sc.target.pos ? enuToW(sc.target.pos) : null;
      if (!tg || tg.distanceTo(W) > Math.max(pad.radius, sc.target.radius || 0) * 1.5) this._addLabel(pad.name || 'PAD', () => W, 'pad');
    }
    if (sc.target && sc.target.pos) {
      const R = sc.target.radius || 5;
      const W = enuToW(sc.target.pos);
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.RingGeometry(R * 0.9, R, 128), ringMat(0x35e0ff, 0.9));
      ring.rotation.x = -Math.PI / 2;
      const inner = new THREE.Mesh(new THREE.RingGeometry(R * 0.28, R * 0.3, 96), ringMat(0x35e0ff, 0.5));
      inner.rotation.x = -Math.PI / 2;
      const glow = new THREE.Mesh(new THREE.CircleGeometry(R * 0.9, 96), ringMat(0x35e0ff, 0.07));
      glow.rotation.x = -Math.PI / 2;
      g.add(ring, inner, glow);
      this.scene.add(g);
      this.targetObj = { group: g, ring, inner, glow, W, up: this.frame.up(W), R };
      this._addLabel('TARGET', () => W, 'target');
    }
    if (this.labelLayer) {
      this.locator = document.createElement('div');
      this.locator.className = 'locator';
      this.locator.innerHTML = '<i></i><span></span>';
      this.locator.querySelector('span').textContent = this.vehicle.name || 'vehicle';
      this.labelLayer.append(this.locator);
    }
  }

  _addLabel(text, getW, kind) {
    if (!this.labelLayer) return;
    const el = document.createElement('div');
    el.className = `marker marker-${kind}`;
    el.innerHTML = '<i></i><span class="t"></span><span class="d"></span>';
    el.querySelector('.t').textContent = text;
    this.labelLayer.append(el);
    this.labels.push({ el, getW, kind, d: el.querySelector('.d') });
  }

  /** Add trail points for frames that appeared since the last call (all of them at load). */
  syncTrail() {
    const r = this.replay, F = r.frames;
    const tmp = new THREE.Vector3();
    const last = r.n - 1;
    for (let i = this._trailAdded; i <= last; i += 1) {
      const live = r.live;
      if (!live && i % this._trailStride !== 0 && i !== last) continue;
      const v = F.vel[i];
      this.trail.addPoint(enuToW(F.pos[i], tmp), F.t[i], Math.hypot(v[0], v[1], v[2]), F.phase ? F.phase[i] : null);
    }
    this._trailAdded = r.n;
  }

  // ------------------------------------------------------------------ per-frame
  /** Horizontal unit vector the vehicle is travelling along around time t. */
  heading(t, basis) {
    const r = this.replay;
    const horiz = (a, b) => {
      const d = b.clone().sub(a);
      d.addScaledVector(basis.up, -d.dot(basis.up));
      return d;
    };
    const tau = Math.max(1.5, 0.03 * (r.tEnd - r.t0));
    const a = enuToW(r.sample(t - tau).pos), b = enuToW(r.sample(t + tau).pos);
    let d = horiz(a, b);
    if (d.length() < 0.3) d = this._globalDirection(basis);
    if (d.lengthSq() < 1e-12) d.copy(basis.north);
    return d.normalize();
  }

  _globalDirection(basis) {
    const r = this.replay;
    const a = enuToW(r.sample(r.t0).pos), b = enuToW(r.sample(r.tEnd).pos);
    const d = b.sub(a);
    d.addScaledVector(basis.up, -d.dot(basis.up));
    return d.length() > 1 ? d : basis.north.clone();
  }

  /** Pose the rocket at time t and describe it for the camera rig. */
  update(t, animTime) {
    const r = this.replay;
    const s = r.sample(t);
    if (!s) return null;
    this.sample = s;
    enuToW(s.pos, this.baseW);
    quatEnuToW(s.quat, this.quatW);
    const basis = this.frame.basis(this.baseW);
    const axis = Y_UP.clone().applyQuaternion(this.quatW);
    this.centerW.copy(this.baseW).addScaledVector(axis, this.L / 2);
    this.rocket.update(s);
    this.trail.setTime(s.t, this.baseW);
    this.env.uTime.value = animTime;

    const sites = this.padObjs.map((p) => p.W).concat(this.targetObj ? [this.targetObj.W] : []);
    this.state = {
      t: s.t,
      sample: s,
      centerW: this.centerW,
      basis,
      axis,
      L: this.L,
      alt: s.alt,
      heading: this.heading(s.t, basis),
      globalHeading: this._globalDirection(basis).normalize(),
      sites,
      startW: enuToW(r.sample(r.t0).pos),
      frame: this.frame,
      spherical: this.frame.spherical,
    };
    return this.state;
  }

  /** Place everything for rendering relative to `origin`; also updates sky/fog/lights. */
  layout(origin, cam, animTime) {
    const { camera, camW } = cam;
    const st = this.state;
    this.rocket.group.position.copy(this.baseW).sub(origin);
    this.rocket.group.quaternion.copy(this.quatW);

    if (this.terrain) this.terrain.layout(origin);
    if (this.plane) this.plane.layout(camW, origin);
    this.trail.layout(origin);

    for (const p of this.padObjs) {
      p.mesh.position.copy(p.W).addScaledVector(p.up, p.offset ?? 0.08).sub(origin);
      p.mesh.quaternion.setFromUnitVectors(Y_UP, p.up);
    }
    if (this.targetObj) {
      const T = this.targetObj;
      if (T.draped) T.group.position.copy(T.W).sub(origin); // geometry already follows the terrain in W orientation
      else {
        T.group.position.copy(T.W).addScaledVector(T.up, 0.3).sub(origin);
        T.group.quaternion.setFromUnitVectors(Y_UP, T.up);
      }
      const pulse = 0.5 + 0.5 * Math.sin(animTime * 2.4);
      T.ring.scale.setScalar(1 + 0.04 * pulse);
      T.ring.material.opacity = 0.55 + 0.4 * pulse;
      T.glow.material.opacity = 0.05 + 0.06 * pulse;
    }

    // lights
    this.sun.position.copy(this.sunDir).multiplyScalar(100);
    this.sun.target.position.set(0, 0, 0);
    this.sun.target.updateMatrixWorld();
    const camUp = this.frame.up(camW);
    this.sky.update(camW, origin, camera, camUp, this.frame.altitude(camW), camera.far);

    // engine glow on the ground when low
    const s = st.sample, th = this.rocket.plume.intensity;
    const alt = Math.max(s.alt, 0);
    const low = 1 - smoothstep(25, 320, alt);
    const flick = 0.82 + 0.18 * Math.sin(animTime * 53) * Math.sin(animTime * 17 + 1.3) + 0.08 * Math.sin(animTime * 131);
    const len = this.rocket.plume.length;
    const I = th * low * flick;
    const gp = this.baseW.clone().addScaledVector(st.basis.up, -alt + clamp(0.4 + 0.2 * alt, 0.4, 5));
    this.env.uLightPos.value.copy(gp).sub(origin);
    this.env.uLightI.value = I * 0.85;
    this.env.uLightR.value = 1.5 + len * 1.4 + 0.1 * alt;
    this.plumeLight.position.copy(this.env.uLightPos.value);
    this.plumeLight.intensity = I * 4 * (len + 0.6) * (len + 0.6);
  }

  /** Project markers into the overlay (call after layout). */
  updateLabels(cam, origin, width, height, pixelsPerMetre, extent = this.L) {
    const { camera } = cam;
    const v = new THREE.Vector3();
    const place = (el, W, visible = true) => {
      v.copy(W).sub(origin).project(camera);
      const ok = visible && v.z < 1 && v.z > -1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      el.style.display = ok ? '' : 'none';
      if (ok) el.style.transform = `translate(${((v.x + 1) / 2) * width}px, ${((1 - v.y) / 2) * height}px)`;
      return ok;
    };
    const rc = this.centerW;
    for (const l of this.labels) {
      const W = l.getW();
      if (place(l.el, W) && l.kind === 'target') {
        const dist = W.distanceTo(rc);
        l.d.textContent = dist >= 1000 ? ` ${(dist / 1000).toFixed(1)} km` : ` ${dist.toFixed(0)} m`;
      }
    }
    if (this.locator) {
      const px = extent * pixelsPerMetre;
      const show = place(this.locator, rc) && px < 26;
      this.locator.style.display = show ? '' : 'none';
    }
  }

  dispose() {
    this.rocket.dispose();
    this.trail.dispose();
    this.terrain?.dispose();
    this.plane?.dispose();
    for (const p of this.padObjs) { p.mesh.geometry.dispose(); p.mesh.material.map?.dispose(); p.mesh.material.dispose(); }
    if (this.labelLayer) this.labelLayer.replaceChildren();
    this.scene.traverse((o) => { if (o.geometry && o.isMesh) o.geometry.dispose(); });
  }
}
