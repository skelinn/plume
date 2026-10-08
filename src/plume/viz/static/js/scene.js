// One 3-D world per replay: atmosphere/sky, ground or terrain, pads, the vehicle, exhaust, dust,
// lights (sun with soft shadows that follow the vehicle, image-based sky light, engine glow),
// the trajectory trail and the engineering overlay.
//
// Per frame:  world.update(t)            -> samples the replay, poses the rocket, reports
//                                           the camera context (centre, local basis, heading ...)
//             rig.update(ctx)            -> picks the floating origin + camera
//             world.layout(origin, cam)  -> places every object at (W - origin) in float32

import * as THREE from 'three';
import { Frame, enuToW, quatEnuToW, Y_UP } from './coords.js';
import { Atmosphere, SUN_E } from './atmosphere.js';
import { Terrain, GroundPlane, fetchTerrain, makeGroundUniforms } from './terrain.js';
import { Rocket } from './rocket.js';
import { Dust, SmokeTrail } from './plume.js';
import { padMesh, disposePad } from './pads.js';
import { launchRail } from './models/hobby.js';
import { Trail } from './trail.js';
import { EngOverlay } from './engineering.js';
import { clamp, smoothstep } from './util.js';

const MAX_TRAIL_POINTS = 14000;
const SUN_EL = 36, SUN_AZ = 215; // degrees: elevation, azimuth from east toward north

export class World {
  /**
   * @param replay Replay
   * @param opts {labelLayer, capture, renderer, quality, engRoot}
   */
  constructor(replay, { labelLayer = null, capture = false, renderer, quality = 'high', engRoot = null } = {}) {
    this.replay = replay;
    this.capture = capture;
    this.labelLayer = labelLayer;
    this.quality = quality;
    const meta = replay.meta, sc = meta.scene;
    this.frame = new Frame(sc);
    this.scene = new THREE.Scene();
    this.vehicle = meta.vehicle;
    this.L = meta.vehicle.length || 5;
    this.D = meta.vehicle.diameter || 0.5;

    // fixed sun, defined in the launch-site tangent frame
    const b0 = this.frame.basis(new THREE.Vector3(0, 0, 0));
    const el = (SUN_EL * Math.PI) / 180, az = (SUN_AZ * Math.PI) / 180;
    this.sunDir = new THREE.Vector3()
      .addScaledVector(b0.up, Math.sin(el))
      .addScaledVector(b0.east, Math.cos(el) * Math.cos(az))
      .addScaledVector(b0.north, Math.cos(el) * Math.sin(az))
      .normalize();

    this.atm = new Atmosphere(renderer, this.frame, this.sunDir, { quality });
    this.atm.onEnv = (tex) => { this.scene.environment = tex; };
    this.scene.add(...this.atm.objects());
    this.scene.environmentIntensity = 1.3;
    // The sky LUT is single-scattering only, which under-lights shadows; a diffuse-only hemisphere term
    // stands in for the missing multiple scattering (sky above, sunlit ground below) without making
    // metal reflections brighter.  Faded towards space in layout().
    this.skyFill = new THREE.HemisphereLight(0x8fb0ff, 0x6b5f50, 0.9);
    this.scene.add(this.skyFill);

    this.sun = new THREE.DirectionalLight(0xffffff, SUN_E);
    this.sun.castShadow = true;
    const sm = quality === 'low' ? 1024 : 2048;
    this.sun.shadow.mapSize.set(sm, sm);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.sun.shadow.radius = 3;
    this.scene.add(this.sun, this.sun.target);
    this.plumeLight = new THREE.PointLight(0xff9a50, 0, 0, 2);
    this.scene.add(this.plumeLight);

    this.groundU = makeGroundUniforms();
    this.rocket = new Rocket(meta.vehicle, replay, this.atm, { quality });
    this.scene.add(this.rocket.group);
    if (this.rocket.chute) this.scene.add(this.rocket.chute.group);
    this.dust = new Dust({ count: quality === 'low' ? 32 : 64 });
    this.scene.add(this.dust.mesh);
    this.smoke = this.rocket.hobby ? new SmokeTrail(this._smokePoints(), Math.max(this.D * 2.5, 0.12)) : null;
    if (this.smoke) this.scene.add(this.smoke.mesh);

    this.trail = new Trail({ maxSpeed: Math.max(replay.maxSpeed(), 1) });
    this.scene.add(this.trail.group);

    // multi-vehicle replays: one rocket (+ trail) per vehicle; the camera, HUD and engineering
    // overlay follow the focused one (setFocus)
    const v0 = replay.vehicles?.[0] || { id: 'vehicle', name: this.vehicle.name, role: 'vehicle', persist: true };
    this.fleet = [{ ...v0, replay, rocket: this.rocket, trail: this.trail, _trailAdded: 0, _trailStride: Math.max(1, Math.ceil(replay.n / MAX_TRAIL_POINTS)), visible: true }];
    for (const v of (replay.vehicles || []).slice(1)) {
      const rk = new Rocket(v.replay.meta.vehicle, v.replay, this.atm, { quality });
      this.scene.add(rk.group);
      const fairing = /fairing/.test(v.role);
      const tr = fairing ? null : new Trail({ maxSpeed: Math.max(v.replay.maxSpeed(), 1) });
      if (tr) this.scene.add(tr.group);
      this.fleet.push({
        ...v, rocket: rk, trail: tr, visible: false, _trailAdded: 0,
        _trailStride: Math.max(1, Math.ceil(v.replay.n / MAX_TRAIL_POINTS)),
        baseW: new THREE.Vector3(), quatW: new THREE.Quaternion(), centerW: new THREE.Vector3(),
      });
    }
    this.focus = this.fleet[0];
    this.vrep = replay; // the focused vehicle's replay

    this.terrain = null;
    this.plane = null;
    this.padObjs = [];
    this.labels = [];
    this.locator = null;
    this.baseW = new THREE.Vector3();
    this.quatW = new THREE.Quaternion();
    this.centerW = new THREE.Vector3();
    this.state = null;
    this.warnings = [];
    this._buildGround();
    this._buildPads();
    this.syncTrail();
    this.eng = new EngOverlay(this, engRoot);
    this._tmp = { v: new THREE.Vector3(), q: new THREE.Quaternion(), c: new THREE.Color() };
  }

  // ------------------------------------------------------------------ construction
  _buildGround() {
    const g = this.replay.meta.scene.ground || { type: 'plane' };
    if (g.type === 'terrain' && g.terrain_id) {
      this.terrain = new Terrain(this.frame, this.atm, this.groundU);
      this.scene.add(this.terrain.group);
    } else if (g.type !== 'none') {
      this.plane = new GroundPlane(this.atm, this.groundU);
      this.scene.add(this.plane.mesh);
    } else {
      this.plane = new GroundPlane(this.atm, this.groundU); // "none": still give the eye a horizon
      this.scene.add(this.plane.mesh);
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
      this._seatPads();
    } catch (e) {
      this.warnings.push(String(e));
      this.scene.remove(this.terrain.group);
      this.terrain = null;
      this.plane = new GroundPlane(this.atm, this.groundU);
      this.scene.add(this.plane.mesh);
    }
  }

  _buildPads() {
    const sc = this.replay.meta.scene;
    const pads = (sc.pads || []).map((p) => ({ ...p }));
    const tg = sc.target && sc.target.pos ? sc.target : null;
    this.targetW = tg ? enuToW(tg.pos) : null;
    this.targetR = tg ? tg.radius || 5 : 0;
    // a target with no pad under it gets a landing-zone slab (it is where the vehicle lands)
    if (tg && !pads.some((p) => enuToW(p.pos).distanceTo(this.targetW) < Math.max(p.radius, tg.radius || 0) * 1.5)) {
      const R = clamp((tg.radius || 10) * 0.4, 6, 20);
      pads.push({ name: 'LZ', pos: tg.pos, radius: R, synthetic: true });
    }
    const hobby = this.rocket.hobby;
    for (const pad of pads) {
      const W = enuToW(pad.pos);
      const isRail = hobby && pad.radius < 4;
      let mesh;
      if (isRail) {
        mesh = launchRail(Math.max(1.25 * this.L, 1.2), this.D / 2 + 0.016, this.rocket.airframe.railPhi);
        this.atmPatchGroup(mesh);
        mesh.traverse((o) => { if (o.isMesh) o.renderOrder = 5; });
      } else mesh = padMesh(pad, this.atm);
      this.scene.add(mesh);
      const obj = { mesh, W, up: this.frame.up(W), pad, rail: isRail, offset: isRail ? 0 : -0.15 + 0.08 };
      if (isRail) {
        // the rail carries the rocket: put it at the rocket's launch position and attitude
        // (measured flights are noisy: average the pre-liftoff samples)
        const r = this.replay, F = r.frames;
        const tl = (r.events.find((e) => e.type === 'ignition')?.t ?? r.t0) + 0.05;
        const acc = [0, 0, 0];
        let nAcc = 0;
        for (let i = 0; i < r.n && (F.t[i] <= tl || nAcc === 0); i++) { acc[0] += F.pos[i][0]; acc[1] += F.pos[i][1]; acc[2] += F.pos[i][2]; nAcc++; }
        obj.W = enuToW(acc.map((x) => x / nAcc));
        obj.q = quatEnuToW(r.sample(r.t0).quat);
      }
      this.padObjs.push(obj);
      if (!pad.synthetic && (!this.targetW || this.targetW.distanceTo(W) > Math.max(pad.radius, this.targetR) * 1.5)) this._addLabel(pad.name || 'PAD', () => W, 'pad');
    }
    if (this.targetW) this._addLabel('TARGET', () => this.targetW, 'target');
    for (const f of this.fleet.slice(1)) {
      if (!f.trail) continue; // jettisoned hardware gets no label
      this._addLabel((f.name || f.id).toUpperCase(), () => f.centerW, 'vehicle', () => f.visible && f !== this.focus);
    }
    if (this.fleet.length > 1) {
      const f0 = this.fleet[0];
      this._addLabel((f0.name || 'vehicle').toUpperCase(), () => f0.centerW || this.centerW, 'vehicle', () => this.focus !== f0);
    }
    this._padUniforms();
    if (this.labelLayer) {
      this.locator = document.createElement('div');
      this.locator.className = 'locator';
      this.locator.innerHTML = '<i></i><span></span>';
      this.locator.querySelector('span').textContent = this.vehicle.name || 'vehicle';
      this.labelLayer.append(this.locator);
    }
  }

  /** Smoke puffs every ~0.6 m of path flown while the (solid) motor burns. */
  _smokePoints() {
    const r = this.replay, F = r.frames, out = [];
    if (!r.n || !F.throttle) return out;
    let last = null;
    for (let i = 0; i < r.n && out.length < 900; i++) {
      if (!(F.throttle[i] > 0.05)) continue;
      const W = enuToW(F.pos[i]);
      if (last && W.distanceTo(last) < Math.max(0.6, this.D * 8)) continue;
      out.push({ W, t: F.t[i] });
      last = W;
    }
    return out;
  }

  atmPatchGroup(g) {
    g.traverse((o) => { if (o.isMesh && o.material?.isMeshStandardMaterial && !o.material.userData.atm) { this.atm.patch(o.material); o.material.userData.atm = true; } });
  }

  _padUniforms() {
    const U = this.groundU;
    let n = 0;
    for (const p of this.padObjs) {
      if (n >= U.uPadP.value.length) break;
      const m = this.frame.toMap(p.W);
      U.uPadP.value[n++].set(m.u, m.v, p.rail ? -0.6 : p.pad.radius);   // < 0: apron only (no slab)
    }
    U.uPadN.value = n;
  }

  /** Seat pads on the highest terrain point of their footprint (the slab grows a skirt below). */
  _seatPads() {
    const F = this.frame, T = this.terrain;
    for (const p of this.padObjs) {
      const R = p.rail ? 1 : p.pad.radius;
      const m = F.toMap(p.W);
      let lo = Infinity, hi = -Infinity;
      const take = (u, v) => { const h = T.heightAt(u, v); if (h != null) { lo = Math.min(lo, h); hi = Math.max(hi, h); } };
      take(m.u, m.v);
      for (const f of [0.5, 1]) for (let k = 0; k < 12; k++) take(m.u + f * R * Math.cos((k * Math.PI) / 6), m.v + f * R * Math.sin((k * Math.PI) / 6));
      if (hi === -Infinity) continue;
      const stood = this._standHeight(p.W, R);
      if (stood !== null) hi = Math.max(lo, Math.min(hi, stood)), m.h = stood;
      // seat on the terrain; a slab never stands on a skirt taller than a few metres (if the replay's
      // pad position and the terrain disagree by more, trust the terrain)
      const top = m.h - hi > 4 ? hi : Math.max(m.h, hi), bottom = Math.min(top, lo);
      F.surface(m.u, m.v, top, p.W);
      p.up = F.up(p.W);
      if (!p.rail) {
        const hgt = top - bottom + 0.4;
        p.mesh.geometry.dispose();
        p.mesh.geometry = new THREE.CylinderGeometry(R, R * 1.01, hgt, 96, 1, false);
        p.offset = -hgt / 2 + 0.08;
      }
    }
    this._padUniforms();
  }

  /**
   * Map height of the ground the vehicle actually stood on near W (median over frames with the
   * footpads within 0.3 m of the ground), or null.  Pads are seated there so the legs meet the slab.
   */
  _standHeight(W, R) {
    const r = this.replay, F = r.frames, Fr = this.frame;
    if (!F.alt) return null;
    const legH = this.vehicle.legs?.count > 0 ? this.vehicle.legs.height || 0 : 0;
    const m0 = Fr.toMap(W), hs = [];
    const b = new THREE.Vector3(), q = new THREE.Quaternion(), ax = new THREE.Vector3();
    const step = Math.max(1, Math.floor(r.n / 4000));
    for (let i = 0; i < r.n; i += step) {
      if (!(F.alt[i] < 0.3)) continue;
      enuToW(F.pos[i], b);
      const m = Fr.toMap(b);
      if (Math.hypot(m.u - m0.u, m.v - m0.v) > R) continue;
      quatEnuToW(F.quat[i], q);
      ax.set(0, 1, 0).applyQuaternion(q);
      const up = Fr.up(b);
      hs.push(m.h - Math.max(F.alt[i], 0) - legH * Math.max(ax.dot(up), 0));
    }
    if (!hs.length) return null;
    hs.sort((x, y) => x - y);
    return hs[Math.floor(hs.length / 2)];
  }

  _addLabel(text, getW, kind, vis = null) {
    if (!this.labelLayer) return;
    const el = document.createElement('div');
    el.className = `marker marker-${kind}`;
    el.innerHTML = '<i></i><span class="t"></span><span class="d"></span>';
    el.querySelector('.t').textContent = text;
    this.labelLayer.append(el);
    this.labels.push({ el, getW, kind, vis, d: el.querySelector('.d') });
  }

  /** Add trail points for frames that appeared since the last call (all of them at load). */
  syncTrail() {
    for (const f of this.fleet) if (f.trail) this._syncTrail(f);
  }

  _syncTrail(f) {
    const r = f.replay, F = r.frames;
    const tmp = new THREE.Vector3();
    const last = r.n - 1;
    // measured flights carry sensor noise: draw their path through a ~1 s moving average
    const k = r.source === 'real' && r.n > 2 ? Math.max(1, Math.round(0.5 / Math.max((r.tEnd - r.t0) / (r.n - 1), 1e-3))) : 0;
    const p = [0, 0, 0];
    for (let i = f._trailAdded; i <= last; i += 1) {
      if (!r.live && i % f._trailStride !== 0 && i !== last) continue;
      const v = F.vel[i];
      let pos = F.pos[i];
      if (k) {
        const a = Math.max(0, i - k), b = Math.min(last, i + k);
        p[0] = p[1] = p[2] = 0;
        for (let j = a; j <= b; j++) { p[0] += F.pos[j][0]; p[1] += F.pos[j][1]; p[2] += F.pos[j][2]; }
        const nn = b - a + 1;
        pos = [p[0] / nn, p[1] / nn, i === 0 ? F.pos[i][2] : p[2] / nn];
      }
      f.trail.addPoint(enuToW(pos, tmp), F.t[i], Math.hypot(v[0], v[1], v[2]), F.phase ? F.phase[i] : null);
    }
    f._trailAdded = r.n;
  }

  /** All trails (one per vehicle that has one). */
  get trails() { return this.fleet.map((f) => f.trail).filter(Boolean); }

  setTrailMode(mode) { for (const tr of this.trails) tr.setColorBy(mode); }

  /** Follow another vehicle of a multi-vehicle replay (id from replay.vehicles). */
  setFocus(id) {
    const f = this.fleet.find((x) => x.id === id) || this.fleet[0];
    if (f === this.focus) return f;
    this.focus = f;
    this.vrep = f.replay;
    this.rocket = f.rocket;
    this.trail = f.trail || this.fleet[0].trail;
    this.vehicle = f.replay.meta.vehicle;
    this.L = this.vehicle.length || 5;
    this.D = this.vehicle.diameter || 0.5;
    f.rocket.group.visible = true;
    if (this.locator) this.locator.querySelector('span').textContent = f.name || this.vehicle.name || 'vehicle';
    return f;
  }

  setQuality(q) {
    this.quality = q;
    for (const f of this.fleet) f.rocket.setQuality(q);
    const sm = q === 'low' ? 1024 : 2048;
    if (this.sun.shadow.mapSize.x !== sm) {
      this.sun.shadow.mapSize.set(sm, sm);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
    this.atm.skyU.uSteps.value = q === 'low' ? 14 : 28;
    this.atm._key = null;
  }

  // ------------------------------------------------------------------ per-frame
  heading(t, basis) {
    const r = this.vrep;
    const horiz = (a, b) => {
      const d = b.clone().sub(a);
      d.addScaledVector(basis.up, -d.dot(basis.up));
      return d;
    };
    const tau = Math.max(1.5, 0.03 * (r.ownTEnd - r.ownT0));
    const a = enuToW(r.sample(t - tau).pos), b = enuToW(r.sample(t + tau).pos);
    let d = horiz(a, b);
    if (d.length() < 0.3) d = this._globalDirection(basis);
    if (d.lengthSq() < 1e-12) d.copy(basis.north);
    return d.normalize();
  }

  _globalDirection(basis) {
    const r = this.vrep;
    const a = enuToW(r.sample(r.ownT0).pos), b = enuToW(r.sample(r.ownTEnd).pos);
    const d = b.sub(a);
    d.addScaledVector(basis.up, -d.dot(basis.up));
    return d.length() > 1 ? d : basis.north.clone();
  }

  /** Pose the rocket at time t and describe it for the camera rig. */
  update(t, animTime) {
    for (const f of this.fleet) if (f !== this.focus) this._poseOther(f, t, animTime);
    const r = this.vrep;
    const s = r.sample(t);
    if (!s) return null;
    this.sample = s;
    enuToW(s.pos, this.baseW);
    quatEnuToW(s.quat, this.quatW);
    const basis = this.frame.basis(this.baseW);
    const axis = Y_UP.clone().applyQuaternion(this.quatW);
    this.axis = axis;
    this.centerW.copy(this.baseW).addScaledVector(axis, this.L / 2);
    const upMesh = basis.up.clone().applyQuaternion(this.quatW.clone().invert());
    const legH = this.vehicle.legs?.count > 0 ? this.vehicle.legs.height || 0 : 0;
    this.aglBase = Math.max(s.alt, 0) + legH * Math.max(axis.dot(basis.up), 0);
    const sunCol = this.atm.sunAt(Math.max(s.alt, 0), basis.up);
    this._tmp.c.copy(sunCol).multiplyScalar(1.0);
    this.throttle = this.rocket.update(s, { upMesh, agl: this.aglBase, time: animTime, sunCol: this._tmp.c });
    this.trail.setTime(s.t, this.baseW);

    const sites = this.padObjs.map((p) => p.W).concat(this.targetW ? [this.targetW] : []);
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
      startW: enuToW(r.sample(r.ownT0).pos),
      frame: this.frame,
      spherical: this.frame.spherical,
      span: Math.max(this.D, 2 * (this.vehicle.legs?.span || 0)),
    };
    // under a parachute, frame rocket + canopy together
    const fc = this.rocket.chuteAt(s.t);
    if (fc > 0) {
      const ext = 1.6 + this.rocket.chute.lineLen + this.rocket.chute.Rc;
      this.state.centerW = this.centerW.clone().addScaledVector(basis.up, 0.45 * ext * fc);
      this.state.L = this.L + ext * fc;
    }
    return this.state;
  }

  /** Pose a vehicle that is not followed: visible inside its track (and after it, if it persists). */
  _poseOther(f, t, animTime) {
    const r = f.replay;
    const vis = r.n > 0 && t >= r.ownT0 - 1e-6 && (t <= r.ownTEnd + 1e-6 || f.persist);
    f.visible = vis;
    f.rocket.group.visible = vis;
    if (f.trail) f.trail.group.visible = t >= r.ownT0;
    if (!vis) return;
    const s = r.sample(t);
    f.baseW = f.baseW || new THREE.Vector3();
    f.quatW = f.quatW || new THREE.Quaternion();
    f.centerW = f.centerW || new THREE.Vector3();
    enuToW(s.pos, f.baseW);
    quatEnuToW(s.quat, f.quatW);
    const basis = this.frame.basis(f.baseW);
    const axis = Y_UP.clone().applyQuaternion(f.quatW);
    const V = r.meta.vehicle;
    f.centerW.copy(f.baseW).addScaledVector(axis, (V.length || 5) / 2);
    const upMesh = basis.up.clone().applyQuaternion(f.quatW.clone().invert());
    const legH = V.legs?.count > 0 ? V.legs.height || 0 : 0;
    const alt = Math.max(s.alt ?? 1e3, 0);
    const agl = alt + legH * Math.max(axis.dot(basis.up), 0);
    const sunCol = this.atm.sunAt(alt, basis.up);
    f.rocket.update(s, { upMesh, agl, time: animTime, sunCol });
    f.trail?.setTime(s.t, f.baseW);
  }

  /** Place everything for rendering relative to `origin`; also updates sky, lights and shadows. */
  layout(origin, cam, animTime, { now = 0 } = {}) {
    const { camera, camW } = cam;
    const st = this.state;
    const R = this.rocket;
    R.group.position.copy(this.baseW).sub(origin);
    R.group.quaternion.copy(this.quatW);
    R.afterLayout();
    for (const f of this.fleet) {
      if (f === this.focus) continue;
      if (f.visible) {
        f.rocket.group.position.copy(f.baseW).sub(origin);
        f.rocket.group.quaternion.copy(f.quatW);
        f.rocket.afterLayout();
      }
      if (f.trail && f.trail !== this.trail) f.trail.layout(origin);
    }

    if (this.terrain) this.terrain.layout(origin);
    if (this.plane) this.plane.layout(camW, origin);
    this.trail.layout(origin);
    for (const p of this.padObjs) {
      p.mesh.position.copy(p.W).addScaledVector(p.up, p.offset ?? 0).sub(origin);
      if (p.q) p.mesh.quaternion.copy(p.q);
      else p.mesh.quaternion.setFromUnitVectors(Y_UP, p.up);
    }
    this.atm.update(camW, origin, camera, { capture: this.capture, now });
    const cm = this.frame.toMap(camW);
    this.groundU.uMapRef.value.set(Math.round(cm.u / 1024) * 1024, Math.round(cm.v / 1024) * 1024);

    // sun + shadow box following the vehicle (and reaching the ground below it)
    const up = st.basis.up;
    const centerR = this.centerW.clone().sub(origin);
    const ext = Math.max(this.L * 0.62, st.span * 0.8, 2.5);
    const sinEl = Math.max(this.sunDir.dot(up), 0.1);
    const reach = Math.min(Math.max(st.alt, 0) / sinEl + this.L * 2, 3000);
    this.sun.position.copy(centerR).addScaledVector(this.sunDir, ext * 2);
    this.sun.target.position.copy(centerR);
    this.sun.target.updateMatrixWorld();
    const sc = this.sun.shadow.camera;
    sc.left = -ext; sc.right = ext; sc.top = ext; sc.bottom = -ext;
    sc.near = 0.05; sc.far = ext * 4 + reach;
    sc.updateProjectionMatrix();
    const sunCol = this.atm.sunAt(Math.max(this.frame.altitude(this.centerW), 0), up);
    this.sun.color.copy(sunCol);
    this.sun.intensity = SUN_E;
    const space = smoothstep(15000, 80000, this.frame.altitude(camW));
    this.skyFill.intensity = 0.95 * (1 - space) + 0.35 * space;
    this.skyFill.color.setRGB(0.56 * (1 - space), 0.69 * (1 - space), 1.0 * (1 - space));
    this.skyFill.groundColor.setRGB(0.42, 0.38, 0.32).lerp(new THREE.Color(0.3, 0.36, 0.45), space);
    this.skyFill.position.copy(up);   // hemisphere "up" = local vertical

    // parachute (hobby): canopy upwind of the descending rocket on its shock cord
    if (R.chute) {
      const f = R.chuteAt(st.t);
      const s = st.sample;
      const v = new THREE.Vector3(...enuToW(s.vel || [0, 0, 0], new THREE.Vector3()).toArray());
      const axisC = up.clone();
      if (v.length() > 1) axisC.addScaledVector(v.normalize(), -0.6).normalize();
      const attach = this.baseW.clone().addScaledVector(st.axis, R.airframe.attachY).sub(origin);
      const anchor = attach.clone().addScaledVector(axisC, 1.6 + R.chute.lineLen);
      R.chute.update(anchor, axisC, attach, f);
    }

    // engine light + ground dust
    const th = R.plume.intensity, len = R.plume.length;
    const exitW = new THREE.Vector3();
    R.plume.group.getWorldPosition(exitW);
    const flick = 0.85 + 0.15 * Math.sin(animTime * 53) * Math.sin(animTime * 17 + 1.3);
    const re = R.re;
    this.plumeLight.position.copy(exitW).addScaledVector(st.axis, -Math.max(Math.min(len * 0.3, 6 * re + 1.5), 4 * re));
    this.plumeLight.intensity = th > 0.01 ? 900 * th * flick * (re / 0.4) ** 2 * (1 - 0.6 * R.plume.vac) * (R.hobby ? 0.4 : 1) : 0;
    const hExit = R.exitHeight ?? 1e9;
    const reachG = len * 1.25 + 4 * re;
    const dustI = th > 0.02 && hExit < reachG ? th * (1 - smoothstep(0.25 * reachG, reachG, hExit)) : 0;
    if (dustI > 0.01 && !R.hobby) {
      const gp = exitW.clone().addScaledVector(up, -hExit);
      const lit = this._tmp.c.copy(sunCol).multiplyScalar(0.75 * Math.max(this.sunDir.dot(up), 0.2) + 0.2);
      const glow = new THREE.Color(1.0, 0.45, 0.15).multiplyScalar(1.2 * th);
      this.dust.update(gp, up, dustI, Math.max(6 * re, 0.6 * len) + 2, animTime, lit, glow);
    } else this.dust.update(new THREE.Vector3(), up, 0, 1, animTime, sunCol, sunCol);

    if (this.smoke) this.smoke.update(st.t, origin, this._tmp.c.copy(sunCol).multiplyScalar(0.9 * Math.max(this.sunDir.dot(up), 0.25) + 0.35));
    this.eng.layout(origin, cam, animTime);
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
      if (place(l.el, W, l.vis ? l.vis() : true) && l.kind === 'target') {
        const dist = W.distanceTo(rc);
        l.d.textContent = dist >= 1000 ? `${(dist / 1000).toFixed(1)} km` : `${dist.toFixed(0)} m`;
      }
    }
    if (this.locator) {
      const px = extent * pixelsPerMetre;
      const show = place(this.locator, rc) && px < 26 && !this.eng.enabled;
      this.locator.style.display = show ? '' : 'none';
    }
    this.eng.updateLabels(cam, origin, width, height);
  }

  dispose() {
    for (const f of this.fleet) { f.rocket.dispose(); f.trail?.dispose(); }
    this.terrain?.dispose();
    this.plane?.dispose();
    this.dust.dispose();
    this.smoke?.dispose();
    this.eng.dispose();
    this.atm.dispose();
    for (const p of this.padObjs) {
      if (p.rail) p.mesh.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
      else disposePad(p.mesh);
    }
    this.sun.shadow.map?.dispose();
    if (this.labelLayer) this.labelLayer.replaceChildren();
  }
}
