// Procedural vehicle assembled from meta.vehicle, so the model always matches the simulated
// geometry: hull, grid fins, landing legs, engine + exhaust and RCS for liquid vehicles; airframe,
// fins, solid-motor flame and parachute for hobby rockets.
//
// Mesh space = body frame after the ENU->three remap (vehicle axis = +y, body y = -z); the hull
// base centre is the group origin, matching the replay's `pos`.  The group is oriented with the
// converted replay quaternion (coords.quatEnuToW).

import * as THREE from 'three';
import * as M from './materials.js';
import { Plume } from './plume.js';
import { Hull } from './models/hull.js';
import { GridFins } from './models/gridfin.js';
import { Legs } from './models/legs.js';
import { Engine } from './models/engine.js';
import { RcsPuffs } from './models/rcs.js';
import { HobbyAirframe, Parachute, isHobby } from './models/hobby.js';
import { FairingHalf, isFairing } from './models/fairing.js';

export class Rocket {
  /**
   * @param vehicle meta.vehicle, @param replay Replay (for derived timing: chute deployment)
   * @param atmosphere Atmosphere (lit materials are patched for aerial perspective)
   */
  constructor(vehicle, replay, atmosphere, { quality = 'high' } = {}) {
    const V = vehicle;
    this.V = V;
    this.L = V.length; this.D = V.diameter;
    const r = V.diameter / 2;
    this.group = new THREE.Group();
    this.group.name = 'rocket';
    this.thrustMax = V.engine?.thrust_max || 0;
    this.hobby = isHobby(V);
    this.soot = M.makeSootUniforms();
    this.soot.uSootTop.value = 0.3 * V.length;
    this.parts = [];
    this.cgZ = this.hobby ? 0.55 * V.length : 0.38 * V.length;

    const mats = {
      carbon: M.patchSoot(M.carbon({ repeat: 30 }), this.soot, { strength: 0.5 }),
      metal: M.aluminium(),
      chrome: M.chrome(),
      dark: M.patchSoot(M.blackPaint(), this.soot, { strength: 0.4 }),
      pad: M.patchSoot(M.aluminium(), this.soot),
      shield: M.heatShield(),
      blanket: M.blanket(),
      ti: M.titanium(),
      clevis: M.darkMetal(),
    };
    this.mats = Object.values(mats);

    if (isFairing(V)) {
      // jettisoned hardware (multi-vehicle replays): shell only, no engine or exhaust
      this.fairing = new FairingHalf(V);
      this.group.add(this.fairing.group);
      this.parts.push(this.fairing);
      this.plume = null;
      this.re = 0.1;
    } else if (this.hobby) {
      this.airframe = new HobbyAirframe(V);
      this.group.add(this.airframe.group);
      this.parts.push(this.airframe);
      this.exitY = this.airframe.exitY;
      this.re = this.airframe.nozzleR;
      this.plume = new Plume(this.re, { solid: true, quality });
      this.plume.group.position.set(0, this.exitY, 0);
      this.group.add(this.plume.group);
      this.chute = new Parachute(0.92);
      this.tDeploy = replay.firstTime('phase', 'descent');
    } else {
      this.hull = new Hull(V, this.soot);
      this.group.add(this.hull.group);
      this.parts.push(this.hull);
      if (V.grid_fins && V.grid_fins.count > 0) {
        this.gridFins = new GridFins(V.grid_fins, r, { ti: mats.ti, dark: mats.dark, metal: mats.metal, clevis: mats.clevis });
        this.group.add(this.gridFins.group);
        this.parts.push(this.gridFins);
      }
      if ((V.legs?.count | 0) > 0) {
        this.legs = new Legs(V.legs, r, V.length, mats);
        this.group.add(this.legs.group);
        this.parts.push(this.legs);
      }
      this.engine = new Engine(V.engine, r, mats);
      this.group.add(this.engine.group);
      this.parts.push(this.engine);
      this.re = this.engine.re;
      this.plume = new Plume(this.re, { solid: false, quality });
      this.plume.group.position.set(0, this.engine.exitY, 0);
      this.engine.gimbal.add(this.plume.group);
      if (this.hull.rcsPods.length) {
        this.rcs = new RcsPuffs(this.hull.rcsPods, V, this.cgZ);
        this.group.add(this.rcs.group);
        this.parts.push(this.rcs);
      }
    }
    // aerial perspective on every lit material
    this.group.traverse((o) => {
      if (!o.isMesh) return;
      const m = o.material;
      if (m && (m.isMeshStandardMaterial) && !m.userData.atm) { atmosphere.patch(m); m.userData.atm = true; }
    });
    if (this.chute) atmosphere.patch(this.chute.canopyMat);
    // draw after the terrain (renderOrder 1-2): detail tiles use an ALWAYS depth test, see terrain.js
    const late = (o) => { if ((o.isMesh || o.isLine) && !o.renderOrder) o.renderOrder = 5; };
    this.group.traverse(late);
    this.chute?.group.traverse(late);
    this.groundL = { n: new THREE.Vector3(), d: 1e9 };
    this._m = new THREE.Matrix4();
  }

  setQuality(q) { this.plume?.setQuality(q); }

  /**
   * Pose moving parts from a replay sample.
   * @param s sample, @param ctx {upMesh: local up in mesh space, agl: height of the hull base above
   *        the ground along up, time: animation time, sunCol: THREE.Color}
   * @returns throttle used (0..1)
   */
  update(s, ctx) {
    let th = s.throttle;
    if (th === undefined) th = this.thrustMax > 0 && s.thrust !== undefined ? s.thrust / this.thrustMax : 0;
    th = Math.max(0, th || 0);
    if (!this.plume) return 0;
    if (this.engine) {
      const [a, b] = s.gimbal || [0, 0];
      this.engine.setGimbal(a, b);
      this.engine.setGlow(Math.min(s._glow ?? th, 1));
    }
    // ground plane in plume-local coordinates (for the splash / clipping)
    const pg = this.plume.group;
    pg.updateMatrix();
    this._m.copy(pg.matrix);
    if (this.engine) this._m.premultiply(this.engine.gimbal.matrix);
    const origin = new THREE.Vector3().setFromMatrixPosition(this._m);
    const inv = this._m.clone().invert();
    const nL = ctx.upMesh.clone().transformDirection(inv);
    // height of the plume origin above ground = agl(base) + origin . up
    const hO = ctx.agl + origin.dot(ctx.upMesh);
    this.groundL.n.copy(nL);
    this.groundL.d = hO;
    this.plume.update(th, s.alt, ctx.time, this.groundL, null, ctx.sunCol);
    this.exitHeight = hO;

    if (this.gridFins) this.gridFins.update(s._fins ?? (s.fins_out === undefined ? 1 : s.fins_out), s.fins);
    if (this.legs) this.legs.update(s.legs_out === undefined ? 1 : s._legs);
    if (this.rcs) this.rcs.update(s.rcs, ctx.sunCol);
    // soot: coverage grows with accumulated dirty burn time
    const burn = s._soot || 0;
    this.soot.uSoot.value = 1 - Math.exp(-burn / 12);
    this.soot.uSootTop.value = this.L * (0.15 + 0.55 * (1 - Math.exp(-burn / 30)));
    return th;
  }

  /** after the group is placed: soot needs the group's inverse world matrix */
  afterLayout() {
    this.group.updateMatrixWorld(true);
    this.soot.uRocketInv.value.copy(this.group.matrixWorld).invert();
  }

  /** Parachute inflation 0..1 at time t (hobby rockets, after the phase switches to descent). */
  chuteAt(t) {
    if (!this.chute || this.tDeploy == null || t < this.tDeploy) return 0;
    return Math.min((t - this.tDeploy) / 0.9, 1);
  }

  dispose() {
    for (const p of this.parts) p.dispose?.();
    this.plume?.dispose();
    this.chute?.dispose();
    for (const m of this.mats) m.dispose();
  }
}
