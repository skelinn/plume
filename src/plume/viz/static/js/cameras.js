// Camera rig: chase / ground / top / free, all expressed around a floating origin.
//
// Each frame the rig picks a float64 `origin` (W-space) close to the camera and positions the
// three.js camera relative to it, so GPU coordinates stay small even 800 km from the launch
// site:
//   chase/top/ground : origin = camera position (camera sits at (0,0,0) in render space)
//   free (orbit)     : origin = rocket centre; OrbitControls orbits/pans around it in render space

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { clamp } from './util.js';

export const MODES = ['chase', 'ground', 'top', 'free'];
const Y = new THREE.Vector3(0, 1, 0);

export class CameraRig {
  constructor(canvas) {
    this.canvas = canvas;
    this.persp = new THREE.PerspectiveCamera(50, 1, 0.1, 1e7);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1e7);
    this.camera = this.persp;
    this.mode = 'chase';
    this.user = { yaw: 0, pitch: 0, zoom: 1 };
    this.onUserChange = null;
    this.smooth = true;
    this._lastOffset = new THREE.Vector3(-8, 4, 8);
    this._freeInit = false;
    this.viewSize = 100; // top view height (m), exposed for the marker logic
    this.persp.position.set(0, 0, 0);

    this.controls = new OrbitControls(this.persp, canvas);
    this.controls.enableDamping = false;
    this.controls.enabled = false;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('change', () => { if (this.mode === 'free') this.onUserChange?.(this); });

    // simple drag/wheel handling for chase / ground / top (free uses OrbitControls)
    let drag = null;
    canvas.addEventListener('pointerdown', (e) => {
      if (this.mode !== 'chase' || e.button !== 0) return;
      drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.user.yaw -= (e.clientX - drag.x) * 0.006;
      this.user.pitch = clamp(this.user.pitch + (e.clientY - drag.y) * 0.004, -1.0, 1.1);
      drag.x = e.clientX; drag.y = e.clientY;
      this.onUserChange?.(this);
    });
    const end = () => { drag = null; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', (e) => {
      if (this.mode === 'free') return;
      e.preventDefault();
      this.user.zoom = clamp(this.user.zoom * Math.exp(e.deltaY * 0.0012), 0.05, 40);
      this.onUserChange?.(this);
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { this.resetUser(); this.onUserChange?.(this); });
  }

  resetUser() {
    this.user = { yaw: 0, pitch: 0, zoom: 1 };
    this._freeInit = false;
  }

  setMode(mode) {
    if (!MODES.includes(mode) || mode === this.mode) return;
    this.mode = mode;
    this.controls.enabled = mode === 'free';
    if (mode === 'free') this._freeInit = false;
  }

  getUserState() {
    return {
      ...this.user,
      free: this.mode === 'free' ? { pos: this.persp.position.clone(), target: this.controls.target.clone() } : null,
    };
  }

  setUserState(s) {
    this.user = { yaw: s.yaw, pitch: s.pitch, zoom: s.zoom };
    if (s.free && this.mode === 'free') {
      this.persp.position.copy(s.free.pos);
      this.controls.target.copy(s.free.target);
      this._freeInit = true;
    }
  }

  /**
   * @param ctx {centerW, basis:{up,east,north}, L, alt, heading, globalHeading, sites, aspect,
   *             spherical, startW}
   * @returns {origin: THREE.Vector3, camera, camW: THREE.Vector3, focusDist}
   */
  update(ctx) {
    const { centerW, basis, L } = ctx;
    const up = basis.up;
    let origin, camW, focus = 10 * L;
    const aspect = ctx.aspect;

    if (this.mode === 'chase') {
      const cam = (this.camera = this.persp);
      const dist = 3.4 * L * this.user.zoom;
      const pitch = 0.32 + this.user.pitch;
      const hd = ctx.heading.clone().applyAxisAngle(up, this.user.yaw);
      const off = up.clone().multiplyScalar(Math.sin(pitch) * dist).addScaledVector(hd, -Math.cos(pitch) * dist);
      camW = centerW.clone().add(off);
      origin = camW.clone();
      this._lastOffset.copy(off);
      cam.position.set(0, 0, 0);
      cam.up.copy(up);
      cam.lookAt(centerW.clone().sub(origin));
      cam.fov = 50;
      focus = dist;
    } else if (this.mode === 'ground') {
      const cam = (this.camera = this.persp);
      let site = ctx.sites.length ? ctx.sites[0] : ctx.startW;
      let best = Infinity;
      for (const s of ctx.sites) {
        const d = s.distanceToSquared(centerW);
        if (d < best) { best = d; site = s; }
      }
      const sUp = ctx.frame.up(site);
      const side = new THREE.Vector3().crossVectors(sUp, ctx.globalHeading).normalize();
      camW = site.clone().addScaledVector(sUp, 2.2).addScaledVector(side, 60 + 12 * L);
      origin = camW.clone();
      cam.position.set(0, 0, 0);
      cam.up.copy(sUp);
      const dist = camW.distanceTo(centerW);
      cam.lookAt(centerW.clone().sub(origin));
      // auto-zoom: keep the vehicle ~1/5 of the frame height
      const fov = 2 * Math.atan((L / 0.3) / (2 * Math.max(dist, 1))) * (180 / Math.PI) * this.user.zoom;
      cam.fov = clamp(fov, 0.1, 70);
      focus = 0.3 * dist;
    } else if (this.mode === 'top') {
      const cam = (this.camera = this.ortho);
      const size = clamp(Math.max(14 * L, 0.8 * Math.max(ctx.alt, 0)) * this.user.zoom, 4, 5e6);
      this.viewSize = size;
      const H = Math.max(size, 100) * 1.5;
      camW = centerW.clone().addScaledVector(up, H);
      origin = camW.clone();
      cam.position.set(0, 0, 0);
      cam.up.copy(basis.north);
      cam.lookAt(centerW.clone().sub(origin));
      cam.left = (-size * aspect) / 2; cam.right = (size * aspect) / 2;
      cam.top = size / 2; cam.bottom = -size / 2;
      cam.near = 1;
      cam.far = H + Math.max(ctx.alt, 0) + 20000 + size;
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld(true);
      return { origin, camera: cam, camW, focusDist: H };
    } else { // free
      const cam = (this.camera = this.persp);
      cam.up.copy(up);
      if (!this._freeInit) {
        cam.position.copy(this._lastOffset.lengthSq() > 0 ? this._lastOffset : new THREE.Vector3(-8, 4, 8));
        this.controls.target.set(0, 0, 0);
        this._freeInit = true;
      }
      this.controls.minDistance = 0.3 * L;
      this.controls.maxDistance = 3e7;
      this.controls.zoomSpeed = 1.2;
      // keep orbit "up" aligned with the local vertical (matters on a spherical Earth)
      if (this.controls._quat) {
        this.controls._quat.setFromUnitVectors(up, Y);
        this.controls._quatInverse.copy(this.controls._quat).invert();
      }
      origin = centerW.clone();
      this.controls.update();
      camW = origin.clone().add(cam.position);
      focus = cam.position.distanceTo(this.controls.target);
      cam.fov = 50;
    }

    const cam = this.camera;
    cam.aspect = aspect;
    cam.near = clamp(focus * 0.01, 0.03, 200);
    cam.far = ctx.spherical ? 2.5e7 : 4e6;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
    return { origin, camera: cam, camW, focusDist: focus };
  }
}
