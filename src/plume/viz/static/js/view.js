// A View = canvas + WebGL renderer + World + camera rig + HUD + marker overlay.
// Compare mode simply instantiates two of them.

import * as THREE from 'three';
import { World } from './scene.js';
import { CameraRig } from './cameras.js';
import { Hud } from './hud.js';
import { el } from './util.js';

export class View {
  constructor(container, { capture = false, compact = false, tag = null } = {}) {
    this.container = container;
    this.capture = capture;
    this.tag = tag;
    this.canvas = el('canvas', { class: 'gl' });
    this.labelLayer = el('div', { class: 'labels' });
    this.hudRoot = el('div', { class: 'hud-root' });
    container.append(this.canvas, this.labelLayer, this.hudRoot);
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: true, preserveDrawingBuffer: capture, powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x05080f, 1);
    this.pixelRatio = capture ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.rig = new CameraRig(this.canvas);
    this.hud = new Hud(this.hudRoot, { compact });
    this.world = null;
    this.replay = null;
    this.w = 1; this.h = 1;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    if (w === this.w && h === this.h && this.renderer.domElement.width) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
  }

  async load(replay) {
    this.world?.dispose();
    this.replay = replay;
    this.world = new World(replay, { labelLayer: this.labelLayer, capture: this.capture });
    this.hud.setReplay(replay, this.tag);
    await this.world.load();
    return this.world.warnings;
  }

  setCamera(mode) { this.rig.setMode(mode); }

  /** Render the world at replay time t. animTime drives plume/target animation. */
  render(t, animTime) {
    const world = this.world;
    if (!world || !this.replay.n) return;
    this.resize();
    const st = world.update(t, animTime);
    if (!st) return;
    st.aspect = this.w / this.h;
    const cam = this.rig.update(st);
    world.layout(cam.origin, cam, animTime);
    const camera = cam.camera;
    world.trail.setCameraInfo(camera.near, this.w * this.pixelRatio, this.h * this.pixelRatio, this.pixelRatio);
    this.renderer.render(world.scene, camera);

    // pixels per metre at the rocket (for the locator marker)
    let ppm;
    if (camera.isOrthographicCamera) ppm = this.h / this.rig.viewSize;
    else {
      const d = Math.max(cam.camW.distanceTo(st.centerW), 1e-3);
      ppm = this.h / (2 * Math.tan((camera.fov * Math.PI) / 360) * d);
    }
    // the locator marker shows when the vehicle is only a few pixels across (top view: its diameter)
    const extent = camera.isOrthographicCamera ? Math.max(world.D, 2 * (world.vehicle.legs?.span || 0)) : world.L;
    world.updateLabels(cam, cam.origin, this.w, this.h, ppm, extent);
    this.hud.update(st.sample, t);
  }

  dispose() {
    this.ro.disconnect();
    this.world?.dispose();
    this.renderer.dispose();
    this.container.replaceChildren();
  }
}
