// A View = canvas + WebGL renderer + HDR post chain + World + camera rig + HUD + overlays.
// Compare mode simply instantiates two of them.

import * as THREE from 'three';
import { World } from './scene.js';
import { CameraRig } from './cameras.js';
import { Hud } from './hud.js';
import { PostFX } from './post.js';
import { el } from './util.js';

export class View {
  constructor(container, { capture = false, compact = false, tag = null, quality = 'high', eng = false } = {}) {
    this.container = container;
    this.capture = capture;
    this.tag = tag;
    this.quality = quality;
    this.eng = eng;
    this.canvas = el('canvas', { class: 'gl' });
    this.labelLayer = el('div', { class: 'labels' });
    this.hudRoot = el('div', { class: 'hud-root' });
    this.engRoot = el('div', { class: 'eng-root' });
    container.append(this.canvas, this.labelLayer, this.engRoot, this.hudRoot);
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: true, preserveDrawingBuffer: capture, powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.toneMapping = THREE.NoToneMapping;    // done in the composite pass (ACES)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.post = new PostFX(this.renderer, { quality });
    this._applyPixelRatio();
    this.rig = new CameraRig(this.canvas);
    this.hud = new Hud(this.hudRoot, { compact });
    this.world = null;
    this.replay = null;
    this.w = 1; this.h = 1;
    this.frameMs = 0;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
  }

  _applyPixelRatio() {
    const dpr = window.devicePixelRatio || 1;
    this.pixelRatio = this.capture ? 1 : Math.min(dpr, this.quality === 'high' ? 1.5 : 1);
    this.renderer.setPixelRatio(this.pixelRatio);
  }

  resize(force = false) {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    if (!force && w === this.w && h === this.h && this.renderer.domElement.width) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    const s = new THREE.Vector2();
    this.renderer.getDrawingBufferSize(s);
    this.post.setSize(s.x, s.y);
  }

  setQuality(q) {
    if (q === this.quality) return;
    this.quality = q;
    this._applyPixelRatio();
    this.post.setQuality(q);
    this.resize(true);
    this.world?.setQuality(q);
  }

  setEng(on) {
    this.eng = on;
    this.world?.eng.setEnabled(on);
  }

  async load(replay) {
    this.world?.dispose();
    this.replay = replay;
    this.world = new World(replay, {
      labelLayer: this.labelLayer, capture: this.capture, renderer: this.renderer, quality: this.quality, engRoot: this.engRoot,
    });
    this.world.eng.setEnabled(this.eng);
    this.hud.setReplay(replay, this.tag);
    await this.world.load();
    return this.world.warnings;
  }

  setCamera(mode) { this.rig.setMode(mode); }

  /** Render the world at replay time t. animTime drives plume/dust animation. */
  render(t, animTime) {
    const world = this.world;
    if (!world || !this.replay.n) return;
    const t0 = performance.now();
    this.resize();
    const st = world.update(t, animTime);
    if (!st) return;
    st.aspect = this.w / this.h;
    const cam = this.rig.update(st);
    world.layout(cam.origin, cam, animTime, { now: t0 / 1000 });
    const camera = cam.camera;
    world.trail.setCameraInfo(camera.near, this.w * this.pixelRatio, this.h * this.pixelRatio, this.pixelRatio);
    this.post.render(world.scene, camera, () => world.eng.render(this.renderer, camera));

    // pixels per metre at the rocket (for the locator marker)
    let ppm;
    if (camera.isOrthographicCamera) ppm = this.h / this.rig.viewSize;
    else {
      const d = Math.max(cam.camW.distanceTo(st.centerW), 1e-3);
      ppm = this.h / (2 * Math.tan((camera.fov * Math.PI) / 360) * d);
    }
    const extent = camera.isOrthographicCamera ? Math.max(world.D, 2 * (world.vehicle.legs?.span || 0)) : world.L;
    world.updateLabels(cam, cam.origin, this.w, this.h, ppm, extent);
    this.hud.update(st.sample, t);
    this.frameMs = 0.9 * this.frameMs + 0.1 * (performance.now() - t0);
  }

  dispose() {
    this.ro.disconnect();
    this.world?.dispose();
    this.post.dispose();
    this.renderer.dispose();
    this.container.replaceChildren();
  }
}
