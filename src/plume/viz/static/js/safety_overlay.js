// Flight-safety overlay for the engineering view (meta.safety, written by `plume safety --embed`
// and by the planner's reliability jobs; see plume.analysis.safety.viewer_overlay):
// instantaneous-impact-point traces and hazard-area outlines on the ground, world ENU metres.
// Thin 1 px lines drawn over the terrain, positions re-based on the camera origin every frame
// (distances reach hundreds of km, so nothing is handed to the GPU in absolute coordinates).

import * as THREE from 'three';
import { enuToW } from './coords.js';

const STYLE = {
  iip_drag: { color: 0xdcc47c, dash: [2400, 1600] },
  iip_vacuum: { color: 0xa3a3a3, dash: [800, 1600] },
  hazard: { color: 0xd08a84, dash: null },
};

export class SafetyOverlay {
  constructor(scene, meta) {
    this.items = [];
    const S = meta?.safety;
    if (!S) return;
    const add = (pts, style, loop) => {
      if (!Array.isArray(pts) || pts.length < 2) return;
      const W = pts.map((p) => enuToW(p));
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(W.length * 3), 3));
      const mat = style.dash
        ? new THREE.LineDashedMaterial({ color: style.color, dashSize: style.dash[0], gapSize: style.dash[1], depthTest: false, depthWrite: false, transparent: true, toneMapped: false })
        : new THREE.LineBasicMaterial({ color: style.color, depthTest: false, depthWrite: false, transparent: true, opacity: 0.95, toneMapped: false });
      const line = new (loop ? THREE.LineLoop : THREE.Line)(g, mat);
      line.frustumCulled = false;
      line.renderOrder = 97;
      scene.add(line);
      this.items.push({ line, W, dashed: !!style.dash });
    };
    add(S.iip_vacuum, STYLE.iip_vacuum, false);
    add(S.iip_drag, STYLE.iip_drag, false);
    for (const h of S.hazards || []) add(h.ring, STYLE.hazard, true);
    this.legend = [
      S.iip_drag && ['IIP trace, drag-aware', STYLE.iip_drag],
      S.iip_vacuum && ['IIP trace, vacuum', STYLE.iip_vacuum],
      (S.hazards || []).length && [`Hazard areas (${S.hazards.length})`, STYLE.hazard],
    ].filter(Boolean);
  }

  get active() { return this.items.length > 0; }

  setVisible(on) { for (const it of this.items) it.line.visible = !!on; }

  layout(origin) {
    for (const it of this.items) {
      const a = it.line.geometry.attributes.position;
      it.W.forEach((p, i) => a.setXYZ(i, p.x - origin.x, p.y - origin.y, p.z - origin.z));
      a.needsUpdate = true;
      it.line.geometry.computeBoundingSphere();
      if (it.dashed) it.line.computeLineDistances();
    }
  }

  /** Legend rows for the engineering panel: [[label, cssColor, dashed], ...]. */
  keyRows() {
    return (this.legend || []).map(([label, s]) => [label, `#${s.color.toString(16).padStart(6, '0')}`, !!s.dash]);
  }
}
