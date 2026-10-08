// Landing / launch pads: a cast concrete slab (procedural texture: slab joints, weathered paint
// markings, blast scorch) on a short skirt that is seated on the terrain; launch rails for small
// rockets.

import * as THREE from 'three';
import { padTextures } from './materials.js';

export function padMesh(pad, atmosphere) {
  const R = pad.radius;
  const t = padTextures(pad.name, R);
  const top = new THREE.MeshStandardMaterial({
    map: t.map, roughnessMap: t.roughnessMap, normalMap: t.normalMap, normalScale: new THREE.Vector2(0.7, 0.7),
    roughness: 1.0, metalness: 0, color: 0xffffff,
  });
  const side = new THREE.MeshStandardMaterial({ color: 0x8c8a85, roughness: 0.95, metalness: 0 });
  atmosphere.patch(top);
  atmosphere.patch(side);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 1.01, 0.3, 96, 1, false), [side, top, side]);
  mesh.receiveShadow = true;
  mesh.castShadow = false;
  mesh.renderOrder = 5;
  mesh.userData.mats = [top, side];
  return mesh;
}

export function disposePad(mesh) {
  mesh.geometry.dispose();
  for (const m of mesh.userData.mats || []) m.dispose();
}
