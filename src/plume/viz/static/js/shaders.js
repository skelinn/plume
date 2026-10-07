// Shared GLSL snippets and the environment uniforms every custom material reads.
//
// Colours in the custom shaders are authored directly in display (sRGB) space and written
// to the framebuffer untouched; only the lit rocket (MeshStandardMaterial) goes through
// three's colour management.  Fog colour == sky horizon colour so the ground melts into it.

import * as THREE from 'three';

export function makeEnvUniforms() {
  return {
    uSunDir: { value: new THREE.Vector3(0.45, 0.62, 0.64).normalize() },
    uFogColor: { value: new THREE.Vector3(0.26, 0.34, 0.45) },
    uFogDensity: { value: 1e-5 },
    // glow thrown on the ground by the engine plume (render-space position)
    uLightPos: { value: new THREE.Vector3(0, -1e9, 0) },
    uLightI: { value: 0 },
    uLightR: { value: 10 },
    uLightCol: { value: new THREE.Vector3(1.0, 0.52, 0.18) },
    uTime: { value: 0 },
  };
}

export const GLSL_COMMON = /* glsl */ `
uniform vec3 uSunDir;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uLightPos;
uniform float uLightI;
uniform float uLightR;
uniform vec3 uLightCol;
uniform float uTime;

float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash31(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}

// Anti-aliased grid lines every 'scale' units; lines that would be denser than a few pixels fade out.
float gridLine(vec2 p, float scale) {
  vec2 c = p / scale;
  vec2 w = max(fwidth(c), vec2(1e-6));
  vec2 g = abs(fract(c - 0.5) - 0.5) / w;
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  float fade = 1.0 - smoothstep(0.05, 0.13, max(w.x, w.y));
  return line * fade;
}
float contourLine(float e, float step) {
  float c = e / step;
  float w = max(fwidth(c), 1e-6);
  float g = abs(fract(c - 0.5) - 0.5) / w;
  float line = 1.0 - min(g, 1.0);
  return line * (1.0 - smoothstep(0.05, 0.14, w));
}

vec3 applyFog(vec3 col, float dist) {
  float d = dist * uFogDensity;
  float f = 1.0 - exp(-d * d);
  return mix(col, uFogColor, clamp(f, 0.0, 1.0));
}

// Orange glow cast on the ground by the engine (point-light stand-in for the custom shaders).
vec3 plumeGlow(vec3 worldRel, vec3 albedo) {
  vec3 d = worldRel - uLightPos;
  float k = uLightI / (1.0 + dot(d, d) / (uLightR * uLightR));
  return uLightCol * k * (0.25 + albedo * 1.4);
}
`;
