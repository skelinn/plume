// Shared GLSL snippets: hashes, value noise, fbm, and anti-aliased line helpers.
//
// Every custom shader renders into the linear HDR target (see post.js); tone mapping and the sRGB
// transfer happen once, in the final composite.

export const GLSL_NOISE = /* glsl */ `
float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash31(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}
float vnoise3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x), mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y);
  float b = mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x), mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y);
  return mix(a, b, f.z);
}
float fbm2(vec2 p) {
  float s = 0.0, a = 0.5;
  // rotate between octaves so the value-noise lattice never lines up (no blocky, axis-aligned edges)
  for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = mat2(1.62, 1.18, -1.18, 1.62) * p + vec2(17.1, 9.2); a *= 0.5; }
  return s / 0.9375;
}
float fbm3(vec3 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * vnoise3(p); p = p * 2.02 + vec3(17.1, 9.2, 4.7); a *= 0.5; }
  return s / 0.9375;
}
// Anti-aliased grid lines every 'scale' units; lines denser than a few pixels fade out.
float gridLine(vec2 p, float scale) {
  vec2 c = p / scale;
  vec2 w = max(fwidth(c), vec2(1e-6));
  vec2 g = abs(fract(c - 0.5) - 0.5) / w;
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  return line * (1.0 - smoothstep(0.05, 0.13, max(w.x, w.y)));
}
float contourLine(float e, float step) {
  float c = e / step;
  float w = max(fwidth(c), 1e-6);
  float g = abs(fract(c - 0.5) - 0.5) / w;
  return (1.0 - min(g, 1.0)) * (1.0 - smoothstep(0.05, 0.14, w));
}
`;
