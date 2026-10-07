// Small shared helpers (math, formatting, DOM).

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/** Largest index i with arr[i] <= x (arr ascending); -1 if x < arr[0]. */
export function bisect(arr, x) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

/** Create an element: el('div', {class:'x', onclick: fn}, child, 'text'). */
export function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}

export function fmtClock(s) {
  const sign = s < 0 ? '-' : '+';
  s = Math.abs(s);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `T${sign}${String(m).padStart(2, '0')}:${sec.toFixed(1).padStart(4, '0')}`;
}

export function fmtDistance(m) {
  if (!Number.isFinite(m)) return '—';
  const a = Math.abs(m);
  if (a >= 10000) return `${(m / 1000).toFixed(2)} km`;
  if (a >= 100) return `${m.toFixed(0)} m`;
  return `${m.toFixed(1)} m`;
}

export function fmtSpeed(v) {
  if (!Number.isFinite(v)) return '—';
  return Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(2)} km/s` : `${v.toFixed(1)} m/s`;
}

export function fmtFixed(v, d = 1) {
  return Number.isFinite(v) ? v.toFixed(d) : '—';
}

/** 0xRRGGBB -> [r,g,b] in 0..1, *without* sRGB->linear conversion (for raw shader output). */
export const srgbVec = (hex) => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
