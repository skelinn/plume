// Capture mode (?capture=1) for deterministic frame-by-frame screenshots, e.g. GIF generation.
//
//   /?capture=1&replay=<id>&camera=chase&t0=0&t1=40&fps=20
//
// The animation clock is disabled (plume flicker / target pulse are driven by replay time) and
// the page exposes
//   window.plumeCapture = { ready: Promise, frameCount, fps, t0, t1, canvas, seek(i): Promise }
// `await ready`, then for i in 0..frameCount-1: `await seek(i)` and screenshot the page.

export function setupCapture(app, params) {
  const fps = Math.max(1, +params.get('fps') || 20);
  const cap = { ready: null, frameCount: 0, fps, t0: 0, t1: 0, canvas: null, time: (i) => cap.t0 + i / fps, seek: null };
  window.plumeCapture = cap;

  const nextFrame = () => new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));

  cap.seek = async (i) => {
    await cap.ready;
    const idx = Math.min(Math.max(Math.round(i), 0), cap.frameCount - 1);
    const t = cap.time(idx);
    app.renderAt(t, t); // animation time == replay time => fully deterministic
    await nextFrame();
    return t;
  };

  cap.ready = (async () => {
    await app.openForCapture();
    const r = app.replays[0];
    cap.t0 = params.has('t0') ? +params.get('t0') : r.t0;
    cap.t1 = params.has('t1') ? +params.get('t1') : r.tEnd;
    cap.t1 = Math.max(cap.t1, cap.t0);
    cap.frameCount = Math.floor((cap.t1 - cap.t0) * fps + 1e-6) + 1;
    cap.canvas = app.canvas();
    app.renderAt(cap.t0, cap.t0);
    await nextFrame();
    document.documentElement.dataset.captureReady = '1';
    return cap;
  })();
  return cap;
}
