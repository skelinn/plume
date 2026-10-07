// Live mode: subscribe to /ws/live?channel=... and hand meta / frames / end messages to the app.
// The server replays what a late joiner missed, so connecting mid-run is fine.

export class LiveSession {
  /**
   * @param handlers {onStatus(status), onMeta(meta), onFrames(frames), onEnd(outcome)}
   * status: 'connecting' | 'idle' (connected, no run yet) | 'streaming' | 'ended' | 'offline'
   */
  constructor(channel, handlers) {
    this.channel = channel || 'default';
    this.h = handlers;
    this.ws = null;
    this.closed = false;
    this.retry = 0;
  }

  connect() {
    this.closed = false;
    this.h.onStatus?.('connecting');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/live?channel=${encodeURIComponent(this.channel)}`);
    this.ws = ws;
    ws.onopen = () => { this.retry = 0; };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      switch (msg.type) {
        case 'idle': this.h.onStatus?.('idle'); break;
        case 'meta': this.h.onStatus?.('streaming'); this.h.onMeta?.(msg.meta); break;
        case 'frames': this.h.onFrames?.(msg.frames); break;
        case 'end': this.h.onStatus?.('ended'); this.h.onEnd?.(msg.outcome); break;
        default: break;
      }
    };
    ws.onclose = () => {
      if (this.closed) return;
      this.h.onStatus?.('offline');
      this.retry = Math.min(this.retry + 1, 6);
      setTimeout(() => { if (!this.closed) this.connect(); }, 500 * 2 ** this.retry);
    };
    ws.onerror = () => ws.close();
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}
