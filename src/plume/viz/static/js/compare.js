// Viewport manager: one view, or two side by side (comparison mode) with synced cameras.

import { View } from './view.js';
import { el } from './util.js';

/** Pick a sensible partner for `a` from the replay list (same vehicle / opposite source). */
export function suggestPartner(a, list) {
  const stem = (s) => (s || '').replace(/\((sim|real)\)/gi, '').replace(/\s+/g, ' ').trim().toLowerCase();
  const others = list.filter((r) => r.id !== a.id);
  if (!others.length) return null;
  const score = (r) => (stem(r.title) === stem(a.title) ? 4 : 0) + (r.source !== a.source ? 2 : 0)
    + (r.id.replace(/_(sim|real)/, '') === a.id.replace(/_(sim|real)/, '') ? 3 : 0);
  return others.reduce((best, r) => (score(r) > score(best) ? r : best), others[0]);
}

export class ViewManager {
  constructor(root, { capture = false } = {}) {
    this.root = root;
    this.capture = capture;
    this.views = [];
    this.camMode = 'chase';
    this.syncing = false;
  }

  get primary() { return this.views[0]; }

  /** (Re)build the views for the given replays. Returns warnings from loading. */
  async show(replays, tags) {
    for (const v of this.views) v.dispose();
    this.views = [];
    this.root.replaceChildren();
    this.root.classList.toggle('compare', replays.length > 1);
    const warnings = [];
    for (let i = 0; i < replays.length; i++) {
      const box = el('div', { class: 'viewport' });
      this.root.append(box);
      const view = new View(box, { capture: this.capture, compact: replays.length > 1, tag: tags?.[i] });
      view.setCamera(this.camMode);
      view.rig.onUserChange = (rig) => this._sync(view, rig);
      this.views.push(view);
    }
    const res = await Promise.all(this.views.map((v, i) => v.load(replays[i])));
    for (const w of res) warnings.push(...w);
    return warnings;
  }

  /** Mirror camera interaction from one view onto the others. */
  _sync(src, rig) {
    if (this.syncing) return;
    this.syncing = true;
    const st = rig.getUserState();
    for (const v of this.views) if (v !== src) v.rig.setUserState(st);
    this.syncing = false;
  }

  setCamera(mode) {
    this.camMode = mode;
    for (const v of this.views) v.setCamera(mode);
  }

  setTrailMode(mode) {
    for (const v of this.views) v.world?.trail.setColorBy(mode);
  }

  render(t, animTime) {
    for (const v of this.views) v.render(t, animTime);
  }
}
