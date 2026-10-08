// Data URLs for the viewer. Served by `plume viz` (FastAPI) by default; a static export
// (`plume export-site`, e.g. on GitHub Pages) sets window.PLUME_STATIC and the same data
// is read from plain files next to index.html.

export const STATIC = typeof window !== 'undefined' && window.PLUME_STATIC === true;

const enc = (id) => id.split('/').map(encodeURIComponent).join('/');

export const api = {
  replays: () => (STATIC ? 'api/replays.json' : '/api/replays'),
  replay: (id) => (STATIC ? `api/replays/${enc(id)}.json` : `/api/replays/${enc(id)}`),
  terrainMeta: (id, maxDim) =>
    STATIC ? `api/terrain/${enc(id)}.${maxDim}.json` : `/api/terrain/${enc(id)}?max_dim=${maxDim}`,
  terrainHeights: (id, maxDim) =>
    STATIC ? `api/terrain/${enc(id)}.${maxDim}.f32` : `/api/terrain/${enc(id)}/heights?max_dim=${maxDim}`,
};
