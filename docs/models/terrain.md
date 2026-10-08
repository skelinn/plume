# Real-world terrain (`plume.terrain.dem`)

Plume can fly cargo hops over **real terrain** built from the Copernicus global digital
elevation model, projected into the mission's local map frame. This page documents the
data source and licence, the projection, the resampling, the accuracy, how synthetic
detail is flagged, landing-hazard analysis, the bundled real route, and the limitations.

**Code:** `src/plume/terrain/dem.py` (library), `src/plume/terrain/cli.py` (`plume terrain …`)
**Verification status:** verified, covering projection round-trips and geodesic distances, tile naming, mosaicking and resampling against an analytic surface, synthetic-detail flagging and hazard maps (`tests/test_dem.py`). Validation of the DEM against surveyed heights is pending.

## Data sources

| Product | Posting | Bucket (public, no auth) | File name |
|---|---|---|---|
| **Copernicus DEM GLO-30** (primary) | 1″ (~30 m) | `https://copernicus-dem-30m.s3.amazonaws.com` | `Copernicus_DSM_COG_10_{N\|S}{lat:02d}_00_{E\|W}{lon:03d}_00_DEM/<same>.tif` |
| Copernicus DEM GLO-90 (fallback) | 3″ (~90 m) | `https://copernicus-dem-90m.s3.amazonaws.com` | `Copernicus_DSM_COG_30_…_DEM/<same>.tif` |

* Tiles are 1° × 1° Cloud-Optimised GeoTIFFs (float32, deflate, EPSG:4326), named after
  their south-west corner (`N32_00_W107_00` covers 32–33° N, 107–106° W). Above 50°
  latitude the tiles get fewer columns; the code reads each tile's own geotransform.
  The grid is pixel-is-point: a tile includes its north-edge row, not its south-edge row.
* GLO-30 COGs carry internal overviews at 2×/4×/8× (GLO-90: 2×/4×). For a coarse grid
  only the coarsest overview whose pixel is ≤ half the requested spacing is read over
  HTTP (`/vsicurl/`, typically 0.6 MB instead of 36 MB per tile).
* **Ocean tiles do not exist** (HTTP 404; 403 is treated the same, as S3 returns it for
  missing keys without list permission). They are recorded with an empty `<tile>.missing`
  marker in the cache and filled with **0 m (sea level)**. With `fallback=True` (default)
  a GLO-30 404 is first retried from GLO-90.
* The cache lives in `data/terrain/dem/cache/` (override: `PLUME_DEM_CACHE`); its
  `.gitignore` keeps `*.tif`, `*.part` and `*.missing` out of git.

### Robust, untrusted downloads

Timeouts (default 60 s), up to 3 retries with exponential back-off, a size cap
(120 MB per tile), Content-Length checking, downloads streamed to `*.part` and atomically
renamed, and every file is validated by opening it **with rasterio only** (single band,
geographic CRS, sane dimensions) — invalid files are deleted. If the bucket cannot be
reached a `DemDownloadError` explains that you are probably offline and where the cache
is; `offline=True` / `--offline` never touches the network.

### Licence and attribution

The Copernicus DEM GLO-30 Public and GLO-90 are distributed free of charge under the
Copernicus DEM licence, which permits any lawful use including commercial use and
redistribution of derived products, **provided this attribution is kept**:

> Contains modified Copernicus Sentinel data / © DLR e.V. 2010-2014 and © Airbus Defence
> and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all
> rights reserved

Every heightmap built from the DEM stores this text in `meta.attribution` (plus
`meta.licence`), and `plume terrain info` prints it. Keep it when redistributing the
bundled `data/terrain/real_*` files or anything derived from them.

## Vertical datum (geoid)

Copernicus heights are **orthometric heights above the EGM2008 geoid** (≈ mean sea
level), not heights above the WGS-84 ellipsoid. Plume uses them as-is: the simulator's
reference surface is a sphere and its atmosphere is indexed by height above sea level, so
MSL heights are the consistent choice. The difference to ellipsoidal heights is the geoid
undulation N (−106 m … +85 m worldwide; roughly −20 to −30 m along the bundled route,
varying by only a few metres over 760 km, so relative heights are barely affected).

`build_heightmap(..., geoid=fn)` is a hook: `fn(lat, lon) -> N` (m, arrays) is added to
every node, e.g. to obtain ellipsoidal heights when combining with GNSS data. No geoid
model ships with Plume (EGM96/EGM2008 grids are large downloads); the default is *no
correction*, recorded in `meta.vertical_datum`.

## Projection

Missions use a **local azimuthal-equidistant projection centred on launch site A on the
WGS-84 ellipsoid**: `+proj=aeqd +lat_0=<A> +lon_0=<A> +datum=WGS84 +units=m`
(`+datum=WGS84` implies `+ellps=WGS84` and avoids a no-op datum shift). `x` = east,
`y` = north, metres. Along every radial from A, the projected range equals the true
geodesic distance — exactly the property of Plume's spherical map coordinates
(`SphericalGravity.surface_point`: `u, v` = east/north arc lengths from A), so a site at
map `(u, v)` is at the right distance and azimuth from A.

```python
from plume.terrain import site_projection
proj = site_projection(32.990, -106.986)       # Pad A
u, v = proj.forward(35.335, -99.225)           # -> (705102.0, 286709.7) m
lat, lon = proj.inverse(u, v)                   # vectorised, scalars in -> scalars out
```

Backends (chosen automatically): `pyproj` (one cached `Transformer` per direction) if
installed; otherwise the same PROJ library through rasterio/GDAL (`rasterio.warp.transform`)
— both ellipsoidal and exact; otherwise a pure-numpy **spherical** fallback with radius
`R_EARTH` (identical to the simulator's own map coordinates; ~0.2 % scale difference to
the ellipsoid at 760 km). `pyproj` is *not* currently installed in the project
environment; the rasterio backend is used and is tested to < 1 mm round-trip and to
agree with an independent Vincenty geodesic to < 1 cm at 761 km.

Away from A the grid axes are A's east/north, not the local east/north: at B (761 km
ENE) they are rotated by the meridian convergence (~4.4°), and the tangential scale is
1 + c²/6 ≈ 1.0024 (c = angular distance). Heights are sampled at the correct geographic
position regardless; only "north-up" of a far-away tile is slightly rotated.

## Mosaicking and resampling

1. The projected grid's perimeter is inverse-projected to a lat/lon box, padded by three
   source pixels plus two grid spacings (room for the warp kernel).
2. The intersecting tiles are fetched (concurrently, 6 threads) and merged with
   `rasterio.merge` onto the tiles' own pixel grid (the crop is snapped to source pixel
   edges, so merging copies pixels exactly — tested to 6e-5 m).
3. The mosaic is warped by GDAL onto the heightmap nodes with **bilinear** resampling and
   an exact per-pixel transform (`tolerance=0`). Heightmap nodes are mapped to GDAL cell
   centres (`heights[j, i]` ↔ pixel `(i, ny-1-j)`), so no half-pixel shift is introduced.
   When the target grid is coarser than the source, GDAL widens the bilinear kernel, so
   downsampling is anti-aliased.

`corridor_heightmap(A, B, width, resolution)` covers the projected bounding box of A and B
padded by `width`, snapped so A lies on a node, and coarsens the resolution automatically
(logged) to keep each side ≤ 4096 cells. `build_heightmap` refuses mosaics above 80 M
source cells and asks for a coarser grid instead.

## Accuracy

| Source of error | Magnitude |
|---|---|
| Copernicus GLO-30 absolute vertical accuracy (spec) | < 4 m LE90; typically 1–2 m RMSE on open flat terrain, worse on steep slopes and under forest |
| Copernicus horizontal accuracy | < 6 m CE90 |
| Resampling (tested on an analytic surface, 10″ source) | ≤ 0.07 m upsampling to 50 m; ≤ 0.64 m (0.08 % of relief) downsampling to 1 km; ≤ 5.5 m from an 8× overview at 2 km |
| Projection (rasterio/PROJ, pyproj) | < 1 mm round-trip; range = geodesic distance to < 1 cm |
| Bundled route (500 m grid from the 8× overview, ~247 m) | features narrower than ~1 km are smoothed; sharp summits are lowered (Sierra Blanca: 3,488 m in the grid vs 3,652 m real) |
| Upsampled landing tiles (30 m → 2–2.5 m) | piecewise-planar between 30 m postings; no real metre-scale detail |

The DEM is a **DSM** (digital *surface* model): it includes buildings, tree canopy,
power-line towers and similar. Example: the nominal Spaceport America coordinate
(32.990 N, 106.975 W) lies on the terminal hangar, which appears as a ~14 m bump — which
is why Pad A was moved 1 km west onto open desert. Run `plume terrain hazard` on any
landing tile to find such structures.

## Synthetic detail (flagged)

30 m data cannot resolve rocks or metre-scale undulation, which matter for touchdown.
`landing_tile(..., synthetic_detail=True)` / `add_synthetic_detail()` therefore add:

* **band-limited undulation**: 1/f^β noise (β = 2.4) restricted to wavelengths shorter
  than `cutoff_wavelength` (default 3 × the source posting, ~93 m for GLO-30) with RMS
  `roughness` (default 0.15 m), so the large-scale shape measured by the DEM is preserved;
* **rocks**: Gaussian bumps, Poisson count `rock_density` per m² (default 1e-4), height
  0.1–0.6 m, radius 0.6–2.0 m.

Whenever this happens the heightmap meta says so — `synthetic_detail: true` and
`synthetic_detail_params` (generator, seed, roughness, cutoff, β, rock density, count,
size ranges, and the note *"procedural detail below DEM resolution -- NOT measured
terrain"*). DEM-only maps have `synthetic_detail: false`. The flag survives
`Heightmap.save`/`load` and is shown by `plume terrain info` / `hazard`.

A **real survey** replaces the synthetic detail: `from_geotiff(path, lat0, lon0, x_range,
y_range, resolution)` (or `landing_tile(..., geotiff=path)`, or `plume terrain fetch
--geotiff survey.tif`) reprojects a LiDAR / drone-photogrammetry GeoTIFF in *any* CRS into
the site projection with `rasterio.warp` (bilinear, nodata-aware, reading lazily from the
file). Incomplete coverage raises unless `fill_value` is given. No synthetic detail is
added (`synthetic_detail: false`); check that the survey's vertical datum matches (use
`vertical_offset`).

## Landing hazards

`hazard_map(hm, footprint_radius=3, max_slope_deg=8, max_roughness=0.1)` fits a plane
`z = a + b x + c y` by least squares over the circular footprint around **every** node
(vectorised with `scipy.ndimage.correlate`; the footprint's symmetric moments decouple
the fit) and returns:

* `slope_deg = atan(|(b, c)|)` — the slope the lander would actually sit on;
* `roughness` — RMS of the detrended heights within the footprint (rocks, ridges);
* `safe` — both below their limits and the footprint fully inside the map;
* `nearest_safe(x, y, clearance=0)` — nearest safe node to a target (optionally requiring
  all nodes within `clearance` to be safe, via a Euclidean distance transform).

The default roughness limit (0.1 m RMS) separates the synthetic micro-relief
(median 0.03 m on a 3 m footprint) from rocks ≳ 0.3 m.

## Command line

```
plume terrain fetch --site LAT,LON [--site2 LAT,LON] --radius-km R --resolution M --out data/terrain/<name>.yaml
                    [--origin LAT,LON] [--product glo30|glo90] [--fmt png|npy]
                    [--synthetic-detail --seed N --roughness 0.15 --rock-density 1e-4]
                    [--flatten-radius M] [--geotiff survey.tif] [--offline]
plume terrain info  data/terrain/<name>.yaml
plume terrain hazard data/terrain/<name>.yaml --footprint 3 --max-slope 8 [--max-roughness 0.1]
                    [--target X,Y] [--clearance M] [--png mask.png]
```

Without `--site2`, `fetch` writes a square of half-width `--radius-km` around `--site` in
the projection centred on `--origin` (default: the site). With `--site2` it writes the
A→B corridor padded by `--radius-km`. Use `--site=-33.9,151.2` for southern latitudes.

## Bundled real route: Spaceport America → Burns Flat

| | Site | lat, lon (WGS-84) | map u, v (m) | height |
|---|---|---|---|---|
| A | open desert 1 km W of the Spaceport America terminal, New Mexico | 32.990, −106.986 | 0, 0 | ~1411 m |
| B | open farmland 2.2 km W of the Clinton-Sherman runway (Oklahoma Air & Space Port), Burns Flat, Oklahoma | 35.335, −99.225 | 705 102.0, 286 709.7 | ~590 m |

Geodesic distance **761.2 km**, initial azimuth 67.9° (ENE). Why this pair: both ends
are FAA-licensed spaceports in sparsely populated country (Jornada del Muerto basin;
western-Oklahoma wheat plains), the distance and heading match the procedural
`demo_hop` (749.5 km ENE) so the same vehicle and guidance apply, both landing areas are
flat (hazard map: 99 % of the 2 km landing tile safe for a 3 m footprint at 8°/0.1 m),
and the track crosses interesting real relief: 118 km out it passes ~1.5 km from the
summit of Sierra Blanca (3,652 m; 3,366 m on the track in the 500 m grid), then the flat
Llano Estacado and the Texas Panhandle. The corridor's
northern edge reaches the Sangre de Cristo range (max 3,848 m — Santa Fe Baldy is
3,847 m, a handy sanity check). There is no ocean in the corridor.

Files (`data/terrain/`):

| File | Grid | Content | Size |
|---|---|---|---|
| `real_route.yaml/.png` | 1652 × 815 @ 500 m, 825 × 407 km, 16-bit PNG (5.4 cm quantisation) | GLO-30 8× overview, 50 tiles | 1.9 MB |
| `real_pad_a.yaml/.png` | 201 × 201 @ 2 m (±200 m) | GLO-30, 30 m pad flattened (`meta.modifications`) | 0.05 MB |
| `real_lz_b.yaml/.png` | 801 × 801 @ 2.5 m (±1 km), in A's projection | GLO-30 + **flagged synthetic detail** (seed 12) | 1.0 MB |

Mission: `configs/missions/real_hop.yaml` (sites carry both `u`/`v` and `lat`/`lon`).
A trial flight of the `cargo_hopper` (250 kg cargo, seed 0) over this terrain landed
27 m from B with 113 kg of propellant left (apogee 184 km, 572 s, ground slope 2.6°).
Regenerate (≈ 105 MB is downloaded once into the cache: 50 overviews of ~0.7 MB for
the route plus the two full 36 MB GLO-30 tiles under the pad and the landing zone):

```
plume terrain fetch --site 32.990,-106.986 --site2 35.335,-99.225 --radius-km 60 --resolution 500 --out data/terrain/real_route.yaml
plume terrain fetch --site 32.990,-106.986 --radius-km 0.2 --resolution 2 --flatten-radius 30 --out data/terrain/real_pad_a.yaml
plume terrain fetch --site 35.335,-99.225 --origin 32.990,-106.986 --radius-km 1 --resolution 2.5 --synthetic-detail --seed 12 --out data/terrain/real_lz_b.yaml
```

## Limitations

* **Geoid**: heights are above EGM2008 (MSL), not the ellipsoid; no geoid model is
  bundled (hook only). Up to ±~100 m vs ellipsoidal heights globally.
* **DSM, not bare earth**: buildings and canopy are part of the surface; water bodies
  are flattened, voids were filled by the producer from other DEMs.
* **Resolution**: 30 m posting at best; anything finer in a landing tile is either
  bilinear interpolation or flagged synthetic detail. The bundled route is 500 m.
* **Projection range**: azimuthal-equidistant distortion grows with distance (tangential
  scale 1 + c²/6: 0.24 % at 760 km, ~1 % at 1,600 km); intended for hops up to a few
  thousand km. Areas crossing the antimeridian are not supported; avoid the poles.
* **Spherical simulator**: Plume's world is a sphere of radius 6371 km; the DEM is mapped
  by geodesic distance/azimuth from A, which is exact radially but cannot represent the
  ellipsoid's curvature differences (≲ 0.3 % in the local "drop" of the horizon).
* **Licence**: attribution required (above); the Copernicus DEM comes without warranty
  and is not a navigation-grade product.
