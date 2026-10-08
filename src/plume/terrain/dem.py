"""Real-world terrain from digital elevation models (DEMs).

Pipeline
--------
1. **Tiles** -- Copernicus DEM GLO-30 (1 arc-second, ~30 m) or GLO-90 (3 arc-second,
   ~90 m) 1 deg x 1 deg Cloud-Optimised GeoTIFFs from the public AWS open-data buckets
   (no authentication). Only the tiles intersecting the requested area are fetched and
   they are cached under ``data/terrain/dem/cache/``. For coarse grids only a COG
   *overview* (2x/4x/8x decimated, typically < 1 MB) is read over HTTP instead of the
   full 36 MB tile. Ocean tiles do not exist in the bucket (HTTP 404) and are treated as
   sea level (0 m).
2. **Mosaic** -- the tiles are merged into one geographic (EPSG:4326) raster cropped to
   the area of interest.
3. **Projection** -- missions use a local azimuthal-equidistant projection centred on
   launch site A on the WGS-84 ellipsoid (``+proj=aeqd``): ``x`` = east, ``y`` = north,
   metres. Distance from A along any radial is the true geodesic distance, which is
   what Plume's spherical map coordinates (``SphericalGravity.surface_point``) assume.
4. **Resampling** -- GDAL bilinear warping onto the regular projected grid used by
   :class:`~plume.terrain.heightmap.Heightmap` (``heights[j, i]`` at
   ``x_min + i*dx``, ``y_min + j*dy``; row 0 = south). When the grid is coarser than the
   source GDAL widens the bilinear kernel, so downsampling is anti-aliased.

Heights are Copernicus orthometric heights (metres above the EGM2008 geoid, i.e. above
mean sea level). An optional ``geoid`` hook converts them to another vertical datum;
see ``docs/models/terrain.md``.

Downloaded files are treated as untrusted data: they are only ever opened by
rasterio/GDAL as rasters, validated (single band, geographic CRS, sane size), and written
atomically (``*.part`` then rename) so an interrupted download never leaves a corrupt
tile in the cache.
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from plume.constants import R_EARTH
from plume.terrain.heightmap import Heightmap

log = logging.getLogger(__name__)

COPERNICUS_ATTRIBUTION = (
    "Contains modified Copernicus Sentinel data / © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
    "European Union and ESA; all rights reserved"
)
COPERNICUS_LICENCE = (
    "Copernicus DEM licence (GLO-30 Public / GLO-90): free of charge, worldwide, for any "
    "lawful use incl. commercial, with attribution"
)
VERTICAL_DATUM = "EGM2008 geoid (orthometric height, m above mean sea level)"
HORIZONTAL_DATUM = "WGS 84"
USER_AGENT = "plume-terrain/0.1 (+https://github.com/plume-sim)"
MAX_TILE_BYTES = 120_000_000  # a GLO-30 tile is ~10-40 MB; refuse anything absurd
MAX_MOSAIC_CELLS = 80_000_000  # ~320 MB float32; ask for a coarser grid beyond this
_ARCSEC_M = 30.87  # metres per arc-second of latitude


@dataclass(frozen=True)
class DemProduct:
    key: str
    title: str
    bucket_url: str
    code: str  # resolution code in the file name: 10 (1") or 30 (3")
    arcsec: float
    overviews: tuple[int, ...]  # decimation factors of the COG's internal overviews

    @property
    def pixel_m(self) -> float:
        return self.arcsec * _ARCSEC_M


PRODUCTS: dict[str, DemProduct] = {
    "glo30": DemProduct(
        "glo30",
        "Copernicus DEM GLO-30",
        "https://copernicus-dem-30m.s3.amazonaws.com",
        "10",
        1.0,
        (2, 4, 8),
    ),
    "glo90": DemProduct(
        "glo90",
        "Copernicus DEM GLO-90",
        "https://copernicus-dem-90m.s3.amazonaws.com",
        "30",
        3.0,
        (2, 4),
    ),
}


class DemDownloadError(RuntimeError):
    """A DEM tile could not be downloaded (network error, offline, corrupt file)."""


class _TileMissing(Exception):
    """The server says the tile does not exist (ocean)."""


# ============================================================================ projection
class SiteProjection:
    """Local azimuthal-equidistant projection centred on a launch site (WGS-84).

    ``forward(lat, lon) -> (x, y)`` and ``inverse(x, y) -> (lat, lon)`` accept scalars or
    arrays (degrees, metres; x = east, y = north). Backends, in order of preference:

    * ``pyproj``   -- one cached :class:`pyproj.Transformer` (ellipsoidal, exact);
    * ``rasterio`` -- the same PROJ library through GDAL (ellipsoidal, exact);
    * ``sphere``   -- pure numpy on a sphere of radius ``R_EARTH``: identical to the
      simulator's own spherical map coordinates; ~0.3 % scale error vs the ellipsoid.
    """

    def __init__(self, lat0: float, lon0: float, backend: str = "auto"):
        if not (-90.0 < lat0 < 90.0) or not (-180.0 <= lon0 <= 180.0):
            raise ValueError(f"invalid site ({lat0}, {lon0})")
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.proj4 = (
            f"+proj=aeqd +lat_0={self.lat0:.10f} +lon_0={self.lon0:.10f} "
            "+datum=WGS84 +units=m +no_defs"
        )
        self._fwd = self._inv = self._crs = None
        if backend == "auto":
            for candidate in ("pyproj", "rasterio", "sphere"):
                if self._init_backend(candidate):
                    break
        elif not self._init_backend(backend):
            raise ImportError(f"projection backend {backend!r} is not available")

    def _init_backend(self, name: str) -> bool:
        if name == "pyproj":
            try:
                from pyproj import CRS, Transformer
            except ImportError:
                return False
            crs = CRS.from_proj4(self.proj4)
            self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        elif name == "rasterio":
            try:
                import rasterio.warp  # noqa: F401
            except ImportError:
                return False
        elif name != "sphere":
            raise ValueError(f"unknown projection backend {name!r}")
        self.backend = name
        return True

    @property
    def crs(self):
        """The projection as a :class:`rasterio.crs.CRS` (for warping)."""
        if self._crs is None:
            from rasterio.crs import CRS

            self._crs = CRS.from_proj4(self.proj4)
        return self._crs

    def forward(self, lat, lon):
        lat_a, lon_a = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        if self.backend == "pyproj":
            x, y = self._fwd.transform(lon_a, lat_a)
        elif self.backend == "rasterio":
            x, y = _rasterio_transform("EPSG:4326", self.crs, lon_a, lat_a)
        else:
            x, y = _sphere_forward(self.lat0, self.lon0, lat_a, lon_a)
        return _out(x, lat), _out(y, lat)

    def inverse(self, x, y):
        x_a, y_a = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self.backend == "pyproj":
            lon, lat = self._inv.transform(x_a, y_a)
        elif self.backend == "rasterio":
            lon, lat = _rasterio_transform(self.crs, "EPSG:4326", x_a, y_a)
        else:
            lat, lon = _sphere_inverse(self.lat0, self.lon0, x_a, y_a)
        return _out(lat, x), _out(lon, x)

    def __repr__(self) -> str:
        return f"SiteProjection({self.lat0:.6f}, {self.lon0:.6f}, backend={self.backend!r})"


def site_projection(lat0: float, lon0: float, backend: str = "auto") -> SiteProjection:
    """Azimuthal-equidistant projection centred on (``lat0``, ``lon0``); see
    :class:`SiteProjection`."""
    return SiteProjection(lat0, lon0, backend)


def _out(a, like):
    a = np.asarray(a, dtype=float)
    return float(a) if np.ndim(like) == 0 else a.reshape(np.shape(like))


def _rasterio_transform(src, dst, xs: np.ndarray, ys: np.ndarray):
    from rasterio.warp import transform

    shape = np.broadcast(xs, ys).shape
    xs, ys = np.broadcast_to(xs, shape).ravel(), np.broadcast_to(ys, shape).ravel()
    if xs.size == 0:
        return np.empty(shape), np.empty(shape)
    ox, oy = transform(src, dst, xs, ys)
    return np.asarray(ox).reshape(shape), np.asarray(oy).reshape(shape)


def _sphere_forward(lat0, lon0, lat, lon, radius=R_EARTH):
    p0, p = np.radians(lat0), np.radians(lat)
    dl = np.radians(lon - lon0)
    hav = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    k = np.where(c > 1e-12, c / np.where(c > 1e-12, np.sin(c), 1.0), 1.0)
    x = radius * k * np.cos(p) * np.sin(dl)
    y = radius * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl))
    return x, y


def _sphere_inverse(lat0, lon0, x, y, radius=R_EARTH):
    p0 = np.radians(lat0)
    rho = np.hypot(x, y)
    c = rho / radius
    safe = np.where(rho > 0, rho, 1.0)
    lat = np.arcsin(np.clip(np.cos(c) * np.sin(p0) + y * np.sin(c) * np.cos(p0) / safe, -1.0, 1.0))
    dl = np.arctan2(x * np.sin(c), rho * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
    lat = np.where(rho > 0, np.degrees(lat), lat0)
    lon = np.where(rho > 0, lon0 + np.degrees(dl), lon0)
    return lat, (lon + 180.0) % 360.0 - 180.0


def geodesic_inverse(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """WGS-84 geodesic distance (m) and initial azimuth (deg) by Vincenty's method
    (sub-millimetre accurate except for nearly antipodal points)."""
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        s_sig = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        if s_sig == 0:
            return 0.0, 0.0
        c_sig = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(s_sig, c_sig)
        s_alpha = cU1 * cU2 * sl / s_sig
        c2_alpha = 1 - s_alpha**2
        c2sm = c_sig - 2 * sU1 * sU2 / c2_alpha if c2_alpha else 0.0
        C = f / 16 * c2_alpha * (4 + f * (4 - 3 * c2_alpha))
        lam_prev = lam
        lam = L + (1 - C) * f * s_alpha * (
            sig + C * s_sig * (c2sm + C * c_sig * (-1 + 2 * c2sm**2))
        )
        if abs(lam - lam_prev) < 1e-13:
            break
    u2 = c2_alpha * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    d_sig = (
        B
        * s_sig
        * (
            c2sm
            + B
            / 4
            * (c_sig * (-1 + 2 * c2sm**2) - B / 6 * c2sm * (-3 + 4 * s_sig**2) * (-3 + 4 * c2sm**2))
        )
    )
    dist = b * A * (sig - d_sig)
    az = math.degrees(math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam)))
    return dist, az % 360.0


# ============================================================================ tiles
def tile_name(lat: float, lon: float, product: str = "glo30") -> str:
    """Copernicus tile name for the 1 deg tile containing (lat, lon) (SW-corner naming)."""
    p = PRODUCTS[product]
    la, lo = math.floor(lat), math.floor(lon)
    ns = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
    ew = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    return f"Copernicus_DSM_COG_{p.code}_{ns}_00_{ew}_00_DEM"


def tile_url(lat: float, lon: float, product: str = "glo30") -> str:
    name = tile_name(lat, lon, product)
    return f"{PRODUCTS[product].bucket_url}/{name}/{name}.tif"


def tiles_for_bounds(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> list[tuple[int, int]]:
    """South-west corners (lat, lon) of all 1 deg tiles intersecting the box."""
    if lon_max - lon_min > 180 or lon_min < -180 or lon_max > 180:
        raise NotImplementedError("areas crossing the antimeridian are not supported")
    lat_min, lat_max = max(lat_min, -90.0), min(lat_max, 89.999999)
    return [
        (la, lo)
        for la in range(math.floor(lat_min), math.floor(lat_max) + 1)
        for lo in range(math.floor(lon_min), min(math.floor(lon_max), 179) + 1)
    ]


def default_cache_dir() -> Path:
    env = os.environ.get("PLUME_DEM_CACHE")
    if env:
        return Path(env)
    try:
        from plume.config import data_root

        return data_root() / "terrain" / "dem" / "cache"
    except FileNotFoundError:
        return Path("data") / "terrain" / "dem" / "cache"


@dataclass
class TileRef:
    lat: int
    lon: int
    product: str
    name: str
    path: Path | None  # None -> no such tile (ocean): sea level
    factor: int = 1  # overview decimation factor of the cached raster


def choose_overview(product: str, resolution: float) -> int:
    """Coarsest COG overview whose pixel is still <= half the target grid spacing."""
    p = PRODUCTS[product]
    best = 1
    for f in p.overviews:
        if p.pixel_m * f <= 0.5 * resolution + 1e-9:
            best = f
    return best


def _http_status(url: str, timeout: float) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def _remote_check(url: str, timeout: float, retries: int) -> None:
    """Raise ``_TileMissing`` for 404/403, ``DemDownloadError`` if unreachable."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            status = _http_status(url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        else:
            if status == 200:
                return
            if status in (403, 404):  # public bucket without listing: missing key -> 403/404
                raise _TileMissing(url)
            last = RuntimeError(f"HTTP {status}")
        if attempt < retries:
            time.sleep(min(2.0**attempt, 8.0))
    raise DemDownloadError(
        f"could not reach {url} ({last}). Are you offline? Cached tiles live in "
        f"{default_cache_dir()} (set PLUME_DEM_CACHE to use another directory)."
    )


def _download(url: str, dest: Path, timeout: float, retries: int) -> None:
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length") or 0)
                if total > MAX_TILE_BYTES:
                    raise DemDownloadError(f"{url}: refusing {total} byte download")
                n = 0
                with open(part, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        n += len(chunk)
                        if n > MAX_TILE_BYTES:
                            raise DemDownloadError(f"{url}: download exceeds size limit")
                        fh.write(chunk)
                if total and n != total:
                    raise OSError(f"truncated download ({n} of {total} bytes)")
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise _TileMissing(url) from e
            last = e
        except DemDownloadError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        finally:
            part.unlink(missing_ok=True)
        if attempt < retries:
            log.info("retrying %s after %s", url, last)
            time.sleep(min(2.0**attempt, 8.0))
    raise DemDownloadError(
        f"failed to download {url} after {retries + 1} attempts ({last}). "
        "Check your internet connection."
    )


def _validate_raster(path: Path) -> None:
    """Open a downloaded file as a raster and check it looks like a DEM tile."""
    import rasterio

    try:
        with rasterio.open(path) as ds:
            ok = (
                ds.count == 1
                and ds.crs is not None
                and ds.crs.is_geographic
                and 1 <= ds.width <= 20_000
                and 1 <= ds.height <= 20_000
                and np.dtype(ds.dtypes[0]).kind in "fiu"
            )
            if not ok:
                raise ValueError("unexpected raster layout")
    except Exception as e:
        path.unlink(missing_ok=True)
        raise DemDownloadError(f"{path.name} is not a valid DEM tile ({e}); removed") from e


def _gdal_http_env(timeout: float, retries: int) -> dict:
    return {
        "GDAL_HTTP_TIMEOUT": str(math.ceil(timeout)),
        "GDAL_HTTP_CONNECTTIMEOUT": str(math.ceil(min(timeout, 20))),
        "GDAL_HTTP_MAX_RETRY": str(retries),
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_USERAGENT": USER_AGENT,
    }


def _write_overview(src_path: str, dest: Path, factor: int, product: str, env: dict) -> None:
    """Read overview ``factor`` of a (local or /vsicurl/) COG and cache it as a GeoTIFF."""
    import rasterio

    level = PRODUCTS[product].overviews.index(factor)
    part = dest.with_name(dest.name + ".part.tif")
    try:
        with rasterio.Env(**env), rasterio.open(src_path, overview_level=level) as ds:
            if ds.count != 1 or ds.crs is None or not ds.crs.is_geographic:
                raise DemDownloadError(f"{src_path}: unexpected raster layout")
            data = ds.read(1).astype(np.float32)
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "width": ds.width,
                "height": ds.height,
                "crs": ds.crs,
                "transform": ds.transform,
                "compress": "deflate",
                "tiled": False,
            }
        with rasterio.open(part, "w", **profile) as out:
            out.write(data, 1)
        os.replace(part, dest)
    except DemDownloadError:
        raise
    except Exception as e:
        raise DemDownloadError(f"failed to read overview of {src_path}: {e}") from e
    finally:
        part.unlink(missing_ok=True)


def fetch_tile(
    lat: int,
    lon: int,
    product: str = "glo30",
    factor: int = 1,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    fallback: bool = True,
    timeout: float = 60.0,
    retries: int = 3,
) -> TileRef:
    """Make one DEM tile available locally and return a reference to it.

    ``factor`` > 1 caches only that COG overview (much smaller). A tile the server does
    not have is recorded with an empty ``<name>.missing`` marker and returned with
    ``path=None`` (sea level). With ``fallback`` a missing GLO-30 tile is retried from
    GLO-90 before being declared ocean. ``offline`` never touches the network.
    """
    if factor != 1 and factor not in PRODUCTS[product].overviews:
        raise ValueError(f"{product} has no overview with factor {factor}")
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    name = tile_name(lat, lon, product)
    full = cache / f"{name}.tif"
    want = full if factor == 1 else cache / f"{name}_ovr{factor}.tif"
    missing = cache / f"{name}.missing"
    ref = TileRef(lat, lon, product, name, want, factor)

    def _fallback_or_sea() -> TileRef:
        if fallback and product == "glo30":
            f90 = choose_overview("glo90", PRODUCTS["glo30"].pixel_m * factor * 2)
            alt = fetch_tile(
                lat,
                lon,
                "glo90",
                f90,
                cache_dir=cache,
                offline=offline,
                fallback=False,
                timeout=timeout,
                retries=retries,
            )
            if alt.path is not None:
                log.warning("tile %s missing from GLO-30; using %s", name, alt.name)
                return alt
        return TileRef(lat, lon, product, name, None, factor)

    if want.exists():
        return ref
    if missing.exists():
        return _fallback_or_sea()
    if factor != 1 and full.exists():  # derive the overview from the cached full tile
        _write_overview(str(full), want, factor, product, {})
        return ref
    if offline:
        raise DemDownloadError(
            f"tile {name} is not in the cache ({cache}) and offline mode is on; "
            "run `plume terrain fetch` while online first"
        )
    url = tile_url(lat, lon, product)
    try:
        if factor == 1:
            log.info("downloading %s", url)
            _download(url, full, timeout, retries)
            _validate_raster(full)
        else:
            _remote_check(url, timeout=min(timeout, 30.0), retries=retries)
            log.info("reading %dx overview of %s", factor, name)
            _write_overview(
                f"/vsicurl/{url}", want, factor, product, _gdal_http_env(timeout, retries)
            )
            _validate_raster(want)
    except _TileMissing:
        log.info("no tile %s (ocean): sea level", name)
        missing.touch()
        return _fallback_or_sea()
    return ref


def fetch_tiles(
    corners: Sequence[tuple[int, int]],
    product: str = "glo30",
    factor: int = 1,
    *,
    max_workers: int = 6,
    **kwargs,
) -> list[TileRef]:
    """Fetch several tiles concurrently (see :func:`fetch_tile`)."""
    if not corners:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(corners)))) as pool:
        futures = [pool.submit(fetch_tile, la, lo, product, factor, **kwargs) for la, lo in corners]
        return [f.result() for f in futures]


# ============================================================================ mosaic / warp
def mosaic_tiles(
    tiles: Sequence[TileRef], bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, object, float]:
    """Merge tiles into one EPSG:4326 float32 array cropped to ``bounds`` =
    (lon_min, lat_min, lon_max, lat_max). Missing tiles are filled with 0 m (sea).
    Returns ``(array, affine_transform, pixel_size_deg)``."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.transform import from_origin

    present = [t for t in tiles if t.path is not None]
    if not present:  # all ocean
        res = 1.0 / 120
        w, s, e, n = bounds
        nx, ny = max(2, math.ceil((e - w) / res) + 1), max(2, math.ceil((n - s) / res) + 1)
        return np.zeros((ny, nx), np.float32), from_origin(w, n, res, res), res
    datasets = [rasterio.open(t.path) for t in present]
    try:
        res = min(min(abs(ds.res[0]), abs(ds.res[1])) for ds in datasets)
        # snap the crop outward onto the source pixel grid so merging copies pixels
        # exactly (an unaligned crop would resample and shift the data by < 1 pixel)
        ref = datasets[0].bounds
        w, s, e, n = bounds
        w = ref.left + math.floor((w - ref.left) / res) * res
        e = ref.left + math.ceil((e - ref.left) / res) * res
        s = ref.top + math.floor((s - ref.top) / res) * res
        n = ref.top + math.ceil((n - ref.top) / res) * res
        cells = ((e - w) / res) * ((n - s) / res)
        if cells > MAX_MOSAIC_CELLS:
            raise ValueError(
                f"mosaic would need {cells / 1e6:.0f} M cells; use a coarser resolution "
                "(a COG overview is then read automatically) or a smaller area"
            )
        same_grid = all(
            math.isclose(abs(ds.res[0]), res, rel_tol=1e-9)
            and math.isclose(abs(ds.res[1]), res, rel_tol=1e-9)
            for ds in datasets
        )
        arr, transform = merge(
            datasets,
            bounds=(w, s, e, n),
            res=res,
            nodata=np.nan,
            dtype="float32",
            resampling=Resampling.nearest if same_grid else Resampling.bilinear,
        )
    finally:
        for ds in datasets:
            ds.close()
    arr = arr[0]
    arr[~np.isfinite(arr)] = 0.0  # ocean / void -> sea level
    return arr, transform, res


@dataclass
class _Grid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)

    def gdal_transform(self):
        from rasterio.transform import Affine

        # GDAL rasters are cell-centred: node (i, j) of the heightmap is the centre of
        # pixel (i, ny-1-j) of a north-up raster.
        return Affine(
            self.dx, 0.0, self.x_min - self.dx / 2, 0.0, -self.dy, self.y_max + self.dy / 2
        )

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.nx),
            np.linspace(self.y_min, self.y_max, self.ny),
        )

    def perimeter(self, n: int = 65) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(self.x_min, self.x_max, n)
        ys = np.linspace(self.y_min, self.y_max, n)
        px = np.concatenate([xs, xs, np.full(n, self.x_min), np.full(n, self.x_max)])
        py = np.concatenate([np.full(n, self.y_min), np.full(n, self.y_max), ys, ys])
        return px, py


def _make_grid(x_range, y_range, resolution: float) -> _Grid:
    x0, x1 = map(float, x_range)
    y0, y1 = map(float, y_range)
    if not (x1 > x0 and y1 > y0 and resolution > 0):
        raise ValueError("need x_range/y_range with max > min and resolution > 0")
    nx = max(2, round((x1 - x0) / resolution) + 1)
    ny = max(2, round((y1 - y0) / resolution) + 1)
    return _Grid(x0, x1, y0, y1, nx, ny)


def _warp(
    source, src_transform, src_crs, src_nodata, proj: SiteProjection, grid: _Grid, resampling
):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.full((grid.ny, grid.nx), np.nan, dtype=np.float32)
    kwargs = {}
    if src_transform is not None:
        kwargs.update(src_transform=src_transform, src_crs=src_crs)
    reproject(
        source=source,
        destination=dst,
        src_nodata=src_nodata,
        dst_transform=grid.gdal_transform(),
        dst_crs=proj.crs,
        dst_nodata=np.nan,
        resampling=Resampling[resampling],
        tolerance=0.0,  # exact per-pixel transform (default approximates to 1/8 px)
        **kwargs,
    )
    return dst[::-1].astype(np.float64)  # row 0 = south


def _latlon_bounds(proj: SiteProjection, grid: _Grid, margin_m: float):
    # lat/lon extremes of a projected rectangle lie on its boundary (away from the poles)
    lat, lon = proj.inverse(*grid.perimeter())
    dlat = margin_m / 111_000.0
    dlon = dlat / max(math.cos(math.radians(float(np.abs(lat).max()) + dlat)), 0.05)
    return (
        float(lat.min()) - dlat,
        float(lat.max()) + dlat,
        float(lon.min()) - dlon,
        float(lon.max()) + dlon,
    )


GeoidFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def build_heightmap(
    lat0: float,
    lon0: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    *,
    product: str = "glo30",
    overview: int | None = None,
    fallback: bool = True,
    cache_dir: Path | None = None,
    offline: bool = False,
    resampling: str = "bilinear",
    geoid: GeoidFn | None = None,
    name: str = "dem",
    timeout: float = 60.0,
    retries: int = 3,
    max_workers: int = 6,
) -> Heightmap:
    """Real terrain on a regular grid in the azimuthal-equidistant projection centred on
    (``lat0``, ``lon0``).

    ``x_range``/``y_range`` are map extents (m, east/north of the centre) and
    ``resolution`` the grid spacing (m). DEM tiles covering the area are fetched (or read
    from the cache), mosaicked and bilinearly resampled. ``overview`` forces a COG
    decimation factor (default: chosen from ``resolution``). ``geoid(lat, lon)`` -- if
    given -- returns a per-node vertical offset (m) added to the orthometric heights
    (e.g. the geoid undulation N to obtain ellipsoidal heights).
    """
    if product not in PRODUCTS:
        raise ValueError(f"unknown DEM product {product!r} (choose from {sorted(PRODUCTS)})")
    proj = site_projection(lat0, lon0)
    grid = _make_grid(x_range, y_range, resolution)
    factor = overview if overview is not None else choose_overview(product, resolution)
    # margin: a few source pixels + the (widened, when downsampling) bilinear kernel
    margin_m = 3 * PRODUCTS[product].pixel_m * factor + 2 * max(grid.dx, grid.dy)
    lat_min, lat_max, lon_min, lon_max = _latlon_bounds(proj, grid, margin_m)
    corners = tiles_for_bounds(lat_min, lat_max, lon_min, lon_max)
    log.info(
        "%s: %d tile(s), %dx overview, grid %dx%d @ %.1f m",
        PRODUCTS[product].title,
        len(corners),
        factor,
        grid.nx,
        grid.ny,
        grid.dx,
    )
    tiles = fetch_tiles(
        corners,
        product,
        factor,
        max_workers=max_workers,
        cache_dir=cache_dir,
        offline=offline,
        fallback=fallback,
        timeout=timeout,
        retries=retries,
    )
    arr, transform, res = mosaic_tiles(tiles, (lon_min, lat_min, lon_max, lat_max))
    heights = _warp(arr, transform, "EPSG:4326", None, proj, grid, resampling)
    bad = ~np.isfinite(heights)
    if bad.any():  # should not happen: the mosaic margin covers the warp kernel
        log.warning("%d grid nodes without DEM coverage; filled from neighbours", bad.sum())
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        heights = heights[tuple(idx)]
    vertical = VERTICAL_DATUM
    if geoid is not None:
        lat, lon = proj.inverse(*grid.nodes())
        heights = heights + np.asarray(geoid(lat, lon), dtype=float)
        vertical = f"{VERTICAL_DATUM} + user geoid hook"
    products_used = sorted({t.product for t in tiles if t.path is not None}) or [product]
    meta = {
        "source": " + ".join(PRODUCTS[p].title for p in products_used),
        "product": product,
        "tiles": [t.name for t in tiles if t.path is not None],
        "sea_level_tiles": [t.name for t in tiles if t.path is None],
        "overview_factor": int(factor),
        "source_resolution_m": round(PRODUCTS[product].pixel_m * factor, 2),
        "source_pixel_deg": float(res),
        "resolution_m": float(grid.dx),
        "resampling": resampling,
        "projection": proj.proj4,
        "origin_lat": proj.lat0,
        "origin_lon": proj.lon0,
        "horizontal_datum": HORIZONTAL_DATUM,
        "vertical_datum": vertical,
        "synthetic_detail": False,
        "attribution": COPERNICUS_ATTRIBUTION,
        "licence": COPERNICUS_LICENCE,
    }
    return Heightmap(heights, grid.x_min, grid.x_max, grid.y_min, grid.y_max, name, meta)


def from_geotiff(
    path: str | Path,
    lat0: float,
    lon0: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    *,
    resampling: str = "bilinear",
    fill_value: float | None = None,
    vertical_offset: float = 0.0,
    name: str | None = None,
    attribution: str | None = None,
) -> Heightmap:
    """Reproject a user-supplied elevation GeoTIFF (LiDAR, drone photogrammetry, any CRS)
    into the site projection. Nodes not covered by valid data raise ``ValueError``
    unless ``fill_value`` is given. No synthetic detail is added."""
    import rasterio

    path = Path(path)
    proj = site_projection(lat0, lon0)
    grid = _make_grid(x_range, y_range, resolution)
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"{path} has no coordinate reference system")
        src_crs = ds.crs.to_string()
        heights = _warp(rasterio.band(ds, 1), None, None, ds.nodata, proj, grid, resampling)
    bad = ~np.isfinite(heights)
    if bad.any():
        if fill_value is None:
            raise ValueError(
                f"{path.name} covers only {100 * (1 - bad.mean()):.1f}% of the requested "
                "grid; shrink the area or pass fill_value"
            )
        heights[bad] = fill_value
    heights = heights + vertical_offset
    meta = {
        "source": f"user GeoTIFF {path.name}",
        "source_crs": src_crs,
        "resolution_m": float(grid.dx),
        "resampling": resampling,
        "projection": proj.proj4,
        "origin_lat": proj.lat0,
        "origin_lon": proj.lon0,
        "horizontal_datum": HORIZONTAL_DATUM,
        "vertical_offset_m": float(vertical_offset),
        "filled_fraction": float(bad.mean()),
        "synthetic_detail": False,
    }
    if attribution:
        meta["attribution"] = attribution
    return Heightmap(
        heights, grid.x_min, grid.x_max, grid.y_min, grid.y_max, name or path.stem, meta
    )


def corridor_heightmap(
    site_a: tuple[float, float],
    site_b: tuple[float, float],
    width: float,
    resolution: float,
    *,
    max_cells: int = 4096,
    name: str = "route",
    **kwargs,
) -> Heightmap:
    """Terrain for a route A -> B (lat/lon, degrees) in the projection centred on A: the
    projected bounding box of both sites padded by ``width`` (m). The resolution is
    coarsened automatically so neither grid dimension exceeds ``max_cells``."""
    proj = site_projection(*site_a)
    xb, yb = proj.forward(*site_b)
    x_range = (min(0.0, xb) - width, max(0.0, xb) + width)
    y_range = (min(0.0, yb) - width, max(0.0, yb) + width)
    span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    if span / resolution + 1 > max_cells:
        coarse = math.ceil(span / (max_cells - 1))
        log.warning(
            "corridor: resolution %.1f m -> %.0f m to keep the grid <= %d cells/side",
            resolution,
            coarse,
            max_cells,
        )
        resolution = float(coarse)
    # snap the extent to the grid so the map origin (site A) lies on a node
    x_range = (
        math.floor(x_range[0] / resolution) * resolution,
        math.ceil(x_range[1] / resolution) * resolution,
    )
    y_range = (
        math.floor(y_range[0] / resolution) * resolution,
        math.ceil(y_range[1] / resolution) * resolution,
    )
    hm = build_heightmap(*site_a, x_range, y_range, resolution, name=name, **kwargs)
    dist, az = geodesic_inverse(*site_a, *site_b)
    hm.meta.update(
        site_a_latlon=[float(site_a[0]), float(site_a[1])],
        site_b_latlon=[float(site_b[0]), float(site_b[1])],
        site_a=[0.0, 0.0],
        site_b=[round(float(xb), 3), round(float(yb), 3)],
        route_length_m=round(dist, 1),
        route_azimuth_deg=round(az, 3),
    )
    return hm


# ============================================================================ detail
def band_limited_noise(
    shape: tuple[int, int], spacing: float, cutoff: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean, unit-RMS 1/f^beta noise containing only wavelengths < ``cutoff`` (m)."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny, d=spacing)[:, None]
    fx = np.fft.rfftfreq(nx, d=spacing)[None, :]
    f = np.hypot(fx, fy)
    fc = 1.0 / cutoff
    t = np.clip((f - 0.5 * fc) / (0.5 * fc), 0.0, 1.0)
    amp = np.where(f > 0, np.maximum(f, 1e-12) ** (-beta / 2.0), 0.0) * t * t * (3 - 2 * t)
    phase = rng.uniform(0, 2 * np.pi, amp.shape)
    out = np.fft.irfft2(amp * np.exp(1j * phase), s=shape)
    out -= out.mean()
    return out / (out.std() + 1e-12)


def add_synthetic_detail(
    hm: Heightmap,
    *,
    seed: int = 0,
    roughness: float = 0.15,
    cutoff_wavelength: float = 90.0,
    beta: float = 2.4,
    rock_density: float = 1.0e-4,
    rock_height: tuple[float, float] = (0.1, 0.6),
    rock_radius: tuple[float, float] = (0.6, 2.0),
) -> Heightmap:
    """Add procedural sub-DEM-resolution detail and FLAG it in ``meta``.

    ``roughness`` is the RMS (m) of band-limited undulation restricted to wavelengths
    shorter than ``cutoff_wavelength`` (default ~3 GLO-30 postings, i.e. what the DEM
    cannot resolve), so the measured large-scale shape is preserved. Rocks are Gaussian
    bumps (``rock_density`` per m^2, height / radius ranges in m).
    """
    rng = np.random.default_rng(seed)
    ny, nx = hm.shape
    dx, dy = hm.dx, hm.dy
    h = hm.heights.copy()
    if roughness > 0:
        h += roughness * band_limited_noise((ny, nx), dx, cutoff_wavelength, beta, rng)
    area = (hm.x_max - hm.x_min) * (hm.y_max - hm.y_min)
    n_rocks = int(rng.poisson(rock_density * area)) if rock_density > 0 else 0
    for _ in range(n_rocks):
        rx = rng.uniform(hm.x_min, hm.x_max)
        ry = rng.uniform(hm.y_min, hm.y_max)
        r = rng.uniform(*rock_radius)
        hr = rng.uniform(*rock_height)
        sig = r / 2.0
        i0, i1 = (
            max(0, int((rx - 3 * sig - hm.x_min) / dx)),
            min(nx, int((rx + 3 * sig - hm.x_min) / dx) + 2),
        )
        j0, j1 = (
            max(0, int((ry - 3 * sig - hm.y_min) / dy)),
            min(ny, int((ry + 3 * sig - hm.y_min) / dy) + 2),
        )
        if i1 <= i0 or j1 <= j0:
            continue
        X = hm.x_min + np.arange(i0, i1) * dx
        Y = hm.y_min + np.arange(j0, j1) * dy
        d2 = (X[None, :] - rx) ** 2 + (Y[:, None] - ry) ** 2
        h[j0:j1, i0:i1] += hr * np.exp(-d2 / (2 * sig * sig))
    meta = dict(hm.meta)
    meta["synthetic_detail"] = True
    meta["synthetic_detail_params"] = {
        "generator": "plume.terrain.dem.add_synthetic_detail",
        "seed": int(seed),
        "roughness_rms_m": float(roughness),
        "cutoff_wavelength_m": float(cutoff_wavelength),
        "spectral_beta": float(beta),
        "rock_density_per_m2": float(rock_density),
        "rock_count": n_rocks,
        "rock_height_m": [float(v) for v in rock_height],
        "rock_radius_m": [float(v) for v in rock_radius],
        "note": "procedural detail below DEM resolution -- NOT measured terrain",
    }
    return Heightmap(h, hm.x_min, hm.x_max, hm.y_min, hm.y_max, hm.name, meta)


def landing_tile(
    site: tuple[float, float],
    half_size: float = 1000.0,
    resolution: float = 2.5,
    *,
    origin: tuple[float, float] | None = None,
    synthetic_detail: bool = True,
    seed: int = 0,
    roughness: float = 0.15,
    cutoff_wavelength: float | None = None,
    rock_density: float = 1.0e-4,
    rock_height: tuple[float, float] = (0.1, 0.6),
    rock_radius: tuple[float, float] = (0.6, 2.0),
    geotiff: str | Path | None = None,
    name: str = "landing_tile",
    **kwargs,
) -> Heightmap:
    """High-resolution square tile (``half_size`` m around ``site`` = (lat, lon)).

    Coordinates are in the projection centred on ``origin`` (default: the site itself),
    so a landing tile at B uses ``origin=site_a``. Without ``geotiff`` the Copernicus DEM
    is upsampled bilinearly; 30 m data cannot resolve rocks or metre-scale undulation, so
    ``synthetic_detail`` adds flagged procedural detail (see
    :func:`add_synthetic_detail`). With ``geotiff`` (LiDAR / drone survey) the survey is
    reprojected instead and no synthetic detail is added.
    """
    origin = origin or site
    proj = site_projection(*origin)
    cx, cy = proj.forward(*site)
    x_range = (cx - half_size, cx + half_size)
    y_range = (cy - half_size, cy + half_size)
    if geotiff is not None:
        hm = from_geotiff(geotiff, *origin, x_range, y_range, resolution, name=name)
    else:
        hm = build_heightmap(*origin, x_range, y_range, resolution, name=name, **kwargs)
        if synthetic_detail:
            src_res = float(hm.meta.get("source_resolution_m", 30.0))
            hm = add_synthetic_detail(
                hm,
                seed=seed,
                roughness=roughness,
                cutoff_wavelength=cutoff_wavelength or 3.0 * src_res,
                rock_density=rock_density,
                rock_height=rock_height,
                rock_radius=rock_radius,
            )
    hm.meta.update(site_latlon=[float(site[0]), float(site[1])], site_xy=[float(cx), float(cy)])
    return hm


# ============================================================================ hazards
@dataclass
class HazardMap:
    """Landing-hazard assessment on a heightmap grid (arrays use ``heights`` layout)."""

    heightmap: Heightmap
    slope_deg: np.ndarray  # slope of the plane fitted over the footprint
    roughness: np.ndarray  # RMS residual (m) about that plane
    safe: np.ndarray  # bool
    footprint_radius: float
    max_slope_deg: float
    max_roughness: float
    valid: np.ndarray = field(repr=False, default=None)  # footprint fully inside the map

    @property
    def safe_fraction(self) -> float:
        return float(self.safe[self.valid].mean()) if self.valid.any() else 0.0

    def nearest_safe(
        self, x: float, y: float, clearance: float = 0.0
    ) -> tuple[float, float, float] | None:
        """Nearest safe node to (x, y): ``(x, y, distance)``, or ``None``. ``clearance``
        (m) additionally requires every node within that distance to be safe."""
        hm = self.heightmap
        ok = self.safe
        if clearance > 0:
            dist_px = ndimage.distance_transform_edt(ok, sampling=(hm.dy, hm.dx))
            ok = ok & (dist_px > clearance)
        jj, ii = np.nonzero(ok)
        if jj.size == 0:
            return None
        xs = hm.x_min + ii * hm.dx
        ys = hm.y_min + jj * hm.dy
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        k = int(np.argmin(d2))
        return float(xs[k]), float(ys[k]), float(math.sqrt(d2[k]))

    def is_safe(self, x: float, y: float) -> bool:
        hm = self.heightmap
        i = round((x - hm.x_min) / hm.dx)
        j = round((y - hm.y_min) / hm.dy)
        if not (0 <= i < hm.shape[1] and 0 <= j < hm.shape[0]):
            return False
        return bool(self.safe[j, i])


def hazard_map(
    heightmap: Heightmap,
    footprint_radius: float = 3.0,
    max_slope_deg: float = 8.0,
    max_roughness: float = 0.1,
) -> HazardMap:
    """Slope and roughness over a circular landing footprint at every node.

    A plane ``z = a + b x + c y`` is least-squares fitted over the footprint disk
    (vectorised with ``scipy.ndimage.correlate``): slope = ``atan(|(b, c)|)`` and
    roughness = RMS of the detrended heights. ``safe`` requires both under their limits
    and the whole footprint inside the map.
    """
    h = heightmap.heights
    dx, dy = heightmap.dx, heightmap.dy
    r = max(footprint_radius, dx, dy)  # at least the 4-neighbours
    ri, rj = math.floor(r / dx + 1e-9), math.floor(r / dy + 1e-9)
    ox = np.arange(-ri, ri + 1)[None, :] * dx
    oy = np.arange(-rj, rj + 1)[:, None] * dy
    w = ((ox**2 + oy**2) <= r * r + 1e-9).astype(float)
    w /= w.sum()
    sxx = float((w * ox**2).sum())
    syy = float((w * oy**2).sum())
    z = h - float(h.mean())
    mean = ndimage.correlate(z, w, mode="nearest")
    b = ndimage.correlate(z, w * ox, mode="nearest") / sxx
    c = ndimage.correlate(z, w * oy, mode="nearest") / syy
    m2 = ndimage.correlate(z * z, w, mode="nearest")
    var = m2 - mean**2 - b * b * sxx - c * c * syy
    rough = np.sqrt(np.clip(var, 0.0, None))
    slope = np.degrees(np.arctan(np.hypot(b, c)))
    valid = np.zeros(h.shape, bool)
    valid[rj : h.shape[0] - rj, ri : h.shape[1] - ri] = True
    safe = valid & (slope <= max_slope_deg) & (rough <= max_roughness) & np.isfinite(h)
    return HazardMap(
        heightmap, slope, rough, safe, footprint_radius, max_slope_deg, max_roughness, valid
    )


__all__ = [
    "COPERNICUS_ATTRIBUTION",
    "PRODUCTS",
    "DemDownloadError",
    "HazardMap",
    "SiteProjection",
    "TileRef",
    "add_synthetic_detail",
    "build_heightmap",
    "choose_overview",
    "corridor_heightmap",
    "default_cache_dir",
    "fetch_tile",
    "fetch_tiles",
    "from_geotiff",
    "geodesic_inverse",
    "hazard_map",
    "landing_tile",
    "mosaic_tiles",
    "site_projection",
    "tile_name",
    "tile_url",
    "tiles_for_bounds",
]
