"""Real-world terrain: projection, DEM tiles, mosaicking/resampling, hazards."""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from plume.terrain import Heightmap, dem  # noqa: E402

# Register the ``network`` marker until it is listed in pyproject.toml (no-op afterwards).
_cfg = getattr(pytest.mark, "_config", None)
if _cfg is not None and not any(m.startswith("network") for m in _cfg.getini("markers")):
    _cfg.addinivalue_line("markers", "network: needs internet access (skipped offline)")

REPO = Path(__file__).resolve().parents[1]
TERRAIN = REPO / "data" / "terrain"
SPACEPORT = (32.990, -106.986)  # real_hop Pad A
BURNS_FLAT = (35.335, -99.225)  # real_hop Site B


def _backends() -> list[str]:
    out = ["rasterio", "sphere"]
    try:
        import pyproj  # noqa: F401

        out.append("pyproj")
    except ImportError:
        pass
    return out


def _online() -> bool:
    try:
        req = urllib.request.Request(dem.tile_url(32, -107, "glo90"), method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# ----------------------------------------------------------------------------- projection
@pytest.mark.parametrize("backend", _backends())
def test_projection_round_trip_sub_mm(backend):
    proj = dem.site_projection(*SPACEPORT, backend=backend)
    rng = np.random.default_rng(0)
    lat = SPACEPORT[0] + rng.uniform(-8, 8, 500)
    lon = SPACEPORT[1] + rng.uniform(-10, 10, 500)
    x, y = proj.forward(lat, lon)
    lat2, lon2 = proj.inverse(x, y)
    x2, y2 = proj.forward(lat2, lon2)
    assert np.max(np.hypot(x2 - x, y2 - y)) < 1e-3
    # metres of latitude / longitude error
    assert np.max(np.abs(lat2 - lat)) * 111_000 < 1e-3
    assert np.max(np.abs((lon2 - lon) * np.cos(np.radians(lat)))) * 111_000 < 1e-3
    # centre maps to the origin, scalars stay scalars
    x0, y0 = proj.forward(*SPACEPORT)
    assert isinstance(x0, float) and abs(x0) < 1e-6 and abs(y0) < 1e-6


def test_aeqd_distance_matches_geodesic_at_750km():
    proj = dem.site_projection(*SPACEPORT)
    assert proj.backend in ("pyproj", "rasterio")  # ellipsoidal
    x, y = proj.forward(*BURNS_FLAT)
    try:
        from pyproj import Geod

        az, _, dist = Geod(ellps="WGS84").inv(
            SPACEPORT[1], SPACEPORT[0], BURNS_FLAT[1], BURNS_FLAT[0]
        )
        az %= 360.0
    except ImportError:
        dist, az = dem.geodesic_inverse(*SPACEPORT, *BURNS_FLAT)
    assert 740e3 < dist < 780e3
    assert abs(math.hypot(x, y) - dist) / dist < 1e-3
    assert abs(math.hypot(x, y) - dist) < 0.01  # actually exact for aeqd radials
    assert abs(math.degrees(math.atan2(x, y)) % 360.0 - az) < 1e-6
    # every radial: projected range == geodesic distance
    for lat, lon in [(36.0, -107.5), (30.0, -100.0), (33.5, -114.0), (28.0, -106.0)]:
        x, y = proj.forward(lat, lon)
        d, _ = dem.geodesic_inverse(*SPACEPORT, lat, lon)
        assert abs(math.hypot(x, y) - d) / d < 1e-6


def test_vincenty_reference_distance():
    # Flinders Peak -> Buninyong (Vincenty 1975 test line): 54 972.271 m
    d, az = dem.geodesic_inverse(
        -(37 + 57 / 60 + 3.72030 / 3600),
        144 + 25 / 60 + 29.52440 / 3600,
        -(37 + 39 / 60 + 10.15610 / 3600),
        143 + 55 / 60 + 35.38390 / 3600,
    )
    assert abs(d - 54972.271) < 0.01
    assert abs(az - (306 + 52 / 60 + 5.37 / 3600)) < 1e-3


def test_sphere_fallback_close_to_ellipsoid():
    e = dem.site_projection(*SPACEPORT, backend="rasterio")
    s = dem.site_projection(*SPACEPORT, backend="sphere")
    xe, ye = e.forward(*BURNS_FLAT)
    xs, ys = s.forward(*BURNS_FLAT)
    assert math.hypot(xs - xe, ys - ye) / math.hypot(xe, ye) < 5e-3


# ----------------------------------------------------------------------------- tiles
def test_tile_naming_and_urls():
    assert dem.tile_name(32.99, -106.975) == "Copernicus_DSM_COG_10_N32_00_W107_00_DEM"
    assert dem.tile_name(35.335, -99.225) == "Copernicus_DSM_COG_10_N35_00_W100_00_DEM"
    assert dem.tile_name(-0.5, -0.5) == "Copernicus_DSM_COG_10_S01_00_W001_00_DEM"
    assert dem.tile_name(0.2, 0.3) == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"
    assert dem.tile_name(-33.9, 151.2, "glo90") == "Copernicus_DSM_COG_30_S34_00_E151_00_DEM"
    assert dem.tile_url(32.5, -106.5) == (
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N32_00_W107_00_DEM/"
        "Copernicus_DSM_COG_10_N32_00_W107_00_DEM.tif"
    )
    assert dem.tile_url(32.5, -106.5, "glo90").startswith(
        "https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N32_00_W107_00_DEM/"
    )
    assert dem.tiles_for_bounds(32.2, 33.4, -107.3, -106.1) == [
        (32, -108),
        (32, -107),
        (33, -108),
        (33, -107),
    ]
    assert dem.choose_overview("glo30", 2.5) == 1
    assert dem.choose_overview("glo30", 500.0) == 8
    assert dem.choose_overview("glo30", 150.0) == 2
    assert dem.choose_overview("glo90", 1000.0) == 4


# ----------------------------------------------------------------------------- synthetic DEM
def _analytic(lat, lon):
    return 1000.0 + 400.0 * np.sin(2 * np.pi * (lon - 20.0) / 0.8) * np.cos(
        2 * np.pi * (lat - 10.0) / 0.6
    )


def _write_tile(cache: Path, lat: int, lon: int, ppd: int = 360) -> Path:
    """A Copernicus-style 1 deg tile (pixel-is-point grid incl. the north edge)."""
    res = 1.0 / ppd
    lats = lat + 1 - np.arange(ppd) * res
    lons = lon + np.arange(ppd) * res
    data = _analytic(lats[:, None], lons[None, :]).astype(np.float32)
    path = cache / f"{dem.tile_name(lat, lon)}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=ppd,
        height=ppd,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(lon - res / 2, lat + 1 + res / 2, res, res),
    ) as ds:
        ds.write(data, 1)
        ds.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
    return path


@pytest.fixture
def synthetic_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for lat in (10, 11):
        for lon in (20, 21):
            _write_tile(cache, lat, lon)
    return cache


def _analytic_on(hm: Heightmap, lat0, lon0):
    proj = dem.site_projection(lat0, lon0)
    X, Y = np.meshgrid(
        np.linspace(hm.x_min, hm.x_max, hm.shape[1]), np.linspace(hm.y_min, hm.y_max, hm.shape[0])
    )
    lat, lon = proj.inverse(X, Y)
    return _analytic(lat, lon)


def test_mosaic_and_resample_match_analytic_surface(synthetic_cache):
    # centred on a 4-tile corner so every seam is crossed
    hm = dem.build_heightmap(
        11.0,
        21.0,
        (-40e3, 40e3),
        (-40e3, 40e3),
        1000.0,
        overview=1,
        cache_dir=synthetic_cache,
        offline=True,
        fallback=False,
    )
    assert hm.shape == (81, 81)
    assert len(hm.meta["tiles"]) == 4 and hm.meta["synthetic_detail"] is False
    assert hm.meta["attribution"] == dem.COPERNICUS_ATTRIBUTION
    err = np.abs(hm.heights - _analytic_on(hm, 11.0, 21.0))
    relief = 800.0
    assert err.max() < 0.005 * relief
    # fine grid (upsampling) as well
    fine = dem.build_heightmap(
        10.5, 20.5, (-3e3, 3e3), (-2e3, 2e3), 50.0, cache_dir=synthetic_cache, offline=True
    )
    assert np.abs(fine.heights - _analytic_on(fine, 10.5, 20.5)).max() < 0.005 * relief
    # a COG overview derived from the cached full tile
    coarse = dem.build_heightmap(
        11.0,
        21.0,
        (-30e3, 30e3),
        (-30e3, 30e3),
        2000.0,
        overview=2,
        cache_dir=synthetic_cache,
        offline=True,
    )
    assert coarse.meta["overview_factor"] == 2
    assert (synthetic_cache / f"{dem.tile_name(10, 20)}_ovr2.tif").exists()
    assert np.abs(coarse.heights - _analytic_on(coarse, 11.0, 21.0)).max() < 0.005 * relief


def test_missing_tiles_are_sea_level_and_offline_errors(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    _write_tile(cache, 10, 20)
    (cache / f"{dem.tile_name(10, 21)}.missing").touch()  # ocean
    hm = dem.build_heightmap(
        10.5,
        21.0,
        (-20e3, 20e3),
        (-10e3, 10e3),
        500.0,
        overview=1,
        cache_dir=cache,
        offline=True,
        fallback=False,
    )
    assert hm.meta["sea_level_tiles"] == [dem.tile_name(10, 21)]
    west = hm.heights[:, : hm.shape[1] // 2 - 4]
    east = hm.heights[:, hm.shape[1] // 2 + 4 :]
    assert np.all(east == 0.0) and west.min() > 500.0
    with pytest.raises(dem.DemDownloadError, match="offline"):
        dem.build_heightmap(
            40.5, 21.0, (-1e3, 1e3), (-1e3, 1e3), 100.0, cache_dir=cache, offline=True
        )


def test_corridor_coarsens_and_records_route(synthetic_cache):
    hm = dem.corridor_heightmap(
        (10.2, 20.2),
        (11.6, 21.7),
        5e3,
        100.0,
        max_cells=400,
        overview=1,
        cache_dir=synthetic_cache,
        offline=True,
    )
    assert max(hm.shape) <= 400
    assert hm.meta["resolution_m"] > 100.0
    xb, yb = hm.meta["site_b"]
    d, _ = dem.geodesic_inverse(10.2, 20.2, 11.6, 21.7)
    assert abs(math.hypot(xb, yb) - d) < 0.01 and abs(hm.meta["route_length_m"] - d) < 0.1
    assert hm.contains(0.0, 0.0) and hm.contains(xb, yb)


def test_from_geotiff_reprojects_utm_survey(tmp_path):
    # a "drone survey" in UTM 33N, 20 m posting, analytic surface in easting/northing
    e0, n0, px, n = 500_000.0, 5_000_000.0, 20.0, 1000

    def surf(e, nn):
        return 300.0 + 50.0 * np.sin(2 * np.pi * e / 5000.0) * np.cos(2 * np.pi * nn / 7000.0)

    ee = e0 + (np.arange(n) + 0.5) * px
    nn = n0 + n * px - (np.arange(n) + 0.5) * px
    path = tmp_path / "survey.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(e0, n0 + n * px, px, px),
        nodata=-9999.0,
    ) as ds:
        ds.write(surf(ee[None, :], nn[:, None]).astype(np.float32), 1)
    lon_c, lat_c = warp_transform("EPSG:32633", "EPSG:4326", [e0 + 10e3], [n0 + 10e3])
    hm = dem.from_geotiff(path, lat_c[0], lon_c[0], (-4e3, 4e3), (-4e3, 4e3), 25.0)
    assert hm.meta["synthetic_detail"] is False and "32633" in hm.meta["source_crs"]
    proj = dem.site_projection(lat_c[0], lon_c[0])
    X, Y = np.meshgrid(
        np.linspace(hm.x_min, hm.x_max, hm.shape[1]), np.linspace(hm.y_min, hm.y_max, hm.shape[0])
    )
    lat, lon = proj.inverse(X, Y)
    e, nn2 = warp_transform("EPSG:4326", "EPSG:32633", lon.ravel(), lat.ravel())
    truth = surf(np.reshape(e, X.shape), np.reshape(nn2, X.shape))
    assert np.abs(hm.heights - truth).max() < 0.005 * 100.0
    with pytest.raises(ValueError, match="covers only"):
        dem.from_geotiff(path, lat_c[0], lon_c[0], (-15e3, 15e3), (-1e3, 1e3), 100.0)
    filled = dem.from_geotiff(
        path, lat_c[0], lon_c[0], (-15e3, 15e3), (-1e3, 1e3), 100.0, fill_value=0.0
    )
    assert 0 < filled.meta["filled_fraction"] < 1
    # a landing tile from the survey carries no synthetic detail
    lz = dem.landing_tile((lat_c[0], lon_c[0]), 500.0, 5.0, geotiff=path, synthetic_detail=True)
    assert lz.meta["synthetic_detail"] is False


# ----------------------------------------------------------------------------- detail
def test_synthetic_detail_is_flagged_and_band_limited(synthetic_cache, tmp_path):
    base = dem.landing_tile(
        (10.5, 20.5), 300.0, 2.5, synthetic_detail=False, cache_dir=synthetic_cache, offline=True
    )
    assert base.meta["synthetic_detail"] is False
    tile = dem.landing_tile(
        (10.5, 20.5), 300.0, 2.5, seed=3, roughness=0.2, cache_dir=synthetic_cache, offline=True
    )
    p = tile.meta["synthetic_detail_params"]
    assert tile.meta["synthetic_detail"] is True and p["seed"] == 3 and p["rock_count"] > 0
    added = tile.heights - base.heights
    assert 0.15 < added.std() < 0.6
    # band-limited: a 3 x cutoff box average of the noise is ~0 (large scales untouched)
    k = int(3 * p["cutoff_wavelength_m"] / 2.5)
    smooth = dem.ndimage.uniform_filter(added, k)[k:-k, k:-k]
    assert np.abs(smooth).max() < 0.1
    # deterministic, and the flag survives save/load
    again = dem.landing_tile(
        (10.5, 20.5), 300.0, 2.5, seed=3, roughness=0.2, cache_dir=synthetic_cache, offline=True
    )
    np.testing.assert_array_equal(again.heights, tile.heights)
    loaded = Heightmap.load(tile.save(tmp_path / "lz.yaml", fmt="npy"))
    assert loaded.meta["synthetic_detail"] is True
    assert loaded.meta["synthetic_detail_params"]["seed"] == 3


# ----------------------------------------------------------------------------- hazards
def _plane(slope_deg: float, n: int = 81, d: float = 1.0) -> Heightmap:
    xs = np.arange(n) * d
    h = np.tan(np.radians(slope_deg)) * xs[None, :] + 0.0 * xs[:, None]
    return Heightmap(h, 0.0, xs[-1], 0.0, xs[-1])


def test_hazard_slope_threshold_on_planes():
    gentle = dem.hazard_map(_plane(5.0), footprint_radius=3.0, max_slope_deg=8.0)
    steep = dem.hazard_map(_plane(10.0), footprint_radius=3.0, max_slope_deg=8.0)
    v = gentle.valid
    assert np.allclose(gentle.slope_deg[v], 5.0, atol=1e-6)
    assert np.allclose(gentle.roughness[v], 0.0, atol=1e-6)
    assert gentle.safe[v].all() and not gentle.safe[~v].any()
    assert not steep.safe.any() and np.allclose(steep.slope_deg[v], 10.0, atol=1e-6)


def test_hazard_kink_rock_and_nearest_safe():
    n, d = 121, 1.0
    xs = np.arange(n) * d - 60.0
    X, Y = np.meshgrid(xs, xs)
    h = np.where(X < 0, X * np.tan(np.radians(4.0)), X * np.tan(np.radians(15.0)))
    h = h + 0.8 * np.exp(-((X + 30) ** 2 + (Y - 20) ** 2) / (2 * 0.8**2))  # a boulder
    hm = Heightmap(h, xs[0], xs[-1], xs[0], xs[-1])
    hz = dem.hazard_map(hm, footprint_radius=3.0, max_slope_deg=8.0, max_roughness=0.1)
    row = hz.safe[60]
    assert row[(xs < -4) & hz.valid[60]].all()
    assert not row[xs > 4].any()
    assert not hz.is_safe(-30.0, 20.0)  # on the boulder
    assert hz.roughness[80, 30] > 0.1
    assert hz.is_safe(-30.0, -20.0)
    # nearest safe point to a target on the steep side is just left of the kink
    x, y, dist = hz.nearest_safe(20.0, 0.0)
    assert -4.0 <= x < 0.0 and abs(y) < 1e-9 and abs(dist - (20.0 - x)) < 1e-9
    # with clearance the boulder neighbourhood is avoided
    x, y, _ = hz.nearest_safe(-30.0, 20.0, clearance=4.0)
    assert math.hypot(x + 30.0, y - 20.0) > 4.0
    assert dem.hazard_map(_plane(20.0)).nearest_safe(0, 0) is None


# ----------------------------------------------------------------------------- bundled data
def test_real_dataset_files_are_sane():
    route_yaml = TERRAIN / "real_route.yaml"
    if not route_yaml.exists():
        pytest.skip("real terrain dataset not generated")
    route = Heightmap.load(route_yaml)
    pad = Heightmap.load(TERRAIN / "real_pad_a.yaml")
    lz = Heightmap.load(TERRAIN / "real_lz_b.yaml")
    assert 200 < route.heights.min() and route.heights.max() < 4500  # NM/TX/OK: no ocean
    assert 1350 < pad.height(0.0, 0.0) < 1450  # Jornada del Muerto basin, ~1410 m
    for hm in (route, pad, lz):
        assert "Copernicus" in hm.meta["attribution"]
        assert hm.meta["origin_lat"] == SPACEPORT[0] and hm.meta["origin_lon"] == SPACEPORT[1]
    assert route.meta["synthetic_detail"] is False and pad.meta["synthetic_detail"] is False
    assert lz.meta["synthetic_detail"] is True
    mission = yaml.safe_load((REPO / "configs" / "missions" / "real_hop.yaml").read_text())
    a, b = mission["launch"], mission["target"]
    assert (a["lat"], a["lon"]) == SPACEPORT and (b["lat"], b["lon"]) == BURNS_FLAT
    proj = dem.site_projection(a["lat"], a["lon"])
    u, v = proj.forward(b["lat"], b["lon"])
    assert abs(u - b["u"]) < 0.5 and abs(v - b["v"]) < 0.5 and a["u"] == a["v"] == 0.0
    assert lz.contains(b["u"], b["v"]) and route.contains(b["u"], b["v"])
    assert 550 < lz.height(b["u"], b["v"]) < 650  # western Oklahoma plains, ~590 m
    # the bundled route and the tiles agree where they overlap (500 m vs 30 m sampling)
    assert abs(route.height(b["u"], b["v"]) - lz.height(b["u"], b["v"])) < 10.0
    assert abs(route.height(0.0, 0.0) - pad.height(0.0, 0.0)) < 10.0
    assert dem.hazard_map(lz, 3.0, 8.0, 0.1).is_safe(b["u"], b["v"])


@pytest.mark.network
def test_network_fetch_real_tiles(tmp_path):
    if not _online():
        pytest.skip("offline")
    hm = dem.build_heightmap(
        *SPACEPORT, (-3e3, 3e3), (-3e3, 3e3), 500.0, product="glo90", cache_dir=tmp_path
    )
    assert 1350 < hm.height(0.0, 0.0) < 1450
    assert hm.meta["overview_factor"] == 2
    assert all(n.startswith("Copernicus_DSM_COG_30_") for n in hm.meta["tiles"])
    # an open-ocean tile does not exist in the bucket -> sea level, remembered
    ref = dem.fetch_tile(30, -140, "glo90", 4, cache_dir=tmp_path, fallback=False)
    assert ref.path is None and (tmp_path / f"{dem.tile_name(30, -140, 'glo90')}.missing").exists()
