# Real-world terrain (Copernicus DEM). rasterio (extra "geo") is imported lazily.
from plume.terrain.dem import (
    COPERNICUS_ATTRIBUTION,
    DemDownloadError,
    HazardMap,
    SiteProjection,
    add_synthetic_detail,
    build_heightmap,
    corridor_heightmap,
    from_geotiff,
    hazard_map,
    landing_tile,
    site_projection,
)
from plume.terrain.heightmap import (
    Heightmap,
    flatten_sites,
    fractal_noise,
    generate_detail_tile,
    generate_regional_map,
)

__all__ = [
    "COPERNICUS_ATTRIBUTION",
    "DemDownloadError",
    "HazardMap",
    "Heightmap",
    "SiteProjection",
    "add_synthetic_detail",
    "build_heightmap",
    "corridor_heightmap",
    "flatten_sites",
    "fractal_noise",
    "from_geotiff",
    "generate_detail_tile",
    "generate_regional_map",
    "hazard_map",
    "landing_tile",
    "site_projection",
]
