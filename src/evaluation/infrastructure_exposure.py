from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import numpy as np
import geopandas as gpd
import osmnx as ox
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

# -----------------------------------------
# Configuration
# -----------------------------------------

IMAGE_NAME = "Mekong_1248200"

MASK_PATH = Path(
    f"experiments/geospatial/predicted_masks/"
    f"{IMAGE_NAME}_water.tif"
)

OUTPUT = Path("experiments/infrastructure")
OUTPUT.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.overpass_url = "https://overpass.private.coffee/api"
ox.settings.requests_timeout = 120
ox.settings.overpass_rate_limit = False
ox.settings.log_console = False

print("GEORESCUE INFRASTRUCTURE EXPOSURE")
print("=" * 55)
print("Image:", IMAGE_NAME, flush=True)

# -----------------------------------------
# Load georeferenced prediction
# -----------------------------------------

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    transform = src.transform
    raster_crs = src.crs

    if raster_crs is None:
        raise ValueError("Prediction raster has no CRS.")

    height, width = mask.shape

    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (width, height),
        transform * (0, height),
    ]

    footprint = Polygon(corners)

if not np.isin(mask, [0, 1, 255]).all():
    raise ValueError("Unexpected water-mask values.")

water_pixels = (mask == 1).astype(np.uint8)

water_geometries = [
    shape(geometry)
    for geometry, value in shapes(
        water_pixels,
        mask=water_pixels.astype(bool),
        transform=transform
    )
    if value == 1
]

water_polygon = (
    unary_union(water_geometries)
    if water_geometries
    else Polygon()
)

print(
    "Predicted water pixels:",
    int(water_pixels.sum()),
    flush=True
)

# -----------------------------------------
# Convert footprint to WGS84 for OSM
# -----------------------------------------

footprint_gdf = gpd.GeoDataFrame(
    {"name": [IMAGE_NAME]},
    geometry=[footprint],
    crs=raster_crs
)

footprint_wgs84 = footprint_gdf.to_crs(
    "EPSG:4326"
).geometry.iloc[0]

# OSMnx requires latitude/longitude polygons.
# A satellite chip should be a reasonably small area.
print("\nDownloading OpenStreetMap features...", flush=True)

try:
    osm = ox.features.features_from_polygon(
        footprint_wgs84,
        tags={
            "building": True,
            "highway": True
        }
    )
except Exception as exc:
    raise RuntimeError(
        "OpenStreetMap download failed. "
        "Check your internet connection and try again. "
        "Do not interpret a download failure as zero infrastructure."
    ) from exc

print("Retrieved OSM elements:", len(osm), flush=True)

# -----------------------------------------
# Prepare spatial geometries
# -----------------------------------------

osm = osm.to_crs(raster_crs)

buildings = osm[
    osm["building"].notna()
    & osm.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()

roads = osm[
    osm["highway"].notna()
    & osm.geometry.geom_type.isin(
        ["LineString", "MultiLineString"]
    )
].copy()

# Clip to the satellite footprint
buildings["geometry"] = buildings.geometry.intersection(
    footprint
)

roads["geometry"] = roads.geometry.intersection(
    footprint
)

buildings = buildings[
    ~buildings.geometry.is_empty
].copy()

roads = roads[
    ~roads.geometry.is_empty
].copy()

# -----------------------------------------
# Calculate exposure
# -----------------------------------------

# Intersect each building/road with predicted water.
# > 0 excludes mere boundary touching.

building_water_overlap = (
    buildings.geometry.intersection(water_polygon)
)

road_water_overlap = (
    roads.geometry.intersection(water_polygon)
)

# Local UTM coordinates for metre-based measurements
metric_crs = footprint_gdf.estimate_utm_crs()

if metric_crs is None:
    raise ValueError("Could not determine a metric CRS.")

roads_metric = roads.to_crs(metric_crs)

road_overlap_gdf = gpd.GeoDataFrame(
    geometry=road_water_overlap,
    crs=raster_crs
).to_crs(metric_crs)

building_overlap_gdf = gpd.GeoDataFrame(
    geometry=building_water_overlap,
    crs=raster_crs
).to_crs(metric_crs)

exposed_buildings = (
    building_overlap_gdf.geometry.area > 0
)

exposed_roads = (
    road_overlap_gdf.geometry.length > 0
)

building_count = int(exposed_buildings.sum())
road_count = int(exposed_roads.sum())

total_road_km = (
    roads_metric.geometry.length.sum() / 1000
)

water_road_km = (
    road_overlap_gdf.geometry.length.sum() / 1000
)

# -----------------------------------------
# Save vector outputs
# -----------------------------------------

water_gdf = gpd.GeoDataFrame(
    {"source": ["U-Net prediction"]},
    geometry=[water_polygon],
    crs=raster_crs
)

water_gdf.to_file(
    OUTPUT / "water_extent.geojson",
    driver="GeoJSON"
)

def export_features(frame, exposed, filename):
    result = gpd.GeoDataFrame(
        {
            "osm_id": [str(idx) for idx in frame.index],
            "exposed": list(map(bool, exposed))
        },
        geometry=frame.geometry.values,
        crs=raster_crs
    )

    result.to_crs("EPSG:4326").to_file(
        OUTPUT / filename,
        driver="GeoJSON"
    )

export_features(
    buildings,
    exposed_buildings,
    "buildings.geojson"
)

export_features(
    roads,
    exposed_roads,
    "roads.geojson"
)

# -----------------------------------------
# Visualize spatial overlays
# -----------------------------------------

fig, ax = plt.subplots(figsize=(10, 9))

extent = (
    min(p[0] for p in corners),
    max(p[0] for p in corners),
    min(p[1] for p in corners),
    max(p[1] for p in corners)
)

background = np.ma.masked_where(
    mask == 255,
    mask.astype(float)
)

ax.imshow(
    background,
    extent=extent,
    origin="upper",
    cmap="Blues",
    vmin=0,
    vmax=1,
    alpha=0.7
)

if len(roads):
    roads.plot(
        ax=ax,
        color="orange",
        linewidth=0.8
    )

if len(buildings):
    buildings.boundary.plot(
        ax=ax,
        color="dimgray",
        linewidth=0.5
    )

if building_count:
    buildings.loc[
        exposed_buildings.values
    ].boundary.plot(
        ax=ax,
        color="red",
        linewidth=1.4
    )

if road_count:
    roads.loc[
        exposed_roads.values
    ].plot(
        ax=ax,
        color="red",
        linewidth=1.7
    )

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_aspect("equal")
ax.set_title(
    f"GeoRescue Infrastructure Exposure\n"
    f"{IMAGE_NAME}"
)

ax.legend(
    handles=[
        Patch(color="steelblue", label="Predicted water"),
        Line2D(
            [0], [0],
            color="orange",
            label="Mapped roads"
        ),
        Line2D(
            [0], [0],
            color="red",
            label="Intersects predicted water"
        )
    ],
    loc="upper right"
)

plt.tight_layout()
plt.savefig(
    OUTPUT / "infrastructure_exposure.png",
    dpi=200
)
plt.close()

# -----------------------------------------
# Save summary and provenance
# -----------------------------------------

summary = {
    "image": IMAGE_NAME,
    "mapped_buildings": len(buildings),
    "buildings_intersecting_water": building_count,
    "mapped_road_features": len(roads),
    "road_features_intersecting_water": road_count,
    "mapped_road_km": round(total_road_km, 3),
    "road_length_intersecting_water_km": round(
        water_road_km, 3
    )
}

with open(
    OUTPUT / "exposure_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary.keys()
    )
    writer.writeheader()
    writer.writerow(summary)

metadata = {
    "image": IMAGE_NAME,
    "osm_download_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "data_source": "OpenStreetMap via OSMnx",
    "prediction_source": str(MASK_PATH),
    "water_threshold": 0.5,
    "notes": (
        "Potential intersections with predicted surface water. "
        "Not verified inundation or infrastructure damage. "
        "OpenStreetMap coverage may be incomplete."
    )
}

with open(
    OUTPUT / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 55)
print("GEORESCUE INFRASTRUCTURE RESULTS")
print("=" * 55)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nINFRASTRUCTURE ANALYSIS COMPLETE!")
print("Outputs:", OUTPUT)


