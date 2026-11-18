from pathlib import Path
import subprocess
import shutil
import json
import csv
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box
from shapely.ops import unary_union

IMAGE = "Mekong_1248200"

ROOT = Path("experiments/infrastructure_overture")
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

MASK_PATH = Path(
    f"experiments/geospatial/predicted_masks/"
    f"{IMAGE}_water.tif"
)

# -----------------------------------
# 1. Load prediction and footprint
# -----------------------------------

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    crs = src.crs
    transform = src.transform
    bounds = src.bounds

    if crs is None:
        raise ValueError("Raster has no CRS.")

    west, south, east, north = transform_bounds(
        crs, "EPSG:4326",
        *bounds,
        densify_pts=21
    )

    footprint = box(*bounds)

    water_geometries = [
        shape(geom)
        for geom, value in shapes(
            (mask == 1).astype("uint8"),
            mask=(mask == 1),
            transform=transform
        )
        if value == 1
    ]

water = (
    unary_union(water_geometries)
    if water_geometries else None
)

if water is None or water.is_empty:
    raise RuntimeError("No predicted water in this mask.")

print("Image:", IMAGE)
print(
    "Bounding box:",
    f"{west:.5f}, {south:.5f}, "
    f"{east:.5f}, {north:.5f}",
    flush=True
)

# -----------------------------------
# 2. Download Overture data
# -----------------------------------

cli = shutil.which("overturemaps")

if cli is None:
    raise RuntimeError(
        "Overture CLI not found. "
        "Run: python -m pip install overturemaps"
    )

bbox = f"{west},{south},{east},{north}"

def download(layer):
    output = DATA / f"{layer}.geojson"

    if output.exists():
        print(f"Using cached {layer} data.")
        return output

    print(f"Downloading {layer} data...", flush=True)

    cmd = [
        cli, "download",
        f"--bbox={bbox}",
        "-f", "geojson",
        f"--type={layer}",
        "-o", str(output)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Overture {layer} download failed. "
            "Check the output above. "
            "Cloud data access may be blocked or unavailable."
        ) from exc

    return output

building_file = download("building")
segment_file = download("segment")

# -----------------------------------
# 3. Load and clip geometries
# -----------------------------------

def load_layer(path, geometry_types):
    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}. "
            "Inspect the downloaded data."
        ) from exc

    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")

    frame = frame.to_crs(crs)

    frame = frame[
        frame.geometry.notna()
        & frame.geometry.geom_type.isin(geometry_types)
    ].copy()

    if frame.empty:
        return frame

    frame["geometry"] = (
        frame.geometry.intersection(footprint)
    )

    frame = frame[
        ~frame.geometry.is_empty
    ].copy()

    return frame.reset_index(drop=True)

buildings = load_layer(
    building_file,
    ["Polygon", "MultiPolygon"]
)

segments = load_layer(
    segment_file,
    ["LineString", "MultiLineString"]
)

# Overture segments include other transportation
# modes. Restrict to road subtype.
if "subtype" not in segments.columns:
    raise ValueError(
        "Overture transportation subtype field missing."
    )

roads = segments[
    segments["subtype"].astype(str) == "road"
].copy().reset_index(drop=True)

print("Buildings loaded:", len(buildings))
print("Road segments loaded:", len(roads))

# -----------------------------------
# 4. Calculate intersections in metres
# -----------------------------------

footprint_gdf = gpd.GeoDataFrame(
    geometry=[footprint], crs=crs
)

metric_crs = footprint_gdf.estimate_utm_crs()

if metric_crs is None:
    raise RuntimeError("Cannot determine metric CRS.")

water_metric = gpd.GeoSeries(
    [water], crs=crs
).to_crs(metric_crs).iloc[0]

buildings_m = buildings.to_crs(metric_crs)
roads_m = roads.to_crs(metric_crs)

building_intersections = (
    buildings_m.geometry.intersection(water_metric)
)

road_intersections = (
    roads_m.geometry.intersection(water_metric)
)

building_flags = (
    building_intersections.area > 0
)

road_flags = (
    road_intersections.length > 0
)

building_count = int(building_flags.sum())
road_count = int(road_flags.sum())

road_length_km = (
    roads_m.geometry.length.sum() / 1000
)

exposed_road_km = (
    road_intersections.length.sum() / 1000
)

# -----------------------------------
# 5. Save the output GeoJSON layers
# -----------------------------------

water_gdf = gpd.GeoDataFrame(
    {"source": ["U-Net"]},
    geometry=[water],
    crs=crs
).to_crs("EPSG:4326")

water_gdf.to_file(
    ROOT / "water.geojson",
    driver="GeoJSON"
)

def save_features(frame, flags, filename):
    result = frame[["geometry"]].copy()
    result["exposed"] = np.asarray(flags, dtype=bool)
    result["feature_id"] = [
        str(i) for i in range(len(result))
    ]

    result.to_crs("EPSG:4326").to_file(
        ROOT / filename,
        driver="GeoJSON"
    )

save_features(
    buildings, building_flags, "buildings.geojson"
)

save_features(
    roads, road_flags, "roads.geojson"
)

# -----------------------------------
# 6. Create visual overlay
# -----------------------------------

fig, ax = plt.subplots(figsize=(10, 9))

gpd.GeoSeries(
    [footprint], crs=crs
).boundary.plot(
    ax=ax, color="black", linewidth=1
)

gpd.GeoSeries(
    [water], crs=crs
).plot(
    ax=ax, color="royalblue", alpha=0.5
)

if len(roads):
    roads.plot(
        ax=ax, color="darkorange", linewidth=0.8
    )

if len(buildings):
    buildings.boundary.plot(
        ax=ax, color="gray", linewidth=0.5
    )

if building_count:
    buildings.loc[
        building_flags.to_numpy()
    ].boundary.plot(
        ax=ax, color="red", linewidth=1.4
    )

if road_count:
    roads.loc[
        road_flags.to_numpy()
    ].plot(
        ax=ax, color="red", linewidth=1.5
    )

ax.set_title(
    f"GeoRescue Infrastructure Exposure\n{IMAGE}"
)
ax.set_aspect("equal")
plt.tight_layout()

plt.savefig(
    ROOT / "exposure_map.png",
    dpi=200
)
plt.close()

# -----------------------------------
# 7. Save results and provenance
# -----------------------------------

summary = {
    "image": IMAGE,
    "data_source": "Overture Maps",
    "mapped_buildings": len(buildings),
    "buildings_intersecting_water": building_count,
    "mapped_road_segments": len(roads),
    "road_segments_intersecting_water": road_count,
    "total_road_km": round(road_length_km, 3),
    "road_length_intersecting_water_km": round(
        exposed_road_km, 3
    )
}

with open(
    ROOT / "exposure_summary.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=summary.keys()
    )
    writer.writeheader()
    writer.writerow(summary)

with open(
    ROOT / "metadata.json",
    "w", encoding="utf-8"
) as f:
    json.dump({
        "image": IMAGE,
        "source": "Overture Maps",
        "retrieved_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "notes": (
            "Intersection with predicted surface water, "
            "not confirmed flood damage. "
            "Vector coverage and feature accuracy vary."
        )
    }, f, indent=2)

print("\n" + "=" * 55)
print("GEORESCUE OVERTURE EXPOSURE RESULTS")
print("=" * 55)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nEXPOSURE ANALYSIS COMPLETE!")
