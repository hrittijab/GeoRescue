from pathlib import Path
import csv
import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from shapely.ops import unary_union

IMAGE = "Mekong_1248200"
ROOT = Path("experiments/infrastructure_overture")

PRED_PATH = Path(
    f"experiments/geospatial/predicted_masks/"
    f"{IMAGE}_water.tif"
)

LABEL_PATH = Path(
    f"data/raw/LabelHand/{IMAGE}_LabelHand.tif"
)

def to_polygons(binary, transform):
    geometries = [
        shape(geometry)
        for geometry, value in shapes(
            binary.astype("uint8"),
            mask=binary,
            transform=transform
        )
        if value == 1
    ]
    return unary_union(geometries)

print("Loading predictions and reference labels...")

with rasterio.open(PRED_PATH) as src:
    predicted = src.read(1)
    raster_crs = src.crs
    raster_transform = src.transform

with rasterio.open(LABEL_PATH) as src:
    if src.crs is None or raster_crs is None:
        raise ValueError(
            "Missing CRS. Cannot safely reproject reference labels."
        )

    reference = np.full(
        predicted.shape,
        -1,
        dtype=np.int16
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=reference,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata if src.nodata is not None else -1,
        dst_transform=raster_transform,
        dst_crs=raster_crs,
        dst_nodata=-1,
        resampling=Resampling.nearest
    )

    print("Reference labels aligned to prediction grid.")
    print(
        "Valid reference pixels:",
        int(np.isin(reference, [0, 1]).sum())
    )

    if not np.isin(reference, [0, 1]).any():
        raise ValueError(
            "No overlapping valid reference pixels after reprojection. "
            "Check the source raster bounds and CRS."
        )

valid = (predicted != 255) & np.isin(reference, [0, 1])

pred_water = (predicted == 1) & valid
ref_water = (reference == 1) & valid

pred_polygon = to_polygons(pred_water, raster_transform)
ref_polygon = to_polygons(ref_water, raster_transform)

buildings = gpd.read_file(
    ROOT / "buildings.geojson"
).to_crs(raster_crs)

roads = gpd.read_file(
    ROOT / "roads.geojson"
).to_crs(raster_crs)

metric_crs = gpd.GeoSeries(
    [pred_polygon.union(ref_polygon).envelope],
    crs=raster_crs
).estimate_utm_crs()

if metric_crs is None:
    raise ValueError("Cannot determine metric CRS.")

polygons_m = gpd.GeoSeries(
    [pred_polygon, ref_polygon],
    crs=raster_crs
).to_crs(metric_crs)

pred_m = polygons_m.iloc[0]
ref_m = polygons_m.iloc[1]

buildings_m = buildings.to_crs(metric_crs)
roads_m = roads.to_crs(metric_crs)

# ---------------------------------------
# Building exposure validation
# ---------------------------------------

pred_building = (
    buildings_m.geometry.intersection(pred_m).area > 0
).to_numpy()

ref_building = (
    buildings_m.geometry.intersection(ref_m).area > 0
).to_numpy()

tp = int(np.sum(pred_building & ref_building))
fp = int(np.sum(pred_building & ~ref_building))
fn = int(np.sum(~pred_building & ref_building))
tn = int(np.sum(~pred_building & ~ref_building))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0

# ---------------------------------------
# Road exposure validation
# ---------------------------------------

pred_road_lengths = (
    roads_m.geometry.intersection(pred_m).length
)

ref_road_lengths = (
    roads_m.geometry.intersection(ref_m).length
)

pred_road_count = int((pred_road_lengths > 0).sum())
ref_road_count = int((ref_road_lengths > 0).sum())

# ---------------------------------------
# Save summary
# ---------------------------------------

results = {
    "image": IMAGE,
    "building_true_positive": tp,
    "building_false_positive": fp,
    "building_false_negative": fn,
    "building_true_negative": tn,
    "predicted_exposed_buildings": int(pred_building.sum()),
    "reference_exposed_buildings": int(ref_building.sum()),
    "building_exposure_precision": round(precision, 4),
    "building_exposure_recall": round(recall, 4),
    "building_exposure_f1": round(f1, 4),
    "predicted_exposed_road_segments": pred_road_count,
    "reference_exposed_road_segments": ref_road_count,
    "predicted_exposed_road_m": round(
        float(pred_road_lengths.sum()), 3
    ),
    "reference_exposed_road_m": round(
        float(ref_road_lengths.sum()), 3
    )
}

output_path = ROOT / "exposure_validation.csv"

with open(
    output_path, "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=results.keys()
    )
    writer.writeheader()
    writer.writerow(results)

print("\n" + "=" * 55)
print("GEORESCUE INFRASTRUCTURE VALIDATION")
print("=" * 55)

for key, value in results.items():
    print(f"{key}: {value}")

print("\nEXPOSURE VALIDATION COMPLETE!")

