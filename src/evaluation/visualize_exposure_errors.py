from pathlib import Path
import csv

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.features import shapes
from rasterio.plot import show
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

IMAGE = "Mekong_1248200"
ROOT = Path("experiments/infrastructure_overture")
OUT = ROOT / "validation"
OUT.mkdir(parents=True, exist_ok=True)

PRED = Path(
    f"experiments/geospatial/predicted_masks/"
    f"{IMAGE}_water.tif"
)

LABEL = Path(
    f"data/raw/LabelHand/{IMAGE}_LabelHand.tif"
)

# --------------------------------------
# 1. Load and align water rasters
# --------------------------------------

with rasterio.open(PRED) as src:
    prediction = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

    if crs is None:
        raise ValueError("Prediction has no CRS.")

with rasterio.open(LABEL) as src:
    if src.crs is None:
        raise ValueError("Reference has no CRS.")

    same_grid = (
        src.crs == crs
        and src.shape == prediction.shape
        and src.transform.almost_equals(transform)
    )

    reference = np.full(
        prediction.shape, -1, dtype=np.int16
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=reference,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=(
            src.nodata if src.nodata is not None else -1
        ),
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=-1,
        resampling=Resampling.nearest
    )

valid = (
    (prediction != 255)
    & np.isin(reference, [0, 1])
)

valid_fraction = float(valid.mean())

if valid_fraction < 0.5:
    raise ValueError(
        "Less than 50% of the prediction grid has "
        "valid overlapping labels. Inspect georeferencing."
    )

print("Same original raster grid:", same_grid)
print("Valid overlapping pixels:", int(valid.sum()))
print("Valid fraction:", round(valid_fraction, 4))

if not same_grid:
    print(
        "WARNING: Reference was reprojected. "
        "Inspect visual alignment before trusting metrics."
    )

pred_water = (prediction == 1) & valid
ref_water = (reference == 1) & valid

# --------------------------------------
# 2. Create water-error classification
# --------------------------------------

# 0: invalid
# 1: true negative
# 2: false positive
# 3: false negative
# 4: true positive

pixel_class = np.zeros(
    prediction.shape, dtype=np.uint8
)

pixel_class[valid & ~pred_water & ~ref_water] = 1
pixel_class[pred_water & ~ref_water] = 2
pixel_class[~pred_water & ref_water] = 3
pixel_class[pred_water & ref_water] = 4

def polygon_from_mask(binary):
    geometries = [
        shape(geom)
        for geom, value in shapes(
            binary.astype(np.uint8),
            mask=binary,
            transform=transform
        )
        if value == 1
    ]

    return (
        unary_union(geometries)
        if geometries else Polygon()
    )

pred_polygon = polygon_from_mask(pred_water)
ref_polygon = polygon_from_mask(ref_water)

# --------------------------------------
# 3. Load existing infrastructure
# --------------------------------------

buildings = gpd.read_file(
    ROOT / "buildings.geojson"
).to_crs(crs)

roads = gpd.read_file(
    ROOT / "roads.geojson"
).to_crs(crs)

metric_crs = gpd.GeoSeries(
    [Polygon([
        (bounds.left, bounds.bottom),
        (bounds.right, bounds.bottom),
        (bounds.right, bounds.top),
        (bounds.left, bounds.top)
    ])],
    crs=crs
).estimate_utm_crs()

if metric_crs is None:
    raise ValueError("Cannot determine metric CRS.")

polygons_m = gpd.GeoSeries(
    [pred_polygon, ref_polygon], crs=crs
).to_crs(metric_crs)

pred_m = polygons_m.iloc[0]
ref_m = polygons_m.iloc[1]

# --------------------------------------
# 4. Classify each building and road
# --------------------------------------

def classify_features(frame, feature_type):
    metric = frame.to_crs(metric_crs)

    pred_intersection = metric.geometry.intersection(
        pred_m
    )

    ref_intersection = metric.geometry.intersection(
        ref_m
    )

    if feature_type == "building":
        pred_flag = pred_intersection.area > 0
        ref_flag = ref_intersection.area > 0
    else:
        pred_flag = pred_intersection.length > 0
        ref_flag = ref_intersection.length > 0

    pred_flag = np.asarray(pred_flag, dtype=bool)
    ref_flag = np.asarray(ref_flag, dtype=bool)

    categories = np.select(
        [
            pred_flag & ref_flag,
            pred_flag & ~ref_flag,
            ~pred_flag & ref_flag
        ],
        [
            "TP", "FP", "FN"
        ],
        default="TN"
    )

    result = frame.copy()
    result["error_type"] = categories
    result["predicted_exposed"] = pred_flag
    result["reference_exposed"] = ref_flag

    return result

buildings = classify_features(buildings, "building")
roads = classify_features(roads, "road")

# --------------------------------------
# 5. Create diagnostic maps
# --------------------------------------

colors = {
    "TP": "#22A06B",
    "FP": "#E6A23C",
    "FN": "#E55353",
    "TN": "#8D98A7"
}

fig, axes = plt.subplots(
    1, 2, figsize=(17, 8)
)

# Left: pixel segmentation errors
cmap = ListedColormap([
    "#FFFFFF",
    "#E8ECF0",
    "#E6A23C",
    "#E55353",
    "#22A06B"
])

norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    cmap.N
)

show(
    pixel_class,
    transform=transform,
    ax=axes[0],
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)

axes[0].set_title(
    "Water Segmentation Errors\n"
    "Green = TP, Red = FN, Orange = FP"
)

# Right: building and road exposure errors
show(
    np.ma.masked_where(
        ~valid,
        np.where(pred_water, 1.0, 0.0)
    ),
    transform=transform,
    ax=axes[1],
    cmap="Blues",
    vmin=0,
    vmax=1,
    alpha=0.45
)

for category in ["TN", "TP", "FP", "FN"]:
    subset = buildings[
        buildings["error_type"] == category
    ]

    if len(subset):
        subset.boundary.plot(
            ax=axes[1],
            color=colors[category],
            linewidth=(
                1.3 if category != "TN" else 0.35
            )
        )

for category in ["TN", "TP", "FP", "FN"]:
    subset = roads[
        roads["error_type"] == category
    ]

    if len(subset):
        subset.plot(
            ax=axes[1],
            color=colors[category],
            linewidth=(
                2.2 if category != "TN" else 0.8
            )
        )

axes[1].set_title(
    "Infrastructure Exposure Errors\n"
    "Building outlines and road segments"
)

legend = [
    Patch(facecolor=colors["TP"], label="True positive"),
    Patch(facecolor=colors["FN"], label="False negative"),
    Patch(facecolor=colors["FP"], label="False positive"),
    Patch(facecolor=colors["TN"], label="True negative"),
]

axes[1].legend(
    handles=legend,
    loc="upper right",
    fontsize=9
)

for ax in axes:
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_aspect("equal")
    ax.set_xlabel("Raster CRS X")
    ax.set_ylabel("Raster CRS Y")

fig.suptitle(
    f"GeoRescue — Exposure Validation: {IMAGE}",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUT / "exposure_error_map.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------
# 6. Save feature classifications
# --------------------------------------

buildings.to_crs("EPSG:4326").to_file(
    OUT / "classified_buildings.geojson",
    driver="GeoJSON"
)

roads.to_crs("EPSG:4326").to_file(
    OUT / "classified_roads.geojson",
    driver="GeoJSON"
)

summary_rows = []

for feature_name, frame in [
    ("buildings", buildings),
    ("road_segments", roads)
]:
    counts = frame["error_type"].value_counts()

    summary_rows.append({
        "feature_type": feature_name,
        "TP": int(counts.get("TP", 0)),
        "FP": int(counts.get("FP", 0)),
        "FN": int(counts.get("FN", 0)),
        "TN": int(counts.get("TN", 0))
    })

with open(
    OUT / "feature_error_summary.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=summary_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nGEORESCUE VISUAL EXPOSURE VALIDATION")
print("=" * 55)

for row in summary_rows:
    print(row)

print("\nMap:", OUT / "exposure_error_map.png")
print("VISUAL EXPOSURE VALIDATION COMPLETE!")
