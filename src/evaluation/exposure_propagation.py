from pathlib import Path
import csv
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box, Polygon
from shapely.ops import unary_union

# ============================================
# GEORESCUE: EXPOSURE ERROR PROPAGATION
# ============================================

IMAGES = [
    "Mekong_1248200",
    "Mekong_1396181",
    "Somalia_12849",
    "Somalia_1087508"
]

OUTPUT = Path("experiments/exposure_propagation")
CACHE = OUTPUT / "infrastructure_cache"

OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

EXISTING = Path("experiments/infrastructure_overture/data")

V1_DIR = Path("experiments/geospatial/predicted_masks")
V2_DIR = Path(
    "experiments/unet_v2/test_evaluation/predicted_masks"
)
LABEL_DIR = Path("data/raw/LabelHand")

# ============================================
# 1. LOAD WATER MASKS
# ============================================

def load_raster(path):
    with rasterio.open(path) as src:
        return (
            src.read(1),
            src.crs,
            src.transform,
            src.bounds
        )


def mask_polygon(binary, transform):
    geometries = [
        shape(geometry)
        for geometry, value in shapes(
            binary.astype(np.uint8),
            mask=binary,
            transform=transform
        )
        if value == 1
    ]

    return unary_union(geometries) if geometries else Polygon()


def calculate_iou(pred, truth, valid):
    tp = int(np.sum(pred & truth & valid))
    fp = int(np.sum(pred & ~truth & valid))
    fn = int(np.sum(~pred & truth & valid))

    denominator = tp + fp + fn

    return (
        tp / denominator
        if denominator else np.nan
    )


# ============================================
# 2. DOWNLOAD OR REUSE INFRASTRUCTURE
# ============================================

def download_layer(image, layer, bbox):
    folder = CACHE / image
    folder.mkdir(parents=True, exist_ok=True)

    destination = folder / f"{layer}.geojson"

    if destination.exists():
        print(f"  Cached {layer} found.", flush=True)
        return destination

    # Reuse the infrastructure already downloaded
    # for the original Mekong pilot.
    original = EXISTING / f"{layer}.geojson"

    if image == "Mekong_1248200" and original.exists():
        shutil.copy2(original, destination)
        print(f"  Reused existing {layer} data.", flush=True)
        return destination

    cli = shutil.which("overturemaps")

    if not cli:
        raise RuntimeError(
            "Overture Maps CLI not installed. "
            "Run: python -m pip install overturemaps"
        )

    print(f"  Downloading {layer}...", flush=True)

    command = [
        cli,
        "download",
        f"--bbox={bbox}",
        "-f", "geojson",
        f"--type={layer}",
        "-o", str(destination)
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {layer} for {image}. "
            "Previously completed outputs are preserved."
        ) from exc

    return destination


def load_features(path, crs, footprint, kind):
    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        # An empty GeoJSON feature collection can
        # occur where Overture has no mapped features.
        # Do not treat other parsing failures as empty.
        import json

        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)

            if (
                content.get("type") == "FeatureCollection"
                and len(content.get("features", [])) == 0
            ):
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs=crs),
                    crs=crs
                )
        except Exception:
            pass

        raise RuntimeError(
            f"Could not read infrastructure file: {path}"
        ) from exc

    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")

    frame = frame.to_crs(crs)

    allowed = (
        ["Polygon", "MultiPolygon"]
        if kind == "building"
        else ["LineString", "MultiLineString"]
    )

    frame = frame[
        frame.geometry.notna()
        & frame.geometry.geom_type.isin(allowed)
    ].copy()

    if kind == "road":
        if "subtype" not in frame.columns:
            if len(frame):
                raise ValueError(
                    f"Missing road subtype in {path}"
                )
        else:
            frame = frame[
                frame["subtype"].astype(str) == "road"
            ].copy()

    if not frame.empty:
        frame["geometry"] = frame.geometry.intersection(
            footprint
        )

        frame = frame[
            ~frame.geometry.is_empty
        ].copy()

    return frame.reset_index(drop=True)


# ============================================
# 3. CLASSIFICATION METRICS
# ============================================

def score_exposure(predicted, reference):
    tp = int(np.sum(predicted & reference))
    fp = int(np.sum(predicted & ~reference))
    fn = int(np.sum(~predicted & reference))
    tn = int(np.sum(~predicted & ~reference))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else np.nan
        )
    }


def exposure_for_geometry(frame, water_geometry, kind):
    if frame.empty:
        return np.zeros(0, dtype=bool), 0.0

    intersection = frame.geometry.intersection(
        water_geometry
    )

    if kind == "building":
        values = intersection.area.to_numpy()
    else:
        values = intersection.length.to_numpy()

    return values > 0, float(values.sum())


# ============================================
# 4. PROCESS EACH IMAGE
# ============================================

results = []

print("GEORESCUE EXPOSURE ERROR PROPAGATION")
print("=" * 60)

for index, name in enumerate(IMAGES, start=1):
    print(f"\n[{index}/{len(IMAGES)}] {name}", flush=True)

    label, label_crs, label_transform, _ = load_raster(
        LABEL_DIR / f"{name}_LabelHand.tif"
    )

    v1, crs, transform, bounds = load_raster(
        V1_DIR / f"{name}_water.tif"
    )

    v2, crs2, transform2, _ = load_raster(
        V2_DIR / f"{name}_water_v2.tif"
    )

    if (
        crs is None
        or label_crs is None
        or crs2 is None
        or label.shape != v1.shape
        or v1.shape != v2.shape
        or label_crs != crs
        or crs != crs2
        or not label_transform.almost_equals(transform)
        or not transform.almost_equals(transform2)
    ):
        raise ValueError(
            f"Raster alignment mismatch: {name}"
        )

    if not np.array_equal(v1 != 255, v2 != 255):
        raise ValueError(
            f"Prediction validity masks differ: {name}"
        )

    valid = (
        (v1 != 255)
        & np.isin(label, [0, 1])
    )

    truth = (label == 1) & valid

    predictions = {
        "V1": (v1 == 1) & valid,
        "V2": (v2 == 1) & valid
    }

    polygons = {
        "reference": mask_polygon(truth, transform),
        "V1": mask_polygon(predictions["V1"], transform),
        "V2": mask_polygon(predictions["V2"], transform)
    }

    west, south, east, north = transform_bounds(
        crs, "EPSG:4326", *bounds,
        densify_pts=21
    )

    bbox = f"{west},{south},{east},{north}"

    footprint = box(*bounds)

    building_file = download_layer(
        name, "building", bbox
    )

    # Somalia road downloads are unavailable in this pilot.
    segment_file = None

    if not name.startswith("Somalia_"):
        segment_file = download_layer(
            name, "segment", bbox
        )
    else:
        print(
            "  Somalia road data: not assessed "
            "(download skipped).",
            flush=True
        )

    buildings = load_features(
        building_file, crs, footprint, "building"
    )

    if segment_file is None:
        roads = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs
        )
        roads_available = False
    else:
        roads = load_features(
            segment_file, crs, footprint, "road"
        )
        roads_available = True

    print(
        f"  Buildings: {len(buildings)}, "
        f"road segments: {len(roads)}",
        flush=True
    )

    footprint_gdf = gpd.GeoSeries(
        [footprint], crs=crs
    )

    metric_crs = footprint_gdf.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError(
            f"No metric CRS for {name}"
        )

    buildings_m = buildings.to_crs(metric_crs)
    roads_m = roads.to_crs(metric_crs)

    water_m = gpd.GeoSeries(
        [
            polygons["reference"],
            polygons["V1"],
            polygons["V2"]
        ],
        crs=crs
    ).to_crs(metric_crs)

    ref_buildings, _ = exposure_for_geometry(
        buildings_m, water_m.iloc[0], "building"
    )

    ref_roads, ref_road_m = exposure_for_geometry(
        roads_m, water_m.iloc[0], "road"
    )

    for model, geom_idx in [("V1", 1), ("V2", 2)]:
        predicted_buildings, _ = exposure_for_geometry(
            buildings_m, water_m.iloc[geom_idx], "building"
        )

        predicted_roads, pred_road_m = exposure_for_geometry(
            roads_m, water_m.iloc[geom_idx], "road"
        )

        scores = score_exposure(
            predicted_buildings, ref_buildings
        )

        road_scores = score_exposure(
            predicted_roads, ref_roads
        )

        results.append({
            "image": name,
            "region": name.rsplit("_", 1)[0],
            "model": model,
            "segmentation_iou": calculate_iou(
                predictions[model], truth, valid
            ),
            "mapped_buildings": len(buildings),
            "reference_exposed_buildings": int(
                ref_buildings.sum()
            ),
            "predicted_exposed_buildings": int(
                predicted_buildings.sum()
            ),
            "building_tp": scores["tp"],
            "building_fp": scores["fp"],
            "building_fn": scores["fn"],
            "building_precision": scores["precision"],
            "building_recall": scores["recall"],
            "building_f1": scores["f1"],
            "mapped_road_segments": len(roads),
            "reference_road_segments": int(
                ref_roads.sum()
            ),
            "predicted_road_segments": int(
                predicted_roads.sum()
            ),
            "road_tp": road_scores["tp"],
            "road_fp": road_scores["fp"],
            "road_fn": road_scores["fn"],
            "reference_exposed_road_m": ref_road_m,
            "predicted_exposed_road_m": pred_road_m
        })

        if not roads_available:
            for key in [
                "mapped_road_segments",
                "reference_road_segments",
                "predicted_road_segments",
                "road_tp",
                "road_fp",
                "road_fn",
                "reference_exposed_road_m",
                "predicted_exposed_road_m"
            ]:
                results[-1][key] = np.nan

        print(
            f"  {model}: IoU="
            f"{results[-1]['segmentation_iou']:.3f}, "
            f"building recall="
            f"{scores['recall']:.3f}",
            flush=True
        )

    # Save completed progress after each image.
    pd.DataFrame(results).to_csv(
        OUTPUT / "exposure_propagation.csv",
        index=False
    )

# ============================================
# 5. COMPARATIVE REPORT
# ============================================

df = pd.DataFrame(results)

pivot = df.pivot(
    index=["image", "region"],
    columns="model",
    values=["segmentation_iou", "building_recall"]
)

comparison = pd.DataFrame({
    "image": pivot.index.get_level_values("image"),
    "region": pivot.index.get_level_values("region"),
    "iou_v1": pivot[("segmentation_iou", "V1")].to_numpy(),
    "iou_v2": pivot[("segmentation_iou", "V2")].to_numpy(),
    "building_recall_v1": pivot[
        ("building_recall", "V1")
    ].to_numpy(),
    "building_recall_v2": pivot[
        ("building_recall", "V2")
    ].to_numpy()
})

comparison["iou_change"] = (
    comparison["iou_v2"] - comparison["iou_v1"]
)

comparison["building_recall_change"] = (
    comparison["building_recall_v2"]
    - comparison["building_recall_v1"]
)

comparison.to_csv(
    OUTPUT / "model_impact.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 6))

valid_pairs = comparison.dropna(
    subset=["iou_change", "building_recall_change"]
)

for region, group in valid_pairs.groupby("region"):
    ax.scatter(
        group["iou_change"] * 100,
        group["building_recall_change"] * 100,
        s=95,
        label=region
    )

ax.axhline(0, color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")

ax.set_xlabel("Change in segmentation IoU (pp)")
ax.set_ylabel("Change in building exposure recall (pp)")
ax.set_title(
    "GeoRescue: Segmentation vs Exposure Improvement"
)
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    OUTPUT / "error_propagation.png",
    dpi=200
)
plt.close()

print("\n" + "=" * 60)
print("GEORESCUE EXPOSURE PROPAGATION RESULTS")
print("=" * 60)

print(
    comparison.round(4).to_string(index=False)
)

print("\nEXPOSURE PROPAGATION PILOT COMPLETE!")
print("Outputs:", OUTPUT)

