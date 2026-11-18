from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import pandas as pd
import torch
import rasterio
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet


SEED = 42
THRESHOLD = 0.5
TARGET = "Mekong_1248200"

ROOT = Path("experiments/unet_v2/test_evaluation")
MASK_DIR = ROOT / "predicted_masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

INFRA = Path("experiments/infrastructure_overture")
RAW = Path("data/raw/S1Hand")

torch.manual_seed(SEED)
torch.set_num_threads(min(4, torch.get_num_threads()))

CHECKPOINTS = {
    "V1": Path("experiments/unet/best_unet.pt"),
    "V2": Path("experiments/unet_v2/best_unet_v2.pt")
}



def counts_from_arrays(pred, truth, valid):
    tp = np.sum(pred & truth & valid)
    fp = np.sum(pred & ~truth & valid)
    fn = np.sum(~pred & truth & valid)
    tn = np.sum(~pred & ~truth & valid)

    return np.array(
        [tp, fp, fn, tn],
        dtype=np.int64
    )


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else float("nan")
    )

    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr
    }


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint.get("config", {})

    model = UNet(
        in_channels=config.get("in_channels", 2),
        base=config.get("base", 16)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded {path} | epoch "
        f"{checkpoint.get('epoch', '?')}",
        flush=True
    )

    return model


models = {
    name: load_model(path)
    for name, path in CHECKPOINTS.items()
}

dataset = FloodDataset("test")

if TARGET not in [
    dataset[i]["name"]
    for i in range(len(dataset))
]:
    raise RuntimeError(
        f"{TARGET} is not in the test split."
    )



overall = defaultdict(
    lambda: np.zeros(4, dtype=np.int64)
)

regional = defaultdict(
    lambda: np.zeros(4, dtype=np.int64)
)

image_rows = []
target_predictions = {}

print("\nEVALUATING U-NET V1 AND V2")
print("=" * 60)

with torch.inference_mode():

    for i in range(len(dataset)):
        sample = dataset[i]

        name = sample["name"]
        region = name.rsplit("_", 1)[0]

        image = sample["image"].unsqueeze(0)
        truth = sample["mask"][0].numpy() >= 0.5
        valid = sample["valid"][0].numpy() > 0.5

        image_scores = {}

        for version, model in models.items():
            logits = model(image)

            probabilities = torch.sigmoid(
                logits
            )[0, 0].numpy()

            pred = probabilities >= THRESHOLD

            counts = counts_from_arrays(
                pred, truth, valid
            )

            overall[version] += counts
            regional[(version, region)] += counts

            scores = metrics(counts)
            image_scores[version] = scores["iou"]

            image_rows.append({
                "image": name,
                "region": region,
                "model": version,
                **scores
            })

            if name == TARGET:
                target_predictions[version] = (
                    pred.copy(),
                    truth.copy(),
                    valid.copy()
                )

            # Save only V2 predictions.
            if version == "V2":
                image_path = (
                    RAW / f"{name}_S1Hand.tif"
                )

                with rasterio.open(image_path) as src:
                    profile = src.profile.copy()

                    if src.crs is None:
                        raise ValueError(
                            f"Missing CRS for {name}"
                        )

                    if src.shape != pred.shape:
                        raise ValueError(
                            f"Raster shape mismatch for {name}"
                        )

                output = np.full(
                    pred.shape, 255, dtype=np.uint8
                )

                output[valid] = 0
                output[pred & valid] = 1

                profile.update(
                    driver="GTiff",
                    count=1,
                    dtype="uint8",
                    nodata=255,
                    compress="lzw"
                )

                with rasterio.open(
                    MASK_DIR / f"{name}_water_v2.tif",
                    "w",
                    **profile
                ) as dst:
                    dst.write(output, 1)

        print(
            f"[{i+1}/{len(dataset)}] {name} | "
            f"V1 IoU={image_scores['V1']:.4f} | "
            f"V2 IoU={image_scores['V2']:.4f}",
            flush=True
        )


summary_rows = []
regional_rows = []

for version in models:
    summary_rows.append({
        "model": version,
        **metrics(overall[version])
    })

for (version, region), counts in sorted(
    regional.items()
):
    regional_rows.append({
        "model": version,
        "region": region,
        **metrics(counts)
    })

pd.DataFrame(summary_rows).to_csv(
    ROOT / "summary.csv",
    index=False
)

pd.DataFrame(regional_rows).to_csv(
    ROOT / "regional.csv",
    index=False
)

pd.DataFrame(image_rows).to_csv(
    ROOT / "per_image.csv",
    index=False
)


print("\nEVALUATING BUILDINGS AND ROADS")
print("=" * 60)

buildings = gpd.read_file(
    INFRA / "buildings.geojson"
)

roads = gpd.read_file(
    INFRA / "roads.geojson"
)

image_path = RAW / f"{TARGET}_S1Hand.tif"

with rasterio.open(image_path) as src:
    crs = src.crs
    transform = src.transform

if crs is None:
    raise ValueError("Target raster has no CRS.")

buildings = buildings.to_crs(crs)
roads = roads.to_crs(crs)


metric_crs = gpd.GeoSeries(
    [buildings.total_bounds],
    crs=crs
) if False else None

from rasterio.transform import array_bounds

h, w = target_predictions["V1"][0].shape
south, west, north, east = (
    None, None, None, None
)

west, south, east, north = array_bounds(
    h, w, transform
)

footprint = gpd.GeoSeries(
    [Polygon([
        (west, south),
        (east, south),
        (east, north),
        (west, north)
    ])],
    crs=crs
)

metric_crs = footprint.estimate_utm_crs()

if metric_crs is None:
    raise ValueError("No appropriate metric CRS.")

buildings_m = buildings.to_crs(metric_crs)
roads_m = roads.to_crs(metric_crs)


def mask_to_polygon(mask):
    geometries = [
        shape(geom)
        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform
        )
        if value == 1
    ]

    if not geometries:
        return Polygon()

    return unary_union(geometries)


def exposure_flags(geometry, kind):
    geometry_m = gpd.GeoSeries(
        [geometry],
        crs=crs
    ).to_crs(metric_crs).iloc[0]

    building_overlap = (
        buildings_m.geometry.intersection(geometry_m)
    )

    road_overlap = (
        roads_m.geometry.intersection(geometry_m)
    )

    return (
        (building_overlap.area > 0).to_numpy(),
        (road_overlap.length > 0).to_numpy(),
        float(road_overlap.length.sum())
    )


# Same reference water and validity region
# for both versions.
_, truth, valid = target_predictions["V1"]

reference_polygon = mask_to_polygon(
    truth & valid
)

ref_buildings, ref_roads, ref_road_m = (
    exposure_flags(reference_polygon, "reference")
)

exposure_rows = []

for version in models:
    pred, _, _ = target_predictions[version]

    predicted_polygon = mask_to_polygon(
        pred & valid
    )

    pred_buildings, pred_roads, pred_road_m = (
        exposure_flags(predicted_polygon, version)
    )

    tp = int(np.sum(pred_buildings & ref_buildings))
    fp = int(np.sum(pred_buildings & ~ref_buildings))
    fn = int(np.sum(~pred_buildings & ref_buildings))
    tn = int(np.sum(~pred_buildings & ~ref_buildings))

    building_counts = np.array(
        [tp, fp, fn, tn],
        dtype=np.int64
    )

    scores = metrics(building_counts)

    road_tp = int(np.sum(pred_roads & ref_roads))
    road_fp = int(np.sum(pred_roads & ~ref_roads))
    road_fn = int(np.sum(~pred_roads & ref_roads))

    exposure_rows.append({
        "model": version,
        "predicted_exposed_buildings": int(
            pred_buildings.sum()
        ),
        "reference_exposed_buildings": int(
            ref_buildings.sum()
        ),
        "building_TP": tp,
        "building_FP": fp,
        "building_FN": fn,
        "building_TN": tn,
        "building_precision": scores["precision"],
        "building_recall": scores["recall"],
        "building_f1": scores["f1"],
        "predicted_exposed_road_segments": int(
            pred_roads.sum()
        ),
        "reference_exposed_road_segments": int(
            ref_roads.sum()
        ),
        "road_TP": road_tp,
        "road_FP": road_fp,
        "road_FN": road_fn,
        "predicted_exposed_road_m": round(
            pred_road_m, 3
        ),
        "reference_exposed_road_m": round(
            ref_road_m, 3
        )
    })

pd.DataFrame(exposure_rows).to_csv(
    ROOT / "exposure_comparison.csv",
    index=False
)



df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(8, 5))

metrics_to_plot = [
    "precision", "recall", "f1", "iou"
]

x = np.arange(len(metrics_to_plot))
width = 0.35

for j, version in enumerate(["V1", "V2"]):
    row = df[df["model"] == version].iloc[0]

    values = [
        row[metric] * 100
        for metric in metrics_to_plot
    ]

    ax.bar(
        x + (j - 0.5) * width,
        values,
        width,
        label=version
    )

ax.set_xticks(x)
ax.set_xticklabels([
    m.upper() for m in metrics_to_plot
])

ax.set_ylabel("Test performance (%)")
ax.set_ylim(0, 100)
ax.set_title("GeoRescue U-Net V1 vs V2")
ax.legend()

plt.tight_layout()
plt.savefig(
    ROOT / "test_comparison.png",
    dpi=200
)
plt.close()



print("\n" + "=" * 60)
print("GEORESCUE U-NET V2 TEST RESULTS")
print("=" * 60)

print(
    pd.DataFrame(summary_rows)
    .round(4)
    .to_string(index=False)
)

print("\nREGIONAL RESULTS")
print(
    pd.DataFrame(regional_rows)
    .round(4)
    .to_string(index=False)
)

print("\nINFRASTRUCTURE EXPOSURE RESULTS")
print(
    pd.DataFrame(exposure_rows)
    .round(4)
    .to_string(index=False)
)

print("Results saved to:", ROOT)
