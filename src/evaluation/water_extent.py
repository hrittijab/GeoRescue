from pathlib import Path
import csv
import json
import numpy as np
import rasterio
import torch

from pyproj import CRS, Geod, Transformer
from rasterio.transform import xy

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet

OUTPUT = Path("experiments/geospatial")
MASK_DIR = OUTPUT / "predicted_masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = Path("experiments/unet/best_unet.pt")
THRESHOLD = 0.5

torch.set_num_threads(min(4, torch.get_num_threads()))

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model = UNet(
    in_channels=checkpoint["config"]["in_channels"],
    base=checkpoint["config"]["base"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = FloodDataset("test")

def pixel_area_m2(src):
    if src.crs is None:
        raise ValueError("Missing CRS")

    crs = CRS.from_user_input(src.crs)
    transform = src.transform

    # Geographic CRS: calculate geodesic area per row.
    if crs.is_geographic:
        geod = crs.get_geod()

        if geod is None:
            geod = Geod(ellps="WGS84")

        areas = np.empty(
            (src.height, src.width),
            dtype=np.float64
        )

        for row in range(src.height):
            for col in range(src.width):
                corners = [
                    transform * (col, row),
                    transform * (col + 1, row),
                    transform * (col + 1, row + 1),
                    transform * (col, row + 1)
                ]

                lons, lats = zip(*corners)

                area, _ = geod.polygon_area_perimeter(
                    lons, lats
                )

                areas[row, col] = abs(area)

        return areas

    # Projected CRS: affine pixel area and linear units.
    if crs.is_projected:
        axes = crs.axis_info

        if len(axes) < 2:
            raise ValueError("Cannot determine CRS units")

        x_factor = axes[0].unit_conversion_factor
        y_factor = axes[1].unit_conversion_factor

        if x_factor is None or y_factor is None:
            raise ValueError("Unknown projected CRS units")

        affine_area = abs(
            transform.a * transform.e
            - transform.b * transform.d
        )

        area_m2 = affine_area * x_factor * y_factor

        return np.full(
            (src.height, src.width),
            area_m2,
            dtype=np.float64
        )

    raise ValueError(f"Unsupported CRS: {crs}")


rows = []

print("Starting geospatial water extent analysis...")
print("Test images:", len(dataset))

for index in range(len(dataset)):
    sample = dataset[index]
    name = sample["name"]
    region = name.rsplit("_", 1)[0]

    image_path = Path("data/raw/S1Hand") / (
        f"{name}_S1Hand.tif"
    )

    with rasterio.open(image_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{name}: Missing georeferencing CRS"
            )

        profile = src.profile.copy()
        area = pixel_area_m2(src)

        crs_string = src.crs.to_string()

    with torch.inference_mode():
        image = sample["image"].unsqueeze(0)
        logits = model(image)

        probability = (
            torch.sigmoid(logits)[0, 0]
            .cpu().numpy()
        )

    valid = sample["valid"][0].numpy() > 0.5
    reference = sample["mask"][0].numpy() > 0.5

    prediction = probability >= THRESHOLD

    predicted_water = prediction & valid
    reference_water = reference & valid

    # Output encoding:
    # 0 = non-water
    # 1 = predicted water
    # 255 = invalid / unknown
    output_mask = np.full(
        prediction.shape, 255, dtype=np.uint8
    )

    output_mask[valid] = 0
    output_mask[predicted_water] = 1

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw"
    )

    mask_path = MASK_DIR / f"{name}_water.tif"

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(output_mask, 1)

    predicted_km2 = (
        area[predicted_water].sum() / 1_000_000
    )

    reference_km2 = (
        area[reference_water].sum() / 1_000_000
    )

    valid_km2 = (
        area[valid].sum() / 1_000_000
    )

    difference_km2 = predicted_km2 - reference_km2

    row = {
        "image": name,
        "region": region,
        "crs": crs_string,
        "valid_area_km2": round(valid_km2, 5),
        "predicted_water_km2": round(predicted_km2, 5),
        "reference_water_km2": round(reference_km2, 5),
        "difference_km2": round(difference_km2, 5),
        "predicted_water_pct": round(
            100 * predicted_km2 / valid_km2, 3
        ) if valid_km2 else 0
    }

    rows.append(row)

    print(
        f"[{index + 1}/{len(dataset)}] {name}: "
        f"Predicted={predicted_km2:.3f} km2 | "
        f"Reference={reference_km2:.3f} km2",
        flush=True
    )

csv_path = OUTPUT / "water_extent.csv"

with open(
    csv_path, "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 55)
print("GEORESCUE GEOSPATIAL ANALYSIS COMPLETE")
print("=" * 55)

total_predicted = sum(
    r["predicted_water_km2"] for r in rows
)

total_reference = sum(
    r["reference_water_km2"] for r in rows
)

print(f"Total predicted water: {total_predicted:.3f} km2")
print(f"Total reference water: {total_reference:.3f} km2")
print(f"CSV saved: {csv_path}")
print(f"GeoTIFF masks saved: {MASK_DIR}")
