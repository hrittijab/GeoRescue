from pathlib import Path
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
FIGURE_DIR = Path("reports/figures")
RESULT_DIR = Path("reports/results")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def display_band(band):
    """Scale finite radar values for visualization."""
    valid = band[np.isfinite(band)]

    if valid.size == 0:
        return np.zeros_like(band)

    low, high = np.percentile(valid, [2, 98])

    if high <= low:
        return np.zeros_like(band)

    scaled = (band - low) / (high - low)
    return np.clip(
        np.nan_to_num(scaled, nan=0),
        0, 1
    )


results = []

for image_path in sorted(IMAGE_DIR.glob("*_S1Hand.tif")):
    name = image_path.name.replace("_S1Hand.tif", "")
    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        print(f"Missing label: {name}")
        continue

    with rasterio.open(image_path) as src:
        image = src.read()
        image_bounds = src.bounds
        image_crs = str(src.crs)

    with rasterio.open(label_path) as src:
        mask = src.read(1)

    if image.shape[1:] != mask.shape:
        raise ValueError(f"Shape mismatch: {name}")

    if image.shape[0] < 2:
        raise ValueError(f"Expected two radar bands: {name}")

    vv = image[0]
    vh = image[1]

    valid_radar = (
        np.isfinite(vv) & np.isfinite(vh)
    )

    valid_labels = (mask == 0) | (mask == 1)
    valid = valid_radar & valid_labels

    water = (mask == 1) & valid
    land = (mask == 0) & valid

    water_count = int(water.sum())
    land_count = int(land.sum())
    valid_count = int(valid.sum())

    water_pct = (
        water_count / valid_count * 100
        if valid_count else 0
    )

    invalid_count = int(mask.size - valid_count)

    results.append({
        "region": name,
        "total_pixels": int(mask.size),
        "valid_pixels": valid_count,
        "water_pixels": water_count,
        "land_pixels": land_count,
        "invalid_pixels": invalid_count,
        "water_percentage": round(water_pct, 2),
        "vv_mean_water_db": (
            round(float(vv[water].mean()), 3)
            if water_count else None
        ),
        "vv_mean_land_db": (
            round(float(vv[land].mean()), 3)
            if land_count else None
        ),
        "vh_mean_water_db": (
            round(float(vh[water].mean()), 3)
            if water_count else None
        ),
        "vh_mean_land_db": (
            round(float(vh[land].mean()), 3)
            if land_count else None
        ),
        "crs": image_crs
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    axes[0].imshow(
        display_band(vv), cmap="gray"
    )
    axes[0].set_title("Sentinel-1 VV")

    axes[1].imshow(
        display_band(vh), cmap="gray"
    )
    axes[1].set_title("Sentinel-1 VH")

    # Invalid pixels gray, land white, water blue
    shown_mask = np.full(mask.shape, 0)
    shown_mask[valid & (mask == 0)] = 1
    shown_mask[valid & (mask == 1)] = 2

    cmap = ListedColormap([
        "#888888", "#f5f5f5", "#2171b5"
    ])

    axes[2].imshow(
        shown_mask, cmap=cmap,
        vmin=0, vmax=2,
        interpolation="nearest"
    )
    axes[2].set_title(
        f"Water Mask ({water_pct:.1f}% water)"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"GeoRescue | {name}",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    output = FIGURE_DIR / f"{name}.png"
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(
        f"{name}: {water_pct:.2f}% water, "
        f"{invalid_count} invalid pixels"
    )
    print(f"  Saved {output}")


if not results:
    raise RuntimeError("No image-label pairs processed.")

csv_path = RESULT_DIR / "dataset_statistics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)


regions = [r["region"] for r in results]
percentages = [r["water_percentage"] for r in results]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(regions, percentages, color="#2878b5")

ax.set_title("GeoRescue: Water Coverage by Sample")
ax.set_ylabel("Water (% of valid pixels)")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", labelrotation=20)

for bar, value in zip(bars, percentages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "water_distribution.png",
    dpi=160,
    bbox_inches="tight"
)
plt.close(fig)

print("\nEDA COMPLETE!")
print(f"Processed {len(results)} image-label pairs")
print(f"Statistics saved to: {csv_path}")
print(f"Figures saved to: {FIGURE_DIR}")
