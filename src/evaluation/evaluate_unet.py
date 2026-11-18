from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet

OUTPUT = Path("experiments/unet/evaluation")
OUTPUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = Path("experiments/unet/best_unet.pt")
DEVICE = torch.device("cpu")
THRESHOLD = 0.5

# Use validation only for now.
# The test split stays out of this development step.
dataset = FloodDataset("val")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model = UNet(
    in_channels=checkpoint["config"]["in_channels"],
    base=checkpoint["config"]["base"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded U-Net checkpoint")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation images:", len(dataset))

def metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    union = tp + fp + fn
    iou = tp / union if union else float("nan")

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else float("nan")
    )

    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr
    }

rows = []
regional_counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))
total_counts = np.zeros(4, dtype=np.int64)

for index in range(len(dataset)):
    sample = dataset[index]
    name = sample["name"]
    region = name.rsplit("_", 1)[0]

    image = sample["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image)
        probability = torch.sigmoid(logits)[0, 0].numpy()

    truth = sample["mask"][0].numpy() > 0.5
    valid = sample["valid"][0].numpy() > 0.5

    prediction = probability >= THRESHOLD

    tp = int(np.sum(prediction & truth & valid))
    fp = int(np.sum(prediction & ~truth & valid))
    fn = int(np.sum(~prediction & truth & valid))
    tn = int(np.sum(~prediction & ~truth & valid))

    counts = np.array([tp, fp, fn, tn])
    regional_counts[region] += counts
    total_counts += counts

    scores = metrics(tp, fp, fn, tn)

    rows.append({
        "image": name,
        "region": region,
        "water_pixels": int(np.sum(truth & valid)),
        "valid_pixels": int(np.sum(valid)),
        **scores
    })

    # Visualization:
    # 0 = correct land, 1 = correct water
    # 2 = false positive, 3 = false negative
    error_map = np.full(truth.shape, np.nan)
    error_map[valid & ~truth & ~prediction] = 0
    error_map[valid & truth & prediction] = 1
    error_map[valid & ~truth & prediction] = 2
    error_map[valid & truth & ~prediction] = 3

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    vv = sample["image"][0].numpy()

    axes[0].imshow(vv, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Sentinel-1 VV")

    truth_view = np.ma.masked_where(~valid, truth)
    axes[1].imshow(
        truth_view, cmap="Blues", vmin=0, vmax=1
    )
    axes[1].set_title("Reference Water")

    pred_view = np.ma.masked_where(~valid, prediction)
    axes[2].imshow(
        pred_view, cmap="Blues", vmin=0, vmax=1
    )
    axes[2].set_title("U-Net Prediction")

    cmap = ListedColormap([
        "#dddddd",
        "#2171b5",
        "#ff9900",
        "#cc2222"
    ])

    axes[3].imshow(
        error_map, cmap=cmap, vmin=0, vmax=3,
        interpolation="nearest"
    )
    axes[3].set_title("Errors: Orange=FP, Red=FN")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{name} | IoU={scores['iou']:.3f}",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT / f"{name}_unet.png",
        dpi=140,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"[{index + 1}/{len(dataset)}] {name}: "
        f"IoU={scores['iou']:.4f}",
        flush=True
    )

# Save per-image evaluation
with open(
    OUTPUT / "image_metrics.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Save aggregated regional evaluation
regional_rows = []

for region, counts in sorted(regional_counts.items()):
    tp, fp, fn, tn = map(int, counts)
    regional_rows.append({
        "region": region,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **metrics(tp, fp, fn, tn)
    })

with open(
    OUTPUT / "regional_metrics.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=regional_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(regional_rows)

tp, fp, fn, tn = map(int, total_counts)
overall = metrics(tp, fp, fn, tn)

print("\n" + "=" * 55)
print("GEORESCUE U-NET VALIDATION EVALUATION")
print("=" * 55)

for metric, value in overall.items():
    print(f"{metric.upper()}: {value:.4f}")

valid_region_ious = [
    r["iou"] for r in regional_rows
    if np.isfinite(r["iou"])
]

print(
    "Regional macro IoU:",
    round(float(np.mean(valid_region_ious)), 4)
)

print("\nRegional results:")
for row in regional_rows:
    print(
        f"{row['region']}: "
        f"IoU={row['iou']:.4f}, "
        f"FPR={row['fpr']:.4f}"
    )

print("\nSaved evaluation files to:", OUTPUT)
print("UNET EVALUATION COMPLETE!")
