from pathlib import Path
from collections import defaultdict
import json
import csv

import numpy as np
import rasterio
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet

SEED = 42
SAMPLES_PER_IMAGE = 2000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUTPUT = Path("experiments/final_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(min(4, torch.get_num_threads()))
rng = np.random.default_rng(SEED)

with open("data/splits/unet_split.json", encoding="utf-8") as f:
    splits = json.load(f)


# ---------------------------------------
# Shared evaluation functions
# ---------------------------------------

def get_region(name):
    return name.rsplit("_", 1)[0]


def get_counts(prob, labels, valid, threshold):
    prediction = (prob >= threshold) & valid
    truth = (labels == 1) & valid

    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth & valid))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth & valid))

    return np.array([tp, fp, fn, tn], dtype=np.int64)


def calculate_metrics(counts):
    tp, fp, fn, tn = map(int, counts)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def load_rf_image(name):
    image_path = Path("data/raw/S1Hand") / f"{name}_S1Hand.tif"
    label_path = Path("data/raw/LabelHand") / f"{name}_LabelHand.tif"

    with rasterio.open(image_path) as src:
        image = src.read([1, 2]).astype(np.float32)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    if image.shape[1:] != label.shape:
        raise ValueError(f"Shape mismatch: {name}")

    valid = (
        np.isfinite(image).all(axis=0)
        & np.isin(label, [0, 1])
    )

    X = np.column_stack([
        image[0][valid],
        image[1][valid]
    ])

    y = label[valid].astype(np.uint8)

    return X, y


# ---------------------------------------
# Train Random Forest
# ---------------------------------------

print("Preparing Random Forest training data...", flush=True)

X_parts = []
y_parts = []

for name in splits["train"]:
    X, y = load_rf_image(name)

    if len(y) == 0:
        continue

    n = min(SAMPLES_PER_IMAGE, len(y))
    idx = rng.choice(len(y), size=n, replace=False)

    X_parts.append(X[idx])
    y_parts.append(y[idx])

X_train = np.concatenate(X_parts)
y_train = np.concatenate(y_parts)

if np.unique(y_train).size < 2:
    raise RuntimeError("Training requires both classes.")

print(f"Training pixels: {len(y_train):,}", flush=True)

rf = RandomForestClassifier(
    n_estimators=60,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Random Forest trained!", flush=True)


# ---------------------------------------
# Select RF threshold on validation data
# ---------------------------------------

print("\nSelecting Random Forest threshold...", flush=True)

validation = defaultdict(list)

for name in splits["val"]:
    X, y = load_rf_image(name)

    if len(y) == 0:
        continue

    probability = rf.predict_proba(X)[:, 1]
    validation[get_region(name)].append((probability, y))

threshold_scores = {}

for threshold in THRESHOLDS:
    region_ious = []

    for region, images in validation.items():
        counts = np.zeros(4, dtype=np.int64)

        for prob, y in images:
            counts += get_counts(
                prob, y,
                np.ones(len(y), dtype=bool),
                threshold
            )

        iou = calculate_metrics(counts)["iou"]

        if np.isfinite(iou):
            region_ious.append(iou)

    threshold_scores[threshold] = (
        float(np.mean(region_ious))
        if region_ious else float("nan")
    )

eligible = [
    t for t in THRESHOLDS
    if np.isfinite(threshold_scores[t])
]

if not eligible:
    raise RuntimeError("No valid validation IoU scores.")

best_threshold = min(
    eligible,
    key=lambda t: (
        -threshold_scores[t],
        abs(t - 0.5),
        t
    )
)

print("Selected RF threshold:", best_threshold)

with open(OUTPUT / "rf_thresholds.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold", "validation_macro_iou"])

    for threshold, score in threshold_scores.items():
        writer.writerow([threshold, score])


# ---------------------------------------
# Load trained U-Net
# ---------------------------------------

print("\nLoading U-Net checkpoint...", flush=True)

checkpoint = torch.load(
    "experiments/unet/best_unet.pt",
    map_location="cpu",
    weights_only=True
)

unet = UNet(
    in_channels=checkpoint["config"]["in_channels"],
    base=checkpoint["config"]["base"]
)

unet.load_state_dict(checkpoint["model_state_dict"])
unet.eval()

print("U-Net loaded from epoch:", checkpoint["epoch"])

# ---------------------------------------
# Evaluate both on identical test images
# ---------------------------------------

test_dataset = FloodDataset("test")

regional = {
    "Random Forest": defaultdict(
        lambda: np.zeros(4, dtype=np.int64)
    ),
    "U-Net": defaultdict(
        lambda: np.zeros(4, dtype=np.int64)
    )
}

image_rows = []

for index in range(len(test_dataset)):
    sample = test_dataset[index]
    name = sample["name"]
    region = get_region(name)

    X, y = load_rf_image(name)

    if len(y) == 0:
        continue

    rf_prob = rf.predict_proba(X)[:, 1]

    rf_counts = get_counts(
        rf_prob, y,
        np.ones(len(y), dtype=bool),
        best_threshold
    )

    with torch.inference_mode():
        tensor = sample["image"].unsqueeze(0)
        logits = unet(tensor)
        unet_prob = torch.sigmoid(logits)[0, 0].numpy()

    labels = sample["mask"][0].numpy()
    valid = sample["valid"][0].numpy() > 0.5

    unet_counts = get_counts(
        unet_prob, labels, valid, 0.5
    )

    for model_name, counts in [
        ("Random Forest", rf_counts),
        ("U-Net", unet_counts)
    ]:
        regional[model_name][region] += counts

        image_rows.append({
            "model": model_name,
            "image": name,
            "region": region,
            **calculate_metrics(counts)
        })

    print(
        f"[{index + 1}/{len(test_dataset)}] "
        f"{name} | RF IoU={calculate_metrics(rf_counts)['iou']:.3f} "
        f"| U-Net IoU={calculate_metrics(unet_counts)['iou']:.3f}",
        flush=True
    )


# ---------------------------------------
# Save and display comparison
# ---------------------------------------

regional_rows = []
summary_rows = []

for model_name, region_counts in regional.items():
    all_counts = np.zeros(4, dtype=np.int64)
    region_ious = []

    for region, counts in sorted(region_counts.items()):
        scores = calculate_metrics(counts)

        regional_rows.append({
            "model": model_name,
            "region": region,
            **scores
        })

        all_counts += counts

        if np.isfinite(scores["iou"]):
            region_ious.append(scores["iou"])

    overall = calculate_metrics(all_counts)

    summary_rows.append({
        "model": model_name,
        "macro_region_iou": float(np.mean(region_ious)),
        **overall
    })


def save_csv(filename, rows):
    if not rows:
        return

    with open(OUTPUT / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


save_csv("image_comparison.csv", image_rows)
save_csv("regional_comparison.csv", regional_rows)
save_csv("summary.csv", summary_rows)

models = [r["model"] for r in summary_rows]
ious = [r["iou"] * 100 for r in summary_rows]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, ious)

ax.set_ylabel("Pixel-aggregated Water IoU (%)")
ax.set_title("GeoRescue: RF vs U-Net — Same Test Split")
ax.set_ylim(0, 100)

for bar, value in zip(bars, ious):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT / "model_comparison.png",
    dpi=180
)
plt.close()

print("\n" + "=" * 55)
print("GEORESCUE FINAL MODEL COMPARISON")
print("=" * 55)

for row in summary_rows:
    print(f"\n{row['model']}")
    for metric in [
        "precision", "recall", "f1",
        "iou", "fpr", "macro_region_iou"
    ]:
        print(f"{metric}: {row[metric]:.4f}")

print("\nMODEL COMPARISON COMPLETE!")
print("Results saved to:", OUTPUT)
