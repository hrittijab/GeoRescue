from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
OUTPUT = Path("experiments/geographic_evaluation")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_IMAGE = 2000
TREES = 60

# ----------------------------------
# Load data
# ----------------------------------

def load_image(path):
    name = path.stem.replace("_S1Hand", "")
    group = name.rsplit("_", 1)[0]

    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    with rasterio.open(path) as src:
        image = src.read([1, 2])

    with rasterio.open(label_path) as src:
        label = src.read(1)

    if image.shape[1:] != label.shape:
        raise ValueError(f"Shape mismatch: {name}")

    vv = image[0]
    vh = image[1]

    valid = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & np.isin(label, [0, 1])
    )

    X = np.column_stack((vv[valid], vh[valid]))
    y = label[valid].astype(np.uint8)

    return group, X, y


files = sorted(IMAGE_DIR.glob("*_S1Hand.tif"))
groups = defaultdict(list)

print("Loading satellite images...", flush=True)

for path in files:
    name = path.stem.replace("_S1Hand", "")
    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        print(f"Skipping missing label: {name}")
        continue

    group, X, y = load_image(path)

    if len(y):
        groups[group].append((path, X, y))

group_names = sorted(groups)

print(f"Images loaded: {sum(map(len, groups.values()))}")
print(f"Geographic groups: {len(group_names)}")

if len(group_names) < 2:
    raise RuntimeError("Need at least two geographic groups.")

# ----------------------------------
# Train one model per held-out group
# ----------------------------------

results = []
result_path = OUTPUT / "evaluation.csv"

columns = [
    "test_group", "test_images", "test_pixels",
    "water_pixels", "water_prevalence",
    "precision", "recall", "f1", "iou",
    "tn", "fp", "fn", "tp"
]

with open(result_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

rng = np.random.default_rng(SEED)

for fold, held_out in enumerate(group_names, 1):
    print(
        f"\n[{fold}/{len(group_names)}] "
        f"Testing on: {held_out}",
        flush=True
    )

    X_train_parts = []
    y_train_parts = []

    for group in group_names:
        if group == held_out:
            continue

        for _, X, y in groups[group]:
            n = min(SAMPLES_PER_IMAGE, len(y))

            indices = rng.choice(
                len(y), size=n, replace=False
            )

            X_train_parts.append(X[indices])
            y_train_parts.append(y[indices])

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    if np.unique(y_train).size < 2:
        raise RuntimeError(
            "Training data contains fewer than two classes."
        )

    print(
        f"Training on {len(X_train):,} pixels...",
        flush=True
    )

    model = RandomForestClassifier(
        n_estimators=TREES,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    total_cm = np.zeros((2, 2), dtype=np.int64)
    test_pixels = 0

    for path, X_test, y_test in groups[held_out]:
        predictions = model.predict(X_test)

        total_cm += confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        test_pixels += len(y_test)

    tn, fp, fn, tp = total_cm.ravel()

    precision = (
        tp / (tp + fp) if (tp + fp) else 0.0
    )
    recall = (
        tp / (tp + fn) if (tp + fn) else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) else 0.0
    )

    water_pixels = int(tp + fn)

    row = {
        "test_group": held_out,
        "test_images": len(groups[held_out]),
        "test_pixels": test_pixels,
        "water_pixels": water_pixels,
        "water_prevalence": round(
            water_pixels / test_pixels, 6
        ),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    results.append(row)

    # Save each fold immediately so progress is retained.
    with open(result_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writerow(row)

    print(
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"IoU={iou:.4f}",
        flush=True
    )

# ----------------------------------
# Summary and visualization
# ----------------------------------

names = [r["test_group"] for r in results]
ious = [r["iou"] for r in results]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(names, ious)

ax.set_title("GeoRescue: Held-Out Geographic Group IoU")
ax.set_ylabel("Water Intersection over Union")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(
    OUTPUT / "geographic_performance.png",
    dpi=200
)
plt.close()

print("\n" + "=" * 55)
print("GEORESCUE GEOGRAPHIC EVALUATION COMPLETE")
print("=" * 55)

for metric in ["precision", "recall", "f1", "iou"]:
    values = [r[metric] for r in results]
    print(f"Macro {metric}: {np.mean(values):.4f}")

print(f"\nSaved: {result_path}")
print(f"Saved: {OUTPUT / 'geographic_performance.png'}")
