from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import rasterio

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
OUTPUT = Path("experiments/model_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_IMAGE = 2000
rng = np.random.default_rng(SEED)

groups = defaultdict(list)

print("Loading satellite data...", flush=True)

for path in sorted(IMAGE_DIR.glob("*_S1Hand.tif")):
    name = path.stem.replace("_S1Hand", "")
    group = name.rsplit("_", 1)[0]
    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        continue

    with rasterio.open(path) as src:
        image = src.read([1, 2])

    with rasterio.open(label_path) as src:
        labels = src.read(1)

    valid = (
        np.isfinite(image[0])
        & np.isfinite(image[1])
        & np.isin(labels, [0, 1])
    )

    X = np.column_stack([
        image[0][valid],
        image[1][valid]
    ]).astype(np.float32)

    y = labels[valid].astype(np.uint8)

    if len(y):
        groups[group].append((X, y))

names = sorted(groups)

if len(names) < 2:
    raise RuntimeError("Need at least two groups.")

print(f"Loaded {sum(map(len, groups.values()))} images")
print(f"Geographic groups: {len(names)}")

# Same sampled training pixels for both configurations.
sampled = {}

for group in names:
    sampled[group] = []

    for X, y in groups[group]:
        n = min(SAMPLES_PER_IMAGE, len(y))
        idx = rng.choice(len(y), size=n, replace=False)
        sampled[group].append((X[idx], y[idx]))

configs = {
    "balanced": "balanced",
    "unweighted": None
}

columns = [
    "configuration", "test_group",
    "precision", "recall", "f1", "iou",
    "false_positive_rate", "water_prevalence"
]

output_file = OUTPUT / "comparison.csv"
results = []

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for config_name, weight in configs.items():
        print(f"\nMODEL: {config_name}", flush=True)

        for fold, held_out in enumerate(names, 1):
            train_X, train_y = [], []

            for group in names:
                if group == held_out:
                    continue

                for X, y in sampled[group]:
                    train_X.append(X)
                    train_y.append(y)

            X_train = np.concatenate(train_X)
            y_train = np.concatenate(train_y)

            model = RandomForestClassifier(
                n_estimators=60,
                max_depth=15,
                min_samples_leaf=5,
                class_weight=weight,
                random_state=SEED,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            cm = np.zeros((2, 2), dtype=np.int64)

            for X_test, y_test in groups[held_out]:
                pred = model.predict(X_test)

                cm += confusion_matrix(
                    y_test, pred, labels=[0, 1]
                )

            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            iou = (
                tp / (tp + fp + fn)
                if tp + fp + fn else 0
            )

            fpr = fp / (fp + tn) if fp + tn else 0
            prevalence = (
                (tp + fn) / cm.sum() if cm.sum() else 0
            )

            row = {
                "configuration": config_name,
                "test_group": held_out,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "iou": round(iou, 4),
                "false_positive_rate": round(fpr, 4),
                "water_prevalence": round(prevalence, 4)
            }

            writer.writerow(row)
            f.flush()
            results.append(row)

            print(
                f"[{fold}/{len(names)}] {held_out}: "
                f"IoU={iou:.4f}, FPR={fpr:.4f}",
                flush=True
            )

print("\nMODEL COMPARISON COMPLETE")

for config_name in configs:
    rows = [
        row for row in results
        if row["configuration"] == config_name
    ]

    print(f"\n{config_name.upper()}")

    for metric in ["precision", "recall", "f1", "iou"]:
        values = [row[metric] for row in rows]
        print(f"{metric}: {np.mean(values):.4f}")

print(f"\nSaved: {output_file}")
