from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import pandas as pd
import rasterio

from sklearn.ensemble import RandomForestClassifier

IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
OUTPUT = Path("experiments/threshold_calibration")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_IMAGE = 2000
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

rng = np.random.default_rng(SEED)
groups = defaultdict(list)

print("Loading satellite data...", flush=True)

# ---------------------------------------
# Load images and sample training pixels
# ---------------------------------------

for path in sorted(IMAGE_DIR.glob("*_S1Hand.tif")):
    name = path.stem.replace("_S1Hand", "")
    group = name.rsplit("_", 1)[0]
    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        continue

    with rasterio.open(path) as src:
        image = src.read([1, 2])

    with rasterio.open(label_path) as src:
        label = src.read(1)

    if image.shape[1:] != label.shape:
        raise ValueError(f"Shape mismatch: {name}")

    valid = (
        np.isfinite(image[0])
        & np.isfinite(image[1])
        & np.isin(label, [0, 1])
    )

    X = np.column_stack([
        image[0][valid],
        image[1][valid]
    ]).astype(np.float32)

    y = label[valid].astype(np.uint8)

    if len(y) == 0:
        continue

    n = min(SAMPLES_PER_IMAGE, len(y))
    indices = rng.choice(len(y), size=n, replace=False)

    groups[group].append({
        "X": X,
        "y": y,
        "X_sample": X[indices],
        "y_sample": y[indices]
    })

names = sorted(groups)

if len(names) < 4:
    raise RuntimeError("Need at least 4 geographic groups.")

print(f"Loaded {sum(len(v) for v in groups.values())} images")
print(f"Geographic groups: {len(names)}")


# ---------------------------------------
# Helper functions
# ---------------------------------------

def train_model(training_groups):
    X_parts = []
    y_parts = []

    for group in training_groups:
        for item in groups[group]:
            X_parts.append(item["X_sample"])
            y_parts.append(item["y_sample"])

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)

    if np.unique(y_train).size < 2:
        raise RuntimeError("Training requires both classes.")

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_group(model, group, threshold):
    tp = fp = tn = fn = 0

    for item in groups[group]:
        probability = model.predict_proba(item["X"])[:, 1]

        # Strict comparison preserves non-water on exact ties.
        prediction = probability > threshold
        truth = item["y"] == 1

        tp += int(np.sum(prediction & truth))
        fp += int(np.sum(prediction & ~truth))
        tn += int(np.sum(~prediction & ~truth))
        fn += int(np.sum(~prediction & truth))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    union = tp + fp + fn
    iou = tp / union if union else float("nan")
    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr,
        "water_prevalence": (
            (tp + fn) / (tp + fp + tn + fn)
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# ---------------------------------------
# Geographic threshold calibration
# ---------------------------------------

results = []
validation_rows = []

output_file = OUTPUT / "calibration_results.csv"

for fold, held_out in enumerate(names):
    print(
        f"\n[{fold + 1}/{len(names)}] "
        f"Held-out test: {held_out}",
        flush=True
    )

    # Select two validation groups, never the test group.
    remaining = [g for g in names if g != held_out]

    offset = fold % len(remaining)
    validation_groups = [
        remaining[offset],
        remaining[(offset + 1) % len(remaining)]
    ]

    calibration_train = [
        g for g in remaining
        if g not in validation_groups
    ]

    print("Validation groups:", validation_groups)

    calibration_model = train_model(calibration_train)

    # Cache validation probabilities to avoid repeated inference.
    validation_cache = {}

    for group in validation_groups:
        validation_cache[group] = [
            (
                calibration_model.predict_proba(item["X"])[:, 1],
                item["y"]
            )
            for item in groups[group]
        ]

    scores = {}

    for threshold in THRESHOLDS:
        group_ious = []

        for group in validation_groups:
            tp = fp = fn = 0

            for probabilities, labels in validation_cache[group]:
                pred = probabilities > threshold
                truth = labels == 1

                tp += int(np.sum(pred & truth))
                fp += int(np.sum(pred & ~truth))
                fn += int(np.sum(~pred & truth))

            union = tp + fp + fn
            if union:
                group_ious.append(tp / union)

        score = (
            float(np.mean(group_ious))
            if group_ious else float("nan")
        )

        scores[threshold] = score

        validation_rows.append({
            "test_group": held_out,
            "validation_groups": ",".join(validation_groups),
            "threshold": threshold,
            "validation_macro_iou": score
        })

    # Select threshold using validation results only.
    eligible = [
        t for t in THRESHOLDS
        if np.isfinite(scores[t])
    ]

    if eligible:
        best_threshold = min(
            eligible,
            key=lambda t: (
                -scores[t],
                abs(t - 0.50),
                t
            )
        )
    else:
        best_threshold = 0.50

    print(
        f"Selected threshold: {best_threshold:.2f}",
        flush=True
    )

    # Retrain using all non-test groups.
    final_model = train_model(remaining)

    # Compare both thresholds on the same final model.
    for variant, threshold in [
        ("baseline_050", 0.50),
        ("calibrated", best_threshold)
    ]:
        metrics = evaluate_group(
            final_model, held_out, threshold
        )

        row = {
            "test_group": held_out,
            "variant": variant,
            "threshold": threshold,
            **metrics
        }

        results.append(row)

        print(
            f"{variant}: "
            f"IoU={metrics['iou']:.4f}, "
            f"FPR={metrics['fpr']:.4f}, "
            f"Recall={metrics['recall']:.4f}",
            flush=True
        )

    pd.DataFrame(results).to_csv(
        output_file, index=False
    )

    pd.DataFrame(validation_rows).to_csv(
        OUTPUT / "validation_scores.csv",
        index=False
    )


# ---------------------------------------
# Final summary
# ---------------------------------------

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("GEORESCUE THRESHOLD CALIBRATION COMPLETE")
print("=" * 60)

for variant in ["baseline_050", "calibrated"]:
    subset = df[df["variant"] == variant]

    print(f"\n{variant.upper()}")

    for metric in [
        "precision", "recall", "f1", "iou", "fpr"
    ]:
        print(
            f"{metric}: {subset[metric].mean():.4f}"
        )

print("\nThresholds selected:")
print(
    df[df["variant"] == "calibrated"][
        ["test_group", "threshold", "iou"]
    ].to_string(index=False)
)

print(f"\nResults saved: {output_file}")
