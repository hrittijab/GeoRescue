from pathlib import Path
import csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix
)



IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
RESULT_DIR = Path("experiments/random_forest")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLES_PER_IMAGE = 12000
N_TREES = 80

rng = np.random.default_rng(RANDOM_SEED)


def load_sample(image_path):
    name = image_path.name.replace("_S1Hand.tif", "")

    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    with rasterio.open(image_path) as src:
        image = src.read()

    with rasterio.open(label_path) as src:
        mask = src.read(1)

    if image.shape[0] < 2:
        raise ValueError(f"Expected two radar bands: {name}")

    if image.shape[1:] != mask.shape:
        raise ValueError(f"Shape mismatch: {name}")

    # VV and VH radar features
    vv = image[0].astype(np.float32)
    vh = image[1].astype(np.float32)

    features = np.stack(
        [vv, vh],
        axis=-1
    )

    # Only valid labels (0=non-water, 1=water)
    valid = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & np.isin(mask, [0, 1])
    )

    X = features[valid]
    y = mask[valid].astype(np.uint8)

    return {
        "name": name,
        "features": features,
        "mask": mask,
        "valid": valid,
        "X": X,
        "y": y
    }


image_paths = sorted(
    IMAGE_DIR.glob("*_S1Hand.tif")
)

samples = [load_sample(p) for p in image_paths]

if len(samples) < 2:
    raise RuntimeError("Need at least two image samples.")

print(f"Loaded {len(samples)} satellite samples.")

for sample in samples:
    print(
        f"{sample['name']}: "
        f"{len(sample['y'])} valid pixels, "
        f"{np.mean(sample['y']) * 100:.2f}% water"
    )


results = []

for test_index, test_sample in enumerate(samples):

    print("\n" + "=" * 55)
    print(f"Testing on: {test_sample['name']}")
    print("=" * 55)

    train_X = []
    train_y = []

    for i, sample in enumerate(samples):
        if i == test_index:
            continue

        n = min(SAMPLES_PER_IMAGE, len(sample["y"]))

        if n == 0:
            continue

        indices = rng.choice(
            len(sample["y"]),
            size=n,
            replace=False
        )

        train_X.append(sample["X"][indices])
        train_y.append(sample["y"][indices])

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)

    X_test = test_sample["X"]
    y_test = test_sample["y"]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            "Training data contains only one class. "
            "Download more diverse images."
        )

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    print("Predicting held-out geographic sample...")
    y_pred = model.predict(X_test)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    iou = jaccard_score(
        y_test, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    results.append({
        "test_region": test_sample["name"],
        "train_pixels": len(y_train),
        "test_pixels": len(y_test),
        "test_water_pixels": int(y_test.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    })

    

    predicted_mask = np.full(
        test_sample["mask"].shape,
        np.nan,
        dtype=np.float32
    )

    actual_mask = predicted_mask.copy()

    predicted_mask[test_sample["valid"]] = y_pred

    actual_mask[test_sample["valid"]] = y_test

    error_map = np.full(
        test_sample["mask"].shape,
        np.nan,
        dtype=np.float32
    )

    # 0 = correct, 1 = false positive,
    # 2 = false negative
    errors = np.zeros_like(y_test, dtype=np.float32)

    errors[(y_test == 0) & (y_pred == 1)] = 1
    errors[(y_test == 1) & (y_pred == 0)] = 2

    error_map[test_sample["valid"]] = errors

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    axes[0].imshow(
        actual_mask, cmap="Blues",
        vmin=0, vmax=1
    )
    axes[0].set_title("Ground Truth")

    axes[1].imshow(
        predicted_mask, cmap="Blues",
        vmin=0, vmax=1
    )
    axes[1].set_title("Random Forest Prediction")

    from matplotlib.colors import ListedColormap

    error_cmap = ListedColormap([
        "#eeeeee",
        "#ff9900",
        "#cc2222"
    ])

    axes[2].imshow(
        error_map,
        cmap=error_cmap,
        vmin=0,
        vmax=2
    )
    axes[2].set_title(
        "Errors: Orange=FP, Red=FN"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{test_sample['name']} | IoU={iou:.3f}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = (
        RESULT_DIR
        / f"{test_sample['name']}_prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


csv_path = RESULT_DIR / "evaluation.csv"

with open(
    csv_path, "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 55)
print("GEORESCUE RANDOM FOREST RESULTS")
print("=" * 55)

for metric in ["precision", "recall", "f1", "iou"]:
    values = [r[metric] for r in results]

    print(
        f"{metric.upper():10}: "
        f"{np.mean(values):.4f} "
        f"+/- {np.std(values):.4f}"
    )

print("\nResults saved to:", csv_path)
print("Prediction maps saved to:", RESULT_DIR)
print("\nBASELINE EXPERIMENT COMPLETE!")

