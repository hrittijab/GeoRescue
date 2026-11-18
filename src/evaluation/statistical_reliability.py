from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# GEORESCUE: STATISTICAL RELIABILITY
# ==========================================

ROOT = Path("experiments/statistical_reliability")
ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = Path("data/splits/unet_split.json")
LABEL_DIR = Path("data/raw/LabelHand")

V1_DIR = Path(
    "experiments/geospatial/predicted_masks"
)

V2_DIR = Path(
    "experiments/unet_v2/test_evaluation/"
    "predicted_masks"
)

N_BOOTSTRAP = 10000
SEED = 42
rng = np.random.default_rng(SEED)

with open(SPLITS, encoding="utf-8") as f:
    splits = json.load(f)

names = splits["test"]

# ==========================================
# 1. Metrics from confusion counts
# ==========================================

def calculate(counts):
    tp, fp, fn, tn = map(int, counts)

    union = tp + fp + fn
    f1_denom = 2 * tp + fp + fn

    return {
        "iou": tp / union if union else np.nan,
        "f1": (
            2 * tp / f1_denom
            if f1_denom else np.nan
        ),
        "precision": (
            tp / (tp + fp)
            if tp + fp else np.nan
        ),
        "recall": (
            tp / (tp + fn)
            if tp + fn else np.nan
        ),
        "fpr": (
            fp / (fp + tn)
            if fp + tn else np.nan
        )
    }


def confusion(pred, truth, valid):
    tp = np.sum(pred & truth & valid)
    fp = np.sum(pred & ~truth & valid)
    fn = np.sum(~pred & truth & valid)
    tn = np.sum(~pred & ~truth & valid)

    return np.array(
        [tp, fp, fn, tn],
        dtype=np.int64
    )


def read_raster(path):
    with rasterio.open(path) as src:
        return (
            src.read(1),
            src.crs,
            src.transform
        )


# ==========================================
# 2. Load paired image predictions
# ==========================================

rows = []
counts_v1 = []
counts_v2 = []
regions = []

print("LOADING PAIRED TEST PREDICTIONS")
print("=" * 60)

for index, name in enumerate(names, start=1):
    region = name.rsplit("_", 1)[0]

    label_path = (
        LABEL_DIR / f"{name}_LabelHand.tif"
    )

    v1_path = (
        V1_DIR / f"{name}_water.tif"
    )

    v2_path = (
        V2_DIR / f"{name}_water_v2.tif"
    )

    for path in [label_path, v1_path, v2_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    label, label_crs, label_transform = read_raster(
        label_path
    )

    v1, v1_crs, v1_transform = read_raster(
        v1_path
    )

    v2, v2_crs, v2_transform = read_raster(
        v2_path
    )

    if not (
        label.shape == v1.shape == v2.shape
        and label_crs == v1_crs == v2_crs
        and label_transform.almost_equals(v1_transform)
        and label_transform.almost_equals(v2_transform)
    ):
        raise ValueError(
            f"Raster alignment mismatch for {name}. "
            "Do not compare unaligned pixels."
        )

    # Both predictions must be valid over identical
    # pixels for a fair paired comparison.
    valid_v1 = (v1 != 255)
    valid_v2 = (v2 != 255)

    if not np.array_equal(valid_v1, valid_v2):
        raise ValueError(
            f"Validity masks differ for {name}."
        )

    valid = (
        valid_v1
        & np.isin(label, [0, 1])
    )

    truth = label == 1

    c1 = confusion(
        v1 == 1, truth, valid
    )

    c2 = confusion(
        v2 == 1, truth, valid
    )

    counts_v1.append(c1)
    counts_v2.append(c2)
    regions.append(region)

    m1 = calculate(c1)
    m2 = calculate(c2)

    # Undefined IoU for images without prediction
    # or reference water is recorded as missing.
    delta = (
        m2["iou"] - m1["iou"]
        if np.isfinite(m1["iou"])
        and np.isfinite(m2["iou"])
        else np.nan
    )

    rows.append({
        "image": name,
        "region": region,
        "reference_water_pixels": int(
            np.sum(truth & valid)
        ),
        "v1_iou": m1["iou"],
        "v2_iou": m2["iou"],
        "iou_difference": delta,
        "v1_recall": m1["recall"],
        "v2_recall": m2["recall"]
    })

    print(
        f"[{index}/{len(names)}] {name} | "
        f"V1={m1['iou']:.4f} "
        f"V2={m2['iou']:.4f}",
        flush=True
    )

counts_v1 = np.asarray(
    counts_v1, dtype=np.int64
)

counts_v2 = np.asarray(
    counts_v2, dtype=np.int64
)

regions = np.asarray(regions)
image_df = pd.DataFrame(rows)

image_df.to_csv(
    ROOT / "per_image_comparison.csv",
    index=False
)

# ==========================================
# 3. Original pooled test results
# ==========================================

observed_v1 = calculate(
    counts_v1.sum(axis=0)
)

observed_v2 = calculate(
    counts_v2.sum(axis=0)
)

print("\nOBSERVED POOLED TEST RESULTS")

for metric in ["iou", "f1", "recall", "precision", "fpr"]:
    print(
        f"{metric:10s} "
        f"V1={observed_v1[metric]:.4f} "
        f"V2={observed_v2[metric]:.4f} "
        f"Difference="
        f"{observed_v2[metric] - observed_v1[metric]:+.4f}"
    )

# ==========================================
# 4. Paired stratified bootstrap
# ==========================================

def run_bootstrap(indices_by_region, iterations):
    """
    Sample images WITH replacement inside each region.
    Use the identical sampled indices for V1 and V2.
    Recompute metrics from pooled confusion counts.
    """
    output = {
        metric: []
        for metric in [
            "iou", "f1", "recall", "precision", "fpr"
        ]
    }

    for _ in range(iterations):
        chosen = []

        for indices in indices_by_region:
            sampled = rng.choice(
                indices,
                size=len(indices),
                replace=True
            )
            chosen.extend(sampled.tolist())

        chosen = np.asarray(chosen, dtype=int)

        a = calculate(
            counts_v1[chosen].sum(axis=0)
        )

        b = calculate(
            counts_v2[chosen].sum(axis=0)
        )

        for metric in output:
            difference = b[metric] - a[metric]
            output[metric].append(difference)

    return {
        metric: np.asarray(values, dtype=float)
        for metric, values in output.items()
    }


unique_regions = sorted(set(regions))

region_indices = [
    np.where(regions == region)[0]
    for region in unique_regions
]

print("\nRunning paired bootstrap...", flush=True)

bootstrap = run_bootstrap(
    region_indices,
    N_BOOTSTRAP
)

# ==========================================
# 5. Summarize uncertainty intervals
# ==========================================

summary = []

def summarize(metric, samples, scope, observed):
    valid_samples = samples[np.isfinite(samples)]

    if len(valid_samples) < 0.95 * len(samples):
        print(
            f"WARNING: Many undefined bootstrap "
            f"values for {scope}/{metric}"
        )

    if len(valid_samples) == 0:
        lower = upper = np.nan
        proportion_positive = np.nan
    else:
        lower, upper = np.percentile(
            valid_samples, [2.5, 97.5]
        )
        proportion_positive = float(
            np.mean(valid_samples > 0)
        )

    return {
        "scope": scope,
        "metric": metric,
        "observed_difference": observed,
        "bootstrap_lower_95": lower,
        "bootstrap_upper_95": upper,
        "bootstrap_fraction_positive": (
            proportion_positive
        ),
        "valid_resamples": len(valid_samples)
    }


for metric, samples in bootstrap.items():
    observed = (
        observed_v2[metric] - observed_v1[metric]
    )

    summary.append(
        summarize(
            metric, samples, "overall", observed
        )
    )

# Separate within-region bootstrap
for region in unique_regions:
    idx = np.where(regions == region)[0]

    regional_bootstrap = run_bootstrap(
        [idx],
        N_BOOTSTRAP
    )

    a = calculate(
        counts_v1[idx].sum(axis=0)
    )

    b = calculate(
        counts_v2[idx].sum(axis=0)
    )

    for metric, samples in regional_bootstrap.items():
        summary.append(
            summarize(
                metric,
                samples,
                region,
                b[metric] - a[metric]
            )
        )

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    ROOT / "bootstrap_intervals.csv",
    index=False
)

# ==========================================
# 6. Image-level wins / losses
# ==========================================

valid_pairs = image_df.dropna(
    subset=["iou_difference"]
)

wins = int(
    (valid_pairs["iou_difference"] > 1e-10).sum()
)

losses = int(
    (valid_pairs["iou_difference"] < -1e-10).sum()
)

ties = int(
    (
        valid_pairs["iou_difference"].abs()
        <= 1e-10
    ).sum()
)

undefined = len(image_df) - len(valid_pairs)

print("\nPAIRED IMAGE-LEVEL COMPARISON")
print(f"V2 improved: {wins}")
print(f"V2 worsened: {losses}")
print(f"Equal IoU: {ties}")
print(f"Undefined IoU pairs: {undefined}")

# ==========================================
# 7. Generate figures
# ==========================================

fig, ax = plt.subplots(figsize=(10, 6))

plot_df = image_df.sort_values(
    "iou_difference"
)

colors = [
    "seagreen" if x > 0
    else "indianred" if x < 0
    else "gray"
    for x in plot_df["iou_difference"]
]

ax.barh(
    plot_df["image"],
    plot_df["iou_difference"] * 100,
    color=colors
)

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel(
    "V2 minus V1 image IoU (percentage points)"
)
ax.set_title(
    "GeoRescue: Paired Test Image Improvements"
)

plt.tight_layout()
plt.savefig(
    ROOT / "image_level_improvements.png",
    dpi=200
)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))

iou_samples = bootstrap["iou"]
iou_samples = iou_samples[
    np.isfinite(iou_samples)
]

ax.hist(
    iou_samples * 100,
    bins=45,
    edgecolor="white"
)

ax.axvline(
    0, linestyle="--",
    color="black",
    label="No improvement"
)

ax.axvline(
    (observed_v2["iou"] - observed_v1["iou"]) * 100,
    color="red",
    linewidth=2,
    label="Observed difference"
)

ax.set_xlabel(
    "V2 minus V1 pooled IoU "
    "(percentage points)"
)
ax.set_ylabel("Bootstrap resamples")
ax.set_title(
    "GeoRescue: Paired Bootstrap IoU Differences"
)
ax.legend()

plt.tight_layout()
plt.savefig(
    ROOT / "bootstrap_iou_distribution.png",
    dpi=200
)
plt.close()

# ==========================================
# 8. Final report
# ==========================================

print("\n" + "=" * 65)
print("GEORESCUE STATISTICAL RELIABILITY RESULTS")
print("=" * 65)

selected = summary_df[
    summary_df["metric"].isin(
        ["iou", "f1", "recall", "fpr"]
    )
]

print(
    selected[
        [
            "scope",
            "metric",
            "observed_difference",
            "bootstrap_lower_95",
            "bootstrap_upper_95"
        ]
    ].round(4).to_string(index=False)
)

print("\nSTATISTICAL RELIABILITY ANALYSIS COMPLETE!")
print("Outputs saved to:", ROOT)
