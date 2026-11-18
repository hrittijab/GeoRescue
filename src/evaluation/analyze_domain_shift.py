from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# GEORESCUE: GEOGRAPHIC DOMAIN SHIFT
# ==========================================

ROOT = Path("experiments/domain_shift")
ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = Path("data/splits/unet_split.json")
RADAR_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")

SEED = 42
SAMPLE_PER_IMAGE = 4000
BINS = np.linspace(-35, 5, 81)

rng = np.random.default_rng(SEED)

with open(SPLITS, encoding="utf-8") as f:
    splits = json.load(f)

rows = []
sample_rows = []

histograms = {}

print("GEORESCUE GEOGRAPHIC DOMAIN SHIFT ANALYSIS")
print("=" * 60)

# ==========================================
# 1. Analyze all 110 image pairs
# ==========================================

for split in ["train", "val", "test"]:
    for name in splits[split]:

        region = name.rsplit("_", 1)[0]

        image_path = (
            RADAR_DIR / f"{name}_S1Hand.tif"
        )

        label_path = (
            LABEL_DIR / f"{name}_LabelHand.tif"
        )

        with rasterio.open(image_path) as src:
            radar = src.read([1, 2]).astype(
                np.float32
            )

        with rasterio.open(label_path) as src:
            labels = src.read(1)

        if radar.shape[1:] != labels.shape:
            raise ValueError(
                f"Image and label shape mismatch: {name}"
            )

        valid = (
            np.isfinite(radar).all(axis=0)
            & np.isin(labels, [0, 1])
        )

        valid_pixels = int(valid.sum())

        if valid_pixels == 0:
            print(f"Skipping {name}: no valid pixels")
            continue

        water_pixels = int(
            ((labels == 1) & valid).sum()
        )

        vv = radar[0][valid]
        vh = radar[1][valid]

        # Keep original finite radar values for
        # descriptive statistics.
        row = {
            "image": name,
            "split": split,
            "region": region,
            "valid_pixels": valid_pixels,
            "water_pixels": water_pixels,
            "water_prevalence": (
                water_pixels / valid_pixels
            ),
            "vv_median": float(np.median(vv)),
            "vh_median": float(np.median(vh)),
            "vv_p10": float(np.percentile(vv, 10)),
            "vv_p90": float(np.percentile(vv, 90)),
            "vh_p10": float(np.percentile(vh, 10)),
            "vh_p90": float(np.percentile(vh, 90)),
            "vv_clip_fraction": float(
                np.mean((vv < -35) | (vv > 5))
            ),
            "vh_clip_fraction": float(
                np.mean((vh < -35) | (vh > 5))
            )
        }

        rows.append(row)

        # Equal maximum sample size per image.
        # This avoids giving larger valid footprints
        # disproportionate weight in histograms.
        n = min(SAMPLE_PER_IMAGE, valid_pixels)

        indices = rng.choice(
            valid_pixels,
            size=n,
            replace=False
        )

        sample_rows.append(
            pd.DataFrame({
                "image": name,
                "split": split,
                "region": region,
                "vv": vv[indices],
                "vh": vh[indices],
                "water": labels[valid][indices]
            })
        )

        print(
            f"{split:5s} | {name:22s} | "
            f"water={row['water_prevalence']:.2%}",
            flush=True
        )

if not rows:
    raise RuntimeError("No valid satellite data found.")

image_df = pd.DataFrame(rows)
sample_df = pd.concat(
    sample_rows,
    ignore_index=True
)

image_df.to_csv(
    ROOT / "image_statistics.csv",
    index=False
)

# ==========================================
# 2. Summarize by region
# ==========================================

regional = (
    image_df.groupby(
        ["split", "region"],
        as_index=False
    )
    .agg(
        images=("image", "count"),
        valid_pixels=("valid_pixels", "sum"),
        water_pixels=("water_pixels", "sum"),
        median_image_vv=("vv_median", "median"),
        median_image_vh=("vh_median", "median"),
        mean_image_water_prevalence=(
            "water_prevalence", "mean"
        ),
        median_image_water_prevalence=(
            "water_prevalence", "median"
        )
    )
)

regional["pooled_water_prevalence"] = (
    regional["water_pixels"]
    / regional["valid_pixels"]
)

regional.to_csv(
    ROOT / "regional_statistics.csv",
    index=False
)

# ==========================================
# 3. Jensen-Shannon divergence
# ==========================================

def histogram(values):
    # Match U-Net's radar clipping interval.
    values = np.clip(values, -35, 5)

    counts, _ = np.histogram(
        values, bins=BINS
    )

    counts = counts.astype(float) + 1e-12
    return counts / counts.sum()


def js_divergence(p, q):
    # Natural logarithm, result between 0 and ln(2).
    m = 0.5 * (p + q)

    return float(
        0.5 * np.sum(p * np.log(p / m))
        + 0.5 * np.sum(q * np.log(q / m))
    )


train_samples = sample_df[
    sample_df["split"] == "train"
]

train_vv = histogram(train_samples["vv"].to_numpy())
train_vh = histogram(train_samples["vh"].to_numpy())

divergence_rows = []

for (split, region), group in sample_df.groupby(
    ["split", "region"]
):
    vv_hist = histogram(group["vv"].to_numpy())
    vh_hist = histogram(group["vh"].to_numpy())

    divergence_rows.append({
        "split": split,
        "region": region,
        "vv_js_divergence": js_divergence(
            train_vv, vv_hist
        ),
        "vh_js_divergence": js_divergence(
            train_vh, vh_hist
        )
    })

divergence = pd.DataFrame(divergence_rows)

divergence.to_csv(
    ROOT / "radar_divergence.csv",
    index=False
)

# ==========================================
# 4. Radar intensity plots
# ==========================================

compare_groups = [
    ("Training", train_samples),
    (
        "Mekong",
        sample_df[
            sample_df["region"] == "Mekong"
        ]
    ),
    (
        "Somalia",
        sample_df[
            sample_df["region"] == "Somalia"
        ]
    )
]

for band in ["vv", "vh"]:
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, group in compare_groups:
        if group.empty:
            continue

        distribution = histogram(
            group[band].to_numpy()
        )

        centers = 0.5 * (
            BINS[:-1] + BINS[1:]
        )

        ax.plot(
            centers,
            distribution,
            label=label,
            linewidth=2
        )

    ax.set_xlabel(
        f"{band.upper()} radar values "
        "(clipped to [-35, 5])"
    )
    ax.set_ylabel("Proportion per histogram bin")
    ax.set_title(
        f"GeoRescue: {band.upper()} Distribution Shift"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        ROOT / f"{band}_distribution.png",
        dpi=200
    )
    plt.close()

# ==========================================
# 5. Water prevalence plot
# ==========================================

ordered = regional.sort_values(
    ["split", "region"]
)

labels = [
    f"{r.region}\n({r.split})"
    for r in ordered.itertuples()
]

values = (
    ordered["pooled_water_prevalence"]
    .to_numpy() * 100
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(range(len(values)), values)
ax.set_xticks(range(len(values)))
ax.set_xticklabels(
    labels,
    rotation=50,
    ha="right"
)

ax.set_ylabel("Labeled water pixels (%)")
ax.set_title(
    "GeoRescue: Water Prevalence by Region"
)

plt.tight_layout()
plt.savefig(
    ROOT / "water_prevalence.png",
    dpi=200
)
plt.close()

# ==========================================
# 6. Join model performance for interpretation
# ==========================================

model_results = Path(
    "experiments/unet_v2/test_evaluation/regional.csv"
)

if model_results.exists():
    performance = pd.read_csv(model_results)

    v2 = performance[
        performance["model"] == "V2"
    ][["region", "iou", "precision", "recall", "fpr"]]

    test_regions = regional[
        regional["split"] == "test"
    ]

    joined = test_regions.merge(
        v2,
        on="region",
        how="left",
        validate="one_to_one"
    )

    joined = joined.merge(
        divergence[
            divergence["split"] == "test"
        ][
            ["region", "vv_js_divergence",
             "vh_js_divergence"]
        ],
        on="region",
        how="left",
        validate="one_to_one"
    )

    joined.to_csv(
        ROOT / "test_domain_shift_vs_performance.csv",
        index=False
    )

# ==========================================
# 7. Summary
# ==========================================

print("\n" + "=" * 60)
print("REGIONAL RADAR AND WATER STATISTICS")
print("=" * 60)

print(
    regional[
        ["split", "region", "images",
         "pooled_water_prevalence",
         "median_image_vv", "median_image_vh"]
    ].round(4).to_string(index=False)
)

print("\n" + "=" * 60)
print("RADAR DISTRIBUTION DIVERGENCE")
print("=" * 60)

print(
    divergence[
        divergence["split"] == "test"
    ].round(5).to_string(index=False)
)

print("\nDOMAIN SHIFT ANALYSIS COMPLETE!")
print("Outputs saved to:", ROOT)
