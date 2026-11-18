from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("experiments/random_forest/evaluation.csv")
OUTPUT = Path("experiments/error_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        "Evaluation CSV missing. "
        "Complete Random Forest training first."
    )

df = pd.read_csv(INPUT)

required = [
    "test_region", "precision", "recall",
    "f1", "iou", "true_negative",
    "false_positive", "false_negative",
    "true_positive", "test_water_pixels"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if len(df) != 5:
    raise RuntimeError(
        f"Expected 5 evaluated regions, found {len(df)}. "
        "Rerun the training script to completion."
    )

df["region"] = df["test_region"].str.split("_").str[0]

df["total_errors"] = (
    df["false_positive"] + df["false_negative"]
)

df["false_positive_rate"] = (
    df["false_positive"] /
    (df["false_positive"] + df["true_negative"])
    .replace(0, np.nan)
)

df["false_negative_rate"] = (
    df["false_negative"] /
    (df["false_negative"] + df["true_positive"])
    .replace(0, np.nan)
)

df["water_prevalence"] = (
    df["test_water_pixels"] /
    df["test_pixels"]
)

# --------------------------------
# Chart 1: Regional model metrics
# --------------------------------

metrics = ["precision", "recall", "f1", "iou"]

ax = df.set_index("region")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

ax.set_title("GeoRescue: Geographic Model Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Held-out region")
ax.legend(title="Metric")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(
    OUTPUT / "regional_performance.png",
    dpi=200
)
plt.close()


# --------------------------------
# Chart 2: Prediction errors
# --------------------------------

ax = df.set_index("region")[
    ["false_positive", "false_negative"]
].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("False Positives vs False Negatives")
ax.set_ylabel("Number of pixels")
ax.set_xlabel("Held-out region")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(
    OUTPUT / "prediction_errors.png",
    dpi=200
)
plt.close()


# --------------------------------
# Chart 3: Water prevalence vs IoU
# --------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    df["water_prevalence"] * 100,
    df["iou"],
    s=100
)

for _, row in df.iterrows():
    ax.annotate(
        row["region"],
        (
            row["water_prevalence"] * 100,
            row["iou"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlabel("Water prevalence (%)")
ax.set_ylabel("Water IoU")
ax.set_title("Water Prevalence vs Model Performance")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT / "prevalence_vs_iou.png",
    dpi=200
)
plt.close()


# --------------------------------
# Save detailed analysis
# --------------------------------

df.to_csv(
    OUTPUT / "error_analysis.csv",
    index=False
)

print("\nGEORESCUE ERROR ANALYSIS")
print("=" * 65)

print(
    df[
        [
            "region", "precision",
            "recall", "f1", "iou",
            "water_prevalence"
        ]
    ].to_string(index=False)
)

print("\nMacro-average metrics:")
for metric in metrics:
    print(f"{metric}: {df[metric].mean():.4f}")

print("\nWorst region by IoU:")
worst = df.loc[df["iou"].idxmin()]
print(worst["region"], worst["iou"])

print("\nFigures and CSV saved to:", OUTPUT)
print("ERROR ANALYSIS COMPLETE!")
