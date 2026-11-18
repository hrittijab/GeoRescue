from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("experiments/geospatial")
df = pd.read_csv(OUTPUT / "water_extent.csv")

required = [
    "image", "region", "predicted_water_km2",
    "reference_water_km2", "difference_km2"
]

if df[required].isna().any().any():
    raise ValueError("Missing values in water extent CSV")

df["absolute_error_km2"] = df["difference_km2"].abs()

regional = df.groupby("region", as_index=False).agg(
    images=("image", "count"),
    predicted_km2=("predicted_water_km2", "sum"),
    reference_km2=("reference_water_km2", "sum"),
    mean_absolute_error_km2=("absolute_error_km2", "mean")
)

regional["bias_km2"] = (
    regional["predicted_km2"]
    - regional["reference_km2"]
)

regional["relative_bias_pct"] = np.where(
    regional["reference_km2"] > 0,
    100 * regional["bias_km2"]
    / regional["reference_km2"],
    np.nan
)

regional.to_csv(
    OUTPUT / "regional_area_summary.csv",
    index=False
)

worst = df.sort_values(
    "absolute_error_km2", ascending=False
).head(10)

worst.to_csv(
    OUTPUT / "largest_area_errors.csv",
    index=False
)

# Regional area comparison
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(regional))
width = 0.35

ax.bar(
    x - width / 2,
    regional["reference_km2"],
    width,
    label="Reference"
)

ax.bar(
    x + width / 2,
    regional["predicted_km2"],
    width,
    label="U-Net prediction"
)

ax.set_xticks(x)
ax.set_xticklabels(regional["region"])
ax.set_ylabel("Summed water area across image footprints (km²)")
ax.set_title("GeoRescue: Regional Water Extent Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT / "regional_area_comparison.png",
    dpi=200
)
plt.close()

# Predicted vs reference area, per image
fig, ax = plt.subplots(figsize=(7, 6))

for region, group in df.groupby("region"):
    ax.scatter(
        group["reference_water_km2"],
        group["predicted_water_km2"],
        label=region,
        s=55,
        alpha=0.8
    )

limit = max(
    df["reference_water_km2"].max(),
    df["predicted_water_km2"].max()
) * 1.05

ax.plot(
    [0, limit], [0, limit],
    linestyle="--",
    label="Perfect area agreement"
)

ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Reference water area (km²)")
ax.set_ylabel("Predicted water area (km²)")
ax.set_title("GeoRescue: Image-Level Area Estimation")
ax.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT / "area_prediction_scatter.png",
    dpi=200
)
plt.close()

print("\nREGIONAL WATER AREA SUMMARY")
print(regional.round(3).to_string(index=False))

print("\nTOP 10 LARGEST AREA ESTIMATION ERRORS")
print(
    worst[
        ["image", "reference_water_km2",
         "predicted_water_km2", "difference_km2"]
    ].round(3).to_string(index=False)
)

print("\nREGIONAL ERROR ANALYSIS COMPLETE!")
