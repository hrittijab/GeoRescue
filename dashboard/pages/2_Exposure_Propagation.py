from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/exposure_propagation"

st.set_page_config(
    page_title="GeoRescue — Exposure Propagation",
    page_icon="🌊",
    layout="wide"
)

st.title("Exposure Error Propagation")

st.caption(
    "How changes in water segmentation affect "
    "building-exposure estimates."
)

st.info(
    "Exploratory four-image diagnostic evaluation. "
    "The images were deliberately selected from previously "
    "examined test regions. These are surface-water "
    "intersections, not verified flood damage."
)

@st.cache_data
def load_results():
    detailed = pd.read_csv(
        RESULTS / "exposure_propagation.csv"
    )

    comparison = pd.read_csv(
        RESULTS / "model_impact.csv"
    )

    return detailed, comparison


try:
    detailed, comparison = load_results()
except FileNotFoundError:
    st.error(
        "Exposure propagation results are missing. "
        "Run the Phase 8.3 evaluation first."
    )
    st.stop()

# ---------------------------------------
# Overview
# ---------------------------------------

valid_recall = comparison.dropna(
    subset=["building_recall_change"]
)

improved_iou = int(
    (comparison["iou_change"] > 0).sum()
)

improved_exposure = int(
    (valid_recall["building_recall_change"] > 0).sum()
)

unchanged_exposure = int(
    (valid_recall["building_recall_change"] == 0).sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Images evaluated", len(comparison))
c2.metric("Images with improved IoU", improved_iou)
c3.metric(
    "Improved building recall",
    f"{improved_exposure}/{len(valid_recall)}"
)
c4.metric(
    "Unchanged building recall",
    unchanged_exposure
)

st.divider()

# ---------------------------------------
# Segmentation versus exposure
# ---------------------------------------

st.subheader("Does better segmentation improve exposure?")

fig, ax = plt.subplots(figsize=(9, 6))

for region, group in valid_recall.groupby("region"):
    ax.scatter(
        group["iou_change"] * 100,
        group["building_recall_change"] * 100,
        s=110,
        label=region
    )

    for _, row in group.iterrows():
        ax.annotate(
            row["image"],
            (
                row["iou_change"] * 100,
                row["building_recall_change"] * 100
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8
        )

ax.axhline(0, color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")

ax.set_xlabel("Change in segmentation IoU (pp)")
ax.set_ylabel("Change in building recall (pp)")
ax.set_title(
    "U-Net V2 minus U-Net V1"
)
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Each point represents one image with defined "
    "building-exposure recall. The small, selected sample "
    "does not support a general correlation estimate."
)

# ---------------------------------------
# Image-level comparison
# ---------------------------------------

st.subheader("Image-level results")

display = comparison.copy()

for column in [
    "iou_v1",
    "iou_v2",
    "building_recall_v1",
    "building_recall_v2",
    "iou_change",
    "building_recall_change"
]:
    display[column] = display[column] * 100

st.dataframe(
    display.round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "All metric columns are percentages or percentage-point "
    "differences. NaN indicates an undefined metric, "
    "not zero performance."
)

# ---------------------------------------
# Selected image details
# ---------------------------------------

st.divider()
st.subheader("Inspect an individual image")

selected = st.selectbox(
    "Select satellite image",
    comparison["image"].tolist()
)

sample = detailed[
    detailed["image"] == selected
].copy()

if len(sample) != 2:
    st.warning(
        "Expected two model records for this image."
    )
else:
    sample = sample.set_index("model")

    v1 = sample.loc["V1"]
    v2 = sample.loc["V2"]

    left, right = st.columns(2)

    with left:
        st.markdown("#### U-Net V1")
        st.metric(
            "Segmentation IoU",
            f"{v1['segmentation_iou'] * 100:.2f}%"
        )
        st.metric(
            "Correctly flagged buildings",
            int(v1["building_tp"])
        )
        st.metric(
            "Missed buildings",
            int(v1["building_fn"])
        )

    with right:
        st.markdown("#### U-Net V2")
        st.metric(
            "Segmentation IoU",
            f"{v2['segmentation_iou'] * 100:.2f}%"
        )
        st.metric(
            "Correctly flagged buildings",
            int(v2["building_tp"])
        )
        st.metric(
            "Missed buildings",
            int(v2["building_fn"])
        )

    st.markdown("#### Detailed evaluation")

    fields = [
        "model",
        "mapped_buildings",
        "reference_exposed_buildings",
        "predicted_exposed_buildings",
        "building_tp",
        "building_fp",
        "building_fn",
        "building_precision",
        "building_recall",
        "building_f1"
    ]

    st.dataframe(
        detailed[
            detailed["image"] == selected
        ][fields],
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------
# Scientific interpretation
# ---------------------------------------

st.divider()
st.subheader("Research findings")

st.markdown(
    """
**Finding 1 — Improvement is not guaranteed downstream.**

For Somalia_12849, segmentation IoU improved while
building-exposure recall remained at zero.

**Finding 2 — Exposure recall remains limited.**

Even when segmentation improved substantially in Mekong,
the model continued missing many buildings that intersected
reference water.

**Finding 3 — Geographic context matters.**

The results differ between regions, consistent with
the wider geographic robustness investigation.
"""
)

st.warning(
    "An infrastructure intersection only indicates overlap "
    "with a predicted or reference water polygon. "
    "It does not establish building damage, road "
    "impassability, or confirmed flood inundation. "
    "Missing infrastructure records and mismatched "
    "observation dates are additional limitations."
)

st.download_button(
    "Download propagation results",
    data=detailed.to_csv(index=False),
    file_name="georescue_exposure_propagation.csv",
    mime="text/csv"
)
