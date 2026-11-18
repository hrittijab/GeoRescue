from pathlib import Path
import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from streamlit_folium import st_folium

# -------------------------------------------
# Configuration
# -------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

IMAGE = "Mekong_1248200"

PREDICTIONS = {
    "U-Net V1": ROOT / (
        "experiments/geospatial/predicted_masks/"
        f"{IMAGE}_water.tif"
    ),
    "U-Net V2": ROOT / (
        "experiments/unet_v2/test_evaluation/"
        f"predicted_masks/{IMAGE}_water_v2.tif"
    )
}

REFERENCE = ROOT / (
    f"data/raw/LabelHand/{IMAGE}_LabelHand.tif"
)

INFRA = ROOT / "experiments/infrastructure_overture"

RESULTS = ROOT / (
    "experiments/unet_v2/test_evaluation"
)

COLORS = {
    "TP": "#16a34a",
    "FN": "#ef4444",
    "FP": "#f59e0b",
    "TN": "#94a3b8"
}

st.set_page_config(
    page_title="GeoRescue — Model Comparison",
    page_icon="🌊",
    layout="wide"
)


# -------------------------------------------
# Cached data loading and intersections
# -------------------------------------------

def raster_polygon(binary, transform):
    geoms = [
        shape(geom)
        for geom, value in shapes(
            binary.astype("uint8"),
            mask=binary,
            transform=transform
        )
        if value == 1
    ]

    return unary_union(geoms) if geoms else Polygon()


@st.cache_data(show_spinner="Loading model exposure layers...")
def load_exposure(version):
    path = PREDICTIONS[version]

    with rasterio.open(path) as src:
        predicted = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

        if crs is None:
            raise ValueError("Prediction raster has no CRS.")

        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", *bounds
        )

    with rasterio.open(REFERENCE) as src:
        if (
            src.crs != crs
            or src.shape != predicted.shape
            or not src.transform.almost_equals(transform)
        ):
            raise ValueError(
                "Reference and prediction grids differ. "
                "Alignment must be verified first."
            )

        reference = src.read(1)

    valid = (
        (predicted != 255)
        & np.isin(reference, [0, 1])
    )

    pred_poly = raster_polygon(
        (predicted == 1) & valid, transform
    )

    ref_poly = raster_polygon(
        (reference == 1) & valid, transform
    )

    buildings = gpd.read_file(
        INFRA / "buildings.geojson"
    ).to_crs(crs)

    roads = gpd.read_file(
        INFRA / "roads.geojson"
    ).to_crs(crs)

    footprint = gpd.GeoSeries(
        [Polygon([
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top)
        ])],
        crs=crs
    )

    metric_crs = footprint.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Cannot determine metric CRS.")

    polygons_metric = gpd.GeoSeries(
        [pred_poly, ref_poly], crs=crs
    ).to_crs(metric_crs)

    pred_metric = polygons_metric.iloc[0]
    ref_metric = polygons_metric.iloc[1]

    def classify(frame, kind):
        metric = frame.to_crs(metric_crs)

        pred_intersection = metric.geometry.intersection(
            pred_metric
        )

        ref_intersection = metric.geometry.intersection(
            ref_metric
        )

        if kind == "building":
            a = pred_intersection.area.to_numpy() > 0
            b = ref_intersection.area.to_numpy() > 0
        else:
            a = pred_intersection.length.to_numpy() > 0
            b = ref_intersection.length.to_numpy() > 0

        categories = np.select(
            [a & b, a & ~b, ~a & b],
            ["TP", "FP", "FN"],
            default="TN"
        )

        result = frame[["geometry"]].copy()
        result["error_type"] = categories

        return result.to_crs("EPSG:4326")

    buildings = classify(buildings, "building")
    roads = classify(roads, "road")

    water = gpd.GeoDataFrame(
        geometry=[pred_poly], crs=crs
    ).to_crs("EPSG:4326")

    # Simplify only the visualization, not the
    # underlying exposure calculations.
    water_m = water.to_crs(metric_crs)
    water_m["geometry"] = water_m.geometry.simplify(
        10, preserve_topology=True
    )
    water_display = water_m.to_crs("EPSG:4326")

    return {
        "buildings": buildings.to_json(),
        "roads": roads.to_json(),
        "water": water_display.to_json(),
        "bounds": [[south, west], [north, east]]
    }


def make_map(data, categories, show_water,
             show_buildings, show_roads):

    bounds = data["bounds"]

    center = [
        (bounds[0][0] + bounds[1][0]) / 2,
        (bounds[0][1] + bounds[1][1]) / 2
    ]

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    if show_water:
        folium.GeoJson(
            json.loads(data["water"]),
            name="Predicted surface water",
            style_function=lambda feature: {
                "fillColor": "#2563eb",
                "color": "#1d4ed8",
                "weight": 1,
                "fillOpacity": 0.45
            }
        ).add_to(m)

    for enabled, key, label in [
        (show_buildings, "buildings", "Buildings"),
        (show_roads, "roads", "Roads")
    ]:
        if not enabled:
            continue

        features = json.loads(data[key])

        features["features"] = [
            f for f in features["features"]
            if f["properties"]["error_type"] in categories
        ]

        if not features["features"]:
            continue

        def style_fn(feature):
            category = feature["properties"]["error_type"]

            return {
                "color": COLORS[category],
                "fillColor": COLORS[category],
                "weight": 3 if category != "TN" else 1,
                "fillOpacity": 0.25
            }

        folium.GeoJson(
            features,
            name=label,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["error_type"],
                aliases=["Exposure classification:"]
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.fit_bounds(bounds)

    return m


# -------------------------------------------
# Dashboard
# -------------------------------------------

st.title("GeoRescue — U-Net Model Comparison")

st.caption(
    "Compare predicted water and infrastructure exposure "
    "for the same satellite image."
)

st.info(
    "This page compares model predictions against labeled "
    "surface water. It does not establish actual flood damage. "
    "The image belongs to the previously explored test set."
)

summary = pd.read_csv(RESULTS / "summary.csv")
regional = pd.read_csv(RESULTS / "regional.csv")
exposure = pd.read_csv(
    RESULTS / "exposure_comparison.csv"
)

st.subheader("Model selector")

version = st.radio(
    "Choose water segmentation model",
    ["U-Net V1", "U-Net V2"],
    horizontal=True,
    index=1
)

short_name = "V1" if version == "U-Net V1" else "V2"

model_metrics = summary[
    summary["model"] == short_name
].iloc[0]

exposure_metrics = exposure[
    exposure["model"] == short_name
].iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Test IoU",
    f"{model_metrics['iou'] * 100:.2f}%"
)

c2.metric(
    "Test recall",
    f"{model_metrics['recall'] * 100:.2f}%"
)

c3.metric(
    "Exposed buildings flagged",
    int(exposure_metrics["predicted_exposed_buildings"])
)

c4.metric(
    "Building exposure recall",
    f"{exposure_metrics['building_recall'] * 100:.2f}%"
)

st.divider()
st.subheader("Interactive water and infrastructure map")

left, right = st.columns([1, 3])

with left:
    show_water = st.checkbox(
        "Predicted water", True
    )
    show_buildings = st.checkbox(
        "Buildings", True
    )
    show_roads = st.checkbox(
        "Road segments", True
    )

    categories = st.multiselect(
        "Exposure classes",
        ["TP", "FN", "FP", "TN"],
        default=["TP", "FN", "FP"]
    )

    st.markdown(
        "🟢 **TP:** Correctly flagged\n\n"
        "🔴 **FN:** Missed exposure\n\n"
        "🟠 **FP:** Incorrectly flagged\n\n"
        "⚪ **TN:** Not exposed"
    )

with right:
    demo_filename = (
        "exposure_v1.json"
        if version == "U-Net V1"
        else "exposure_v2.json"
    )

    demo_path = (
        ROOT / "dashboard/demo_data" / demo_filename
    )

    if demo_path.exists():
        data = json.loads(
            demo_path.read_text(encoding="utf-8")
        )
        st.caption("Displaying precomputed exposure layers.")
    else:
        data = load_exposure(version)

    m = make_map(
        data,
        categories,
        show_water,
        show_buildings,
        show_roads
    )

    st_folium(
        m,
        height=620,
        returned_objects=[],
        key=f"model_map_{short_name}"
    )

st.divider()

st.subheader("V1 versus V2: Test performance")

performance = summary.set_index("model")[
    ["precision", "recall", "f1", "iou", "fpr"]
].T * 100

st.bar_chart(performance)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.subheader("Geographic performance")

region_pivot = regional.pivot(
    index="region",
    columns="model",
    values="iou"
) * 100

st.bar_chart(region_pivot)

st.dataframe(
    regional,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Infrastructure exposure improvements")

display = exposure[
    [
        "model",
        "predicted_exposed_buildings",
        "reference_exposed_buildings",
        "building_TP",
        "building_FP",
        "building_FN",
        "building_precision",
        "building_recall",
        "building_f1",
        "predicted_exposed_road_m",
        "reference_exposed_road_m"
    ]
]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Infrastructure exposure metrics apply only to "
    "Mekong_1248200, using the same saved Overture Maps "
    "features. Model scores above summarize all 20 test images."
)

st.warning(
    "U-Net V2 improves exposure recall on this example, "
    "but neither model is suitable for operational "
    "flood-response decisions without further validation."
)

st.download_button(
    "Download model comparison",
    data=summary.to_csv(index=False),
    file_name="georescue_model_comparison.csv",
    mime="text/csv"
)

