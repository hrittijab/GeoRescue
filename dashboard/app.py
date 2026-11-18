from pathlib import Path
import json

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "experiments/infrastructure_overture"
VALIDATION = INFRA / "validation"
GEOSPATIAL = ROOT / "experiments/geospatial"
COMPARISON = ROOT / "experiments/final_comparison"

st.set_page_config(
    page_title="GeoRescue",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "TP": "#16a34a",
    "FP": "#f59e0b",
    "FN": "#ef4444",
    "TN": "#94a3b8"
}


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def read_geojson(path):
    return gpd.read_file(path).to_crs("EPSG:4326")


def metric_pct(value):
    return f"{value * 100:.2f}%"


def error_style(feature):
    category = feature["properties"].get("error_type", "TN")
    return {
        "color": COLORS.get(category, "#64748b"),
        "weight": 3 if category != "TN" else 1,
        "fillColor": COLORS.get(category, "#64748b"),
        "fillOpacity": 0.30 if category != "TN" else 0.08
    }


def make_map(show_water, show_buildings, show_roads, categories):
    water = read_geojson(str(INFRA / "water.geojson"))

    # Simplification is used only for dashboard rendering.
    # Original analysis geometries remain unchanged.
    water_display = (
        water.to_crs("EPSG:3857")
        .simplify(12, preserve_topology=True)
    )

    water_display = gpd.GeoDataFrame(
        geometry=water_display,
        crs="EPSG:3857"
    ).to_crs("EPSG:4326")

    bounds = water.total_bounds
    center = [
        (bounds[1] + bounds[3]) / 2,
        (bounds[0] + bounds[2]) / 2
    ]

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    if show_water:
        folium.GeoJson(
            json.loads(water_display.to_json()),
            name="Predicted surface water",
            style_function=lambda feature: {
                "fillColor": "#2563eb",
                "color": "#1d4ed8",
                "weight": 1,
                "fillOpacity": 0.45
            }
        ).add_to(m)

    for enabled, filename, label in [
        (
            show_buildings,
            "classified_buildings.geojson",
            "Buildings"
        ),
        (
            show_roads,
            "classified_roads.geojson",
            "Roads"
        )
    ]:
        if not enabled:
            continue

        frame = read_geojson(
            str(VALIDATION / filename)
        )

        frame = frame[
            frame["error_type"].isin(categories)
        ].copy()

        if frame.empty:
            continue

        data = json.loads(
            frame[["error_type", "geometry"]].to_json()
        )

        folium.GeoJson(
            data,
            name=label,
            style_function=error_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["error_type"],
                aliases=["Exposure classification:"]
            )
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Fit to the satellite-image footprint, not just
    # the portion containing predicted water.
    import rasterio
    from rasterio.warp import transform_bounds

    mask_path = (
        GEOSPATIAL
        / "predicted_masks/Mekong_1248200_water.tif"
    )

    with rasterio.open(mask_path) as src:
        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds
        )

    m.fit_bounds([[south, west], [north, east]])

    return m


st.sidebar.title("🌊 GeoRescue")
st.sidebar.caption("Geospatial AI & Infrastructure Exposure")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Exposure Map",
        "Regional Analytics",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: Sen1Floods11")
st.sidebar.caption("Models: U-Net and Random Forest")
st.sidebar.caption("Infrastructure: Overture Maps")
st.sidebar.caption("Prototype • Historical observations")

if page == "Overview":
    st.title("GeoRescue")
    st.subheader(
        "Satellite-Based Water Mapping and "
        "Infrastructure Exposure Analytics"
    )

    st.info(
        "Research prototype. Surface-water predictions "
        "are not confirmed flood events or damage assessments."
    )

    extent = read_csv(
        str(GEOSPATIAL / "water_extent.csv")
    )

    exposure = read_csv(
        str(INFRA / "exposure_summary.csv")
    ).iloc[0]

    validation = read_csv(
        str(VALIDATION / "feature_error_summary.csv")
    )

    building_row = validation[
        validation["feature_type"] == "buildings"
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Predicted water",
        f"{extent['predicted_water_km2'].sum():.2f} km²"
    )

    c2.metric(
        "Reference water",
        f"{extent['reference_water_km2'].sum():.2f} km²"
    )

    c3.metric(
        "Flagged buildings",
        int(exposure["buildings_intersecting_water"])
    )

    c4.metric(
        "Missed reference-exposed buildings",
        int(building_row["FN"])
    )

    st.caption(
        "Water totals sum 20 test-image footprints and "
        "may double-count geographic overlap. "
        "Infrastructure counts describe one Mekong image."
    )

    st.divider()
    st.subheader("Project pipeline")

    st.markdown(
        "**Sentinel-1 SAR → Water segmentation → "
        "Geographic area estimation → Infrastructure "
        "intersection → Exposure validation**"
    )

    st.subheader("Key finding")
    st.warning(
        "In Mekong_1248200, U-Net identified 20 of "
        "110 buildings intersecting reference water. "
        "Building exposure recall was 18.18%, "
        "despite relatively strong exposure precision."
    )

    st.markdown(
        "This project investigates whether satellite "
        "segmentation accuracy translates into useful "
        "infrastructure-exposure estimates."
    )

elif page == "Exposure Map":
    st.title("Interactive Infrastructure Exposure Map")

    st.caption(
        "Image: Mekong_1248200 | "
        "Geospatial reference: Sentinel-1 water mask | "
        "Infrastructure: Overture Maps"
    )

    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Layers")
        show_water = st.checkbox(
            "Predicted surface water",
            value=True
        )
        show_buildings = st.checkbox(
            "Building footprints",
            value=True
        )
        show_roads = st.checkbox(
            "Road segments",
            value=True
        )

        categories = st.multiselect(
            "Exposure classifications",
            ["TP", "FN", "FP", "TN"],
            default=["TP", "FN", "FP"]
        )

        st.markdown(
            "🟢 **TP:** Correctly flagged\n\n"
            "🔴 **FN:** Missed exposure\n\n"
            "🟠 **FP:** Incorrectly flagged\n\n"
            "⚪ **TN:** No exposure in either overlay"
        )

        st.caption(
            "Classification is relative to labeled "
            "surface water, not verified flood damage."
        )

    with c2:
        geo_map = make_map(
            show_water,
            show_buildings,
            show_roads,
            categories
        )

        st_folium(
            geo_map,
            height=620,
            width=None,
            returned_objects=[],
            key="georescue_map"
        )

    st.divider()
    st.subheader("Exposure error visualization")

    image_path = VALIDATION / "exposure_error_map.png"
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.caption(
        "The display map simplifies water geometry for "
        "rendering. Exposure metrics use original geometries."
    )

elif page == "Regional Analytics":
    st.title("Regional Water Extent Analysis")

    regional = read_csv(
        str(GEOSPATIAL / "regional_area_summary.csv")
    )

    st.subheader("Predicted vs reference water area")

    chart = regional.set_index("region")[
        ["predicted_km2", "reference_km2"]
    ].rename(
        columns={
            "predicted_km2": "Predicted",
            "reference_km2": "Reference"
        }
    )

    st.bar_chart(chart)

    st.subheader("Regional error summary")
    st.dataframe(
        regional,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Image-level water area")
    extent = read_csv(
        str(GEOSPATIAL / "water_extent.csv")
    )

    selected_region = st.selectbox(
        "Select region",
        sorted(extent["region"].unique())
    )

    selected = extent[
        extent["region"] == selected_region
    ]

    st.dataframe(
        selected[
            [
                "image",
                "predicted_water_km2",
                "reference_water_km2",
                "difference_km2"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download water-area results",
        data=extent.to_csv(index=False),
        file_name="georescue_water_extent.csv",
        mime="text/csv"
    )

elif page == "Model Evaluation":
    st.title("Machine Learning Evaluation")

    summary = read_csv(
        str(COMPARISON / "summary.csv")
    )

    st.subheader("Random Forest vs U-Net")

    metrics = [
        "precision", "recall", "f1", "iou", "fpr"
    ]

    display = summary.set_index("model")[metrics] * 100
    st.bar_chart(display.T)

    st.dataframe(
        summary[
            ["model", "precision", "recall",
             "f1", "iou", "fpr", "macro_region_iou"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Both models evaluated on the same 20-image "
        "test split. Previous experimentation involved "
        "these geographic groups, so results represent "
        "internal evaluation rather than a fully "
        "independent benchmark."
    )

    st.divider()
    st.subheader("Downstream exposure validation")

    results = read_csv(
        str(INFRA / "exposure_validation.csv")
    ).iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Building precision",
        metric_pct(results["building_exposure_precision"])
    )

    c2.metric(
        "Building recall",
        metric_pct(results["building_exposure_recall"])
    )

    c3.metric(
        "Building F1",
        metric_pct(results["building_exposure_f1"])
    )

    st.markdown(
        f"**True positives:** "
        f"{int(results['building_true_positive'])}  \n"
        f"**False positives:** "
        f"{int(results['building_false_positive'])}  \n"
        f"**False negatives:** "
        f"{int(results['building_false_negative'])}"
    )

    st.warning(
        "High pixel-level performance does not "
        "guarantee reliable building-level exposure "
        "detection. These metrics are from a single image."
    )

st.sidebar.divider()
st.sidebar.caption(
    "GeoRescue • Applied Data Science Prototype"
)
