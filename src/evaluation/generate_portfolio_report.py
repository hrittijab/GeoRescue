from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "reports" / "portfolio"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "test_models":
        "experiments/unet_v2/test_evaluation/summary.csv",
    "test_regions":
        "experiments/unet_v2/test_evaluation/regional.csv",
    "exposure":
        "experiments/unet_v2/test_evaluation/exposure_comparison.csv",
    "bootstrap":
        "experiments/statistical_reliability/bootstrap_intervals.csv",
    "domain":
        "experiments/domain_shift/test_domain_shift_vs_performance.csv",
    "validation":
        "experiments/unet_v2/comparison/summary.csv",
    "rf_comparison":
        "experiments/final_comparison/summary.csv",
    "water_area":
        "experiments/geospatial/regional_area_summary.csv"
}

data = {}

for name, relative_path in SOURCES.items():
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Required experiment result missing: {path}"
        )

    data[name] = pd.read_csv(path)

print("Loaded", len(data), "experiment result files.")

def percent(value):
    return f"{float(value) * 100:.2f}%"

def pp(value):
    return f"{float(value) * 100:+.2f} pp"

def as_md(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(x) for x in row) + " |"
        )
    return "\n".join(lines)

test = data["test_models"].set_index("model")
validation = data["validation"].set_index("model")

v1 = test.loc["V1"]
v2 = test.loc["V2"]

model_rows = [
    [
        "U-Net V1",
        percent(v1["precision"]),
        percent(v1["recall"]),
        percent(v1["f1"]),
        percent(v1["iou"]),
        percent(v1["fpr"])
    ],
    [
        "U-Net V2",
        percent(v2["precision"]),
        percent(v2["recall"]),
        percent(v2["f1"]),
        percent(v2["iou"]),
        percent(v2["fpr"])
    ]
]

region_rows = []

for _, row in data["test_regions"].iterrows():
    region_rows.append([
        row["model"],
        row["region"],
        percent(row["precision"]),
        percent(row["recall"]),
        percent(row["iou"]),
        percent(row["fpr"])
    ])

exposure_rows = []

for _, row in data["exposure"].iterrows():
    exposure_rows.append([
        row["model"],
        int(row["predicted_exposed_buildings"]),
        int(row["reference_exposed_buildings"]),
        int(row["building_TP"]),
        int(row["building_FP"]),
        int(row["building_FN"]),
        percent(row["building_recall"]),
        f"{row['predicted_exposed_road_m']:.2f} m"
    ])

bootstrap_rows = []

for _, row in data["bootstrap"].iterrows():
    if row["metric"] not in ["iou", "recall", "fpr", "f1"]:
        continue
    bootstrap_rows.append([
        row["scope"],
        row["metric"].upper(),
        pp(row["observed_difference"]),
        pp(row["bootstrap_lower_95"]),
        pp(row["bootstrap_upper_95"])
    ])

domain_rows = []

for _, row in data["domain"].iterrows():
    domain_rows.append([
        row["region"],
        percent(row["pooled_water_prevalence"]),
        f"{row['vv_js_divergence']:.5f}",
        f"{row['vh_js_divergence']:.5f}",
        percent(row["iou"])
    ])

rf = data["rf_comparison"]
rf_rows = []

for _, row in rf.iterrows():
    rf_rows.append([
        row["model"],
        percent(row["precision"]),
        percent(row["recall"]),
        percent(row["f1"]),
        percent(row["iou"])
    ])

water_rows = []

for _, row in data["water_area"].iterrows():
    water_rows.append([
        row["region"],
        str(int(row["images"])),
        f"{row['predicted_km2']:.3f}",
        f"{row['reference_km2']:.3f}",
        f"{row['relative_bias_pct']:+.2f}%"
    ])

generated = datetime.now(
    timezone.utc
).strftime("%Y-%m-%d %H:%M UTC")

report = f"""# GeoRescue: Geospatial Water Segmentation and Infrastructure Exposure

**Technical evaluation report**  
Generated: {generated}  
Status: Experimental research and portfolio prototype

## Abstract

GeoRescue is a geospatial machine learning prototype that
segments surface water in Sentinel-1 synthetic aperture radar
(SAR) imagery and estimates potential exposure of mapped
infrastructure.

The project combines classical machine learning, convolutional
neural networks, geographic evaluation, area estimation, vector
intersection, and an interactive dashboard.

On the internal 20-image test split, U-Net V2 achieved
{percent(v2["iou"])} pixel-aggregated IoU, compared with
{percent(v1["iou"])} for U-Net V1. Recall improved by
{pp(v2["recall"] - v1["recall"])}, while false-positive rate
also increased by {pp(v2["fpr"] - v1["fpr"])}.

Performance varied substantially across geographic groups.
The project therefore evaluates not only segmentation
accuracy but also downstream building and road exposure.

## 1. Research question

**How reliably can SAR-based surface-water segmentation
support infrastructure-exposure estimation across different
geographic environments?**

Subquestions:

- Does a U-Net improve upon a classical Random Forest?
- Does an updated U-Net training recipe improve recall and IoU?
- How differently do models perform across geographic groups?
- Do pixel-level improvements translate to improved
  infrastructure-exposure estimates?

## 2. Dataset and preprocessing

Dataset: Sen1Floods11, using 110 manually labeled Sentinel-1
SAR image chips available in the local project.

Each chip has two input channels, VV and VH, and categorical
labels corresponding to non-water (0), water (1), and invalid
(-1) pixels.

The fixed experimental split contains:

| Split | Images | Geographic groups |
| --- | ---: | --- |
| Training | 70 | Seven filename-prefix groups |
| Validation | 20 | India and Spain |
| Test | 20 | Mekong and Somalia |

The input pipeline clips radar values to [-35, 5],
normalizes them to [0, 1], and excludes invalid pixels
from segmentation loss and evaluation.

**Split limitation:** Geographic groups were derived from
filename prefixes. They have not been independently verified
as event-level independence units.

## 3. Methods

### 3.1 Classical baseline

A Random Forest uses VV and VH pixel features.
The same fixed data split was used for the later direct
Random Forest versus U-Net comparison.

### 3.2 U-Net V1

A compact two-channel U-Net with approximately 1.94 million
parameters was trained using masked binary cross-entropy
and patch-based optimization.

### 3.3 U-Net V2

The second version retains the same architecture and adds
random flips and 90-degree rotations, combined BCE and Dice
loss, additional training epochs, and validation checkpoint
selection.

V2's best checkpoint was selected using validation IoU.
This experiment changes multiple training factors together;
therefore it is not an ablation study.

### 3.4 Evaluation

Metrics include precision, recall, F1, IoU, and
false-positive rate, calculated from valid labeled pixels.

Pooled metrics aggregate confusion counts across pixels,
while regional metrics are calculated separately.

All reported model comparisons should use the precise
split and evaluation procedure associated with each file.

## 4. Baseline comparison

### Random Forest versus U-Net V1

{as_md(
    ["Model", "Precision", "Recall", "F1", "IoU"],
    rf_rows
)}

This comparison uses the fixed test split and the
Random Forest configuration chosen using validation data.

## 5. U-Net V1 versus V2

### Validation results

{as_md(
    ["Model", "IoU", "F1", "Recall", "FPR"],
    [
        [
            name,
            percent(validation.loc[name, "iou"]),
            percent(validation.loc[name, "f1"]),
            percent(validation.loc[name, "recall"]),
            percent(validation.loc[name, "fpr"])
        ]
        for name in ["U-Net V1", "U-Net V2"]
    ]
)}

### Test results

{as_md(
    ["Model", "Precision", "Recall", "F1", "IoU", "FPR"],
    model_rows
)}

The improvement in test IoU was
{pp(v2["iou"] - v1["iou"])}. Recall increased, but
precision decreased and the false-positive rate rose.

### Regional test results

{as_md(
    ["Model", "Region", "Precision", "Recall", "IoU", "FPR"],
    region_rows
)}

Somalia remained substantially more difficult than Mekong
for both models. This cautions against drawing
global generalization claims from the pooled metrics.

## 6. Geographic domain shift

Analysis compared radar intensity distributions and
labeled water prevalence across training and test groups.

{as_md(
    ["Region", "Water prevalence", "VV JS", "VH JS", "V2 IoU"],
    domain_rows
)}

Jensen-Shannon divergence measures descriptive differences
between sampled radar histograms and the training
distribution. It does not establish causal explanations
for geographic performance gaps.

Low water prevalence and altered radar distributions
are plausible contributors to the low Somalia precision,
but additional controlled experiments would be required.

## 7. Geospatial water-area estimation

Predicted water masks were exported as georeferenced
GeoTIFF files. Water-covered area was estimated using
geospatial pixel geometry.

{as_md(
    ["Region", "Images", "Predicted km²", "Reference km²",
     "Relative bias"],
    water_rows
)}

These area totals are summed across image footprints.
Overlapping footprints could double-count coverage.
The outputs represent surface water, not necessarily
new flood inundation.

## 8. Infrastructure exposure

Geographic polygons from predicted water masks were
intersected with Overture Maps building footprints and
road segments for Mekong_1248200.

Exposure was validated against the labeled water mask.

{as_md(
    ["Model", "Flagged buildings", "Reference buildings",
     "TP", "FP", "FN", "Building recall", "Road length flagged"],
    exposure_rows
)}

These measurements are proxy water-intersection estimates,
not independently verified inundation or damage.
Infrastructure coverage and timestamps may not match
the satellite imagery.

## 9. Bootstrap reliability analysis

A paired, region-stratified bootstrap used 10,000
image resamples, preserving the Mekong and Somalia
sample counts and pairing V1/V2 predictions.

{as_md(
    ["Scope", "Metric", "Observed difference",
     "Lower 95%", "Upper 95%"],
    bootstrap_rows
)}

Overall recall improvements are consistently positive
under this resampling procedure. The interval for pooled
IoU improvement includes zero, so the evidence is less
conclusive for IoU.

These are exploratory conditional intervals, not
confidence intervals for performance in unseen countries.
Only two geographic test groups were used.

## 10. Limitations and validity threats

1. Only 110 image chips are available in this experiment.
2. Regional splitting uses filename-prefix groups;
   event independence has not been independently verified.
3. Test regions were examined during the development
   cycle, so later evaluations are diagnostic rather
   than independent holdout validation.
4. Some image chips contain very little water,
   making precision and IoU sensitive to false positives.
5. The infrastructure analysis covers one selected
   Mekong image, not the full test dataset.
6. Overture feature coverage and capture times may differ
   from those of the Sentinel-1 images.
7. Water intersections do not demonstrate confirmed
   flooding, passability, population exposure, or damage.
8. V2 modifies augmentation, loss, and training duration
   together; attribution of improvement is unresolved.
9. Bootstrap uncertainty does not account fully for
   spatial dependence or variation among new geographic
   regions.

## 11. Future work

- Verify event-level grouping and acquire genuinely
  independent geographic evaluation data.
- Evaluate exposure on a broader set of image footprints.
- Compare model confidence calibration and decision thresholds
  selected only using development data.
- Study controlled ablations of augmentation, loss, and
  training duration.
- Add pre-event water observations to distinguish persistent
  water from newly inundated areas.
- Quantify uncertainty in geographic area and infrastructure
  intersections.

## 12. Conclusion

GeoRescue demonstrates an end-to-end applied machine learning
workflow spanning SAR image segmentation, geographic
generalization analysis, geospatial area estimation,
infrastructure intersections, and interactive visualization.

The key finding is that improvements in pixel-level recall
can increase downstream infrastructure-exposure recall,
but performance is highly dependent on geographic conditions
and comes with false-positive trade-offs.

The prototype is intended for research and educational use,
not operational disaster-response decisions.
"""

(OUT / "technical_report.md").write_text(
    report, encoding="utf-8"
)

# -----------------------------------------
# Reproducibility guide
# -----------------------------------------

guide = """# GeoRescue Reproducibility Guide

## Environment

- Python 3.13 was used during local development.
- Windows PowerShell is the demonstrated environment.
- U-Net training and inference were performed on CPU.
- Dependencies include PyTorch, Rasterio, NumPy, Pandas,
  scikit-learn, Matplotlib, GeoPandas, pyproj, OSMnx,
  Overture Maps, Streamlit, Folium, and streamlit-folium.

Create a virtual environment and install dependencies
from the repository's verified requirements file.
The current environment should also be captured using
`python -m pip freeze` before publishing.

## Dataset

Place the authorized Sen1Floods11 source imagery under:

- `data/raw/S1Hand/`
- `data/raw/LabelHand/`

Use the image names and preprocessing expected by
`src/data/flood_dataset.py`.

The fixed split is:
`data/splits/unet_split.json`

Do not regenerate the split while reproducing reported
metrics. Check the dataset provider's redistribution
license before publishing data files.

## Training

Original model:

```powershell
python -m src.models.train_unet
```

Improved model:

```powershell
python -m src.models.train_unet_v2
```

The original and improved checkpoints are stored in
different folders.

## Core evaluation

```powershell
python -m src.evaluation.compare_rf_unet
python -m src.evaluation.compare_unet_versions
python -m src.evaluation.evaluate_unet_v2_test
```

## Geospatial analysis

```powershell
python -m src.evaluation.water_extent
python -m src.evaluation.analyze_geospatial
python -m src.evaluation.analyze_domain_shift
python -m src.evaluation.statistical_reliability
```

## Infrastructure analysis

The Overture download requires internet access only
when the data has not already been saved.

```powershell
python -m src.evaluation.infrastructure_overture
python -m src.evaluation.validate_exposure
python -m src.evaluation.visualize_exposure_errors
```

Infrastructure results depend on the available Overture
dataset version and downloaded feature coverage.

## Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Open the local Streamlit address shown in the terminal.

## Evaluation cautions

The test split was examined during iterative development.
Accordingly, final test comparisons are internal
diagnostic results, not an untouched external benchmark.

Do not infer global disaster-response capability from
this experiment.

## Reproducing the report

```powershell
python -m src.evaluation.generate_portfolio_report
```

This regenerates the technical report from saved result
CSVs. The README and existing experiments are unchanged.
"""

(OUT / "reproducibility.md").write_text(
    guide, encoding="utf-8"
)

# -----------------------------------------
# Result manifest
# -----------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {}

for name, relative_path in SOURCES.items():
    path = ROOT / relative_path
    files[name] = {
        "path": relative_path,
        "sha256": sha256(path),
        "bytes": path.stat().st_size
    }

manifest = {
    "generated_utc": generated,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "files": files
}

(OUT / "results_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("GEORESCUE REPORT GENERATION COMPLETE")
print("=" * 60)
print("Technical report:", OUT / "technical_report.md")
print("Reproducibility guide:", OUT / "reproducibility.md")
print("Results manifest:", OUT / "results_manifest.json")
print("Existing models and dashboard unchanged.")
