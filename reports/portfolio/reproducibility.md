# GeoRescue Reproducibility Guide

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
