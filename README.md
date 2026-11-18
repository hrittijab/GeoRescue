

# GeoRescue
### Geospatial AI for Surface-Water Mapping and Infrastructure Exposure Analysis

GeoRescue is an applied machine learning and geospatial analytics project that uses Sentinel-1 synthetic aperture radar (SAR) imagery to identify surface water and estimate potential infrastructure exposure.

The project investigates an important question:

**Do improvements in water segmentation accuracy lead to more reliable building and road exposure estimates across different geographic environments?**

## Key Results

| Metric | U-Net V1 | U-Net V2 |
|---|---:|---:|
| Test IoU | 55.79% | 59.36% |
| Test precision | 81.58% | 74.33% |
| Test recall | 63.83% | 74.68% |
| Test F1 | 71.62% | 74.50% |
| False-positive rate | 2.59% | 4.64% |

U-Net V2 improved recall by **10.85 percentage points**, but the false-positive rate also increased.

A paired, region-stratified bootstrap (10,000 resamples) produced an exploratory 95% interval of **-1.65 to +7.30 percentage points** for the overall IoU improvement. The interval includes zero.

These results are internal diagnostic evaluations, not an independent global benchmark.

## Infrastructure Exposure

On the selected Mekong image `Mekong_1248200`, GeoRescue combined predicted water polygons with Overture Maps building footprints and road segments.

| Exposure metric | V1 | V2 |
|---|---:|---:|
| Reference-exposed buildings | 110 | 110 |
| Correctly flagged buildings | 20 | 44 |
| Missed buildings | 90 | 66 |
| Building exposure recall | 18.18% | 40.00% |
| Predicted road length intersecting water | 10.61 m | 84.04 m |

Reference water intersected approximately 853.46 metres of mapped roads.

These results represent geographic intersections with labeled surface water, not independently verified flood damage.

## Geographic Generalization

Geographic evaluation revealed substantial performance differences:

| Test region | U-Net V2 IoU | Water prevalence |
|---|---:|---:|
| Mekong | 76.51% | 25.56% |
| Somalia | 11.73% | 3.28% |

Radar-distribution analysis revealed larger Jensen-Shannon divergence from the training data in Somalia than in Mekong.

This suggests that geographic distribution shift and water-class imbalance may contribute to poor performance, although causality has not been established.

## System Architecture

```text
Sentinel-1 SAR (VV + VH)
         |
         v
Raster Preprocessing
         |
         v
+-------------------------+
| Random Forest Baseline  |
| U-Net V1 / U-Net V2     |
+-------------------------+
         |
         v
Georeferenced Water Masks
         |
         +----> Pixel-Level Evaluation
         |
         +----> Geographic Domain Shift Analysis
         |
         v
Geospatial Water-Area Estimation
         |
         v
Overture Maps Buildings + Roads
         |
         v
Infrastructure Exposure Analysis
         |
         v
Streamlit Interactive Dashboard
```

## Technology Stack

**Machine learning:** Python, PyTorch, scikit-learn, NumPy, Pandas

**Geospatial:** Rasterio, GeoPandas, Shapely, pyproj, Overture Maps

**Visualization:** Matplotlib, Streamlit, Folium

**Data:** Sen1Floods11 Sentinel-1 SAR imagery

## Dataset and Evaluation

The project uses 110 manually labeled SAR image chips.

- Training: 70 images across seven geographic groups
- Validation: 20 images from India and Spain
- Test: 20 images from Mekong and Somalia

The U-Net architecture contains approximately 1.94 million parameters.

V2 introduces spatial augmentation, combined BCE and Dice loss, and validation-based checkpoint selection.

Geographic groups are based on filename prefixes and have not been independently verified as event-level splits. Test regions were examined during development, so follow-up results are reported as diagnostic.

## Run the Dashboard

Create and activate a Python virtual environment, then install the required project dependencies.

Start the application:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard presents:

- Interactive water and infrastructure exposure maps
- U-Net V1 versus V2 model comparisons
- Regional segmentation performance
- Water-area estimation and error analysis
- Building and road exposure classifications

Some dashboard pages require saved experiment outputs and locally downloaded geographic layers. The repository is not yet a fully self-contained installation.

## Technical Documentation

- [Technical Report](reports/portfolio/technical_report.md)
- [Reproducibility Guide](reports/portfolio/reproducibility.md)

## Limitations

- The current dataset is small and geographically limited.
- Somalia results show substantial generalization failures.
- Infrastructure exposure was assessed on one selected image.
- Overture infrastructure coverage and observation dates may not match the satellite data.
- Predictions identify surface water, not necessarily newly flooded areas.
- Infrastructure intersections are not verified damage assessments.
- The experiments are intended for research and educational use, not operational disaster response.

## Future Work

- Independent event-level and geographic validation
- Pre-event versus post-event flood-change detection
- Model calibration and uncertainty estimation
- Broader infrastructure exposure evaluation
- Controlled training ablations
- Reproducible dataset and experiment packaging

## Project Status

**Working research prototype:** model training, evaluation, geospatial analytics, and dashboard completed. Scientific validation and release packaging remain ongoing.
'@ | Set-Content -Encoding UTF8 README.md

Write-Host "Professional GeoRescue README created!"
