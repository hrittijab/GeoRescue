# GeoRescue: Geospatial Water Segmentation and Infrastructure Exposure

**Technical evaluation report**  
Generated: 2026-11-17 17:42 UTC  
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
59.36% pixel-aggregated IoU, compared with
55.79% for U-Net V1. Recall improved by
+10.85 pp, while false-positive rate
also increased by +2.05 pp.

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

| Model | Precision | Recall | F1 | IoU |
| --- | --- | --- | --- | --- |
| Random Forest | 71.78% | 70.26% | 71.01% | 55.05% |
| U-Net | 81.58% | 63.83% | 71.62% | 55.79% |

This comparison uses the fixed test split and the
Random Forest configuration chosen using validation data.

## 5. U-Net V1 versus V2

### Validation results

| Model | IoU | F1 | Recall | FPR |
| --- | --- | --- | --- | --- |
| U-Net V1 | 59.52% | 74.62% | 65.61% | 1.84% |
| U-Net V2 | 61.84% | 76.42% | 71.68% | 2.86% |

### Test results

| Model | Precision | Recall | F1 | IoU | FPR |
| --- | --- | --- | --- | --- | --- |
| U-Net V1 | 81.58% | 63.83% | 71.62% | 55.79% | 2.59% |
| U-Net V2 | 74.33% | 74.68% | 74.50% | 59.36% | 4.64% |

The improvement in test IoU was
+3.57 pp. Recall increased, but
precision decreased and the false-positive rate rose.

### Regional test results

| Model | Region | Precision | Recall | IoU | FPR |
| --- | --- | --- | --- | --- | --- |
| V1 | Mekong | 98.12% | 68.37% | 67.49% | 0.45% |
| V1 | Somalia | 14.52% | 22.63% | 9.70% | 4.51% |
| V2 | Mekong | 96.68% | 78.57% | 76.51% | 0.93% |
| V2 | Somalia | 14.32% | 39.33% | 11.73% | 7.97% |

Somalia remained substantially more difficult than Mekong
for both models. This cautions against drawing
global generalization claims from the pooled metrics.

## 6. Geographic domain shift

Analysis compared radar intensity distributions and
labeled water prevalence across training and test groups.

| Region | Water prevalence | VV JS | VH JS | V2 IoU |
| --- | --- | --- | --- | --- |
| Mekong | 25.56% | 0.03782 | 0.03574 | 76.51% |
| Somalia | 3.28% | 0.09753 | 0.06196 | 11.73% |

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

| Region | Images | Predicted km² | Reference km² | Relative bias |
| --- | --- | --- | --- | --- |
| Mekong | 10 | 43.192 | 61.986 | -30.32% |
| Somalia | 10 | 10.880 | 6.981 | +55.85% |

These area totals are summed across image footprints.
Overlapping footprints could double-count coverage.
The outputs represent surface water, not necessarily
new flood inundation.

## 8. Infrastructure exposure

Geographic polygons from predicted water masks were
intersected with Overture Maps building footprints and
road segments for Mekong_1248200.

Exposure was validated against the labeled water mask.

| Model | Flagged buildings | Reference buildings | TP | FP | FN | Building recall | Road length flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 20 | 110 | 20 | 0 | 90 | 18.18% | 10.61 m |
| V2 | 44 | 110 | 44 | 0 | 66 | 40.00% | 84.04 m |

These measurements are proxy water-intersection estimates,
not independently verified inundation or damage.
Infrastructure coverage and timestamps may not match
the satellite imagery.

## 9. Bootstrap reliability analysis

A paired, region-stratified bootstrap used 10,000
image resamples, preserving the Mekong and Somalia
sample counts and pairing V1/V2 predictions.

| Scope | Metric | Observed difference | Lower 95% | Upper 95% |
| --- | --- | --- | --- | --- |
| overall | IOU | +3.57 pp | -1.65 pp | +7.30 pp |
| overall | F1 | +2.88 pp | -1.55 pp | +5.59 pp |
| overall | RECALL | +10.85 pp | +7.82 pp | +13.02 pp |
| overall | FPR | +2.05 pp | +1.11 pp | +3.17 pp |
| Mekong | IOU | +9.02 pp | +3.71 pp | +11.67 pp |
| Mekong | F1 | +6.10 pp | +2.60 pp | +8.19 pp |
| Mekong | RECALL | +10.20 pp | +6.81 pp | +12.42 pp |
| Mekong | FPR | +0.48 pp | +0.08 pp | +1.10 pp |
| Somalia | IOU | +2.03 pp | -1.08 pp | +4.36 pp |
| Somalia | F1 | +3.31 pp | -1.81 pp | +6.87 pp |
| Somalia | RECALL | +16.70 pp | +9.84 pp | +19.01 pp |
| Somalia | FPR | +3.46 pp | +1.71 pp | +5.63 pp |

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


## Additional Analysis

[Exposure Error Propagation Supplement](exposure_propagation.md)
