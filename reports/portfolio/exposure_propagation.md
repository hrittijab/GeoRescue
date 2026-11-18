# GeoRescue: Exposure Error Propagation

## Research question

Do improvements in satellite water-segmentation accuracy consistently improve downstream infrastructure-exposure estimates?

## Experimental design

Four previously examined test images were selected for exploratory analysis: two from Mekong and two from Somalia. U-Net V1 and V2 were evaluated against the same water labels and infrastructure features within each image.

Infrastructure records were retrieved from Overture Maps. Somalia road measurements were not assessed where the transportation download was unavailable.

## Results

| Image | Region | V1 IoU | V2 IoU | IoU change | V1 building recall | V2 building recall |
|---|---|---:|---:|---:|---:|---:|
| Mekong_1248200 | Mekong | 65.43% | 78.63% | +13.20 pp | 18.18% | 40.00% |
| Mekong_1396181 | Mekong | 59.95% | 70.86% | +10.91 pp | 5.39% | 13.17% |
| Somalia_1087508 | Somalia | 19.90% | 18.90% | -1.01 pp | Undefined | Undefined |
| Somalia_12849 | Somalia | 12.27% | 18.33% | +6.06 pp | 0.00% | 0.00% |

## Infrastructure coverage

| Image | Mapped buildings | Reference-exposed buildings | Road data available |
|---|---:|---:|---|
| Mekong_1248200 | 1187 | 110 | Yes |
| Mekong_1396181 | 1432 | 501 | Yes |
| Somalia_12849 | 993 | 8 | No |
| Somalia_1087508 | 979 | 0 | No |

## Interpretation

Both selected Mekong images improved in segmentation IoU and building-exposure recall. However, substantial numbers of reference-exposed buildings remained missed.

Somalia_12849 provides an informative counterexample: segmentation IoU improved without improving building exposure recall. The Somalia_1087508 building-recall comparison is undefined and cannot be treated as zero.

These examples illustrate why pixel-level segmentation metrics should be supplemented with downstream task-specific evaluation.

## Limitations

- The four images were deliberately selected after previous test-set inspection.
- Building exposure is inferred from intersection with reference water labels, not independently verified damage.
- Overture infrastructure coverage may be incomplete.
- Infrastructure observations may not match satellite dates.
- Road data are not available for every selected image.
- One or more undefined recall values reduce the number of interpretable paired comparisons.
- The experiment does not establish a general causal or statistical relationship between IoU and exposure recall.

## Conclusion

The pilot indicates that segmentation improvements can improve downstream building-exposure recall, but such improvements are not automatic. Wider, independently sampled evaluation is required.
