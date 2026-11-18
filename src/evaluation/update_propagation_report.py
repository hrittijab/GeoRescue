from pathlib import Path
import pandas as pd

ROOT = Path("reports/portfolio")
ROOT.mkdir(parents=True, exist_ok=True)

SOURCE = Path(
    "experiments/exposure_propagation"
)

comparison = pd.read_csv(
    SOURCE / "model_impact.csv"
)

details = pd.read_csv(
    SOURCE / "exposure_propagation.csv"
)

def pct(value):
    if pd.isna(value):
        return "Undefined"
    return f"{value * 100:.2f}%"

def change(value):
    if pd.isna(value):
        return "Undefined"
    return f"{value * 100:+.2f} pp"

lines = [
    "# GeoRescue: Exposure Error Propagation",
    "",
    "## Research question",
    "",
    "Do improvements in satellite water-segmentation "
    "accuracy consistently improve downstream "
    "infrastructure-exposure estimates?",
    "",
    "## Experimental design",
    "",
    "Four previously examined test images were selected "
    "for exploratory analysis: two from Mekong and two "
    "from Somalia. U-Net V1 and V2 were evaluated against "
    "the same water labels and infrastructure features "
    "within each image.",
    "",
    "Infrastructure records were retrieved from Overture "
    "Maps. Somalia road measurements were not assessed "
    "where the transportation download was unavailable.",
    "",
    "## Results",
    "",
    "| Image | Region | V1 IoU | V2 IoU | IoU change "
    "| V1 building recall | V2 building recall |",
    "|---|---|---:|---:|---:|---:|---:|"
]

for _, r in comparison.iterrows():
    lines.append(
        f"| {r['image']} | {r['region']} "
        f"| {pct(r['iou_v1'])} "
        f"| {pct(r['iou_v2'])} "
        f"| {change(r['iou_change'])} "
        f"| {pct(r['building_recall_v1'])} "
        f"| {pct(r['building_recall_v2'])} |"
    )

lines.extend([
    "",
    "## Infrastructure coverage",
    "",
    "| Image | Mapped buildings | "
    "Reference-exposed buildings | Road data available |",
    "|---|---:|---:|---|"
])

for image, group in details.groupby("image", sort=False):
    r = group.iloc[0]

    road_available = (
        "No" if pd.isna(r["mapped_road_segments"])
        else "Yes"
    )

    lines.append(
        f"| {image} "
        f"| {int(r['mapped_buildings'])} "
        f"| {int(r['reference_exposed_buildings'])} "
        f"| {road_available} |"
    )

lines.extend([
    "",
    "## Interpretation",
    "",
    "Both selected Mekong images improved in segmentation "
    "IoU and building-exposure recall. However, substantial "
    "numbers of reference-exposed buildings remained missed.",
    "",
    "Somalia_12849 provides an informative counterexample: "
    "segmentation IoU improved without improving building "
    "exposure recall. The Somalia_1087508 building-recall "
    "comparison is undefined and cannot be treated as zero.",
    "",
    "These examples illustrate why pixel-level segmentation "
    "metrics should be supplemented with downstream "
    "task-specific evaluation.",
    "",
    "## Limitations",
    "",
    "- The four images were deliberately selected after "
    "previous test-set inspection.",
    "- Building exposure is inferred from intersection with "
    "reference water labels, not independently verified damage.",
    "- Overture infrastructure coverage may be incomplete.",
    "- Infrastructure observations may not match satellite dates.",
    "- Road data are not available for every selected image.",
    "- One or more undefined recall values reduce the number "
    "of interpretable paired comparisons.",
    "- The experiment does not establish a general causal or "
    "statistical relationship between IoU and exposure recall.",
    "",
    "## Conclusion",
    "",
    "The pilot indicates that segmentation improvements can "
    "improve downstream building-exposure recall, but such "
    "improvements are not automatic. Wider, independently "
    "sampled evaluation is required.",
    ""
])

supplement = ROOT / "exposure_propagation.md"

supplement.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

main = ROOT / "technical_report.md"

if main.exists():
    report = main.read_text(encoding="utf-8")

    link = (
        "[Exposure Error Propagation Supplement]"
        "(exposure_propagation.md)"
    )

    if link not in report:
        report += (
            "\n\n## Additional Analysis\n\n"
            f"{link}\n"
        )

        main.write_text(
            report,
            encoding="utf-8"
        )

print("EXPOSURE PROPAGATION REPORT UPDATED!")
print("Supplement:", supplement)
print("Main report:", main)
