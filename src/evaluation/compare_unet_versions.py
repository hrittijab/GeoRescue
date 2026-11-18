from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet

OUTPUT = Path("experiments/unet_v2/comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(min(4, torch.get_num_threads()))

CHECKPOINTS = {
    "U-Net V1": "experiments/unet/best_unet.pt",
    "U-Net V2": "experiments/unet_v2/best_unet_v2.pt"
}

def load_model(path):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    config = checkpoint.get("config", {})

    model = UNet(
        in_channels=config.get("in_channels", 2),
        base=config.get("base", 16)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    return model


def calculate(counts):
    tp, fp, fn, tn = map(int, counts)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr
    }


models = {
    name: load_model(path)
    for name, path in CHECKPOINTS.items()
}

dataset = FloodDataset("val")

overall = defaultdict(
    lambda: np.zeros(4, dtype=np.int64)
)

regions = defaultdict(
    lambda: np.zeros(4, dtype=np.int64)
)

image_rows = []

print("VALIDATION MODEL COMPARISON")
print("=" * 55)

with torch.inference_mode():
    for i in range(len(dataset)):
        sample = dataset[i]

        name = sample["name"]
        region = name.rsplit("_", 1)[0]

        image = sample["image"].unsqueeze(0)
        truth = sample["mask"][0].numpy() >= 0.5
        valid = sample["valid"][0].numpy() > 0.5

        for model_name, model in models.items():
            logits = model(image)
            probabilities = torch.sigmoid(
                logits
            )[0, 0].numpy()

            pred = probabilities >= 0.5

            tp = int(np.sum(pred & truth & valid))
            fp = int(np.sum(pred & ~truth & valid))
            fn = int(np.sum(~pred & truth & valid))
            tn = int(np.sum(~pred & ~truth & valid))

            counts = np.array(
                [tp, fp, fn, tn], dtype=np.int64
            )

            overall[model_name] += counts
            regions[(model_name, region)] += counts

            image_rows.append({
                "image": name,
                "region": region,
                "model": model_name,
                **calculate(counts)
            })

        print(
            f"[{i + 1}/{len(dataset)}] {name}",
            flush=True
        )


summary_rows = []

for model_name in CHECKPOINTS:
    summary_rows.append({
        "model": model_name,
        **calculate(overall[model_name])
    })

regional_rows = []

for (model_name, region), counts in sorted(
    regions.items()
):
    regional_rows.append({
        "model": model_name,
        "region": region,
        **calculate(counts)
    })


pd.DataFrame(summary_rows).to_csv(
    OUTPUT / "summary.csv", index=False
)

pd.DataFrame(regional_rows).to_csv(
    OUTPUT / "regional.csv", index=False
)

pd.DataFrame(image_rows).to_csv(
    OUTPUT / "per_image.csv", index=False
)

# Per-region IoU comparison
regional_df = pd.DataFrame(regional_rows)

pivot = regional_df.pivot(
    index="region",
    columns="model",
    values="iou"
)

ax = (pivot * 100).plot(
    kind="bar", figsize=(9, 5)
)

ax.set_ylabel("IoU (%)")
ax.set_xlabel("Validation region")
ax.set_title("GeoRescue: U-Net V1 vs V2")
ax.set_ylim(0, 100)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUTPUT / "regional_comparison.png",
    dpi=200
)
plt.close()

print("\n" + "=" * 55)
print("OVERALL VALIDATION METRICS")
print("=" * 55)
print(
    pd.DataFrame(summary_rows)
    .round(4)
    .to_string(index=False)
)

print("\nREGIONAL VALIDATION METRICS")
print(
    pd.DataFrame(regional_rows)
    .round(4)
    .to_string(index=False)
)

print("\nUNET V1 VS V2 COMPARISON COMPLETE!")
print("Results saved to:", OUTPUT)
