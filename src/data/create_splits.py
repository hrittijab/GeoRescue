from pathlib import Path
from collections import defaultdict
import json
import random

IMAGE_DIR = Path("data/raw/S1Hand")
LABEL_DIR = Path("data/raw/LabelHand")
OUTPUT = Path("data/splits/unet_split.json")

groups = defaultdict(list)

for image_path in sorted(IMAGE_DIR.glob("*_S1Hand.tif")):
    name = image_path.name.replace("_S1Hand.tif", "")
    label_path = LABEL_DIR / f"{name}_LabelHand.tif"

    if not label_path.exists():
        continue

    group = name.rsplit("_", 1)[0]
    groups[group].append(name)

names = sorted(groups)

if len(names) < 5:
    raise RuntimeError("At least five groups required.")

rng = random.Random(42)
rng.shuffle(names)

test_groups = names[:2]
val_groups = names[2:4]
train_groups = names[4:]

def collect(selected):
    return [
        name
        for group in selected
        for name in groups[group]
    ]

splits = {
    "train": collect(train_groups),
    "val": collect(val_groups),
    "test": collect(test_groups),
    "groups": {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups
    }
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. "
        "Keep the existing split for reproducibility."
    )

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print("\nGEORESCUE DATA SPLITS")
print("=" * 45)

for split in ["train", "val", "test"]:
    print(
        f"{split.upper():5}: "
        f"{len(splits[split])} images | "
        f"groups: {splits['groups'][split]}"
    )

print("\nSaved:", OUTPUT)
