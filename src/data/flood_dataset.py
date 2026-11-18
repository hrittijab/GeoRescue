import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


class FloodDataset(Dataset):
    def __init__(self, split="train"):
        split_file = Path("data/splits/unet_split.json")

        with open(split_file, encoding="utf-8") as f:
            splits = json.load(f)

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")

        self.names = splits[split]
        self.image_dir = Path("data/raw/S1Hand")
        self.label_dir = Path("data/raw/LabelHand")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image_path = (
            self.image_dir / f"{name}_S1Hand.tif"
        )
        label_path = (
            self.label_dir / f"{name}_LabelHand.tif"
        )

        with rasterio.open(image_path) as src:
            image = src.read([1, 2]).astype(np.float32)

        with rasterio.open(label_path) as src:
            label = src.read(1)

        if image.shape[1:] != label.shape:
            raise ValueError(f"Shape mismatch: {name}")

        valid_radar = np.isfinite(image).all(axis=0)
        valid_label = np.isin(label, [0, 1])
        valid = valid_radar & valid_label

        # Fixed dB clipping for the initial experiment.
        # These are preprocessing parameters, not
        # statistics estimated from test imagery.
        image = np.nan_to_num(
            image, nan=-35.0,
            posinf=5.0, neginf=-35.0
        )

        image = np.clip(image, -35.0, 5.0)
        image = (image + 35.0) / 40.0

        # Invalid radar locations become zero input.
        image[:, ~valid_radar] = 0.0

        # Binary target and separate validity mask.
        mask = (label == 1).astype(np.float32)

        return {
            "image": torch.from_numpy(
                image.copy()
            ),
            "mask": torch.from_numpy(
                mask[None].copy()
            ),
            "valid": torch.from_numpy(
                valid[None].astype(np.float32)
            ),
            "name": name
        }


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dataset = FloodDataset(split)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        print(f"\n{split.upper()} DATASET")
        print("Images:", len(dataset))

        if len(dataset) == 0:
            continue

        batch = next(iter(loader))

        print("Image:", batch["image"].shape)
        print("Mask:", batch["mask"].shape)
        print("Valid:", batch["valid"].shape)

        print(
            "Input range:",
            float(batch["image"].min()),
            float(batch["image"].max())
        )

        print(
            "Valid pixels:",
            int(batch["valid"].sum().item())
        )

        assert torch.isfinite(batch["image"]).all()
        assert torch.isfinite(batch["mask"]).all()
        assert torch.isfinite(batch["valid"]).all()

    print("\nPYTORCH DATASET READY!")
