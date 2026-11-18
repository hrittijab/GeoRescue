from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet, masked_bce_loss

# ----------------------------------
# Configuration
# ----------------------------------

SEED = 42
EPOCHS = 3
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001

OUTPUT = Path("experiments/unet")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))

print("Training device:", device, flush=True)


# ----------------------------------
# Patch dataset for CPU training
# ----------------------------------

class PatchDataset(Dataset):
    def __init__(self):
        self.dataset = FloodDataset("train")

    def __len__(self):
        return len(self.dataset) * PATCHES_PER_IMAGE

    def __getitem__(self, index):
        sample = self.dataset[index % len(self.dataset)]

        image = sample["image"]
        mask = sample["mask"]
        valid = sample["valid"]

        height, width = image.shape[-2:]

        if height < PATCH_SIZE or width < PATCH_SIZE:
            raise ValueError("Image smaller than patch size")

        # Prefer patches with enough valid pixels.
        for _ in range(12):
            y = random.randint(0, height - PATCH_SIZE)
            x = random.randint(0, width - PATCH_SIZE)

            patch_valid = valid[
                :, y:y + PATCH_SIZE, x:x + PATCH_SIZE
            ]

            if patch_valid.mean().item() >= 0.25:
                break

        sl = (
            slice(None),
            slice(y, y + PATCH_SIZE),
            slice(x, x + PATCH_SIZE)
        )

        return {
            "image": image[sl].contiguous(),
            "mask": mask[sl].contiguous(),
            "valid": valid[sl].contiguous()
        }


train_dataset = PatchDataset()
val_dataset = FloodDataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Full images for validation.
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset.dataset))
print("Training patches per epoch:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ----------------------------------
# Model and optimizer
# ----------------------------------

model = UNet(in_channels=2, base=16).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ----------------------------------
# Validation metrics
# ----------------------------------

@torch.no_grad()
def validate():
    model.eval()

    tp = fp = fn = tn = 0
    total_loss = 0.0
    total_valid = 0.0

    for batch in val_loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        valid = batch["valid"].to(device)

        logits = model(image)

        pixel_losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, mask, reduction="none"
        )

        total_loss += (pixel_losses * valid).sum().item()
        total_valid += valid.sum().item()

        predicted = torch.sigmoid(logits) >= 0.5
        truth = mask >= 0.5
        usable = valid > 0.5

        tp += int((predicted & truth & usable).sum().item())
        fp += int((predicted & ~truth & usable).sum().item())
        fn += int((~predicted & truth & usable).sum().item())
        tn += int((~predicted & ~truth & usable).sum().item())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0

    return {
        "loss": total_loss / max(total_valid, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


# ----------------------------------
# Training loop
# ----------------------------------

history = []
best_iou = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    total_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid = batch["valid"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = masked_bce_loss(
            logits, masks, valid
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    train_loss = running_loss / max(total_batches, 1)

    metrics = validate()

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": metrics["loss"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_iou": metrics["iou"]
    }

    history.append(row)

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
        f"\nTrain loss: {train_loss:.4f}"
        f"\nVal loss:   {metrics['loss']:.4f}"
        f"\nVal IoU:    {metrics['iou']:.4f}"
        f"\nVal F1:     {metrics['f1']:.4f}"
        f"\nVal recall: {metrics['recall']:.4f}",
        flush=True
    )

    if metrics["iou"] > best_iou:
        best_iou = metrics["iou"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_iou": best_iou,
                "config": {
                    "in_channels": 2,
                    "base": 16,
                    "patch_size": PATCH_SIZE
                }
            },
            OUTPUT / "best_unet.pt"
        )

        print("Best model saved!", flush=True)

    with open(
        OUTPUT / "training_history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)


print("\n" + "=" * 50)
print("GEORESCUE U-NET TRAINING COMPLETE")
print("=" * 50)
print(f"Best validation IoU: {best_iou:.4f}")
print(f"Model saved: {OUTPUT / 'best_unet.pt'}")
print(f"History saved: {OUTPUT / 'training_history.csv'}")
