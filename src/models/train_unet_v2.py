from pathlib import Path
import csv
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.data.flood_dataset import FloodDataset
from src.models.unet import UNet

# ============================================
# GEORESCUE: IMPROVED U-NET TRAINING
# ============================================

SEED = 42
EPOCHS = 10
PATIENCE = 3
MIN_DELTA = 0.002
BATCH_SIZE = 4
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 2
LEARNING_RATE = 0.0007

OUTPUT = Path("experiments/unet_v2")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(4, torch.get_num_threads()))
DEVICE = torch.device("cpu")

# ============================================
# 1. DATA AUGMENTATION
# ============================================

class AugmentedFloodDataset(Dataset):
    def __init__(self):
        self.dataset = FloodDataset("train")

    def __len__(self):
        return len(self.dataset) * PATCHES_PER_IMAGE

    def __getitem__(self, index):
        sample = self.dataset[index % len(self.dataset)]

        image = sample["image"].clone()
        mask = sample["mask"].clone()
        valid = sample["valid"].clone()

        h, w = image.shape[-2:]

        if h < PATCH_SIZE or w < PATCH_SIZE:
            raise ValueError("Image smaller than patch size.")

        top = random.randint(0, h - PATCH_SIZE)
        left = random.randint(0, w - PATCH_SIZE)

        image = image[
            :, top:top+PATCH_SIZE, left:left+PATCH_SIZE
        ]

        mask = mask[
            :, top:top+PATCH_SIZE, left:left+PATCH_SIZE
        ]

        valid = valid[
            :, top:top+PATCH_SIZE, left:left+PATCH_SIZE
        ]

        # Apply identical transforms to image,
        # mask, and validity mask.
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
            valid = torch.flip(valid, dims=[-1])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
            valid = torch.flip(valid, dims=[-2])

        k = random.randint(0, 3)

        if k:
            image = torch.rot90(image, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
            valid = torch.rot90(valid, k, [-2, -1])

        return (
            image.contiguous(),
            mask.contiguous(),
            valid.contiguous()
        )


# ============================================
# 2. BCE + DICE LOSS
# ============================================

def combined_loss(logits, target, valid):
    valid = (valid > 0.5).float()
    target = target.float().clamp(0, 1)

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none"
    )

    bce = (bce * valid).sum() / valid.sum().clamp_min(1)

    probabilities = torch.sigmoid(logits)

    # Soft Dice over valid pixels only.
    intersection = (
        probabilities * target * valid
    ).sum()

    denominator = (
        (probabilities * valid).sum()
        + (target * valid).sum()
    )

    dice_loss = 1 - (
        (2 * intersection + 1)
        / (denominator + 1)
    )

    return 0.5 * bce + 0.5 * dice_loss


# ============================================
# 3. DATA LOADERS
# ============================================

train_data = AugmentedFloodDataset()
val_data = FloodDataset("val")

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Device:", DEVICE)
print("Training images:", len(train_data.dataset))
print("Validation images:", len(val_data))
print("Training patches per epoch:", len(train_data))

# ============================================
# 4. MODEL
# ============================================

model = UNet(
    in_channels=2,
    base=16
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================
# 5. VALIDATION
# ============================================

def validate():
    model.eval()

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():
        for batch in val_loader:
            image = batch["image"].to(DEVICE)
            target = batch["mask"].to(DEVICE)
            valid = batch["valid"].to(DEVICE)

            logits = model(image)

            loss = combined_loss(
                logits, target, valid
            )

            total_loss += float(loss.item())

            prediction = torch.sigmoid(logits) >= 0.5
            truth = target >= 0.5
            usable = valid > 0.5

            total_tp += int(
                (prediction & truth & usable).sum()
            )
            total_fp += int(
                (prediction & ~truth & usable).sum()
            )
            total_fn += int(
                (~prediction & truth & usable).sum()
            )

    denom_iou = total_tp + total_fp + total_fn
    denom_f1 = 2 * total_tp + total_fp + total_fn

    iou = (
        total_tp / denom_iou
        if denom_iou else float("nan")
    )

    f1 = (
        2 * total_tp / denom_f1
        if denom_f1 else float("nan")
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0
    )

    return {
        "val_loss": total_loss / len(val_loader),
        "val_iou": iou,
        "val_f1": f1,
        "val_recall": recall
    }


# ============================================
# 6. TRAINING WITH EARLY STOPPING
# ============================================

best_iou = -1.0
best_epoch = 0
no_improvement = 0

history = []

print("\nTRAINING IMPROVED U-NET")
print("=" * 60, flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for image, mask, valid in train_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        valid = valid.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(image)
        loss = combined_loss(logits, mask, valid)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        epoch_loss += float(loss.item())

    train_loss = epoch_loss / len(train_loader)

    metrics = validate()
    val_iou = metrics["val_iou"]

    result = {
        "epoch": epoch,
        "train_loss": train_loss,
        **metrics
    }

    history.append(result)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {metrics['val_loss']:.4f} | "
        f"Val IoU: {val_iou:.4f} | "
        f"Val F1: {metrics['val_f1']:.4f} | "
        f"Val Recall: {metrics['val_recall']:.4f}",
        flush=True
    )

    if np.isfinite(val_iou) and val_iou > best_iou + MIN_DELTA:
        best_iou = val_iou
        best_epoch = epoch
        no_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_validation_iou": best_iou,
                "config": {
                    "in_channels": 2,
                    "base": 16,
                    "patch_size": PATCH_SIZE,
                    "loss": "0.5 BCE + 0.5 Dice",
                    "augmentation": "flips + rotations"
                }
            },
            OUTPUT / "best_unet_v2.pt"
        )

        print("  Saved new best checkpoint!", flush=True)

    else:
        no_improvement += 1

        if no_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}",
                flush=True
            )
            break


# ============================================
# 7. SAVE TRAINING HISTORY
# ============================================

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

print("\n" + "=" * 60)
print("GEORESCUE U-NET V2 TRAINING COMPLETE")
print("=" * 60)
print("Best epoch:", best_epoch)
print("Best validation IoU:", round(best_iou, 4))
print("Checkpoint:", OUTPUT / "best_unet_v2.pt")
print("History:", OUTPUT / "training_history.csv")
