import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, base=16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            base * 8, base * 16
        )

        self.up4 = nn.ConvTranspose2d(
            base * 16, base * 8, 2, stride=2
        )
        self.dec4 = DoubleConv(
            base * 16, base * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(
            base * 8, base * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(
            base * 4, base * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2
        )
        self.dec1 = DoubleConv(base * 2, base)

        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(
            torch.cat([self.up4(b), e4], dim=1)
        )
        d3 = self.dec3(
            torch.cat([self.up3(d4), e3], dim=1)
        )
        d2 = self.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )
        d1 = self.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

        # Raw logits; sigmoid is applied for inference.
        return self.output(d1)


def masked_bce_loss(logits, targets, valid):
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    return (
        (loss * valid).sum()
        / valid.sum().clamp(min=1.0)
    )


if __name__ == "__main__":
    from src.data.flood_dataset import FloodDataset

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = UNet().to(device)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {parameters:,}")

    dataset = FloodDataset("train")
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    valid = sample["valid"].unsqueeze(0).to(device)

    print("Input shape:", tuple(image.shape))

    # Test full-resolution forward pass.
    model.eval()
    with torch.no_grad():
        logits = model(image)

        print("Output shape:", tuple(logits.shape))

        assert logits.shape == mask.shape

        loss = masked_bce_loss(
            logits, mask, valid
        )

        print("Initial masked BCE loss:", loss.item())

    # Test backward propagation on a small crop.
    model.train()
    model.zero_grad(set_to_none=True)

    crop_image = image[:, :, :128, :128]
    crop_mask = mask[:, :, :128, :128]
    crop_valid = valid[:, :, :128, :128]

    crop_logits = model(crop_image)

    train_loss = masked_bce_loss(
        crop_logits, crop_mask, crop_valid
    )

    train_loss.backward()

    print("Backward pass: successful")
    print("\nUNET ARCHITECTURE READY!")
