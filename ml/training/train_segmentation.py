"""
Trains the optional U-Net tumor-segmentation model defined in
backend/app/ml/unet.py. This requires a dataset with PIXEL-LEVEL tumor
masks (e.g. BraTS, or a Kaggle brain-MRI segmentation dataset), which is
different from the plain classification dataset used by train.py.

Expected dataset layout:

    dataset/segmentation/
        images/*.png
        masks/*.png      (same filenames as images/, binary tumor masks)

Usage:
    cd ml/training
    python train_segmentation.py --data-dir ../../dataset/segmentation --epochs 25
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
from app.ml.unet import UNet  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = sorted(os.listdir(images_dir))
        self.img_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, fname)).convert("L")
        return self.img_tf(image), (self.mask_tf(mask) > 0.5).float()


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset/segmentation")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="../../backend/models")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")
    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        print(f"ERROR: expected '{images_dir}' and '{masks_dir}' to exist.")
        print("Segmentation is optional -- skip this script if you don't have a masked dataset.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(images_dir, masks_dir)
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = bce(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output, "segmentation_unet.pt")
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, checkpoint_path)
            print(f"  -> Saved new best checkpoint to {checkpoint_path}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
