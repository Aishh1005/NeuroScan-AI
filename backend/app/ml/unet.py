"""
A compact, real (not simulated) U-Net implementation for optional tumor
segmentation. This architecture must be trained on a dataset that provides
pixel-level tumor masks (e.g. the BraTS dataset or a Kaggle brain-MRI
segmentation dataset with masks) before it can produce real segmentations.

If no trained segmentation weights are present, the inference layer
(app/ml/inference.py) intentionally disables this feature rather than
fabricating a tumor outline -- see SEGMENTATION_MODEL_PATH handling.
"""
import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.down1 = double_conv(in_channels, base)
        self.down2 = double_conv(base, base * 2)
        self.down3 = double_conv(base * 2, base * 4)
        self.down4 = double_conv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = double_conv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.upconv4 = double_conv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.upconv3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.upconv2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.upconv1 = double_conv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bn = self.bottleneck(self.pool(d4))

        u4 = self.up4(bn)
        u4 = self.upconv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4)
        u3 = self.upconv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.upconv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.upconv1(torch.cat([u1, d1], dim=1))

        return self.out_conv(u1)
