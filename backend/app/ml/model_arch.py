"""
Real, trainable model architectures for brain MRI tumor classification.

Supported architectures:
- ResNet50
- EfficientNet-B0
- VGG16

VGG16 is used by the current NeuroScan AI classification pipeline.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(
    architecture: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True
) -> nn.Module:

    architecture = architecture.lower()

    # ============================================================
    # RESNET50
    # ============================================================

    if architecture == "resnet50":

        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        return model


    # ============================================================
    # EFFICIENTNET-B0
    # ============================================================

    if architecture == "efficientnet":

        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        return model


    # ============================================================
    # VGG16
    # ============================================================

    if architecture == "vgg16":

        weights = (
            models.VGG16_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = models.vgg16(
            weights=weights
        )

        in_features = model.classifier[6].in_features

        model.classifier[6] = nn.Linear(
            in_features,
            num_classes
        )

        return model


    # ============================================================
    # UNSUPPORTED ARCHITECTURE
    # ============================================================

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )


def get_target_layer(
    model: nn.Module,
    architecture: str
):
    """
    Returns the convolutional layer used as the
    Grad-CAM target layer.
    """

    architecture = architecture.lower()


    # ============================================================
    # RESNET50
    # ============================================================

    if architecture == "resnet50":

        return model.layer4[-1]


    # ============================================================
    # EFFICIENTNET
    # ============================================================

    if architecture == "efficientnet":

        return model.features[-1]


    # ============================================================
    # VGG16
    # ============================================================

    if architecture == "vgg16":

        return model.features[-1]


    # ============================================================
    # UNSUPPORTED ARCHITECTURE
    # ============================================================

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )