
import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------
# Make backend/app importable
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

BACKEND_DIR = os.path.join(
    PROJECT_ROOT,
    "backend"
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.ml.model_arch import get_model


# ---------------------------------------------------------
# ImageNet normalization
# ---------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------

def build_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        ),
    ])

    return train_transform, val_transform


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Classifier Training"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../dataset"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet",
        choices=[
            "resnet50",
            "efficientnet"
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../../backend/models"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    train_dir = os.path.abspath(
        os.path.join(args.data_dir, "train")
    )

    val_dir = os.path.abspath(
        os.path.join(args.data_dir, "val")
    )

    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(train_dir):

        print("ERROR: Training folder not found:")
        print(train_dir)
        sys.exit(1)

    if not os.path.isdir(val_dir):

        print("ERROR: Validation folder not found:")
        print(val_dir)
        sys.exit(1)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("=" * 60)
    print("BRAIN MRI TUMOR CLASSIFIER")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Architecture : {args.arch}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("Loading datasets...")

    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    train_classes = train_dataset.classes
    val_classes = val_dataset.classes

    if train_classes != val_classes:

        print("ERROR: Training and validation classes differ.")
        print("Training classes:", train_classes)
        print("Validation classes:", val_classes)
        sys.exit(1)

    class_names = train_classes

    print()
    print("Classes:", class_names)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print()

    # -----------------------------------------------------
    # DataLoader
    #
    # num_workers=0 is intentional for Windows stability.
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        "Training batches per epoch:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print()

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("Loading model...")

    model = get_model(
        architecture=args.arch,
        num_classes=len(class_names),
        pretrained=True
    )

    model = model.to(device)

    print("Model loaded successfully.")
    print()

    # -----------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    log_file = os.path.join(
        os.path.dirname(__file__),
        "training_log.csv"
    )

    best_val_accuracy = 0.0

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    with open(
        log_file,
        "w",
        newline=""
    ) as logfile:

        writer = csv.writer(logfile)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "seconds"
        ])

        for epoch in range(
            1,
            args.epochs + 1
        ):

            print()
            print("=" * 60)
            print(
                f"STARTING EPOCH {epoch}/{args.epochs}"
            )
            print("=" * 60)

            start_time = time.time()

            # -------------------------------------------------
            # Training phase
            # -------------------------------------------------

            model.train()

            total_train_loss = 0.0
            train_correct = 0
            train_total = 0

            total_batches = len(train_loader)

            for batch_number, (images, labels) in enumerate(
                train_loader,
                start=1
            ):

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                batch_size = images.size(0)

                total_train_loss += (
                    loss.item() * batch_size
                )

                predictions = outputs.argmax(
                    dim=1
                )

                train_correct += (
                    predictions == labels
                ).sum().item()

                train_total += batch_size

                # Show progress every 10 batches
                if (
                    batch_number % 10 == 0
                    or batch_number == total_batches
                ):

                    current_loss = (
                        total_train_loss
                        / train_total
                    )

                    current_accuracy = (
                        train_correct
                        / train_total
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    print(
                        f"Batch "
                        f"{batch_number}/{total_batches} | "
                        f"Loss: {current_loss:.4f} | "
                        f"Accuracy: {current_accuracy:.4f} | "
                        f"Time: {elapsed:.1f}s",
                        flush=True
                    )

            train_loss = (
                total_train_loss
                / train_total
            )

            train_accuracy = (
                train_correct
                / train_total
            )

            # -------------------------------------------------
            # Validation phase
            # -------------------------------------------------

            print()
            print("Starting validation...")

            model.eval()

            total_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for batch_number, (
                    images,
                    labels
                ) in enumerate(
                    val_loader,
                    start=1
                ):

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    batch_size = images.size(0)

                    total_val_loss += (
                        loss.item()
                        * batch_size
                    )

                    predictions = outputs.argmax(
                        dim=1
                    )

                    val_correct += (
                        predictions == labels
                    ).sum().item()

                    val_total += batch_size

                    if (
                        batch_number % 10 == 0
                        or batch_number == len(val_loader)
                    ):

                        print(
                            f"Validation batch "
                            f"{batch_number}/"
                            f"{len(val_loader)}",
                            flush=True
                        )

            val_loss = (
                total_val_loss
                / val_total
            )

            val_accuracy = (
                val_correct
                / val_total
            )

            scheduler.step(
                val_accuracy
            )

            elapsed = (
                time.time()
                - start_time
            )

            # -------------------------------------------------
            # Epoch result
            # -------------------------------------------------

            print()
            print("=" * 60)
            print(
                f"EPOCH {epoch}/{args.epochs} COMPLETE"
            )
            print("=" * 60)
            print(
                f"Train Loss     : {train_loss:.4f}"
            )
            print(
                f"Train Accuracy : {train_accuracy:.4f}"
            )
            print(
                f"Val Loss       : {val_loss:.4f}"
            )
            print(
                f"Val Accuracy   : {val_accuracy:.4f}"
            )
            print(
                f"Time           : {elapsed:.1f} seconds"
            )
            print("=" * 60)

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                round(elapsed, 1)
            ])

            logfile.flush()

            # -------------------------------------------------
            # Save best checkpoint
            # -------------------------------------------------

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                checkpoint_path = os.path.join(
                    output_dir,
                    f"classifier_{args.arch}.pt"
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "architecture":
                            args.arch,

                        "class_names":
                            class_names,

                        "model_version":
                            "v1",

                        "val_accuracy":
                            val_accuracy
                    },
                    checkpoint_path
                )

                print()
                print("BEST MODEL SAVED")
                print(
                    "File:",
                    checkpoint_path
                )
                print()

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print()
    print("Checkpoint location:")

    print(
        os.path.abspath(
            os.path.join(
                output_dir,
                f"classifier_{args.arch}.pt"
            )
        )
    )

    print()


if __name__ == "__main__":
    main()
