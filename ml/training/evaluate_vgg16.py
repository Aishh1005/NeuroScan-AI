import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# This file is inside:
# ml/training/evaluate_vgg16.py

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "classifier_vgg16.pt"
)

MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "vgg16_class_mapping.json"
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "clean_combined_dataset",
    "Testing"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "vgg16_evaluation"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

NUM_CLASSES = 4


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("VGG16 COMPLETE TEST SET EVALUATION")
print("=" * 70)

print("Device:", DEVICE)
print("Test dataset:", TEST_DIR)
print("Model:", MODEL_PATH)


# ============================================================
# 4. CHECK REQUIRED FILES/FOLDERS
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nVGG16 model was not found:\n{MODEL_PATH}"
    )

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"\nTesting dataset was not found:\n{TEST_DIR}"
    )


# ============================================================
# 5. LOAD CLASS MAPPING
# ============================================================

if os.path.isfile(MAPPING_PATH):

    try:
        with open(
            MAPPING_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            mapping = json.load(f)

        print("\nClass mapping file:")
        print(mapping)

    except Exception as e:

        print(
            "\nCould not read class mapping file."
        )

        print(
            "Using the class order from training."
        )

else:

    print(
        "\nClass mapping file not found."
    )

    print(
        "Using the class order from training."
    )


print("\nClass order:")

for index, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {index} -> {class_name}"
    )


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = ImageFolder(
    TEST_DIR,
    transform=test_transform
)


print(
    "Images found:",
    len(test_dataset)
)

print(
    "ImageFolder classes:",
    test_dataset.classes
)


# ============================================================
# 8. VERIFY CLASS ORDER
# ============================================================

expected_folder_classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

if test_dataset.classes != expected_folder_classes:

    raise ValueError(
        "\nClass folder order does not match expected order.\n"
        f"Found: {test_dataset.classes}\n"
        f"Expected: {expected_folder_classes}"
    )


# ============================================================
# 9. CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 10. CREATE VGG16 MODEL
# ============================================================

print("\nCreating VGG16 model...")

model = vgg16(weights=None)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    NUM_CLASSES
)


# ============================================================
# 11. LOAD TRAINED MODEL
# ============================================================

print("Loading trained VGG16 weights...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Model checkpoint loaded successfully."
    )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Model weights loaded successfully."
    )


model = model.to(DEVICE)

model.eval()


# ============================================================
# 12. DISPLAY TRAINING RESULTS
# ============================================================

if isinstance(checkpoint, dict):

    if "best_validation_accuracy" in checkpoint:

        print(
            "\nBest validation accuracy from training: "
            f"{checkpoint['best_validation_accuracy'] * 100:.2f}%"
        )

    if "test_accuracy" in checkpoint:

        print(
            "Test accuracy recorded during training: "
            f"{checkpoint['test_accuracy'] * 100:.2f}%"
        )


# ============================================================
# 13. RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS ON ALL TEST IMAGES")
print("=" * 70)

all_true_labels = []
all_predictions = []

total_images = len(test_dataset)

processed = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_true_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        processed += images.size(0)

        print(
            f"\rProcessed: {processed}/{total_images}",
            end=""
        )


print("\n\nPrediction process completed.")


# ============================================================
# 14. OVERALL ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    all_true_labels,
    all_predictions
)

correct_predictions = sum(
    true == pred
    for true, pred in zip(
        all_true_labels,
        all_predictions
    )
)

wrong_predictions = (
    total_images - correct_predictions
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Total test images : {total_images}"
)

print(
    f"Correct predictions: {correct_predictions}"
)

print(
    f"Wrong predictions  : {wrong_predictions}"
)

print(
    f"Test accuracy      : "
    f"{overall_accuracy * 100:.2f}%"
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows = Actual class"
)

print(
    "Columns = Predicted class\n"
)

print(
    "                 Predicted"
)

print(
    "             Glioma  Meningioma  NoTumor  Pituitary"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:12s}"
        f"{cm[i, 0]:8d}"
        f"{cm[i, 1]:12d}"
        f"{cm[i, 2]:10d}"
        f"{cm[i, 3]:11d}"
    )


# ============================================================
# 16. CLASS-WISE ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE ACCURACY")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    total_actual = cm[i].sum()

    correct_actual = cm[i, i]

    if total_actual > 0:

        class_accuracy = (
            correct_actual /
            total_actual
        ) * 100

    else:

        class_accuracy = 0.0

    print(
        f"{class_name.capitalize():12s}: "
        f"{correct_actual}/{total_actual} "
        f"correct "
        f"({class_accuracy:.2f}%)"
    )


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_true_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 18. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VGG16 Brain Tumor Classification Evaluation\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Total test images: {total_images}\n"
    )

    f.write(
        f"Correct predictions: {correct_predictions}\n"
    )

    f.write(
        f"Wrong predictions: {wrong_predictions}\n"
    )

    f.write(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(report)


# ============================================================
# 19. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "VGG16 Brain Tumor Classification - Confusion Matrix"
)

plt.colorbar()

tick_marks = range(NUM_CLASSES)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


# Put numbers inside the matrix

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()


confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. SAVE RAW CONFUSION MATRIX
# ============================================================

confusion_text_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.txt"
)

with open(
    confusion_text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Rows = Actual class\n"
    )

    f.write(
        "Columns = Predicted class\n\n"
    )

    f.write(
        "Classes: "
        + ", ".join(CLASS_NAMES)
        + "\n\n"
    )

    for row in cm:

        f.write(
            " ".join(
                str(value)
                for value in row
            )
            + "\n"
        )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Overall test accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Correct: {correct_predictions}"
)

print(
    f"Wrong  : {wrong_predictions}"
)

print("\nFiles saved:")

print(
    f"Classification report:\n"
    f"{report_path}"
)

print(
    f"\nConfusion matrix image:\n"
    f"{confusion_matrix_path}"
)

print(
    f"\nConfusion matrix text:\n"
    f"{confusion_text_path}"
)

print("\nDone.")