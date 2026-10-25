import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# test_vgg16.py is inside:
# ml/training/

# Go from:
# ml/training
#      ↑
# ml
#      ↑
# project root

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


# ============================================================
# 2. CHOOSE ONE MRI IMAGE
# ============================================================

# For the first test, this automatically selects
# the first image from the glioma testing folder.

TEST_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "clean_combined_dataset",
    "Testing",
    "glioma"
)


# ============================================================
# 3. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("VGG16 SINGLE MRI TEST")
print("=" * 60)

print("Device:", DEVICE)
print("Model :", MODEL_PATH)


# ============================================================
# 5. CHECK MODEL FILE
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nVGG16 model was not found:\n{MODEL_PATH}\n"
    )


# ============================================================
# 6. LOAD VGG16
# ============================================================

print("\nLoading VGG16 model...")

model = vgg16(weights=None)

# Change final layer to 4 classes
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    4
)


# ============================================================
# 7. LOAD TRAINED WEIGHTS
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Our training script saved a checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model checkpoint loaded successfully.")

    if "best_validation_accuracy" in checkpoint:
        print(
            "Best validation accuracy: "
            f"{checkpoint['best_validation_accuracy'] * 100:.2f}%"
        )

    if "test_accuracy" in checkpoint:
        print(
            "Test accuracy from training: "
            f"{checkpoint['test_accuracy'] * 100:.2f}%"
        )

else:
    # Backup case if the file contains only state_dict
    model.load_state_dict(checkpoint)
    print("Model weights loaded successfully.")


model = model.to(DEVICE)
model.eval()


# ============================================================
# 8. LOAD CLASS MAPPING IF AVAILABLE
# ============================================================

if os.path.isfile(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        print("\nClass mapping file found:")
        print(mapping)

    except Exception:
        print("\nClass mapping file could not be read.")
        print("Using the class order from training script.")

else:
    print("\nClass mapping file not found.")
    print("Using the class order from training script.")


print("\nClass order used by this test:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} -> {name}")


# ============================================================
# 9. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 10. FIND ONE MRI IMAGE
# ============================================================

if not os.path.isdir(TEST_IMAGE_DIR):
    raise FileNotFoundError(
        f"\nTesting folder was not found:\n{TEST_IMAGE_DIR}\n"
    )


valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

image_files = [
    f
    for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
]

if len(image_files) == 0:
    raise FileNotFoundError(
        f"\nNo image files found in:\n{TEST_IMAGE_DIR}\n"
    )


image_files.sort()

IMAGE_PATH = os.path.join(
    TEST_IMAGE_DIR,
    image_files[0]
)

print("\nMRI image selected:")
print(IMAGE_PATH)


# ============================================================
# 11. LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("\nOriginal image size:", image.size)
print("Image mode:", image.mode)


# ============================================================
# 12. PREPROCESS IMAGE
# ============================================================

input_tensor = transform(image)

# Add batch dimension
input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)

print("Input tensor shape:", input_tensor.shape)


# ============================================================
# 13. RUN VGG16 PREDICTION
# ============================================================

print("\nRunning VGG16 prediction...")

with torch.no_grad():
    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = probabilities[0][i].item() * 100

    print(
        f"{class_name.capitalize():12s}: "
        f"{probability:.2f}%"
    )


predicted_class = CLASS_NAMES[predicted_index]

predicted_probability = (
    probabilities[0][predicted_index].item() * 100
)


print("\n" + "-" * 60)

print(
    f"Prediction : {predicted_class.upper()}"
)

print(
    f"Confidence : {predicted_probability:.2f}%"
)

print("-" * 60)

print("\nTest completed successfully.")