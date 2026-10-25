import os
import json
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "clean_combined_dataset"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "Training"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "Testing"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 4

EPOCHS = 5

LEARNING_RATE = 0.0001

VALIDATION_SPLIT = 0.20

RANDOM_SEED = 42

NUM_CLASSES = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 70)
print("NEUROSCAN AI - VGG16 BRAIN TUMOR CLASSIFIER")
print("=" * 70)

print("Device       :", device)
print("Image size   :", IMAGE_SIZE)
print("Batch size   :", BATCH_SIZE)
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)

print("=" * 70)
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(TRAIN_DIR):

    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.isdir(TEST_DIR):

    raise FileNotFoundError(
        f"Testing folder not found:\n{TEST_DIR}"
    )


# ============================================================
# CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

print("Expected class order:")

for index, name in enumerate(CLASS_NAMES):

    print(
        f"  {index} -> {name}"
    )

print()


# ============================================================
# TRANSFORMS
# ============================================================

# ------------------------------------------------------------
# Training transform
#
# Training images receive augmentation so that the model
# sees slightly different versions of training images.
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ------------------------------------------------------------
# Validation / testing transform
#
# NO random augmentation here.
# ------------------------------------------------------------

eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD TRAINING DATASET FOR SPLITTING
# ============================================================

print("Loading training dataset...")

# Dataset used only to determine image paths and labels.
base_dataset = datasets.ImageFolder(
    TRAIN_DIR
)


print(
    "Folder classes detected:",
    base_dataset.classes
)


if set(base_dataset.classes) != set(CLASS_NAMES):

    raise ValueError(
        "\nDataset classes do not match expected classes.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {base_dataset.classes}\n"
    )


# ============================================================
# CREATE CLASS MAPPING
# ============================================================

folder_classes = base_dataset.classes

folder_to_target = {
    folder_index: CLASS_NAMES.index(class_name)
    for folder_index, class_name
    in enumerate(folder_classes)
}


# ============================================================
# CREATE TARGET LABELS
# ============================================================

all_targets = [
    folder_to_target[label]
    for _, label in base_dataset.samples
]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

total_train_images = len(
    base_dataset
)

validation_size = int(
    total_train_images
    * VALIDATION_SPLIT
)

training_size = (
    total_train_images
    - validation_size
)


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)


all_indices = torch.randperm(
    total_train_images,
    generator=generator
).tolist()


train_indices = all_indices[
    :training_size
]

val_indices = all_indices[
    training_size:
]


print()
print(
    "Total training-folder images:",
    total_train_images
)

print(
    "Training images:",
    len(train_indices)
)

print(
    "Validation images:",
    len(val_indices)
)

print()


# ============================================================
# CREATE TWO DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

# Training dataset
train_dataset_full = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

# Validation dataset
val_dataset_full = datasets.ImageFolder(
    TRAIN_DIR,
    transform=eval_transform
)


# ============================================================
# REMAPPED DATASET
# ============================================================

class RemappedSubset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        indices,
        mapping
    ):

        self.dataset = dataset

        self.indices = indices

        self.mapping = mapping


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, index):

        real_index = self.indices[index]

        image, label = self.dataset[
            real_index
        ]

        new_label = self.mapping[label]

        return image, new_label


# ============================================================
# FINAL TRAIN / VALIDATION DATASETS
# ============================================================

train_dataset = RemappedSubset(
    train_dataset_full,
    train_indices,
    folder_to_target
)

val_dataset = RemappedSubset(
    val_dataset_full,
    val_indices,
    folder_to_target
)


# ============================================================
# VERIFY TRAINING CLASS COUNTS
# ============================================================

train_class_counts = [
    0
    for _ in range(NUM_CLASSES)
]


for index in train_indices:

    original_label = base_dataset.targets[index]

    new_label = folder_to_target[
        original_label
    ]

    train_class_counts[
        new_label
    ] += 1


print("Training class distribution:")

for class_name, count in zip(
    CLASS_NAMES,
    train_class_counts
):

    print(
        f"  {class_name}: {count}"
    )

print()


# ============================================================
# LOAD TEST DATASET
#
# IMPORTANT:
# Test data is never used during training or validation.
# ============================================================

print("Loading test dataset...")

test_dataset_original = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)


print(
    "Test folder classes:",
    test_dataset_original.classes
)


if set(test_dataset_original.classes) != set(CLASS_NAMES):

    raise ValueError(
        "\nTest dataset classes do not match expected classes.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {test_dataset_original.classes}\n"
    )


test_folder_to_target = {
    folder_index: CLASS_NAMES.index(class_name)
    for folder_index, class_name
    in enumerate(test_dataset_original.classes)
}


test_dataset = RemappedSubset(
    test_dataset_original,
    list(
        range(
            len(test_dataset_original)
        )
    ),
    test_folder_to_target
)


print(
    "Test images:",
    len(test_dataset)
)

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE VGG16
# ============================================================

print(
    "Loading VGG16 ImageNet pretrained model..."
)

weights = VGG16_Weights.IMAGENET1K_V1

model = models.vgg16(
    weights=weights
)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

in_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(
    in_features,
    NUM_CLASSES
)


model = model.to(device)


print(
    "VGG16 loaded successfully."
)

print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_images_for_weights = sum(
    train_class_counts
)


class_weights = [

    total_images_for_weights
    / (
        NUM_CLASSES * count
    )

    for count in train_class_counts
]


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("Class weights:")

for class_name, weight in zip(
    CLASS_NAMES,
    class_weights
):

    print(
        f"  {class_name}: "
        f"{weight.item():.4f}"
    )

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=1,
    factor=0.5
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for batch_number, (
        images,
        labels
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        batch_size = images.size(0)


        running_loss += (
            loss.item()
            * batch_size
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += batch_size


        if (
            batch_number % 100 == 0
            or batch_number == len(train_loader)
        ):

            current_loss = (
                running_loss
                / total
            )

            current_accuracy = (
                correct
                / total
            )


            print(
                f"  Batch "
                f"{batch_number}/"
                f"{len(train_loader)} | "
                f"Loss: "
                f"{current_loss:.4f} | "
                f"Accuracy: "
                f"{current_accuracy:.4f}",
                flush=True
            )


    epoch_loss = (
        running_loss
        / total
    )


    epoch_accuracy = (
        correct
        / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION / TEST FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            batch_size = images.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size


    loss = (
        running_loss
        / total
    )


    accuracy = (
        correct
        / total
    )


    return (
        loss,
        accuracy
    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

best_model_state = None


print()

print("=" * 70)

print(
    "STARTING VGG16 TRAINING"
)

print("=" * 70)

print()


for epoch in range(
    1,
    EPOCHS + 1
):

    print()

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        "-" * 70
    )


    start_time = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print()

    print(
        "Running validation..."
    )


    val_loss, val_accuracy = (
        evaluate(
            val_loader
        )
    )


    scheduler.step(
        val_accuracy
    )


    elapsed = (
        time.time()
        - start_time
    )


    print()

    print(
        "=" * 70
    )


    print(
        f"Epoch {epoch}/{EPOCHS}"
    )


    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )


    print(
        f"Train Accuracy : "
        f"{train_accuracy:.4f}"
    )


    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )


    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.4f}"
    )


    print(
        f"Time           : "
        f"{elapsed:.1f} seconds"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL IN MEMORY
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


        print()

        print(
            "New best validation model found."
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# FINAL TEST EVALUATION
#
# Test data is used ONLY here.
# ============================================================

print()

print("=" * 70)

print(
    "FINAL TEST EVALUATION"
)

print("=" * 70)


test_loss, test_accuracy = (
    evaluate(
        test_loader
    )
)


print()

print(
    f"Test Loss     : "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy : "
    f"{test_accuracy:.4f}"
)


print()


# ============================================================
# SAVE VGG16 MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "classifier_vgg16.pt"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "architecture":
            "vgg16",

        "class_names":
            CLASS_NAMES,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE,

        "model_version":
            "v1",

        "best_validation_accuracy":
            best_val_accuracy,

        "test_accuracy":
            test_accuracy
    },
    model_path
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

mapping_path = os.path.join(
    MODEL_DIR,
    "vgg16_class_mapping.json"
)


with open(
    mapping_path,
    "w"
) as file:

    json.dump(
        {
            "0": "glioma",
            "1": "meningioma",
            "2": "notumor",
            "3": "pituitary"
        },
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "=" * 70
)

print(
    "VGG16 TRAINING COMPLETE"
)

print(
    "=" * 70
)

print()

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "Final test accuracy:",
    f"{test_accuracy:.4f}"
)

print()

print(
    "Model saved at:"
)

print(
    model_path
)

print()

print(
    "Class mapping saved at:"
)

print(
    mapping_path
)

print()

print(
    "=" * 70
)