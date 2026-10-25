import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


# ============================================================
# BRISC DATASET PATH
# ============================================================

# CHANGE THIS PATH if your downloaded brisc2025 folder
# is located somewhere else.

BRISC_ROOT = os.path.join(
    PROJECT_ROOT,
    "brisc2025",
)

TRAIN_IMAGES_DIR = os.path.join(
    BRISC_ROOT,
    "segmentation_task",
    "train",
    "images",
)

TRAIN_MASKS_DIR = os.path.join(
    BRISC_ROOT,
    "segmentation_task",
    "train",
    "masks",
)

TEST_IMAGES_DIR = os.path.join(
    BRISC_ROOT,
    "segmentation_task",
    "test",
    "images",
)

TEST_MASKS_DIR = os.path.join(
    BRISC_ROOT,
    "segmentation_task",
    "test",
    "masks",
)


# ============================================================
# OUTPUT
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_tumor_segmenter.keras",
)

SAMPLE_DIR = os.path.join(
    MODEL_DIR,
    "segmentation_samples",
)

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True,
)


# ============================================================
# CHECK DATASET
# ============================================================

print("\n========================================")
print("CHECKING BRISC SEGMENTATION DATASET")
print("========================================")

required_directories = [
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
]

for directory in required_directories:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            "\nDataset folder not found:\n"
            f"{directory}\n\n"
            "Please check BRISC_ROOT in this file."
        )

print("Dataset folders found.")


# ============================================================
# GET PAIRED FILES
# ============================================================

def get_pairs(
    images_dir,
    masks_dir,
):

    image_files = [
        file
        for file in os.listdir(images_dir)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files.sort()

    pairs = []

    for image_file in image_files:

        image_stem = os.path.splitext(
            image_file
        )[0]

        mask_file = (
            image_stem
            + ".png"
        )

        mask_path = os.path.join(
            masks_dir,
            mask_file,
        )

        image_path = os.path.join(
            images_dir,
            image_file,
        )

        if os.path.exists(mask_path):

            pairs.append(
                (
                    image_path,
                    mask_path,
                )
            )

        else:

            print(
                "WARNING: mask not found for:",
                image_file,
            )

    return pairs


# ============================================================
# LOAD PAIRS
# ============================================================

train_pairs = get_pairs(
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
)

test_pairs = get_pairs(
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
)

print(
    "Training image-mask pairs:",
    len(train_pairs),
)

print(
    "Testing image-mask pairs:",
    len(test_pairs),
)

if len(train_pairs) == 0:

    raise RuntimeError(
        "No training image/mask pairs found."
    )

if len(test_pairs) == 0:

    raise RuntimeError(
        "No testing image/mask pairs found."
    )


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    train_pairs
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

validation_count = max(
    1,
    int(
        len(train_pairs)
        * 0.10
    ),
)

validation_pairs = train_pairs[
    :validation_count
]

training_pairs = train_pairs[
    validation_count:
]

print(
    "Training pairs:",
    len(training_pairs),
)

print(
    "Validation pairs:",
    len(validation_pairs),
)

print(
    "Testing pairs:",
    len(test_pairs),
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(
    image_path,
    mask_path,
):

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3,
    )

    image = tf.image.resize(
        image,
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = (
        image
        / 255.0
    )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    mask = tf.io.read_file(
        mask_path
    )

    mask = tf.image.decode_png(
        mask,
        channels=1,
    )

    mask = tf.image.resize(
        mask,
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        method="nearest",
    )

    mask = tf.cast(
        mask,
        tf.float32,
    )

    # --------------------------------------------------------
    # CONVERT MASK TO 0 / 1
    # --------------------------------------------------------

    mask = tf.where(
        mask > 0,
        1.0,
        0.0,
    )

    return (
        image,
        mask,
    )


# ============================================================
# DATASET CREATION
# ============================================================

def make_dataset(
    pairs,
    shuffle=False,
):

    image_paths = [
        pair[0]
        for pair in pairs
    ]

    mask_paths = [
        pair[1]
        for pair in pairs
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            mask_paths,
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(pairs),
                500,
            ),
            seed=SEED,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=1,
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        1
    )

    return dataset


train_dataset = make_dataset(
    training_pairs,
    shuffle=True,
)

validation_dataset = make_dataset(
    validation_pairs,
    shuffle=False,
)

test_dataset = make_dataset(
    test_pairs,
    shuffle=False,
)


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(
    y_true,
    y_pred,
):

    y_true = tf.cast(
        y_true,
        tf.float32,
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32,
    )

    y_pred = tf.cast(
        y_pred >= 0.5,
        tf.float32,
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
    )

    denominator = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
    )

    dice = (
        2.0 * intersection
        + 1.0
    ) / (
        denominator
        + 1.0
    )

    return dice


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred,
):

    smooth = 1.0

    y_true = tf.cast(
        y_true,
        tf.float32,
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32,
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
    )

    denominator = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return 1.0 - dice


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    y_true,
    y_pred,
):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    bce = tf.reduce_mean(
        bce
    )

    return (
        bce
        +
        dice_loss(
            y_true,
            y_pred,
        )
    )


# ============================================================
# U-NET BLOCK
# ============================================================

def conv_block(
    x,
    filters,
):

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
    )(x)

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
    )(x)

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    return x


# ============================================================
# U-NET MODEL
# ============================================================

def build_unet():

    inputs = tf.keras.layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )
    )

    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    c1 = conv_block(
        inputs,
        16,
    )

    p1 = tf.keras.layers.MaxPooling2D(
        2
    )(c1)

    c2 = conv_block(
        p1,
        32,
    )

    p2 = tf.keras.layers.MaxPooling2D(
        2
    )(c2)

    c3 = conv_block(
        p2,
        64,
    )

    p3 = tf.keras.layers.MaxPooling2D(
        2
    )(c3)

    # --------------------------------------------------------
    # BOTTLENECK
    # --------------------------------------------------------

    c4 = conv_block(
        p3,
        128,
    )

    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    u5 = tf.keras.layers.UpSampling2D(
        2
    )(c4)

    u5 = tf.keras.layers.Concatenate()(
        [
            u5,
            c3,
        ]
    )

    c5 = conv_block(
        u5,
        64,
    )

    u6 = tf.keras.layers.UpSampling2D(
        2
    )(c5)

    u6 = tf.keras.layers.Concatenate()(
        [
            u6,
            c2,
        ]
    )

    c6 = conv_block(
        u6,
        32,
    )

    u7 = tf.keras.layers.UpSampling2D(
        2
    )(c6)

    u7 = tf.keras.layers.Concatenate()(
        [
            u7,
            c1,
        ]
    )

    c7 = conv_block(
        u7,
        16,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
    )(c7)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainTumorUNet",
    )


# ============================================================
# BUILD MODEL
# ============================================================

print(
    "\n========================================"
)

print(
    "BUILDING U-NET"
)

print(
    "========================================"
)

model = build_unet()

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss=combined_loss,
    metrics=[
        dice_coefficient,
    ],
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_dice_coefficient",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coefficient",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print(
    "\n========================================"
)

print(
    "STARTING SEGMENTATION TRAINING"
)

print(
    "========================================"
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best segmentation model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss":
            combined_loss,
        "dice_coefficient":
            dice_coefficient,
        "dice_loss":
            dice_loss,
    },
    compile=False,
)


# ============================================================
# TEST
# ============================================================

print(
    "\n========================================"
)

print(
    "TESTING SEGMENTATION MODEL"
)

print(
    "========================================"
)

model.compile(
    loss=combined_loss,
    metrics=[
        dice_coefficient,
    ],
)

test_results = model.evaluate(
    test_dataset,
    verbose=1,
)

print(
    "\nTest results:",
    test_results,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "\n========================================"
)

print(
    "SEGMENTATION MODEL SAVED"
)

print(
    "========================================"
)

print(
    MODEL_PATH
)


# ============================================================
# SAVE SAMPLE PREDICTIONS
# ============================================================

def save_sample_predictions(
    model,
    pairs,
    count=5,
):

    sample_pairs = pairs[:count]

    for index, (
        image_path,
        mask_path,
    ) in enumerate(
        sample_pairs
    ):

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        original_size = image.size

        resized = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        image_array = (
            np.array(
                resized
            ).astype(
                np.float32
            )
            / 255.0
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0,
        )[0, ..., 0]

        prediction_mask = (
            prediction >= 0.5
        ).astype(
            np.uint8
        ) * 255

        prediction_image = Image.fromarray(
            prediction_mask
        )

        prediction_image = (
            prediction_image.resize(
                original_size
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        original_output = os.path.join(
            SAMPLE_DIR,
            f"sample_{index + 1}_original.png",
        )

        mask_output = os.path.join(
            SAMPLE_DIR,
            f"sample_{index + 1}_prediction.png",
        )

        image.save(
            original_output
        )

        prediction_image.save(
            mask_output
        )

    print(
        "\nSample predictions saved to:"
    )

    print(
        SAMPLE_DIR
    )


save_sample_predictions(
    model,
    test_pairs,
)