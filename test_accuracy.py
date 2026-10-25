from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "dataset" / "Testing"
MODEL_PATH = BASE_DIR / "models" / "brain_mri_model.keras"

# --------------------------------------------------
# 2. Settings
# --------------------------------------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8

print("Loading model...")
model = load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Test directory: {TEST_DIR}")
print()

# --------------------------------------------------
# 3. Load testing dataset
# --------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)
print("Number of test images:", len(test_ds.file_paths))
print()

# --------------------------------------------------
# 4. Apply the same VGG16 preprocessing
# --------------------------------------------------
def preprocess_dataset(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels

test_ds = test_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# --------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------
print("Evaluating model on test dataset...")

loss, accuracy = model.evaluate(test_ds, verbose=1)

print()
print("========================================")
print("FINAL TEST RESULT")
print("========================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("========================================")

# --------------------------------------------------
# 6. Get predictions
# --------------------------------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --------------------------------------------------
# 7. Calculate correct predictions
# --------------------------------------------------
correct = np.sum(y_true == y_pred)
total = len(y_true)

print()
print(f"Correct predictions : {correct}")
print(f"Total test images   : {total}")
print(f"Accuracy calculation: ({correct} / {total}) × 100")

# --------------------------------------------------
# 8. Classification report
# --------------------------------------------------
print()
print("========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------
print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(confusion_matrix(y_true, y_pred))