import os
from app.ml.inference import ClassifierService


IMAGE_PATH = os.path.abspath(
    os.path.join(
        "..",
        "..",
        "clean_combined_dataset",
        "Testing",
        "glioma",
        "Te-gl_1.jpg"
    )
)

print("========================================")
print("NeuroScan AI - Backend Inference Test")
print("========================================")

print("Image:", IMAGE_PATH)
print("Image exists:", os.path.isfile(IMAGE_PATH))

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )

print("\nRunning complete inference...\n")

result = ClassifierService.predict_with_gradcam(IMAGE_PATH)

print("========================================")
print("INFERENCE RESULT")
print("========================================")

print("Predicted class:", result["predicted_class"])
print("Confidence:", result["confidence"])

print("\nClass probabilities:")
for class_name, probability in result["class_probabilities"].items():
    print(f"  {class_name}: {probability}")

print("\nTumor detected:", result["tumor_detected"])
print("Is brain MRI:", result["is_brain_mri"])
print("Brain MRI probability:", result["brain_mri_probability"])

print("\nGrad-CAM path:", result["gradcam_path"])

print("Model name:", result["model_name"])
print("Model version:", result["model_version"])

print("\n========================================")
print("TEST COMPLETED")
print("========================================")