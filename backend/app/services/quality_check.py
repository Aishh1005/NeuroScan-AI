"""
Real, computed MRI image quality checks. These are simple, explainable
heuristics -- NOT a clinical image-quality certification. They exist to
reject obviously unusable uploads (corrupted, blank, tiny, extremely
blurry, or badly exposed images) before they reach the model.
"""
import numpy as np
import cv2
from PIL import Image

MIN_DIMENSION = 64
BLUR_THRESHOLD = 15.0          # Laplacian variance below this = too blurry
BLANK_STD_THRESHOLD = 3.0      # pixel std-dev below this = near-blank image
DARK_MEAN_THRESHOLD = 8.0      # mean brightness below this = essentially black
BRIGHT_MEAN_THRESHOLD = 247.0  # mean brightness above this = essentially white/blown-out


def run_quality_check(pil_image: Image.Image) -> dict:
    reasons = []

    width, height = pil_image.size
    fmt = (pil_image.format or "UNKNOWN").upper()

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        reasons.append(f"Image resolution is too small ({width}x{height}).")

    gray = np.array(pil_image.convert("L"), dtype=np.uint8)

    # Sharpness via variance of Laplacian (standard, real, computed metric)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_score = float(laplacian.var())
    if sharpness_score < BLUR_THRESHOLD:
        reasons.append("Image appears too blurry for reliable analysis.")

    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    is_blank_or_near_empty = brightness_std < BLANK_STD_THRESHOLD
    if is_blank_or_near_empty:
        reasons.append("Image appears blank or has almost no visual content.")

    if brightness_mean < DARK_MEAN_THRESHOLD:
        reasons.append("Image is too dark / underexposed.")
    if brightness_mean > BRIGHT_MEAN_THRESHOLD:
        reasons.append("Image is overexposed / washed out.")

    passed = len(reasons) == 0

    return {
        "width": width,
        "height": height,
        "format": fmt,
        "sharpness_score": round(sharpness_score, 2),
        "brightness_mean": round(brightness_mean, 2),
        "is_blank_or_near_empty": is_blank_or_near_empty,
        "passed": passed,
        "reasons": reasons,
    }
