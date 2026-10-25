"""
Real preprocessing pipeline: the same resize/normalize steps the model
actually consumes (see app/ml/inference.py::preprocess_transform), plus a
saved visual preview (resized, un-normalized) so the frontend can show the
user what the model "sees" before/after preprocessing.
"""
import os
import uuid
from PIL import Image


def save_preprocessed_preview(pil_image: Image.Image, output_dir: str, size=(224, 224)) -> str:
    os.makedirs(output_dir, exist_ok=True)
    preview = pil_image.convert("RGB").resize(size)
    filename = f"preprocessed_{uuid.uuid4().hex}.png"
    path = os.path.join(output_dir, filename)
    preview.save(path)
    return path
