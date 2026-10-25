"""
NeuroScan AI - Model Inference Services

Pipeline:

    Uploaded image
          |
          v
    Brain MRI Validator
    (TensorFlow / Keras)
          |
          v
    Valid Brain MRI?
       /       \
     NO         YES
     |           |
   Reject      VGG16
               (PyTorch)
                 |
                 v
          4-class prediction
                 |
                 v
           PyTorch Grad-CAM
                 |
                 v
       Tumor detected?
                 |
                 v
          U-Net segmentation
          (TensorFlow / Keras)

Classification classes:

    0 -> glioma
    1 -> meningioma
    2 -> notumor
    3 -> pituitary

Important:
    - Only brain MRI images are sent to the VGG16 classifier.
    - Non-brain images are rejected by the validator.
    - VGG16 uses 224 x 224 input.
    - The trained VGG16 model is PyTorch.
    - Validator and U-Net remain TensorFlow/Keras.
"""


# ============================================================
# IMPORTS
# ============================================================
import open_clip
import os
import uuid
import json

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

import tensorflow as tf

from PIL import Image, ImageFilter

from app.ml.model_arch import get_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)


# ============================================================
# BACKEND PATH
# ============================================================

BACKEND_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)


# ============================================================
# MODEL DIRECTORIES
# ============================================================

TF_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

PYTORCH_MODEL_DIR = os.path.join(
    BACKEND_DIR,
    "models",
)


# ============================================================
# MODEL FILE PATHS
# ============================================================

VALIDATOR_MODEL_PATH = os.path.join(
    TF_MODEL_DIR,
    "brain_mri_validator.keras",
)

SEGMENTATION_MODEL_PATH = os.path.join(
    TF_MODEL_DIR,
    "brain_tumor_segmenter.keras",
)

CLASSIFIER_MODEL_PATH = os.path.join(
    PYTORCH_MODEL_DIR,
    "classifier_vgg16.pt",
)

CLASS_MAPPING_PATH = os.path.join(
    PYTORCH_MODEL_DIR,
    "vgg16_class_mapping.json",
)


# ============================================================
# IMAGE SIZES
# ============================================================

VALIDATOR_IMAGE_SIZE = (
    224,
    224,
)

CLASSIFIER_IMAGE_SIZE = (
    224,
    224,
)

SEGMENTATION_IMAGE_SIZE = (
    128,
    128,
)


# ============================================================
# THRESHOLDS
# ============================================================

VALIDATOR_THRESHOLD = 0.90

# Additional pretrained anatomy screening
ANATOMY_MODEL_NAME = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

ANATOMY_MIN_MARGIN = 0.02

SEGMENTATION_THRESHOLD = 0.40


# ============================================================
# PYTORCH DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class ModelNotFoundError(Exception):
    """
    Raised when a required model file cannot be found.
    """

    pass


class InvalidBrainMRIError(Exception):
    """
    Raised when the uploaded image is not recognized
    as a valid brain MRI.
    """

    pass


# ============================================================
# CLASSIFIER SERVICE
# ============================================================

class ClassifierService:
    """
    Handles:

        1. Brain MRI validation
        2. VGG16 classification
        3. PyTorch Grad-CAM
    """

    _validator_model = None

    _classifier_model = None

    _class_names = None

    _checkpoint = None

    _loaded = False

    # Additional BiomedCLIP anatomy screening model
    _anatomy_model = None
    _anatomy_preprocess = None
    _anatomy_tokenizer = None
    _anatomy_text_features = None


    # ========================================================
    # LOAD ALL CLASSIFIER MODELS
    # ========================================================

    @classmethod
    def _load_models(cls):
        """
        Loads:

            - TensorFlow brain MRI validator
            - PyTorch VGG16 classifier
            - class mapping
        """

        if cls._loaded:
            return


        # ====================================================
        # CHECK BRAIN MRI VALIDATOR
        # ====================================================

        if not os.path.isfile(
            VALIDATOR_MODEL_PATH
        ):

            raise ModelNotFoundError(
                "Brain MRI validator model was not found at: "
                f"{VALIDATOR_MODEL_PATH}"
            )


        # ====================================================
        # CHECK VGG16 MODEL
        # ====================================================

        if not os.path.isfile(
            CLASSIFIER_MODEL_PATH
        ):

            raise ModelNotFoundError(
                "VGG16 classifier model was not found at: "
                f"{CLASSIFIER_MODEL_PATH}"
            )


        # ====================================================
        # LOAD BRAIN MRI VALIDATOR
        # ====================================================

        print(
            "Loading brain MRI validator..."
        )

        cls._validator_model = (
            tf.keras.models.load_model(
                VALIDATOR_MODEL_PATH,
                compile=False,
            )
        )

        # ====================================================
        # LOAD BIOMEDCLIP ANATOMY SCREENING MODEL
        # ====================================================

        print("Loading BiomedCLIP anatomy screening model...")

        try:
            (
                cls._anatomy_model,
                cls._anatomy_preprocess,
            ) = open_clip.create_model_from_pretrained(
                ANATOMY_MODEL_NAME
            )

            cls._anatomy_model = cls._anatomy_model.to(DEVICE)
            cls._anatomy_model.eval()

            cls._anatomy_tokenizer = open_clip.get_tokenizer(
                ANATOMY_MODEL_NAME
            )

            brain_prompts = [
                "An axial MRI scan of the human brain.",
                "A sagittal MRI scan of the human brain.",
                "A coronal MRI scan of the human brain.",
                "A T1 weighted MRI image of the brain.",
                "A T2 weighted MRI image of the brain.",
                "A medical brain MRI showing brain anatomy.",
            ]

            non_brain_prompts = [
                "An MRI scan of a human breast.",
                "A mammography image of a human breast.",
                "A CT scan of the lungs.",
                "An MRI scan of the spine.",
                "An MRI scan of a human knee.",
                "An MRI scan of the abdomen.",
                "A medical image of the chest.",
                "An X-ray image of a human body part.",
                "A photograph of a person.",
                "A photograph of an object or scene.",
                "A medical scan that is not a brain MRI.",
            ]

            all_prompts = brain_prompts + non_brain_prompts
            text_tokens = cls._anatomy_tokenizer(all_prompts).to(DEVICE)

            with torch.no_grad():
                text_embeddings = cls._anatomy_model.encode_text(text_tokens)
                text_embeddings = F.normalize(
                    text_embeddings.float(),
                    dim=-1,
                )

                brain_count = len(brain_prompts)
                brain_features = text_embeddings[:brain_count].mean(dim=0)
                non_brain_features = text_embeddings[brain_count:].mean(dim=0)

                brain_features = F.normalize(brain_features, dim=0)
                non_brain_features = F.normalize(non_brain_features, dim=0)

                cls._anatomy_text_features = torch.stack(
                    [brain_features, non_brain_features],
                    dim=0,
                )

            print("BiomedCLIP anatomy screening model loaded.")

        except Exception as exc:
            cls._anatomy_model = None
            cls._anatomy_preprocess = None
            cls._anatomy_tokenizer = None
            cls._anatomy_text_features = None

            raise RuntimeError(
                "Could not load BiomedCLIP anatomy screening model. "
                "Check the internet connection and open_clip_torch "
                f"installation. Details: {exc}"
            ) from exc


        # ====================================================
        # LOAD PYTORCH VGG16 CHECKPOINT
        # ====================================================

        print(
            "Loading trained PyTorch VGG16 classifier..."
        )

        cls._checkpoint = torch.load(
            CLASSIFIER_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )


        # ====================================================
        # GET CLASS NAMES FROM CHECKPOINT
        # ====================================================

        checkpoint_class_names = (
            cls._checkpoint.get(
                "class_names"
            )
        )

        if checkpoint_class_names:

            cls._class_names = list(
                checkpoint_class_names
            )


        # ====================================================
        # FALLBACK TO CLASS MAPPING JSON
        # ====================================================

        elif os.path.isfile(
            CLASS_MAPPING_PATH
        ):

            with open(
                CLASS_MAPPING_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                mapping = json.load(file)


            if (
                isinstance(
                    mapping,
                    dict,
                )
                and
                "class_names" in mapping
            ):

                cls._class_names = list(
                    mapping["class_names"]
                )


            elif isinstance(
                mapping,
                dict,
            ):

                try:

                    cls._class_names = [
                        mapping[str(index)]
                        for index in range(
                            len(mapping)
                        )
                    ]

                except Exception:

                    cls._class_names = None


            elif isinstance(
                mapping,
                list,
            ):

                cls._class_names = list(
                    mapping
                )


        # ====================================================
        # FINAL CLASS NAME FALLBACK
        # ====================================================

        if not cls._class_names:

            cls._class_names = [
                "glioma",
                "meningioma",
                "notumor",
                "pituitary",
            ]


        # ====================================================
        # NUMBER OF CLASSES
        # ====================================================

        num_classes = len(
            cls._class_names
        )


        # ====================================================
        # CREATE VGG16 ARCHITECTURE
        # ====================================================

        cls._classifier_model = get_model(
            architecture="vgg16",
            num_classes=num_classes,
            pretrained=False,
        )


        # ====================================================
        # GET TRAINED WEIGHTS
        # ====================================================

        state_dict = (
            cls._checkpoint.get(
                "model_state_dict"
            )
        )

        if state_dict is None:

            raise ModelNotFoundError(
                "The VGG16 checkpoint does not contain "
                "'model_state_dict'."
            )


        # ====================================================
        # LOAD TRAINED WEIGHTS
        # ====================================================

        cls._classifier_model.load_state_dict(
            state_dict
        )


        # ====================================================
        # MOVE MODEL TO CPU/GPU
        # ====================================================

        cls._classifier_model.to(
            DEVICE
        )


        # ====================================================
        # EVALUATION MODE
        # ====================================================

        cls._classifier_model.eval()


        # ====================================================
        # MARK MODELS AS LOADED
        # ====================================================

        cls._loaded = True


        # ====================================================
        # INFORMATION
        # ====================================================

        print(
            "Brain MRI validator loaded successfully."
        )

        print(
            "PyTorch VGG16 classifier loaded successfully."
        )

        print(
            f"Device: {DEVICE}"
        )

        print(
            f"Classes: {cls._class_names}"
        )


    # ========================================================
    # CHECK MODEL AVAILABILITY
    # ========================================================

    @classmethod
    def is_available(cls):
        """
        Returns True when both the brain MRI validator
        and VGG16 classifier exist.
        """

        return (
            os.path.isfile(
                VALIDATOR_MODEL_PATH
            )
            and
            os.path.isfile(
                CLASSIFIER_MODEL_PATH
            )
        )


    # ========================================================
    # VALIDATE BRAIN MRI
    # ========================================================

    @classmethod
    def validate_brain_mri(cls, pil_image: Image.Image):
        """
        Validate an uploaded image using two checks:
        1. The existing TensorFlow/Keras brain MRI validator.
        2. BiomedCLIP anatomy screening.

        The image is accepted only if both checks pass.
        VGG16 is not run when validation fails.
        """
        cls._load_models()

        if (
            cls._anatomy_model is None
            or cls._anatomy_preprocess is None
            or cls._anatomy_text_features is None
        ):
            raise RuntimeError(
                "Anatomy screening is unavailable. "
                "Classification has been stopped for safety."
            )

        image = pil_image.convert("RGB")

        validator_image = image.resize(
            VALIDATOR_IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            validator_image,
            dtype=np.float32,
        )
        image_array = np.expand_dims(image_array, axis=0)

        # Check 1: existing Keras validator
        prediction = cls._validator_model.predict(
            image_array,
            verbose=0,
        )

        validator_probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        validator_pass = (
            np.isfinite(validator_probability)
            and VALIDATOR_THRESHOLD <= validator_probability <= 1.0
        )

        # Check 2: BiomedCLIP anatomy screening
        anatomy_image = cls._anatomy_preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = cls._anatomy_model.encode_image(anatomy_image)
            image_features = F.normalize(image_features.float(), dim=-1)

            similarities = (
                image_features @ cls._anatomy_text_features.T
            )[0]

            brain_similarity = float(similarities[0].item())
            non_brain_similarity = float(similarities[1].item())

        similarity_margin = brain_similarity - non_brain_similarity

        anatomy_pass = (
            np.isfinite(brain_similarity)
            and np.isfinite(non_brain_similarity)
            and similarity_margin >= ANATOMY_MIN_MARGIN
        )

        # Both checks must pass before the image reaches VGG16.
        is_brain_mri = bool(validator_pass and anatomy_pass)

        print(
            "Brain MRI validation:",
            {
                "validator_probability": round(validator_probability, 4),
                "validator_pass": bool(validator_pass),
                "brain_similarity": round(brain_similarity, 4),
                "non_brain_similarity": round(non_brain_similarity, 4),
                "similarity_margin": round(similarity_margin, 4),
                "anatomy_pass": bool(anatomy_pass),
                "final_result": is_brain_mri,
            },
        )

        return {
            "is_brain_mri": is_brain_mri,
            "brain_mri_probability": validator_probability,
            "anatomy_screen_pass": bool(anatomy_pass),
            "brain_similarity": brain_similarity,
            "non_brain_similarity": non_brain_similarity,
            "similarity_margin": similarity_margin,
        }


    # ========================================================
    # PREPROCESS IMAGE FOR VGG16
    # ========================================================

    @classmethod
    def _preprocess_for_vgg16(
        cls,
        pil_image: Image.Image,
    ):
        """
        Preprocessing used by the trained VGG16 model.

        Training preprocessing:

            Resize(224, 224)
            ToTensor()
            ImageNet normalization
        """

        # ====================================================
        # RGB
        # ====================================================

        image = pil_image.convert(
            "RGB"
        )


        # ====================================================
        # RESIZE
        # ====================================================

        image = image.resize(
            CLASSIFIER_IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )


        # ====================================================
        # PIL -> NUMPY
        # ====================================================

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )


        # ====================================================
        # HWC -> CHW
        # ====================================================

        image_array = np.transpose(
            image_array,
            (
                2,
                0,
                1,
            ),
        )


        # ====================================================
        # 0-255 -> 0-1
        # ====================================================

        image_array = (
            image_array / 255.0
        )


        # ====================================================
        # NUMPY -> PYTORCH TENSOR
        # ====================================================

        tensor = torch.from_numpy(
            image_array
        )


        # ====================================================
        # IMAGENET NORMALIZATION
        # ====================================================

        mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )


        std = torch.tensor(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )


        tensor = (
            tensor - mean
        ) / std


        # ====================================================
        # ADD BATCH DIMENSION
        # ====================================================

        tensor = tensor.unsqueeze(
            0
        )


        # ====================================================
        # CPU/GPU
        # ====================================================

        tensor = tensor.to(
            DEVICE
        )


        return tensor


    # ========================================================
    # FIND LAST VGG16 CONVOLUTIONAL LAYER
    # ========================================================

    @classmethod
    def _find_last_conv_layer(
        cls,
        model,
    ):
        """
        Finds the final Conv2d layer inside VGG16.
        """

        last_conv = None

        for layer in model.features:

            if isinstance(
                layer,
                torch.nn.Conv2d,
            ):

                last_conv = layer


        if last_conv is None:

            raise RuntimeError(
                "Could not find a Conv2d layer "
                "inside VGG16."
            )


        return last_conv


    # ========================================================
    # PYTORCH GRAD-CAM
    # ========================================================

    @classmethod
    def _generate_gradcam(
        cls,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        predicted_class_index: int,
        output_dir: str,
    ):
        """
        Generate a Grad-CAM-style explanation image for the
        trained PyTorch VGG16 classifier.

        Important:
            - The trained classifier weights are NOT changed.
            - The input image is NOT changed.
            - Only explanation/visualization code runs here.
            - If standard positive Grad-CAM is empty, safe
              visualization fallbacks are used so the API can
              still return an explanation image.
        """

        model = cls._classifier_model

        if model is None:
            raise RuntimeError(
                "VGG16 classifier model is not loaded."
            )

        # VGG16 normally uses inplace ReLU. Disable inplace
        # operations because explanation hooks need the tensors
        # to remain usable during autograd.
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        target_layer = cls._find_last_conv_layer(model)

        captured_activation = []

        def forward_hook(module, module_input, module_output):
            # Keep the real tensor connected to the computation graph.
            # Do NOT detach it here; autograd needs this connection.
            captured_activation.append(module_output)

            if module_output.requires_grad:
                module_output.retain_grad()

        forward_handle = target_layer.register_forward_hook(
            forward_hook
        )

        try:
            model.zero_grad(set_to_none=True)
            input_tensor = input_tensor.to(DEVICE)

            # Gradients must be enabled for Grad-CAM even if the caller
            # happens to be inside an inference/no_grad context.
            with torch.enable_grad():
                output = model(input_tensor)

                if not captured_activation:
                    raise RuntimeError(
                        "Grad-CAM activation was not captured."
                    )

                activation_tensor = captured_activation[0]

                class_score = output[
                    0,
                    predicted_class_index,
                ]

                # Ask autograd directly for the gradient of the target
                # class score with respect to the final convolution output.
                gradient = torch.autograd.grad(
                    outputs=class_score,
                    inputs=activation_tensor,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                if gradient is None:
                    # Some PyTorch versions/graph paths may not return the
                    # gradient through autograd.grad. Try the retained grad.
                    gradient = activation_tensor.grad

                if gradient is None:
                    raise RuntimeError(
                        "Grad-CAM gradient was not captured."
                    )

                activation = (
                    activation_tensor
                    .detach()
                    .clone()
                )

                gradient = (
                    gradient
                    .detach()
                    .clone()
                )

                # ------------------------------------------------------
                # Standard Grad-CAM
                # ------------------------------------------------------
                weights = gradient.mean(
                    dim=(2, 3),
                    keepdim=True,
                )

                cam_tensor = (
                    weights * activation
                ).sum(
                    dim=1,
                    keepdim=True,
                )

                cam_tensor = F.relu(cam_tensor)

                cam_np = (
                    cam_tensor[0, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                cam_np = np.nan_to_num(
                    cam_np,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                raw_min = float(cam_np.min())
                raw_max = float(cam_np.max())

                print(
                    f"Grad-CAM standard raw range: {raw_min} to {raw_max}"
                )

                # ------------------------------------------------------
                # Fallback 1
                # ------------------------------------------------------
                # If ReLU removes everything, use the magnitude of
                # gradient × activation. This still uses the model's
                # class-specific gradient information.
                if raw_max <= 1e-10:
                    fallback_tensor = (
                        torch.abs(gradient)
                        * torch.abs(activation)
                    ).sum(
                        dim=1,
                        keepdim=True,
                    )

                    cam_np = (
                        fallback_tensor[0, 0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    cam_np = np.nan_to_num(
                        cam_np,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    fallback_max = float(cam_np.max())

                    print(
                        "Grad-CAM standard map was zero; "
                        f"gradient-activation fallback range: "
                        f"{float(cam_np.min())} to {fallback_max}"
                    )

                # ------------------------------------------------------
                # Fallback 2
                # ------------------------------------------------------
                # If the gradient path is numerically empty, use the
                # final convolution activation magnitude. This is a
                # visualization fallback only; classification is unchanged.
                if float(cam_np.max()) <= 1e-10:
                    activation_fallback = torch.abs(
                        activation
                    ).mean(
                        dim=1,
                        keepdim=True,
                    )

                    cam_np = (
                        activation_fallback[0, 0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    cam_np = np.nan_to_num(
                        cam_np,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    print(
                        "Grad-CAM gradient map was also zero; "
                        "using activation visualization fallback."
                    )

            # ----------------------------------------------------------
            # Normalize the CAM robustly.
            # ----------------------------------------------------------
            positive_values = cam_np[cam_np > 0]

            if positive_values.size > 0:
                low_value = float(
                    np.percentile(positive_values, 5)
                )
                high_value = float(
                    np.percentile(positive_values, 99)
                )
            else:
                low_value = float(cam_np.min())
                high_value = float(cam_np.max())

            if high_value > low_value + 1e-10:
                cam_np = (
                    cam_np - low_value
                ) / (
                    high_value - low_value
                )
            elif high_value > 1e-10:
                cam_np = cam_np / high_value
            else:
                # Do not fail the whole classification if the explanation
                # map is numerically flat. Return a valid neutral image.
                cam_np = np.zeros_like(cam_np, dtype=np.float32)

            cam_np = np.clip(cam_np, 0.0, 1.0)

        finally:
            forward_handle.remove()
            model.zero_grad(set_to_none=True)

        # ============================================================
        # ORIGINAL IMAGE
        # ============================================================
        original_image = pil_image.convert("RGB")
        original_width, original_height = original_image.size

        # ============================================================
        # RESIZE + SMOOTH HEATMAP
        # ============================================================
        cam_image = Image.fromarray(
            np.uint8(cam_np * 255),
            mode="L",
        )

        cam_image = cam_image.resize(
            (original_width, original_height),
            Image.Resampling.BICUBIC,
        )

        cam_image = cam_image.filter(
            ImageFilter.GaussianBlur(radius=1.5)
        )

        heatmap = (
            np.asarray(cam_image, dtype=np.float32) / 255.0
        )
        heatmap = np.clip(heatmap, 0.0, 1.0)

        # ============================================================
        # SOFT BRAIN-REGION MASK
        # ============================================================
        gray_array = np.asarray(
            original_image.convert("L"),
            dtype=np.float32,
        )

        background_threshold = max(
            10.0,
            float(np.percentile(gray_array, 10)),
        )

        brain_mask = (
            gray_array > background_threshold
        ).astype(np.float32)

        brain_mask_image = Image.fromarray(
            np.uint8(brain_mask * 255),
            mode="L",
        )

        brain_mask_image = brain_mask_image.filter(
            ImageFilter.GaussianBlur(radius=2.0)
        )

        brain_mask = (
            np.asarray(brain_mask_image, dtype=np.float32) / 255.0
        )

        # Keep some heatmap visible rather than hard-removing pixels.
        heatmap = heatmap * (
            0.35 + 0.65 * brain_mask
        )

        heatmap_max = float(heatmap.max())
        if heatmap_max > 1e-10:
            heatmap = heatmap / heatmap_max

        heatmap = np.clip(heatmap, 0.0, 1.0)

        print(
            "Grad-CAM final heatmap range: "
            f"{float(heatmap.min())} to {float(heatmap.max())}"
        )

        # ============================================================
        # CREATE A CLEAR JET-LIKE HEATMAP
        # ============================================================
        h = heatmap

        # Piecewise RGB map:
        # blue -> cyan -> green -> yellow -> red
        red = np.where(
            h < 0.5,
            0.0,
            np.where(h < 0.75, (h - 0.5) * 4.0, 1.0),
        )

        green = np.where(
            h < 0.25,
            h * 4.0,
            np.where(h < 0.75, 1.0, 1.0 - (h - 0.75) * 4.0),
        )

        blue = np.where(
            h < 0.25,
            1.0,
            np.where(h < 0.5, 1.0 - (h - 0.25) * 4.0, 0.0),
        )

        heatmap_rgb = np.uint8(
            np.clip(
                np.stack([red, green, blue], axis=2) * 255.0,
                0.0,
                255.0,
            )
        )

        heatmap_rgba = Image.fromarray(
            heatmap_rgb,
            mode="RGB",
        ).convert("RGBA")

        # ============================================================
        # OVERLAY TRANSPARENCY
        # ============================================================
        # Keep low-activation regions mostly transparent and make
        # strong regions clearly visible.
        alpha = (
            35.0 + heatmap * 200.0
        )
        alpha = np.uint8(np.clip(alpha, 0.0, 255.0))

        heatmap_rgba.putalpha(
            Image.fromarray(alpha, mode="L")
        )

        original_rgba = original_image.convert("RGBA")

        overlay = Image.alpha_composite(
            original_rgba,
            heatmap_rgba,
        )

        # ============================================================
        # SAVE
        # ============================================================
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        filename = f"gradcam_{uuid.uuid4().hex}.png"
        output_path = os.path.join(
            output_dir,
            filename,
        )

        overlay.convert("RGB").save(
            output_path,
            format="PNG",
        )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                "Grad-CAM image could not be saved."
            )

        print(
            "Grad-CAM generated successfully:",
            output_path,
        )

        return output_path


    # ========================================================
    # COMPLETE PREDICTION PIPELINE
    # ========================================================

    @classmethod
    def predict_with_gradcam(
        cls,
        pil_image: Image.Image,
        output_dir: str,
    ):
        """
        Complete prediction pipeline.

        1. Load models
        2. Validate brain MRI
        3. Resize to 224 x 224
        4. Run VGG16
        5. Calculate probabilities
        6. Determine class
        7. Generate Grad-CAM
        8. Return API-compatible result
        """

        # ====================================================
        # LOAD MODELS
        # ====================================================

        cls._load_models()


        # ====================================================
        # STEP 1
        # BRAIN MRI VALIDATION
        # ====================================================

        validation_result = (
            cls.validate_brain_mri(
                pil_image
            )
        )


        # ====================================================
        # REJECT NON-BRAIN IMAGE
        # ====================================================

        if not validation_result[
            "is_brain_mri"
        ]:

            raise InvalidBrainMRIError(
                "Invalid image. Please upload "
                "a brain MRI image."
            )


        # ====================================================
        # STEP 2
        # PREPROCESS IMAGE
        # ====================================================

        input_tensor = (
            cls._preprocess_for_vgg16(
                pil_image
            )
        )


        # ====================================================
        # STEP 3
        # VGG16 CLASSIFICATION
        # ====================================================

        cls._classifier_model.zero_grad(
            set_to_none=True
        )


        output = cls._classifier_model(
            input_tensor
        )


        # ====================================================
        # STEP 4
        # SOFTMAX
        # ====================================================

        probabilities = (
            torch.softmax(
                output,
                dim=1,
            )
        )


        probabilities = (
            probabilities[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )


        # ====================================================
        # STEP 5
        # FIND PREDICTED CLASS
        # ====================================================

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        predicted_class = (
            cls._class_names[
                predicted_index
            ]
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        class_probabilities = {}


        for index, class_name in enumerate(
            cls._class_names
        ):

            class_probabilities[
                class_name
            ] = float(
                probabilities[
                    index
                ]
            )


        # ====================================================
        # STEP 6
        # DETERMINE TUMOR STATUS
        # ====================================================

        normalized_class_name = (
            predicted_class
            .lower()
            .replace(
                "_",
                "",
            )
            .replace(
                " ",
                "",
            )
        )


        tumor_detected = (
            normalized_class_name
            not in {
                "notumor",
                "normal",
            }
        )


        # ====================================================
        # STEP 7
        # GENERATE GRAD-CAM
        # ====================================================

        gradcam_path = None


        try:

            gradcam_path = (
                cls._generate_gradcam(
                    pil_image=pil_image,
                    input_tensor=input_tensor,
                    predicted_class_index=predicted_index,
                    output_dir=output_dir,
                )
            )

        except Exception as exc:

            # Grad-CAM failure should not destroy
            # the classification result.

            print(
                "Grad-CAM generation failed:",
                str(exc),
            )

            gradcam_path = None


        # ====================================================
        # MODEL VERSION
        # ====================================================

        model_version = (
            cls._checkpoint.get(
                "model_version",
                "v1",
            )
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "class_probabilities":
                class_probabilities,

            "tumor_detected":
                bool(
                    tumor_detected
                ),

            "gradcam_path":
                gradcam_path,

            "model_name":
                "VGG16",

            "model_version":
                f"pytorch-{model_version}-gradcam",

            "is_brain_mri":
                True,

            "brain_mri_probability":
                validation_result[
                    "brain_mri_probability"
                ],
        }


# ============================================================
# U-NET SEGMENTATION SERVICE
# ============================================================

class SegmentationService:
    """
    TensorFlow/Keras U-Net segmentation service.

    The U-Net remains separate from the VGG16 classifier.
    """

    _model = None

    _loaded = False


    # ========================================================
    # LOAD U-NET
    # ========================================================

    @classmethod
    def _load_model(cls):

        if cls._loaded:
            return


        # ====================================================
        # CHECK MODEL
        # ====================================================

        if not os.path.isfile(
            SEGMENTATION_MODEL_PATH
        ):

            raise ModelNotFoundError(
                "Segmentation model was not found at: "
                f"{SEGMENTATION_MODEL_PATH}"
            )


        # ====================================================
        # LOAD MODEL
        # ====================================================

        print(
            "Loading U-Net segmentation model..."
        )


        cls._model = (
            tf.keras.models.load_model(
                SEGMENTATION_MODEL_PATH,
                compile=False,
            )
        )


        cls._loaded = True


        print(
            "U-Net segmentation model loaded successfully."
        )


    # ========================================================
    # CHECK U-NET AVAILABILITY
    # ========================================================

    @classmethod
    def is_available(cls):

        return os.path.isfile(
            SEGMENTATION_MODEL_PATH
        )


    # ========================================================
    # U-NET PREDICTION
    # ========================================================

    @classmethod
    def predict(
        cls,
        pil_image: Image.Image,
        output_dir: str,
    ):
        """
        Generates an estimated tumor segmentation mask.
        """

        # ====================================================
        # LOAD U-NET
        # ====================================================

        cls._load_model()


        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        original_image = (
            pil_image.convert(
                "RGB"
            )
        )


        original_width, original_height = (
            original_image.size
        )


        # ====================================================
        # RESIZE
        # ====================================================

        resized_image = (
            original_image.resize(
                SEGMENTATION_IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
        )


        # ====================================================
        # IMAGE -> NUMPY
        # ====================================================

        image_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )


        # ====================================================
        # NORMALIZE 0-255 -> 0-1
        # ====================================================

        image_array = (
            image_array / 255.0
        )


        # ====================================================
        # ADD BATCH DIMENSION
        # ====================================================

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )


        # ====================================================
        # U-NET PREDICTION
        # ====================================================

        prediction = (
            cls._model.predict(
                image_array,
                verbose=0,
            )
        )


        # ====================================================
        # CONVERT OUTPUT
        # ====================================================

        prediction = np.asarray(
            prediction
        )


        # ====================================================
        # HANDLE OUTPUT SHAPE
        # ====================================================

        if prediction.ndim == 4:

            mask = prediction[
                0,
                :,
                :,
                0,
            ]

        elif prediction.ndim == 3:

            mask = prediction[
                0
            ]

        else:

            raise RuntimeError(
                "Unexpected U-Net output shape: "
                f"{prediction.shape}"
            )


        # ====================================================
        # RESIZE MASK TO ORIGINAL IMAGE
        # ====================================================

        mask_image = Image.fromarray(
            np.uint8(
                mask * 255
            )
        )


        mask_image = mask_image.resize(
            (
                original_width,
                original_height,
            ),
            Image.Resampling.BILINEAR,
        )


        mask = (
            np.asarray(
                mask_image,
                dtype=np.float32,
            )
            / 255.0
        )


        # ====================================================
        # THRESHOLD
        # ====================================================

        binary_mask = (
            mask
            >= SEGMENTATION_THRESHOLD
        )


        # ====================================================
        # SMOOTH MASK
        # ====================================================

        binary_mask_image = (
            Image.fromarray(
                np.uint8(
                    binary_mask * 255
                )
            )
        )


        binary_mask_image = (
            binary_mask_image.filter(
                ImageFilter.MaxFilter(
                    5
                )
            )
        )


        binary_mask_image = (
            binary_mask_image.filter(
                ImageFilter.MinFilter(
                    5
                )
            )
        )


        binary_mask = (
            np.asarray(
                binary_mask_image
            )
            > 127
        )


        # ====================================================
        # COUNT MASK PIXELS
        # ====================================================

        mask_pixels = int(
            np.sum(
                binary_mask
            )
        )


        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        original_array = np.asarray(
            original_image,
            dtype=np.uint8,
        )


        overlay_array = (
            original_array.copy()
        )


        # ====================================================
        # RED CHANNEL
        # ====================================================

        overlay_array[
            binary_mask,
            0
        ] = 255


        # ====================================================
        # GREEN CHANNEL
        # ====================================================

        overlay_array[
            binary_mask,
            1
        ] = (
            overlay_array[
                binary_mask,
                1
            ]
            * 0.35
        ).astype(
            np.uint8
        )


        # ====================================================
        # BLUE CHANNEL
        # ====================================================

        overlay_array[
            binary_mask,
            2
        ] = (
            overlay_array[
                binary_mask,
                2
            ]
            * 0.35
        ).astype(
            np.uint8
        )


        # ====================================================
        # CREATE IMAGE
        # ====================================================

        overlay_image = Image.fromarray(
            overlay_array
        )


        # ====================================================
        # OUTPUT DIRECTORY
        # ====================================================

        os.makedirs(
            output_dir,
            exist_ok=True,
        )


        # ====================================================
        # OUTPUT FILE
        # ====================================================

        filename = (
            f"segmentation_{uuid.uuid4().hex}.png"
        )


        output_path = os.path.join(
            output_dir,
            filename,
        )


        # ====================================================
        # SAVE
        # ====================================================

        overlay_image.save(
            output_path
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "segmentation_path":
                output_path,

            "mask_pixels":
                mask_pixels,

            "segmentation_available":
                True,
        }