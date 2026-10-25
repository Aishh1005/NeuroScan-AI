import os
import uuid
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from app.config import settings


# ============================================================
# MODEL PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

# ------------------------------------------------------------
# Brain MRI validator
# ------------------------------------------------------------

VALIDATOR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_mri_validator.keras",
)

# ------------------------------------------------------------
# Brain tumor classifier
# ------------------------------------------------------------

CLASSIFIER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_mri_model.keras",
)

# ------------------------------------------------------------
# Class names
# ------------------------------------------------------------

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json",
)

# ------------------------------------------------------------
# Brain tumor segmentation model
# ------------------------------------------------------------

SEGMENTATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_tumor_segmenter.keras",
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

VALIDATOR_IMAGE_SIZE = (
    224,
    224,
)

CLASSIFIER_IMAGE_SIZE = (
    128,
    128,
)

SEGMENTATION_IMAGE_SIZE = (
    128,
    128,
)

VALIDATOR_THRESHOLD = 0.90

# Segmentation probability threshold.
# The segmentation model outputs values between 0 and 1.
SEGMENTATION_THRESHOLD = 0.40


# ============================================================
# MODEL ERRORS
# ============================================================

class ModelNotFoundError(Exception):
    """Raised when a required trained model is missing."""
    pass


class InvalidBrainMRIError(Exception):
    """Raised when the uploaded image is not a brain MRI."""
    pass


# ============================================================
# CLASSIFIER SERVICE
# ============================================================

class ClassifierService:

    _validator_model = None
    _classifier_model = None
    _class_names = None

    # ========================================================
    # LOAD MODELS
    # ========================================================

    @classmethod
    def _load_models(cls):

        if (
            cls._validator_model is not None
            and cls._classifier_model is not None
            and cls._class_names is not None
        ):
            return

        # ----------------------------------------------------
        # CHECK VALIDATOR
        # ----------------------------------------------------

        if not os.path.exists(
            VALIDATOR_MODEL_PATH
        ):
            raise ModelNotFoundError(
                "Brain MRI validator not found at: "
                f"{VALIDATOR_MODEL_PATH}"
            )

        # ----------------------------------------------------
        # CHECK CLASSIFIER
        # ----------------------------------------------------

        if not os.path.exists(
            CLASSIFIER_MODEL_PATH
        ):
            raise ModelNotFoundError(
                "Brain MRI classifier not found at: "
                f"{CLASSIFIER_MODEL_PATH}"
            )

        # ----------------------------------------------------
        # CHECK CLASS NAMES
        # ----------------------------------------------------

        if not os.path.exists(
            CLASS_NAMES_PATH
        ):
            raise ModelNotFoundError(
                "Class names file not found at: "
                f"{CLASS_NAMES_PATH}"
            )

        # ----------------------------------------------------
        # LOAD VALIDATOR
        # ----------------------------------------------------

        print(
            "Loading brain MRI validator..."
        )

        cls._validator_model = (
            tf.keras.models.load_model(
                VALIDATOR_MODEL_PATH,
                compile=False,
            )
        )

        # ----------------------------------------------------
        # LOAD CLASSIFIER
        # ----------------------------------------------------

        print(
            "Loading brain tumor classifier..."
        )

        cls._classifier_model = (
            tf.keras.models.load_model(
                CLASSIFIER_MODEL_PATH,
                compile=False,
            )
        )

        # ----------------------------------------------------
        # LOAD CLASS NAMES
        # ----------------------------------------------------

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            cls._class_names = json.load(
                file
            )

        print(
            "Models loaded successfully."
        )

        print(
            "Classes:",
            cls._class_names,
        )

    # ========================================================
    # MODEL AVAILABILITY
    # ========================================================

    @classmethod
    def is_available(cls) -> bool:

        return (
            os.path.exists(
                VALIDATOR_MODEL_PATH
            )
            and os.path.exists(
                CLASSIFIER_MODEL_PATH
            )
            and os.path.exists(
                CLASS_NAMES_PATH
            )
        )

    # ========================================================
    # VALIDATE BRAIN MRI
    # ========================================================

    @classmethod
    def validate_brain_mri(
        cls,
        pil_image: Image.Image,
    ):

        cls._load_models()

        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        image = pil_image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(
            VALIDATOR_IMAGE_SIZE
        )

        # ----------------------------------------------------
        # NUMPY
        # ----------------------------------------------------

        image_array = np.array(
            image
        ).astype(
            "float32"
        )

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # ----------------------------------------------------
        # VALIDATOR PREDICTION
        # ----------------------------------------------------

        brain_probability = float(
            cls._validator_model.predict(
                image_array,
                verbose=0,
            )[0][0]
        )

        print(
            "Brain MRI probability: "
            f"{brain_probability:.4f}"
        )

        is_brain_mri = (
            brain_probability
            >= VALIDATOR_THRESHOLD
        )

        return {
            "is_brain_mri": is_brain_mri,
            "brain_mri_probability": (
                brain_probability
            ),
        }

    # ========================================================
    # FIND VGG16 BASE MODEL
    # ========================================================

    @classmethod
    def _find_vgg16_base_model(cls):

        for layer in (
            cls._classifier_model.layers
        ):

            if isinstance(
                layer,
                tf.keras.Model,
            ):

                return layer

        raise RuntimeError(
            "VGG16 base model was not found "
            "inside the classifier."
        )

    # ========================================================
    # FIND LAST CONVOLUTION LAYER
    # ========================================================

    @classmethod
    def _find_last_conv_layer(
        cls,
        base_model,
    ):

        conv_layers = []

        for layer in base_model.layers:

            if isinstance(
                layer,
                tf.keras.layers.Conv2D,
            ):

                conv_layers.append(
                    layer
                )

        if not conv_layers:

            raise RuntimeError(
                "No Conv2D layer was found "
                "inside VGG16."
            )

        target_layer = conv_layers[-1]

        print(
            "Grad-CAM target layer:",
            target_layer.name,
        )

        return target_layer

    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    @classmethod
    def _generate_gradcam(
        cls,
        image_array,
        original_image,
        pred_idx,
        output_dir,
    ):

        # ----------------------------------------------------
        # 1. FIND VGG16
        # ----------------------------------------------------

        base_model = (
            cls._find_vgg16_base_model()
        )

        print(
            "VGG16 base model:",
            base_model.name,
        )

        # ----------------------------------------------------
        # 2. FIND LAST CONVOLUTION
        # ----------------------------------------------------

        target_layer = (
            cls._find_last_conv_layer(
                base_model
            )
        )

        # ----------------------------------------------------
        # 3. FIND VGG16 INDEX
        # ----------------------------------------------------

        base_model_index = None

        for index, layer in enumerate(
            cls._classifier_model.layers
        ):

            if layer is base_model:

                base_model_index = index
                break

        if base_model_index is None:

            raise RuntimeError(
                "VGG16 base model could not "
                "be located in classifier."
            )

        # ----------------------------------------------------
        # 4. CLASSIFIER HEAD
        # ----------------------------------------------------

        head_layers = (
            cls._classifier_model.layers[
                base_model_index + 1:
            ]
        )

        print(
            "Classifier head layers:",
            len(head_layers),
        )

        # ----------------------------------------------------
        # 5. FEATURE MODEL
        # ----------------------------------------------------

        feature_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                target_layer.output,
                base_model.output,
            ],
        )

        # ----------------------------------------------------
        # 6. IMAGE TENSOR
        # ----------------------------------------------------

        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32,
        )

        # ----------------------------------------------------
        # 7. GRADIENT
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, features = (
                feature_model(
                    image_tensor,
                    training=False,
                )
            )

            x = features

            for layer in head_layers:

                x = layer(
                    x,
                    training=False,
                )

            predictions = x

            class_score = predictions[
                :,
                pred_idx,
            ]

        # ----------------------------------------------------
        # 8. GET GRADIENTS
        # ----------------------------------------------------

        gradients = tape.gradient(
            class_score,
            conv_outputs,
        )

        if gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients could "
                "not be calculated."
            )

        # ----------------------------------------------------
        # 9. GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        pooled_gradients = (
            tf.reduce_mean(
                gradients,
                axis=(0, 1, 2),
            )
        )

        conv_outputs = conv_outputs[0]

        # ----------------------------------------------------
        # 10. HEATMAP
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_outputs
            * pooled_gradients,
            axis=-1,
        )

        heatmap = tf.nn.relu(
            heatmap
        )

        heatmap = heatmap.numpy()

        print(
            "Grad-CAM raw minimum:",
            float(
                np.min(
                    heatmap
                )
            ),
        )

        print(
            "Grad-CAM raw maximum:",
            float(
                np.max(
                    heatmap
                )
            ),
        )

        if np.max(
            heatmap
        ) <= 1e-10:

            raise RuntimeError(
                "Grad-CAM produced an empty heatmap."
            )

        # ----------------------------------------------------
        # 11. RESIZE HEATMAP
        # ----------------------------------------------------

        original_width, original_height = (
            original_image.size
        )

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            [
                original_height,
                original_width,
            ],
            method="bilinear",
        )

        heatmap = (
            heatmap
            .numpy()
            .squeeze()
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            None,
        )

        # ----------------------------------------------------
        # 12. BRAIN MASK
        # ----------------------------------------------------

        grayscale = np.array(
            original_image.convert(
                "L"
            )
        ).astype(
            np.float32
        )

        maximum_intensity = float(
            np.max(
                grayscale
            )
        )

        if maximum_intensity > 0:

            brain_threshold = max(
                8.0,
                maximum_intensity * 0.08,
            )

            brain_mask = (
                grayscale
                > brain_threshold
            ).astype(
                np.float32
            )

        else:

            brain_mask = np.ones_like(
                grayscale,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # SMOOTH MASK
        # ----------------------------------------------------

        brain_mask_image = Image.fromarray(
            (
                brain_mask
                * 255
            ).astype(
                np.uint8
            )
        )

        brain_mask_image = (
            brain_mask_image.filter(
                ImageFilter.MaxFilter(
                    11
                )
            )
        )

        brain_mask_image = (
            brain_mask_image.filter(
                ImageFilter.GaussianBlur(
                    2
                )
            )
        )

        brain_mask = (
            np.array(
                brain_mask_image
            ).astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # 13. REMOVE BACKGROUND ACTIVATION
        # ----------------------------------------------------

        heatmap = (
            heatmap
            * brain_mask
        )

        # ----------------------------------------------------
        # 14. KEEP STRONG ACTIVATION
        # ----------------------------------------------------

        positive_values = heatmap[
            heatmap > 0
        ]

        if positive_values.size > 0:

            cutoff = float(
                np.percentile(
                    positive_values,
                    65,
                )
            )

            heatmap = np.where(
                heatmap >= cutoff,
                heatmap,
                0,
            )

        # ----------------------------------------------------
        # 15. NORMALIZE
        # ----------------------------------------------------

        max_value = float(
            np.max(
                heatmap
            )
        )

        if max_value <= 1e-10:

            raise RuntimeError(
                "No useful Grad-CAM activation "
                "remained inside the brain region."
            )

        heatmap = (
            heatmap
            / max_value
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # 16. SMOOTH
        # ----------------------------------------------------

        heatmap_image = Image.fromarray(
            (
                heatmap
                * 255
            ).astype(
                np.uint8
            )
        )

        heatmap_image = (
            heatmap_image.filter(
                ImageFilter.GaussianBlur(
                    4
                )
            )
        )

        heatmap = (
            np.array(
                heatmap_image
            ).astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # 17. COLORS
        # ----------------------------------------------------

        red = np.clip(
            heatmap * 2.0,
            0.0,
            1.0,
        )

        green = np.clip(
            4.0
            * heatmap
            * (
                1.0
                - heatmap
            ),
            0.0,
            1.0,
        )

        blue = np.zeros_like(
            heatmap
        )

        heatmap_rgb = np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )

        heatmap_rgb = (
            heatmap_rgb
            * 255
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # 18. ORIGINAL IMAGE
        # ----------------------------------------------------

        original_array = np.array(
            original_image.convert(
                "RGB"
            )
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # 19. OVERLAY
        # ----------------------------------------------------

        alpha = (
            0.15
            + (
                0.75
                * heatmap
            )
        )

        alpha = alpha[
            ...,
            np.newaxis
        ]

        overlay = (
            original_array
            * (
                1.0
                - alpha
            )
            +
            heatmap_rgb.astype(
                np.float32
            )
            * alpha
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(
            np.uint8
        )

        gradcam_image = Image.fromarray(
            overlay
        )

        # ----------------------------------------------------
        # 20. SAVE
        # ----------------------------------------------------

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        gradcam_filename = (
            f"gradcam_"
            f"{uuid.uuid4().hex}.png"
        )

        gradcam_path = os.path.join(
            output_dir,
            gradcam_filename,
        )

        gradcam_image.save(
            gradcam_path
        )

        print(
            "Grad-CAM saved:",
            gradcam_path,
        )

        return gradcam_path

    # ========================================================
    # CLASSIFICATION + GRAD-CAM
    # ========================================================

    @classmethod
    def predict_with_gradcam(
        cls,
        pil_image: Image.Image,
        output_dir: str,
    ):

        cls._load_models()

        # ----------------------------------------------------
        # 1. BRAIN MRI VALIDATION
        # ----------------------------------------------------

        validation_result = (
            cls.validate_brain_mri(
                pil_image
            )
        )

        if not validation_result[
            "is_brain_mri"
        ]:

            raise InvalidBrainMRIError(
                "The uploaded image does not "
                "appear to be a valid brain MRI. "
                "Please upload a brain MRI image."
            )

        # ----------------------------------------------------
        # 2. PREPARE IMAGE
        # ----------------------------------------------------

        rgb_image = (
            pil_image.convert(
                "RGB"
            )
        )

        resized_image = (
            rgb_image.resize(
                CLASSIFIER_IMAGE_SIZE
            )
        )

        image_array = np.array(
            resized_image
        ).astype(
            "float32"
        )

        # ----------------------------------------------------
        # 3. VGG16 PREPROCESSING
        # ----------------------------------------------------

        image_array = (
            tf.keras.applications.vgg16
            .preprocess_input(
                image_array
            )
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # ----------------------------------------------------
        # 4. CLASSIFICATION
        # ----------------------------------------------------

        predictions = (
            cls._classifier_model.predict(
                image_array,
                verbose=0,
            )[0]
        )

        pred_idx = int(
            np.argmax(
                predictions
            )
        )

        predicted_class = (
            cls._class_names[
                pred_idx
            ]
        )

        confidence = float(
            predictions[
                pred_idx
            ]
        )

        class_probabilities = {

            cls._class_names[i]:
            float(
                predictions[i]
            )

            for i in range(
                len(
                    cls._class_names
                )
            )
        }

        tumor_detected = (
            predicted_class.lower()
            not in (
                "notumor",
                "no_tumor",
                "no tumor",
                "normal",
            )
        )

        # ----------------------------------------------------
        # 5. OUTPUT DIRECTORY
        # ----------------------------------------------------

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # 6. GRAD-CAM
        # ----------------------------------------------------

        gradcam_path = (
            cls._generate_gradcam(
                image_array=image_array,
                original_image=rgb_image,
                pred_idx=pred_idx,
                output_dir=output_dir,
            )
        )

        # ----------------------------------------------------
        # 7. RETURN
        # ----------------------------------------------------

        return {
            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "class_probabilities":
                class_probabilities,

            "tumor_detected":
                tumor_detected,

            "gradcam_path":
                gradcam_path,

            "model_name":
                "VGG16",

            "model_version":
                "keras-v4-gradcam",

            "is_brain_mri":
                True,

            "brain_mri_probability":
                validation_result[
                    "brain_mri_probability"
                ],
        }


# ============================================================
# SEGMENTATION SERVICE
# ============================================================

class SegmentationService:

    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "brain_tumor_segmenter.keras",
    )

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    IMAGE_SIZE = 128

    # --------------------------------------------------------
    # SEGMENTATION THRESHOLD
    # --------------------------------------------------------

    THRESHOLD = 0.40

    # --------------------------------------------------------
    # LOADED MODEL
    # --------------------------------------------------------

    _model = None

    # ========================================================
    # LOAD SEGMENTATION MODEL
    # ========================================================

    @classmethod
    def _load_model(cls):

        if cls._model is not None:
            return

        if not os.path.exists(
            cls.MODEL_PATH
        ):

            raise ModelNotFoundError(
                "Brain tumor segmentation model "
                "was not found at: "
                f"{cls.MODEL_PATH}"
            )

        print(
            "Loading brain tumor segmentation model..."
        )

        cls._model = (
            tf.keras.models.load_model(
                cls.MODEL_PATH,
                compile=False,
            )
        )

        print(
            "Brain tumor segmentation model loaded."
        )

    # ========================================================
    # CHECK AVAILABILITY
    # ========================================================

    @classmethod
    def is_available(
        cls,
    ) -> bool:

        return os.path.exists(
            cls.MODEL_PATH
        )

    # ========================================================
    # CREATE SEGMENTATION IMAGE
    # ========================================================

    @classmethod
    def predict_mask(
        cls,
        pil_image: Image.Image,
        output_dir: str,
    ):

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        cls._load_model()

        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        original_image = (
            pil_image.convert(
                "RGB"
            )
        )

        original_width, original_height = (
            original_image.size
        )

        original_array = np.array(
            original_image
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # RESIZE FOR MODEL
        # ----------------------------------------------------

        resized_image = (
            original_image.resize(
                (
                    cls.IMAGE_SIZE,
                    cls.IMAGE_SIZE,
                )
            )
        )

        image_array = np.array(
            resized_image
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # SAME PREPROCESSING USED DURING
        # SEGMENTATION TRAINING
        # ----------------------------------------------------

        image_array = (
            image_array
            / 255.0
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        print(
            "Running tumor segmentation..."
        )

        prediction = (
            cls._model.predict(
                image_array,
                verbose=0,
            )
        )

        # ----------------------------------------------------
        # REMOVE BATCH + CHANNEL
        # ----------------------------------------------------

        prediction = (
            prediction[
                0,
                ...,
                0
            ]
        )

        print(
            "Segmentation probability "
            f"minimum: {float(np.min(prediction)):.4f}"
        )

        print(
            "Segmentation probability "
            f"maximum: {float(np.max(prediction)):.4f}"
        )

        # ----------------------------------------------------
        # RESIZE TO ORIGINAL MRI SIZE
        # ----------------------------------------------------

        prediction_tensor = (
            tf.convert_to_tensor(
                prediction[
                    ...,
                    np.newaxis
                ],
                dtype=tf.float32,
            )
        )

        prediction_tensor = (
            tf.image.resize(
                prediction_tensor,
                [
                    original_height,
                    original_width,
                ],
                method="bilinear",
            )
        )

        prediction = (
            prediction_tensor
            .numpy()
            .squeeze()
        )

        prediction = np.clip(
            prediction,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # CREATE BINARY MASK
        # ----------------------------------------------------

        mask = (
            prediction
            >= cls.THRESHOLD
        )

        mask_pixels = int(
            np.sum(mask)
        )

        print(
            "Predicted tumor pixels:",
            mask_pixels,
        )

        # ----------------------------------------------------
        # EMPTY RESULT
        # ----------------------------------------------------

        if mask_pixels == 0:

            raise RuntimeError(
                "The segmentation model did not "
                "detect a tumor region in this image."
            )

        # ----------------------------------------------------
        # SOFT STRENGTH
        # ----------------------------------------------------

        denominator = (
            1.0
            - cls.THRESHOLD
        )

        if denominator <= 0:

            denominator = 1.0

        strength = np.where(
            prediction >= cls.THRESHOLD,
            (
                prediction
                - cls.THRESHOLD
            )
            / denominator,
            0.0,
        )

        strength = np.clip(
            strength,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # REMOVE VERY SMALL NOISE
        # ----------------------------------------------------

        mask_image = Image.fromarray(
            (
                mask * 255
            ).astype(
                np.uint8
            )
        )

        # Small smoothing
        mask_image = (
            mask_image.filter(
                ImageFilter.MaxFilter(
                    5
                )
            )
        )

        mask_image = (
            mask_image.filter(
                ImageFilter.MinFilter(
                    5
                )
            )
        )

        cleaned_mask = (
            np.array(
                mask_image
            )
            > 127
        )

        # Use cleaned mask
        strength = np.where(
            cleaned_mask,
            strength,
            0.0,
        )

        mask_pixels = int(
            np.sum(
                cleaned_mask
            )
        )

        # ----------------------------------------------------
        # CHECK AGAIN
        # ----------------------------------------------------

        if mask_pixels == 0:

            raise RuntimeError(
                "The predicted segmentation "
                "became empty after noise removal."
            )

        # ----------------------------------------------------
        # RED TUMOR COLOR
        # ----------------------------------------------------

        tumor_color = np.zeros_like(
            original_array
        )

        tumor_color[..., 0] = 255.0
        tumor_color[..., 1] = 30.0
        tumor_color[..., 2] = 30.0

        # ----------------------------------------------------
        # OVERLAY STRENGTH
        # ----------------------------------------------------

        alpha = (
            strength
            * 0.70
        )

        alpha = alpha[
            ...,
            np.newaxis
        ]

        # ----------------------------------------------------
        # FINAL OVERLAY
        # ----------------------------------------------------

        overlay = (
            original_array
            * (
                1.0
                - alpha
            )
            +
            tumor_color
            * alpha
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(
            np.uint8
        )

        segmentation_image = (
            Image.fromarray(
                overlay
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        filename = (
            f"segmentation_"
            f"{uuid.uuid4().hex}.png"
        )

        segmentation_path = os.path.join(
            output_dir,
            filename,
        )

        segmentation_image.save(
            segmentation_path
        )

        print(
            "Tumor segmentation image saved:",
            segmentation_path,
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {
            "segmentation_path":
                segmentation_path,

            "mask_pixels":
                mask_pixels,

            "segmentation_available":
                True,
        }


# ============================================================
# END OF FILE
# ============================================================