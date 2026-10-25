# Project Documentation

## 1. Problem Statement

Manual interpretation of brain MRI scans for tumor detection is time-consuming and depends heavily on
radiologist expertise and availability, particularly in resource-constrained settings. Deep-learning
image classifiers can assist this process, but "black box" predictions are difficult for clinicians and
researchers to trust or audit. There is a need for an integrated platform that not only classifies brain
tumors from MRI images, but also explains its predictions, communicates uncertainty honestly, and keeps a
reproducible record of every analysis.

## 2. Existing System

Most academic brain-tumor-classification projects stop at a standalone Jupyter notebook: a model is
trained and its accuracy reported, with no user-facing application, no persistence of results, no
explainability layer, and no confidence communication. Where web demos exist, they typically lack
authentication, per-user history, or any indication of when a prediction should not be trusted.

## 3. Proposed System

NeuroScan AI is a full-stack, authenticated web application that wraps a trained deep-learning classifier
in a complete, auditable pipeline: upload → image-quality validation → preprocessing → classification →
explainability (Grad-CAM) → optional tumor localization (U-Net) → confidence scoring → persisted history →
PDF reporting. Every stage's output is visible to the user, and every result page carries an explicit
research-only disclaimer.

## 4. Objectives

1. Provide secure, authenticated access to a brain-MRI analysis tool.
2. Reject clearly unusable uploads before they reach the model (quality validation).
3. Classify tumors using a transfer-learning CNN trained on a real, cited dataset.
4. Make the model's reasoning inspectable via Grad-CAM, not just a bare label.
5. Communicate prediction confidence honestly, including a "low confidence — review recommended" state.
6. Persist a per-user, searchable analysis history.
7. Generate a professional PDF report suitable for academic submission or portfolio demonstration.
8. Avoid every form of fabrication: no fake weights, no fake metrics, no fake segmentation masks.

## 5. Methodology

- **Data**: A public, licensed brain-MRI tumor classification dataset (see `dataset/README.md`), split by
  patient/source where possible to avoid leakage, in an `ImageFolder`-compatible layout.
- **Preprocessing**: Resize to 224×224, ImageNet-mean/std normalization, RGB channel conversion.
- **Model**: ResNet50 (primary) / EfficientNet-B0 (secondary) initialized with ImageNet weights, fine-tuned
  with a replaced classification head via `ml/training/train.py`.
- **Explainability**: Grad-CAM computed from the gradients of the predicted class score with respect to the
  final convolutional layer's activations (`backend/app/ml/gradcam.py`).
- **Segmentation (optional)**: A from-scratch U-Net (`backend/app/ml/unet.py`) trained with a combined
  BCE + Dice loss on a masked dataset, if the researcher supplies one.
- **Evaluation**: Accuracy, macro-precision/recall/F1, ROC-AUC (one-vs-rest), and confusion matrix computed
  on a held-out test split via `ml/training/evaluate.py` and persisted to the database.

## 6. System Architecture

See `README.md` §3 for the layered diagram (React frontend → FastAPI backend → SQLAlchemy/SQLite →
PyTorch ML layer). The backend is organized by responsibility (api / auth / services / ml) rather than by
technical layer alone, to keep the ML pipeline modular and independently testable.

## 7. ML Pipeline

```
MRI upload
  → file validation (type, size, corruption)
  → quality check (resolution, blur, brightness, blank-image detection)
  → preprocessing (resize, normalize, tensor conversion)
  → classifier inference (ResNet50/EfficientNet)
  → Grad-CAM (attention heatmap over the predicted class)
  → [optional] U-Net segmentation, only if tumor detected and a trained segmentation checkpoint exists
  → confidence-level classification (high ≥ 90%, moderate 70–89%, low < 70%)
  → persisted Analysis record
```

## 8. Database Design

| Table | Purpose |
|---|---|
| `users` | Accounts, hashed passwords, role (user/admin), active flag |
| `analyses` | One row per upload: quality status, prediction, confidence, probabilities, image paths |
| `reports` | Generated PDF file paths, linked 1:1 to an analysis |
| `audit_logs` | Login/logout/registration/analysis/report/admin action trail |
| `model_metrics` | Real evaluation results written by `evaluate.py`; empty until training happens |

## 9. API Architecture

RESTful JSON API under `/api/*`, versionless (single current version), JWT bearer authentication on all
non-auth endpoints, role-gated admin endpoints (`/api/admin/*`). See `README.md` §8 for the full endpoint
table, or `/docs` for live OpenAPI documentation.

## 10. Use-Case Description

- **Researcher (user role)**: registers, logs in, uploads MRI images, reviews AI-assisted predictions and
  Grad-CAM explanations, downloads PDF reports, reviews their own history.
- **Administrator**: monitors system-wide usage and classification distribution, manages account status,
  reviews the audit log. Administrators are not shown other users' images by default — only aggregate
  statistics and metadata (see §23 of the original specification: no casual access to per-user MRI content).

## 11. Limitations

See `README.md` §11 for the full list (no bundled weights/dataset, heuristic quality checks, Grad-CAM is
not medical evidence, password reset is a placeholder, SQLite by default).

## 12. Future Scope

- Multi-modal MRI sequence support (T1/T2/FLAIR) instead of single-image classification
- Ensemble or uncertainty-calibrated confidence (e.g. Monte Carlo dropout) instead of softmax confidence alone
- DICOM ingestion in addition to JPG/PNG
- Federated or privacy-preserving training across institutions
- A clinician-review workflow (second-opinion sign-off) before any report is finalized
