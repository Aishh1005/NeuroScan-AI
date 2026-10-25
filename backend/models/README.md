# Trained Model Files

This folder is where the backend loads trained weights from at runtime. It
is intentionally empty in this delivery — **no fabricated weights are
included.**

## Required for tumor classification

Place a checkpoint produced by `ml/training/train.py` here:

```
backend/models/classifier_resnet50.pt
```
(or `classifier_efficientnet.pt` if you set `CLASSIFIER_ARCHITECTURE=efficientnet`
and point `CLASSIFIER_MODEL_PATH` at it in `.env`)

Until this file exists, any upload will complete image-quality-check and
preprocessing successfully, then the API will return:

> "Trained model not found. Please place the trained classifier checkpoint
> at '...'. See ml/training/train.py and the README for training
> instructions."

## Optional, for tumor localization

```
backend/models/segmentation_unet.pt
```

Produced by `ml/training/train_segmentation.py`. If absent, the
segmentation/localization step is simply skipped for every analysis — no
fake tumor outline is ever drawn.

## Checkpoint format

Both training scripts save a dict:
```python
{
  "model_state_dict": ...,
  "architecture": "resnet50",   # classifier only
  "class_names": [...],         # classifier only
  "model_version": "v1",        # classifier only
}
```
