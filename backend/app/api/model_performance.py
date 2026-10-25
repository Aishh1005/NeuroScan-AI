import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.db_models import ModelMetric, User
from app.auth.security import get_current_user
from app.ml.inference import ClassifierService, SegmentationService

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/performance")
def model_performance(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    metrics = db.query(ModelMetric).order_by(ModelMetric.trained_at.desc()).all()

    if not metrics:
        return {
            "available": False,
            "message": "Model evaluation results are not yet available.",
            "classifier_model_present": ClassifierService.is_available(),
            "segmentation_model_present": SegmentationService.is_available(),
            "models": [],
        }

    result = []
    for m in metrics:
        result.append({
            "model_name": m.model_name,
            "architecture": m.architecture,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "roc_auc": m.roc_auc,
            "confusion_matrix": json.loads(m.confusion_matrix) if m.confusion_matrix else None,
            "num_classes": m.num_classes,
            "train_dataset": m.train_dataset,
            "test_dataset": m.test_dataset,
            "trained_at": m.trained_at.isoformat(),
        })

    return {
        "available": True,
        "classifier_model_present": ClassifierService.is_available(),
        "segmentation_model_present": SegmentationService.is_available(),
        "models": result,
    }
