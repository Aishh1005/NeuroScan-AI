import json
import os
import uuid
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from PIL import Image, UnidentifiedImageError

from app.database.session import get_db
from app.config import settings
from app.models.db_models import User, Analysis, AuditLog
from app.schemas.schemas import AnalysisOut, DashboardStats
from app.auth.security import get_current_user
from app.services.quality_check import run_quality_check
from app.services.preprocessing import save_preprocessed_preview

from app.ml.inference import (
    ClassifierService,
    SegmentationService,
    ModelNotFoundError,
    InvalidBrainMRIError,
)


router = APIRouter(prefix="/api/analysis", tags=["analysis"])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"

    if confidence >= 0.70:
        return "moderate"

    return "low"


def _analysis_to_out(analysis: Analysis) -> AnalysisOut:
    probabilities = (
        json.loads(analysis.class_probabilities)
        if analysis.class_probabilities
        else None
    )

    return AnalysisOut(
        id=analysis.id,
        original_filename=analysis.original_filename,
        quality_status=analysis.quality_status,
        status=analysis.status,
        failure_reason=analysis.failure_reason,
        tumor_detected=analysis.tumor_detected,
        predicted_class=analysis.predicted_class,
        confidence=analysis.confidence,
        confidence_level=analysis.confidence_level,
        class_probabilities=probabilities,
        model_name=analysis.model_name,
        model_version=analysis.model_version,
        segmentation_available=analysis.segmentation_available,
        created_at=analysis.created_at,
    )


@router.post("/upload", response_model=AnalysisOut)
async def upload_and_analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ---------------------------------------------------------
    # 1. Basic file validation
    # ---------------------------------------------------------

    ext = (
        file.filename.rsplit(".", 1)[-1].lower()
        if file.filename and "." in file.filename
        else ""
    )

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JPG, JPEG, or PNG.",
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File exceeds the maximum allowed size of "
                f"{settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    # ---------------------------------------------------------
    # 2. Validate image file
    # ---------------------------------------------------------

    try:
        import io

        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()

        # Reopen image after verify()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is corrupted or not a valid image.",
        )

    # ---------------------------------------------------------
    # 3. Save original upload
    # ---------------------------------------------------------

    user_dir = os.path.join(
        settings.UPLOAD_DIR,
        str(current_user.id),
    )

    os.makedirs(user_dir, exist_ok=True)

    stored_filename = f"{uuid.uuid4().hex}.{ext}"
    stored_path = os.path.join(user_dir, stored_filename)

    with open(stored_path, "wb") as output_file:
        output_file.write(contents)

    # ---------------------------------------------------------
    # 4. Image quality check
    # ---------------------------------------------------------

    quality = run_quality_check(pil_image)

    analysis = Analysis(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_image_path=stored_path,
        quality_status="passed" if quality["passed"] else "failed",
        quality_details=json.dumps(quality),
        status="pending",
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    if not quality["passed"]:
        analysis.status = "failed"
        analysis.failure_reason = (
            "Image quality is insufficient for reliable analysis. "
            + " ".join(quality["reasons"])
        )

        db.commit()
        db.refresh(analysis)

        return _analysis_to_out(analysis)

    # ---------------------------------------------------------
    # 5. Save preprocessing preview
    # ---------------------------------------------------------

    preview_dir = os.path.join(user_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    save_preprocessed_preview(
        pil_image,
        preview_dir,
    )

    # ---------------------------------------------------------
    # 6. Brain MRI validation + classification + Grad-CAM
    # ---------------------------------------------------------

    gradcam_dir = os.path.join(user_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    try:
        result = ClassifierService.predict_with_gradcam(
            pil_image,
            gradcam_dir,
        )

    # ---------------------------------------------------------
    # 6A. Invalid brain MRI image
    # ---------------------------------------------------------

    except InvalidBrainMRIError as e:
        analysis.status = "failed"
        analysis.failure_reason = (
            "This is not a brain MRI. Please upload a brain MRI image."
        )

        db.commit()
        db.refresh(analysis)

        db.add(
            AuditLog(
                user_id=current_user.id,
                action="invalid_brain_mri",
                detail=str(e),
            )
        )

        db.commit()

        return _analysis_to_out(analysis)

    # ---------------------------------------------------------
    # 6B. Model missing
    # ---------------------------------------------------------

    except ModelNotFoundError as e:
        analysis.status = "failed"
        analysis.failure_reason = str(e)

        db.commit()
        db.refresh(analysis)

        db.add(
            AuditLog(
                user_id=current_user.id,
                action="inference_failed",
                detail=str(e),
            )
        )

        db.commit()

        return _analysis_to_out(analysis)

    # ---------------------------------------------------------
    # 6C. Other unexpected inference error
    # ---------------------------------------------------------

    except Exception as e:
        print(f"INFERENCE ERROR: {repr(e)}")

        analysis.status = "failed"
        analysis.failure_reason = (
            "An unexpected error occurred while analyzing the image."
        )

        db.commit()
        db.refresh(analysis)

        db.add(
            AuditLog(
                user_id=current_user.id,
                action="inference_error",
                detail=str(e),
            )
        )

        db.commit()

        return _analysis_to_out(analysis)

    # ---------------------------------------------------------
    # 7. Save classification result
    # ---------------------------------------------------------

    confidence_level = _confidence_level(
        result["confidence"]
    )

    analysis.status = "completed"
    analysis.tumor_detected = result["tumor_detected"]
    analysis.predicted_class = result["predicted_class"]
    analysis.confidence = result["confidence"]
    analysis.confidence_level = confidence_level
    analysis.class_probabilities = json.dumps(
        result["class_probabilities"]
    )
    analysis.gradcam_image_path = result["gradcam_path"]
    analysis.model_name = result["model_name"]
    analysis.model_version = result["model_version"]

    # ---------------------------------------------------------
    # 8. Optional tumor segmentation
    # ---------------------------------------------------------

    if (
        result["tumor_detected"]
        and SegmentationService.is_available()
    ):
        try:
            segmentation_dir = os.path.join(
                user_dir,
                "segmentation",
            )

            segmentation_result = (
                SegmentationService.predict(
                    pil_image,
                    segmentation_dir,
                )
            )

            analysis.segmentation_image_path = (
                segmentation_result["segmentation_path"]
            )

            analysis.segmentation_available = True

        except ModelNotFoundError:
            analysis.segmentation_available = False

        except Exception as e:
            print(f"SEGMENTATION ERROR: {repr(e)}")
            analysis.segmentation_available = False

    # ---------------------------------------------------------
    # 9. Save completed analysis
    # ---------------------------------------------------------

    db.commit()
    db.refresh(analysis)

    db.add(
        AuditLog(
            user_id=current_user.id,
            action="analysis_completed",
            detail=(
                f"Analysis {analysis.id}: "
                f"{analysis.predicted_class} "
                f"({analysis.confidence:.2%})"
            ),
        )
    )

    db.commit()

    return _analysis_to_out(analysis)


# -------------------------------------------------------------
# Analysis history
# -------------------------------------------------------------

@router.get(
    "/history",
    response_model=list[AnalysisOut],
)
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    analyses = (
        db.query(Analysis)
        .filter(
            Analysis.user_id == current_user.id
        )
        .order_by(
            Analysis.created_at.desc()
        )
        .all()
    )

    return [
        _analysis_to_out(analysis)
        for analysis in analyses
    ]


# -------------------------------------------------------------
# Dashboard statistics
# -------------------------------------------------------------

@router.get(
    "/dashboard",
    response_model=DashboardStats,
)
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    analyses = (
        db.query(Analysis)
        .filter(
            Analysis.user_id == current_user.id,
            Analysis.status == "completed",
        )
        .order_by(
            Analysis.created_at.desc()
        )
        .all()
    )

    total = len(analyses)

    tumor_count = sum(
        1
        for analysis in analyses
        if analysis.tumor_detected
    )

    no_tumor_count = total - tumor_count

    average_confidence = (
        sum(
            analysis.confidence
            for analysis in analyses
        ) / total
        if total
        else None
    )

    class_distribution = defaultdict(int)

    for analysis in analyses:
        if analysis.predicted_class:
            class_distribution[
                analysis.predicted_class
            ] += 1

    history_by_day = defaultdict(int)

    for analysis in analyses:
        day = analysis.created_at.strftime("%Y-%m-%d")
        history_by_day[day] += 1

    history_over_time = [
        {
            "date": day,
            "count": count,
        }
        for day, count in sorted(
            history_by_day.items()
        )
    ]

    most_recent = (
        _analysis_to_out(analyses[0])
        if analyses
        else None
    )

    return DashboardStats(
        total_analyses=total,
        tumor_detected_count=tumor_count,
        no_tumor_count=no_tumor_count,
        average_confidence=average_confidence,
        most_recent=most_recent,
        class_distribution=dict(class_distribution),
        history_over_time=history_over_time,
    )


# -------------------------------------------------------------
# Get one analysis
# -------------------------------------------------------------

@router.get(
    "/{analysis_id}",
    response_model=AnalysisOut,
)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id
        )
        .first()
    )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    if (
        analysis.user_id != current_user.id
        and current_user.role != "admin"
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this analysis.",
        )

    return _analysis_to_out(analysis)


# -------------------------------------------------------------
# Get analysis images
# -------------------------------------------------------------

@router.get(
    "/{analysis_id}/image/{image_type}"
)
def get_analysis_image(
    analysis_id: str,
    image_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from fastapi.responses import FileResponse

    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id
        )
        .first()
    )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    if (
        analysis.user_id != current_user.id
        and current_user.role != "admin"
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this analysis.",
        )

    path_map = {
        "original": analysis.stored_image_path,
        "gradcam": analysis.gradcam_image_path,
        "segmentation": analysis.segmentation_image_path,
    }

    path = path_map.get(image_type)

    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Image of type '{image_type}' "
                "is not available."
            ),
        )

    return FileResponse(path)