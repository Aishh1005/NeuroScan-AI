from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.db_models import User, Analysis, AuditLog
from app.schemas.schemas import UserOut
from app.auth.security import get_current_admin


router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------
# GET ALL USERS
# ---------------------------------------------------------
@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    return (
        db.query(User)
        .order_by(User.created_at.desc())
        .all()
    )


# ---------------------------------------------------------
# GET PENDING DOCTORS
# ---------------------------------------------------------
@router.get("/doctors/pending", response_model=list[UserOut])
def list_pending_doctors(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    return (
        db.query(User)
        .filter(
            User.role == "doctor",
            User.verification_status == "pending",
        )
        .order_by(User.created_at.desc())
        .all()
    )


# ---------------------------------------------------------
# APPROVE DOCTOR
# ---------------------------------------------------------
@router.patch("/doctors/{user_id}/approve")
def approve_doctor(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    doctor = (
        db.query(User)
        .filter(
            User.id == user_id,
            User.role == "doctor",
        )
        .first()
    )

    if not doctor:
        raise HTTPException(
            status_code=404,
            detail="Doctor not found.",
        )

    if doctor.verification_status == "approved":
        raise HTTPException(
            status_code=400,
            detail="Doctor is already approved.",
        )

    doctor.verification_status = "approved"
    doctor.is_active = True

    db.commit()

    db.add(
        AuditLog(
            user_id=admin.id,
            action="approve_doctor",
            detail=f"Doctor {doctor.id} approved by admin.",
        )
    )

    db.commit()

    return {
        "message": "Doctor approved successfully.",
        "doctor_id": doctor.id,
        "verification_status": doctor.verification_status,
    }


# ---------------------------------------------------------
# REJECT DOCTOR
# ---------------------------------------------------------
@router.patch("/doctors/{user_id}/reject")
def reject_doctor(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    doctor = (
        db.query(User)
        .filter(
            User.id == user_id,
            User.role == "doctor",
        )
        .first()
    )

    if not doctor:
        raise HTTPException(
            status_code=404,
            detail="Doctor not found.",
        )

    if doctor.verification_status == "rejected":
        raise HTTPException(
            status_code=400,
            detail="Doctor is already rejected.",
        )

    doctor.verification_status = "rejected"

    db.commit()

    db.add(
        AuditLog(
            user_id=admin.id,
            action="reject_doctor",
            detail=f"Doctor {doctor.id} rejected by admin.",
        )
    )

    db.commit()

    return {
        "message": "Doctor rejected.",
        "doctor_id": doctor.id,
        "verification_status": doctor.verification_status,
    }


# ---------------------------------------------------------
# TOGGLE USER ACTIVE STATUS
# ---------------------------------------------------------
@router.patch("/users/{user_id}/toggle-active")
def toggle_active(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    if user.id == admin.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot deactivate your own admin account.",
        )

    user.is_active = not user.is_active

    db.commit()

    db.add(
        AuditLog(
            user_id=admin.id,
            action="toggle_user_active",
            detail=(
                f"User {user.id} set to "
                f"is_active={user.is_active}"
            ),
        )
    )

    db.commit()

    return {
        "message": "User status updated.",
        "is_active": user.is_active,
    }


# ---------------------------------------------------------
# ADMIN STATISTICS
# ---------------------------------------------------------
@router.get("/statistics")
def statistics(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    total_users = db.query(User).count()

    total_analyses = (
        db.query(Analysis)
        .filter(Analysis.status == "completed")
        .count()
    )

    tumor_positive = (
        db.query(Analysis)
        .filter(
            Analysis.status == "completed",
            Analysis.tumor_detected == True,  # noqa: E712
        )
        .count()
    )

    tumor_negative = total_analyses - tumor_positive

    class_dist = defaultdict(int)

    completed_analyses = (
        db.query(Analysis)
        .filter(Analysis.status == "completed")
        .all()
    )

    for analysis in completed_analyses:
        if analysis.predicted_class:
            class_dist[analysis.predicted_class] += 1

    recent = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(10)
        .all()
    )

    recent_out = [
        {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.confidence,
            "status": analysis.status,
            "created_at": analysis.created_at.isoformat(),
        }
        for analysis in recent
    ]

    return {
        "total_users": total_users,
        "total_analyses": total_analyses,
        "tumor_positive": tumor_positive,
        "tumor_negative": tumor_negative,
        "class_distribution": dict(class_dist),
        "recent_analyses": recent_out,
    }


# ---------------------------------------------------------
# AUDIT LOGS
# ---------------------------------------------------------
@router.get("/logs")
def get_logs(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(200)
        .all()
    )

    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "detail": log.detail,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]