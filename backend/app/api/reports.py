import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.config import BASE_DIR
from app.models.db_models import User, Analysis, Report, AuditLog
from app.auth.security import get_current_user
from app.services.report_generator import generate_pdf_report

router = APIRouter(prefix="/api/reports", tags=["reports"])

REPORTS_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


@router.post("/generate/{analysis_id}")
def generate_report(analysis_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="You do not have access to this analysis.")

    filename = f"report_{analysis.id}.pdf"
    output_path = os.path.join(REPORTS_DIR, filename)

    try:
        generate_pdf_report(analysis, analysis.user, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    existing = db.query(Report).filter(Report.analysis_id == analysis.id).first()
    if existing:
        existing.file_path = output_path
    else:
        db.add(Report(analysis_id=analysis.id, file_path=output_path))
    db.commit()

    db.add(AuditLog(user_id=current_user.id, action="report_generated", detail=f"Analysis {analysis.id}"))
    db.commit()

    return {"message": "Report generated successfully.", "analysis_id": analysis.id}


@router.get("/{analysis_id}")
def download_report(analysis_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="You do not have access to this analysis.")

    report = db.query(Report).filter(Report.analysis_id == analysis_id).first()
    if not report or not os.path.exists(report.file_path):
        raise HTTPException(status_code=404, detail="Report has not been generated yet.")

    return FileResponse(report.file_path, media_type="application/pdf",
                         filename=f"brain_mri_report_{analysis_id}.pdf")
