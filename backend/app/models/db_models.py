import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Float,
    Text,
    Integer,
)
from sqlalchemy.orm import relationship

from app.database.session import Base


def gen_uuid():
    return str(uuid.uuid4())


# =========================================================
# USER
# =========================================================
class User(Base):
    __tablename__ = "users"

    id = Column(
        String,
        primary_key=True,
        default=gen_uuid,
    )

    full_name = Column(
        String,
        nullable=False,
    )

    email = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password = Column(
        String,
        nullable=False,
    )

    # "user" is kept as the database default so we do not
    # accidentally break existing parts of the project.
    # New doctor registrations will be created with role="doctor"
    # in the auth registration code later.
    role = Column(
        String,
        default="user",
    )

    # -----------------------------------------------------
    # DOCTOR VERIFICATION
    # -----------------------------------------------------
    # Doctor's unique ID used during registration.
    # Nullable because the admin account does not need one.
    doctor_id = Column(
        String,
        unique=True,
        nullable=True,
        index=True,
    )

    # Possible values:
    # pending
    # approved
    # rejected
    verification_status = Column(
        String,
        default="pending",
        nullable=False,
    )

    is_active = Column(
        Boolean,
        default=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    analyses = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# =========================================================
# DEMO DOCTOR VERIFICATION RECORDS
# =========================================================
class DemoDoctor(Base):
    __tablename__ = "demo_doctors"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Example:
    # NS-DOC-001
    doctor_id = Column(
        String,
        unique=True,
        nullable=False,
        index=True,
    )

    # Example:
    # Dr. Ananya Sharma
    name = Column(
        String,
        nullable=False,
    )

    # Example:
    # NSV-7K42P
    verification_code = Column(
        String,
        nullable=False,
    )

    # For our academic demo this will normally be "verified".
    status = Column(
        String,
        default="verified",
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# =========================================================
# ANALYSIS
# =========================================================
class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(
        String,
        primary_key=True,
        default=gen_uuid,
    )

    user_id = Column(
        String,
        ForeignKey("users.id"),
        nullable=False,
    )

    original_filename = Column(
        String,
        nullable=False,
    )

    stored_image_path = Column(
        String,
        nullable=False,
    )

    gradcam_image_path = Column(
        String,
        nullable=True,
    )

    segmentation_image_path = Column(
        String,
        nullable=True,
    )

    quality_status = Column(
        String,
        nullable=False,
    )

    quality_details = Column(
        Text,
        nullable=True,
    )

    status = Column(
        String,
        default="pending",
    )

    failure_reason = Column(
        String,
        nullable=True,
    )

    tumor_detected = Column(
        Boolean,
        nullable=True,
    )

    predicted_class = Column(
        String,
        nullable=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    class_probabilities = Column(
        Text,
        nullable=True,
    )

    confidence_level = Column(
        String,
        nullable=True,
    )

    model_name = Column(
        String,
        nullable=True,
    )

    model_version = Column(
        String,
        nullable=True,
    )

    segmentation_available = Column(
        Boolean,
        default=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    user = relationship(
        "User",
        back_populates="analyses",
    )

    report = relationship(
        "Report",
        back_populates="analysis",
        uselist=False,
        cascade="all, delete-orphan",
    )


# =========================================================
# REPORT
# =========================================================
class Report(Base):
    __tablename__ = "reports"

    id = Column(
        String,
        primary_key=True,
        default=gen_uuid,
    )

    analysis_id = Column(
        String,
        ForeignKey("analyses.id"),
        nullable=False,
    )

    file_path = Column(
        String,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    analysis = relationship(
        "Analysis",
        back_populates="report",
    )


# =========================================================
# AUDIT LOG
# =========================================================
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(
        String,
        primary_key=True,
        default=gen_uuid,
    )

    user_id = Column(
        String,
        ForeignKey("users.id"),
        nullable=True,
    )

    action = Column(
        String,
        nullable=False,
    )

    detail = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# =========================================================
# MODEL METRICS
# =========================================================
class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    model_name = Column(
        String,
        nullable=False,
    )

    architecture = Column(
        String,
        nullable=False,
    )

    accuracy = Column(
        Float,
        nullable=True,
    )

    precision = Column(
        Float,
        nullable=True,
    )

    recall = Column(
        Float,
        nullable=True,
    )

    f1_score = Column(
        Float,
        nullable=True,
    )

    roc_auc = Column(
        Float,
        nullable=True,
    )

    confusion_matrix = Column(
        Text,
        nullable=True,
    )

    num_classes = Column(
        Integer,
        nullable=True,
    )

    train_dataset = Column(
        String,
        nullable=True,
    )

    test_dataset = Column(
        String,
        nullable=True,
    )

    trained_at = Column(
        DateTime,
        default=datetime.utcnow,
    )