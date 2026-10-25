from datetime import datetime
from typing import Optional, Dict, List

from pydantic import BaseModel, EmailStr, field_validator


# =========================================================
# USER REGISTRATION
# =========================================================

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str

    # Doctor verification details
    doctor_id: str
    verification_code: str

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v):
        v = v.strip()

        if len(v) < 3:
            raise ValueError(
                "Full name must be at least 3 characters long."
            )

        return v

    @field_validator("doctor_id")
    @classmethod
    def validate_doctor_id(cls, v):
        v = v.strip()

        if not v:
            raise ValueError(
                "Doctor ID is required."
            )

        return v

    @field_validator("verification_code")
    @classmethod
    def validate_verification_code(cls, v):
        v = v.strip()

        if not v:
            raise ValueError(
                "Verification code is required."
            )

        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError(
                "Password must be at least 8 characters long."
            )

        if not any(c.isupper() for c in v):
            raise ValueError(
                "Password must contain at least one uppercase letter."
            )

        if not any(c.islower() for c in v):
            raise ValueError(
                "Password must contain at least one lowercase letter."
            )

        if not any(c.isdigit() for c in v):
            raise ValueError(
                "Password must contain at least one digit."
            )

        if not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?"
            for c in v
        ):
            raise ValueError(
                "Password must contain at least one special character."
            )

        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data:
            if v != info.data["password"]:
                raise ValueError(
                    "Passwords do not match."
                )

        return v


# =========================================================
# LOGIN
# =========================================================

class UserLogin(BaseModel):
    email: EmailStr
    password: str


# =========================================================
# USER OUTPUT
# =========================================================

class UserOut(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    role: str

    # Doctor information
    doctor_id: Optional[str] = None
    verification_status: Optional[str] = None

    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# =========================================================
# LOGIN TOKEN
# =========================================================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# =========================================================
# CHANGE PASSWORD
# =========================================================

class ChangePassword(BaseModel):
    current_password: str
    new_password: str


# =========================================================
# ANALYSIS
# =========================================================

class QualityDetails(BaseModel):
    width: int
    height: int
    format: str
    sharpness_score: float
    brightness_mean: float
    is_blank_or_near_empty: bool
    passed: bool
    reasons: List[str] = []


class AnalysisOut(BaseModel):
    id: str
    original_filename: str
    quality_status: str
    status: str
    failure_reason: Optional[str] = None
    tumor_detected: Optional[bool] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    segmentation_available: bool
    created_at: datetime

    class Config:
        from_attributes = True


# =========================================================
# DASHBOARD STATISTICS
# =========================================================

class DashboardStats(BaseModel):
    total_analyses: int
    tumor_detected_count: int
    no_tumor_count: int
    average_confidence: Optional[float]
    most_recent: Optional[AnalysisOut]
    class_distribution: Dict[str, int]
    history_over_time: List[Dict]