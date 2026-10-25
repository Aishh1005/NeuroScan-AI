from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.db_models import User, DemoDoctor, AuditLog
from app.schemas.schemas import (
    UserCreate,
    UserLogin,
    UserOut,
    Token,
    ChangePassword,
)
from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)


router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
)


# =========================================================
# DOCTOR REGISTRATION
# =========================================================

@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
):
    # -----------------------------------------------------
    # Check whether email already exists
    # -----------------------------------------------------

    email = payload.email.lower().strip()

    existing = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists.",
        )

    # -----------------------------------------------------
    # Clean verification information
    # -----------------------------------------------------

    doctor_id = payload.doctor_id.strip()
    verification_code = payload.verification_code.strip().upper()
    full_name = payload.full_name.strip()

    # -----------------------------------------------------
    # Find demo doctor record
    # -----------------------------------------------------

    demo_doctor = (
        db.query(DemoDoctor)
        .filter(
            DemoDoctor.doctor_id == doctor_id
        )
        .first()
    )

    if not demo_doctor:
        raise HTTPException(
            status_code=400,
            detail="Invalid Doctor ID or verification details.",
        )

    # -----------------------------------------------------
    # Check demo doctor name
    # -----------------------------------------------------

    if (
        demo_doctor.name.strip().lower()
        != full_name.lower()
    ):
        raise HTTPException(
            status_code=400,
            detail="Doctor name does not match the verification record.",
        )

    # -----------------------------------------------------
    # Check verification code
    # -----------------------------------------------------

    if (
        demo_doctor.verification_code.strip().upper()
        != verification_code
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid verification code.",
        )

    # -----------------------------------------------------
    # Check verification record status
    # -----------------------------------------------------

    if demo_doctor.status != "verified":
        raise HTTPException(
            status_code=400,
            detail="This doctor verification record is not active.",
        )

    # -----------------------------------------------------
    # Create doctor account
    # -----------------------------------------------------

    user = User(
        full_name=full_name,
        email=email,
        hashed_password=hash_password(payload.password),

        # New public registrations are doctors.
        role="doctor",

        doctor_id=doctor_id,

        # Admin must approve the account before login.
        verification_status="pending",

        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # -----------------------------------------------------
    # Create audit log
    # -----------------------------------------------------

    db.add(
        AuditLog(
            user_id=user.id,
            action="doctor_register",
            detail=(
                f"Doctor registration created for "
                f"{user.doctor_id}. Awaiting admin approval."
            ),
        )
    )

    db.commit()

    return user


# =========================================================
# LOGIN
# =========================================================

@router.post(
    "/login",
    response_model=Token,
)
def login(
    payload: UserLogin,
    db: Session = Depends(get_db),
):
    email = payload.email.lower().strip()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    # -----------------------------------------------------
    # Verify password
    # -----------------------------------------------------

    if not verify_password(
        payload.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    # -----------------------------------------------------
    # Check active status
    # -----------------------------------------------------

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="This account has been deactivated.",
        )

    # -----------------------------------------------------
    # Doctor approval check
    # -----------------------------------------------------

    if user.role == "doctor":

        if user.verification_status == "pending":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your doctor account is waiting for "
                    "administrator approval."
                ),
            )

        if user.verification_status == "rejected":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your doctor registration was rejected "
                    "by the administrator."
                ),
            )

        if user.verification_status != "approved":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your doctor account has not been approved."
                ),
            )

    # -----------------------------------------------------
    # Create JWT token
    # -----------------------------------------------------

    token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role,
        }
    )

    # -----------------------------------------------------
    # Audit log
    # -----------------------------------------------------

    db.add(
        AuditLog(
            user_id=user.id,
            action="login",
            detail="User logged in.",
        )
    )

    db.commit()

    return Token(
        access_token=token,
        user=user,
    )


# =========================================================
# SWAGGER / OAUTH2 LOGIN
# =========================================================

@router.post("/oauth-login")
def oauth_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    email = form_data.username.lower().strip()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    # -----------------------------------------------------
    # Verify password
    # -----------------------------------------------------

    if not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    # -----------------------------------------------------
    # Check active status
    # -----------------------------------------------------

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="This account has been deactivated.",
        )

    # -----------------------------------------------------
    # Doctor approval check
    # -----------------------------------------------------

    if user.role == "doctor":

        if user.verification_status == "pending":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your doctor account is waiting for "
                    "administrator approval."
                ),
            )

        if user.verification_status == "rejected":
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your doctor registration was rejected "
                    "by the administrator."
                ),
            )

        if user.verification_status != "approved":
            raise HTTPException(
                status_code=403,
                detail="Your doctor account has not been approved.",
            )

    # -----------------------------------------------------
    # Create token
    # -----------------------------------------------------

    token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role,
        }
    )

    # -----------------------------------------------------
    # Audit log
    # -----------------------------------------------------

    db.add(
        AuditLog(
            user_id=user.id,
            action="login",
            detail="User logged in through Swagger OAuth2.",
        )
    )

    db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
    }


# =========================================================
# CURRENT USER
# =========================================================

@router.get(
    "/me",
    response_model=UserOut,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user


# =========================================================
# CHANGE PASSWORD
# =========================================================

@router.post("/change-password")
def change_password(
    payload: ChangePassword,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(
        payload.current_password,
        current_user.hashed_password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect.",
        )

    if len(payload.new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="New password must be at least 8 characters.",
        )

    current_user.hashed_password = hash_password(
        payload.new_password
    )

    db.commit()

    db.add(
        AuditLog(
            user_id=current_user.id,
            action="change_password",
            detail="Password changed.",
        )
    )

    db.commit()

    return {
        "message": "Password updated successfully."
    }


# =========================================================
# LOGOUT
# =========================================================

@router.post("/logout")
def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.add(
        AuditLog(
            user_id=current_user.id,
            action="logout",
            detail="User logged out.",
        )
    )

    db.commit()

    return {
        "message": "Logged out successfully."
    }