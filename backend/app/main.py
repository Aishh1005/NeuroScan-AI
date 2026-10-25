import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.session import Base, engine, SessionLocal
from app.models.db_models import User, DemoDoctor
from app.auth.security import hash_password

from app.api import (
    auth,
    analysis,
    reports,
    admin,
    model_performance,
)


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("brain_tumor_app")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Brain Tumor Detection & Classification API",
    description=(
        "AI-assisted research platform for brain MRI analysis. "
        "Not for clinical diagnosis."
    ),
    version="1.0.0",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STARTUP EVENT
# =========================================================

@app.on_event("startup")
def on_startup():
    """
    Create database tables and bootstrap the admin account
    and demo doctor verification record.
    """

    # -----------------------------------------------------
    # Create all database tables
    # -----------------------------------------------------
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # =================================================
        # CREATE / VERIFY ADMIN ACCOUNT
        # =================================================

        admin_user = (
            db.query(User)
            .filter(
                User.email == settings.ADMIN_EMAIL.lower()
            )
            .first()
        )

        if not admin_user:
            admin_user = User(
                full_name=settings.ADMIN_NAME,
                email=settings.ADMIN_EMAIL.lower(),
                hashed_password=hash_password(
                    settings.ADMIN_PASSWORD
                ),
                role="admin",
                doctor_id=None,
                verification_status="approved",
                is_active=True,
            )

            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

            logger.info(
                "Bootstrap admin account created: %s",
                settings.ADMIN_EMAIL,
            )

        else:
            # Make sure the existing admin remains approved.
            admin_user.role = "admin"
            admin_user.verification_status = "approved"
            admin_user.is_active = True

            db.commit()

            logger.info(
                "Bootstrap admin account already exists: %s",
                settings.ADMIN_EMAIL,
            )

        # =================================================
        # CREATE DEMO DOCTOR VERIFICATION RECORD
        # =================================================

        demo_doctor = (
            db.query(DemoDoctor)
            .filter(
                DemoDoctor.doctor_id == "NS-DOC-001"
            )
            .first()
        )

        if not demo_doctor:
            demo_doctor = DemoDoctor(
                doctor_id="NS-DOC-001",
                name="Dr. Ananya Sharma",
                verification_code="NSV-7K42P",
                status="verified",
            )

            db.add(demo_doctor)
            db.commit()

            logger.info(
                "Demo doctor verification record created: NS-DOC-001"
            )

        else:
            logger.info(
                "Demo doctor verification record already exists: NS-DOC-001"
            )

    except Exception:
        db.rollback()

        logger.exception(
            "Error while initializing the database"
        )

        raise

    finally:
        db.close()


# =========================================================
# GLOBAL EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Handle unexpected server errors without exposing
    internal stack traces to the client.
    """

    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred."
        },
    )


# =========================================================
# API ROUTERS
# =========================================================

app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(admin.router)
app.include_router(model_performance.router)


# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def home():
    """
    Basic homepage route to confirm that the API is running.
    """

    return {
        "message": "Brain Tumor Detection API is running successfully",
        "documentation": "/docs",
        "health_check": "/api/health",
        "version": "1.0.0",
    }


# =========================================================
# HEALTH CHECK ROUTE
# =========================================================

@app.get("/api/health")
def health_check():
    """
    Check whether the backend service is running.
    """

    return {
        "status": "ok",
        "service": "brain-tumor-detection-api",
    }