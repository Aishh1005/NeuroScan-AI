import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    SECRET_KEY: str = "insecure-dev-key-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/app.db"

    FRONTEND_ORIGIN: str = "http://localhost:5173"

    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 10

    # ML model
    CLASSIFIER_MODEL_PATH: str = str(
        BASE_DIR / "models" / "classifier_efficientnet.pt"
    )

    SEGMENTATION_MODEL_PATH: str = str(
        BASE_DIR / "models" / "segmentation_unet.pt"
    )

    CLASSIFIER_ARCHITECTURE: str = "efficientnet"

    CLASS_NAMES: str = "glioma,meningioma,notumor,pituitary"

    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "ChangeMe123!"
    ADMIN_NAME: str = "System Admin"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def class_names_list(self):
        return [
            c.strip()
            for c in self.CLASS_NAMES.split(",")
            if c.strip()
        ]


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(Path(settings.CLASSIFIER_MODEL_PATH).parent, exist_ok=True)