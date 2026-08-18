from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "YOLO-CMFM Inference API")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    model_name: str = os.getenv("MODEL_NAME", "YOLOv11-CMFM")
    model_device: str = os.getenv("MODEL_DEVICE", "0")
    model_imgsz: int = int(os.getenv("MODEL_IMGSZ", "640"))
    model_load_on_startup: bool = _env_bool("MODEL_LOAD_ON_STARTUP", True)
    model_source_dir: Path = _env_path(
        "MODEL_SOURCE_DIR", PROJECT_ROOT / "third_party" / "YOLOv11-CMFM"
    )
    model_weights: Path = _env_path("MODEL_WEIGHTS", PROJECT_ROOT / "weights" / "best.pt")
    runtime_dir: Path = _env_path("RUNTIME_DIR", PROJECT_ROOT / "backend" / "runtime")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    result_ttl_seconds: int = int(os.getenv("RESULT_TTL_SECONDS", "3600"))
    gpu_concurrency: int = max(1, int(os.getenv("GPU_CONCURRENCY", "1")))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def results_dir(self) -> Path:
        return self.runtime_dir / "results"

    @property
    def requests_dir(self) -> Path:
        return self.runtime_dir / "requests"


settings = Settings()

