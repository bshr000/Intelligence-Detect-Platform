from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .model_service import InvalidImagePairError, ModelService, ModelUnavailableError
from .schemas import DetectionResponse, HealthResponse, ModelInfoResponse


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService(settings)
    app.state.model_service = service
    await run_in_threadpool(service.initialize)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Visible-SAR four-channel object detection service based on YOLOv11-CMFM.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


def _validate_filename(file: UploadFile, label: str) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{label} must use one of these extensions: {supported}",
        )


async def _read_limited(file: UploadFile, label: str) -> bytes:
    _validate_filename(file, label)
    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{label} exceeds the {settings.max_upload_mb} MB limit.",
        )
    return content


@app.get("/", tags=["service"])
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/api/v1/health/live", response_model=HealthResponse, tags=["health"])
async def live(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        model_ready=service.ready,
        detail=service.detail,
    )


@app.get("/api/v1/health/ready", response_model=HealthResponse, tags=["health"])
async def ready(request: Request):
    service = _service(request)
    payload = HealthResponse(
        status="ok" if service.ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        model_ready=service.ready,
        detail=service.detail,
    )
    if not service.ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.get("/api/v1/models/current", response_model=ModelInfoResponse, tags=["model"])
async def model_info(request: Request) -> ModelInfoResponse:
    service = _service(request)
    return ModelInfoResponse(
        name=settings.model_name,
        ready=service.ready,
        device=settings.model_device,
        weights=str(settings.model_weights),
        source=str(settings.model_source_dir),
        detail=service.detail,
    )


async def _detect(
    request: Request,
    rgb_image: UploadFile,
    sar_image: UploadFile,
    confidence: float,
    iou: float,
    imgsz: int,
) -> DetectionResponse:
    rgb_bytes = await _read_limited(rgb_image, "RGB image")
    sar_bytes = await _read_limited(sar_image, "SAR image")
    service = _service(request)
    try:
        return await service.infer(rgb_bytes, sar_bytes, confidence, iou, imgsz)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImagePairError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        await rgb_image.close()
        await sar_image.close()


@app.post(
    "/api/v1/detections",
    response_model=DetectionResponse,
    tags=["inference"],
)
async def create_detection(
    request: Request,
    rgb_image: Annotated[UploadFile, File(description="Visible RGB image")],
    sar_image: Annotated[UploadFile, File(description="Spatially aligned SAR image")],
    confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    iou: Annotated[float, Form(ge=0.0, le=1.0)] = 0.70,
    imgsz: Annotated[int, Form(ge=32, le=4096)] = settings.model_imgsz,
) -> DetectionResponse:
    return await _detect(request, rgb_image, sar_image, confidence, iou, imgsz)


@app.post("/detect", response_model=DetectionResponse, include_in_schema=False)
async def legacy_detection(
    request: Request,
    rgb_image: Annotated[UploadFile, File()],
    sar_image: Annotated[UploadFile, File()],
    confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    iou: Annotated[float, Form(ge=0.0, le=1.0)] = 0.70,
    imgsz: Annotated[int, Form(ge=32, le=4096)] = settings.model_imgsz,
) -> DetectionResponse:
    return await _detect(request, rgb_image, sar_image, confidence, iou, imgsz)


@app.get("/api/v1/results/{request_id}.png", tags=["inference"])
async def result_image(request_id: UUID) -> FileResponse:
    path = settings.results_dir / f"{request_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result image not found or expired.")
    return FileResponse(path, media_type="image/png")
