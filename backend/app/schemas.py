from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class Detection(BaseModel):
    id: str
    class_id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    bbox_pixels: tuple[float, float, float, float]


class DetectionResponse(BaseModel):
    request_id: str
    model: str
    device: str
    inference_time: float = Field(description="Total inference time in seconds")
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    image_width: int
    image_height: int
    result_image_url: str
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    service: str
    version: str
    model_ready: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    ready: bool
    device: str
    weights: str
    source: str
    detail: str | None = None

