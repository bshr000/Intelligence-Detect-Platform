from __future__ import annotations

import asyncio
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from .config import Settings
from .schemas import BoundingBox, Detection, DetectionResponse


class ModelUnavailableError(RuntimeError):
    pass


class InvalidImagePairError(ValueError):
    pass


@dataclass(slots=True)
class DecodedPair:
    visible: Any
    sar: Any
    width: int
    height: int


class ModelService:
    """Thin adapter around the original YOLOv11-CMFM inference flow."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.model: Any | None = None
        self.ready = False
        self.detail = "Model has not been loaded."
        self._semaphore = asyncio.Semaphore(config.gpu_concurrency)

    def initialize(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.requests_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired_results()

        if not self.config.model_load_on_startup:
            self.detail = "Model loading is disabled by MODEL_LOAD_ON_STARTUP."
            return
        if not self.config.model_source_dir.is_dir():
            self.detail = f"Model source directory not found: {self.config.model_source_dir}"
            return
        if not self.config.model_weights.is_file():
            self.detail = f"Model weights not found: {self.config.model_weights}"
            return

        source = str(self.config.model_source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.config.model_weights))
        except Exception as exc:  # surfaced through readiness instead of killing the API
            self.detail = f"Model load failed: {exc}"
            self.model = None
            return

        self.ready = True
        self.detail = None

    def cleanup_expired_results(self) -> None:
        cutoff = time.time() - self.config.result_ttl_seconds
        for path in self.config.results_dir.glob("*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    @staticmethod
    def decode_pair(rgb_bytes: bytes, sar_bytes: bytes) -> DecodedPair:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ModelUnavailableError(
                "OpenCV and NumPy are required. Install the model and backend dependencies first."
            ) from exc

        visible = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        sar = cv2.imdecode(np.frombuffer(sar_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if visible is None:
            raise InvalidImagePairError("RGB image cannot be decoded.")
        if sar is None:
            raise InvalidImagePairError("SAR image cannot be decoded.")

        height, width = visible.shape[:2]
        sar_height, sar_width = sar.shape[:2]
        if (width, height) != (sar_width, sar_height):
            raise InvalidImagePairError(
                "RGB and SAR images must have identical width and height because the model expects spatially aligned inputs."
            )
        return DecodedPair(visible=visible, sar=sar, width=width, height=height)

    async def infer(
        self,
        rgb_bytes: bytes,
        sar_bytes: bytes,
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> DetectionResponse:
        if not self.ready or self.model is None:
            raise ModelUnavailableError(self.detail or "Model is not ready.")

        pair = await run_in_threadpool(self.decode_pair, rgb_bytes, sar_bytes)
        async with self._semaphore:
            return await run_in_threadpool(
                self._predict, pair, confidence, iou, imgsz
            )

    def _predict(
        self,
        pair: DecodedPair,
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> DetectionResponse:
        import cv2

        request_id = str(uuid.uuid4())
        request_root = self.config.requests_dir / request_id
        visible_dir = request_root / "visible"
        infrared_dir = request_root / "infrared"
        visible_dir.mkdir(parents=True, exist_ok=False)
        infrared_dir.mkdir(parents=True, exist_ok=False)
        visible_path = visible_dir / "input.png"
        infrared_path = infrared_dir / "input.png"

        try:
            if not cv2.imwrite(str(visible_path), pair.visible):
                raise RuntimeError("Failed to stage the RGB image.")
            if not cv2.imwrite(str(infrared_path), pair.sar):
                raise RuntimeError("Failed to stage the SAR image.")

            started = time.perf_counter()
            results = self.model.predict(
                source=str(visible_path),
                imgsz=imgsz,
                device=self.config.model_device,
                conf=confidence,
                iou=iou,
                show=False,
                save=False,
                save_frames=False,
                use_simotm="RGBT",
                channels=4,
                verbose=False,
            )
            elapsed = time.perf_counter() - started
            if not results:
                raise RuntimeError("The model returned no inference result.")

            result = results[0]
            detections = self._serialize_detections(result, pair.width, pair.height)
            rgb_output = pair.visible.copy()
            sar_output = cv2.cvtColor(pair.sar, cv2.COLOR_GRAY2BGR)
            self._draw_detections(rgb_output, detections)
            self._draw_detections(sar_output, detections)

            rgb_result_path = self.config.results_dir / f"{request_id}-rgb.png"
            sar_result_path = self.config.results_dir / f"{request_id}-sar.png"
            if not cv2.imwrite(str(rgb_result_path), rgb_output):
                raise RuntimeError("Failed to write the RGB detection result image.")
            if not cv2.imwrite(str(sar_result_path), sar_output):
                raise RuntimeError("Failed to write the SAR detection result image.")

            speed = getattr(result, "speed", {}) or {}
            self.cleanup_expired_results()
            return DetectionResponse(
                request_id=request_id,
                model=self.config.model_name,
                device=self.config.model_device,
                inference_time=elapsed,
                preprocess_ms=float(speed.get("preprocess", 0.0)),
                inference_ms=float(speed.get("inference", elapsed * 1000)),
                postprocess_ms=float(speed.get("postprocess", 0.0)),
                image_width=pair.width,
                image_height=pair.height,
                result_image_url=f"/api/v1/results/{request_id}.png",
                rgb_result_image_url=f"/api/v1/results/{request_id}/rgb.png",
                sar_result_image_url=f"/api/v1/results/{request_id}/sar.png",
                detections=detections,
            )
        finally:
            shutil.rmtree(request_root, ignore_errors=True)

    @staticmethod
    def _serialize_detections(result: Any, width: int, height: int) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        names = getattr(result, "names", {})
        output: list[Detection] = []

        for index, (coords, score, raw_class_id) in enumerate(
            zip(xyxy, confidences, class_ids, strict=True)
        ):
            x1, y1, x2, y2 = (float(value) for value in coords)
            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))
            class_id = int(raw_class_id)
            if isinstance(names, dict):
                label = str(names.get(class_id, class_id))
            else:
                label = str(names[class_id]) if class_id < len(names) else str(class_id)

            output.append(
                Detection(
                    id=f"det-{index}",
                    class_id=class_id,
                    label=label,
                    confidence=float(score),
                    bbox=BoundingBox(
                        x=x1 / width,
                        y=y1 / height,
                        width=max(0.0, x2 - x1) / width,
                        height=max(0.0, y2 - y1) / height,
                    ),
                    bbox_pixels=(x1, y1, x2, y2),
                )
            )
        return output

    @staticmethod
    def _draw_detections(image: Any, detections: list[Detection]) -> None:
        import cv2

        colors = ((47, 157, 245), (64, 196, 99), (232, 140, 55), (188, 86, 220))
        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.bbox_pixels)
            color = colors[detection.class_id % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(
                image,
                text,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
