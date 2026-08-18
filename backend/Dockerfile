ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/YOLOv11-CMFM ./third_party/YOLOv11-CMFM
COPY backend/requirements.txt ./backend/requirements.txt

RUN python -m pip install -e ./third_party/YOLOv11-CMFM \
    && python -m pip install -r ./backend/requirements.txt

COPY backend ./backend
COPY weights/README.md ./weights/README.md

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

