# Model weights

Place the trained checkpoint at `weights/best.pt`, or set `MODEL_WEIGHTS` to an
absolute checkpoint path. Weight files are intentionally ignored by Git.

The API remains live without a checkpoint, while `/api/v1/health/ready` returns
HTTP 503 until the model can be loaded.

