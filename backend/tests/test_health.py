from fastapi.testclient import TestClient

from app.main import app


def test_live_endpoint_stays_available_without_weights() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "YOLO-CMFM Inference API"


def test_ready_endpoint_reports_missing_weights() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["model_ready"] is False


def test_detection_endpoint_reports_unavailable_model_without_weights() -> None:
    files = {
        "rgb_image": ("visible.png", b"placeholder", "image/png"),
        "sar_image": ("sar.png", b"placeholder", "image/png"),
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/detections", files=files)

    assert response.status_code == 503
    assert "weights not found" in response.json()["detail"].lower()
