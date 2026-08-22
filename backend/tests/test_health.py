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


def test_cors_allows_local_frontend_on_dynamic_port() -> None:
    origin = "http://localhost:3001"
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"Origin": origin},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_does_not_allow_untrusted_origin() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"Origin": "https://example.invalid"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
