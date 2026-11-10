"""Unit and integration tests for service health check endpoints."""

from fastapi.testclient import TestClient


def test_root_health_endpoint(client: TestClient) -> None:
    """Verify that root /health returns 200 OK and valid health payload."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "RelayCore"
    assert "version" in data
    assert "uptime_seconds" in data
    assert "components" in data
    assert "groq_provider" in data["components"]

    # Verify custom middleware headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_api_v1_health_endpoint(client: TestClient) -> None:
    """Verify that /api/v1/health returns matching healthy payload."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["environment"] == "testing"


def test_custom_request_id_header(client: TestClient) -> None:
    """Verify that incoming X-Request-ID is preserved across execution."""
    custom_id = "test-custom-req-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
