"""Integration tests for health check endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestLivenessCheck:
    """Tests for GET /api/health endpoint."""

    def test_returns_healthy(self, client):
        """Test liveness check returns healthy status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data

    def test_returns_correct_version(self, client):
        """Test liveness check returns correct version."""
        from millm import __version__

        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["version"] == __version__


class TestReadinessCheck:
    """Tests for GET /api/health/ready endpoint."""

    def test_returns_degraded_when_no_model(self, client):
        """Test readiness returns degraded when no model is loaded."""
        response = client.get("/api/health/ready")

        # Should be 200 (degraded is still ready)
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["status"] in ["healthy", "degraded"]
        assert data["model_loaded"] is False

    def test_includes_component_health(self, client):
        """Test readiness check includes component health."""
        response = client.get("/api/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert len(data["components"]) > 0

        # Check component structure
        for component in data["components"]:
            assert "name" in component
            assert "status" in component


class TestDetailedHealthCheck:
    """Tests for GET /api/health/detailed endpoint."""

    def test_returns_detailed_info(self, client):
        """Test detailed health check returns all information."""
        response = client.get("/api/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data
        assert "circuit_breakers" in data
        assert "model_loaded" in data
        assert "sae_attached" in data

    def test_includes_circuit_breaker_status(self, client):
        """Test detailed health includes circuit breaker status."""
        response = client.get("/api/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert len(data["circuit_breakers"]) > 0

        # Check circuit breaker structure
        cb = data["circuit_breakers"][0]
        assert "name" in cb
        assert "state" in cb
        assert "failure_count" in cb
        assert "is_open" in cb


class TestCircuitBreakerStatus:
    """Tests for GET /api/health/circuits endpoint."""

    def test_returns_circuit_breakers(self, client):
        """Test circuit breakers endpoint returns all circuits."""
        response = client.get("/api/health/circuits")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Should include huggingface circuit
        names = [cb["name"] for cb in data]
        assert "huggingface" in names

    def test_circuit_breaker_structure(self, client):
        """Test circuit breaker response structure."""
        response = client.get("/api/health/circuits")

        assert response.status_code == 200
        for cb in response.json():
            assert "name" in cb
            assert "state" in cb
            assert "failure_count" in cb
            assert "is_open" in cb


class TestResetCircuitBreaker:
    """Tests for POST /api/health/circuits/{name}/reset endpoint."""

    def test_resets_existing_circuit(self, client):
        """Test resetting an existing circuit breaker."""
        response = client.post("/api/health/circuits/huggingface/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "huggingface"
        assert data["state"] == "closed"
        assert data["failure_count"] == 0
        assert data["is_open"] is False

    def test_returns_404_for_unknown_circuit(self, client):
        """Test resetting unknown circuit returns 404."""
        response = client.post("/api/health/circuits/unknown/reset")

        assert response.status_code == 404


class TestHealthCheckWorkflow:
    """Integration tests for health check workflow."""

    def test_liveness_then_readiness(self, client):
        """Test typical health check workflow."""
        # Liveness check (kubernetes style)
        liveness = client.get("/api/health")
        assert liveness.status_code == 200
        assert liveness.json()["status"] == "healthy"

        # Readiness check
        readiness = client.get("/api/health/ready")
        assert readiness.status_code == 200
        assert readiness.json()["ready"] is True

    def test_detailed_health_for_debugging(self, client):
        """Test detailed health check for debugging."""
        response = client.get("/api/health/detailed")

        assert response.status_code == 200
        data = response.json()

        # Should have all debugging info
        assert data["version"] is not None
        assert data["timestamp"] is not None
        assert len(data["components"]) > 0
        assert len(data["circuit_breakers"]) > 0


class TestMetrics:
    """Tests for GET /api/health/metrics endpoint."""

    def test_returns_metrics(self, client):
        """Test metrics endpoint returns all expected fields."""
        response = client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.json()

        # Request metrics
        assert "total_requests" in data
        assert "active_requests" in data
        assert "request_errors" in data

        # Model metrics
        assert "model_loaded" in data
        assert "model_load_count" in data
        assert "model_unload_count" in data

        # SAE metrics
        assert "sae_attached" in data

        # Steering metrics
        assert "steering_enabled" in data
        assert "active_features" in data

        # Monitoring metrics
        assert "monitoring_enabled" in data
        assert "monitored_features" in data

        # Circuit breaker metrics
        assert "circuit_breaker_open" in data
        assert "circuit_breaker_trips" in data

        # System metrics
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_metrics_types(self, client):
        """Test metrics have correct types."""
        response = client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.json()

        # Check types
        assert isinstance(data["total_requests"], int)
        assert isinstance(data["active_requests"], int)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))

    def test_uptime_increases(self, client):
        """Test uptime metric increases between calls."""
        import time

        response1 = client.get("/api/health/metrics")
        uptime1 = response1.json()["uptime_seconds"]

        time.sleep(0.1)  # Brief pause

        response2 = client.get("/api/health/metrics")
        uptime2 = response2.json()["uptime_seconds"]

        assert uptime2 >= uptime1


class TestPrometheusMetrics:
    """Tests for GET /api/health/metrics/prometheus endpoint."""

    def test_returns_prometheus_format(self, client):
        """Test prometheus endpoint returns text format."""
        response = client.get("/api/health/metrics/prometheus")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_contains_expected_metrics(self, client):
        """Test prometheus metrics contain expected counters and gauges."""
        response = client.get("/api/health/metrics/prometheus")

        content = response.text

        # Check for expected metric names
        assert "millm_requests_total" in content
        assert "millm_requests_active" in content
        assert "millm_request_errors_total" in content
        assert "millm_model_loaded" in content
        assert "millm_model_loads_total" in content
        assert "millm_circuit_breaker_open" in content
        assert "millm_uptime_seconds" in content

    def test_contains_help_and_type(self, client):
        """Test prometheus metrics contain HELP and TYPE comments."""
        response = client.get("/api/health/metrics/prometheus")

        content = response.text

        # Check for Prometheus metadata
        assert "# HELP millm_requests_total" in content
        assert "# TYPE millm_requests_total counter" in content
        assert "# TYPE millm_requests_active gauge" in content

    def test_valid_prometheus_format(self, client):
        """Test prometheus metrics are in valid format."""
        response = client.get("/api/health/metrics/prometheus")

        content = response.text
        lines = [line for line in content.split("\n") if line and not line.startswith("#")]

        # Each non-empty, non-comment line should be metric_name value
        for line in lines:
            parts = line.split()
            assert len(parts) >= 2, f"Invalid line: {line}"
            # Value should be numeric
            try:
                float(parts[-1])
            except ValueError:
                pytest.fail(f"Non-numeric value in line: {line}")
