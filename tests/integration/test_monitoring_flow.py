"""Integration tests for monitoring flow.

Tests the complete monitoring workflow through the monitoring API,
including configuration, history retrieval, and statistics.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from millm.api.dependencies import get_monitoring_service
from millm.api.routes.management.monitoring import router


@pytest.fixture
def mock_monitoring_state():
    """Create a mock monitoring state."""
    return {
        "enabled": True,
        "sae_attached": True,
        "sae_id": "test-sae-123",
        "monitored_features": [1234, 5678, 9012],
        "history_size": 1000,
        "history_count": 50,
    }


@pytest.fixture
def mock_activation_records():
    """Create mock activation records."""
    return [
        {
            "timestamp": "2024-01-15T12:00:01Z",
            "request_id": "req-001",
            "token_position": 0,
            "activations": {1234: 0.82, 5678: 0.45},
            "top_k": [(1234, 0.82), (5678, 0.45)],
        },
        {
            "timestamp": "2024-01-15T12:00:02Z",
            "request_id": "req-001",
            "token_position": 1,
            "activations": {1234: 0.91, 5678: 0.52},
            "top_k": [(1234, 0.91), (5678, 0.52)],
        },
    ]


@pytest.fixture
def mock_statistics():
    """Create mock feature statistics."""
    return {
        "features": [
            {
                "feature_idx": 1234,
                "count": 100,
                "mean": 0.75,
                "std": 0.12,
                "min": 0.45,
                "max": 0.98,
                "active_ratio": 0.85,
            },
            {
                "feature_idx": 5678,
                "count": 100,
                "mean": 0.42,
                "std": 0.08,
                "min": 0.22,
                "max": 0.65,
                "active_ratio": 0.72,
            },
        ],
        "total_activations": 100,
        "since": "2024-01-15T12:00:00Z",
    }


@pytest.fixture
def mock_monitoring_service(mock_monitoring_state, mock_activation_records, mock_statistics):
    """Create a mock monitoring service."""
    service = MagicMock()

    # State operations
    service.get_state = MagicMock(return_value=mock_monitoring_state)
    service.configure = MagicMock()
    service.set_enabled = MagicMock()

    # History operations
    service.get_history = MagicMock(return_value=mock_activation_records)
    service.clear_history = MagicMock(return_value=50)

    # Statistics operations
    service.get_statistics = MagicMock(return_value=mock_statistics)
    service.reset_statistics = MagicMock(return_value=2)
    service.get_top_features = MagicMock(return_value=mock_statistics["features"])

    return service


@pytest.fixture
def app_with_mock_service(mock_monitoring_service):
    """Create a test app with mocked service."""
    app = FastAPI()
    app.include_router(router)

    # Override the service dependency
    app.dependency_overrides[get_monitoring_service] = lambda: mock_monitoring_service

    return app


@pytest.fixture
def client(app_with_mock_service):
    """Create a test client."""
    return TestClient(app_with_mock_service)


class TestGetMonitoringState:
    """Tests for GET /api/monitoring endpoint."""

    def test_returns_monitoring_state(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test that endpoint returns current monitoring state."""
        response = client.get("/api/monitoring")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enabled"] is True
        assert data["data"]["sae_attached"] is True
        assert data["data"]["monitored_features"] == [1234, 5678, 9012]
        assert data["data"]["history_size"] == 1000
        assert data["data"]["history_count"] == 50

    def test_returns_disabled_state(self, client, mock_monitoring_service):
        """Test returns disabled state correctly."""
        mock_monitoring_service.get_state.return_value = {
            "enabled": False,
            "sae_attached": False,
            "sae_id": None,
            "monitored_features": [],
            "history_size": 1000,
            "history_count": 0,
        }

        response = client.get("/api/monitoring")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is False
        assert data["data"]["sae_attached"] is False


class TestConfigureMonitoring:
    """Tests for POST /api/monitoring/configure endpoint."""

    def test_configures_monitoring(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test configuring monitoring parameters."""
        response = client.post(
            "/api/monitoring/configure",
            json={
                "enabled": True,
                "features": [1234, 5678],
                "history_size": 500,
            },
        )

        assert response.status_code == 200
        mock_monitoring_service.configure.assert_called_once_with(
            enabled=True,
            features=[1234, 5678],
            history_size=500,
        )

    def test_configures_with_partial_params(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test configuring with only some parameters."""
        response = client.post(
            "/api/monitoring/configure",
            json={"enabled": True},
        )

        assert response.status_code == 200
        mock_monitoring_service.configure.assert_called_once()


class TestToggleMonitoring:
    """Tests for POST /api/monitoring/enable endpoint."""

    def test_enables_monitoring(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test enabling monitoring."""
        response = client.post(
            "/api/monitoring/enable",
            json={"enabled": True},
        )

        assert response.status_code == 200
        mock_monitoring_service.set_enabled.assert_called_once_with(True)

    def test_disables_monitoring(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test disabling monitoring."""
        mock_monitoring_state["enabled"] = False

        response = client.post(
            "/api/monitoring/enable",
            json={"enabled": False},
        )

        assert response.status_code == 200
        mock_monitoring_service.set_enabled.assert_called_once_with(False)


class TestGetActivationHistory:
    """Tests for GET /api/monitoring/history endpoint."""

    def test_returns_history(self, client, mock_monitoring_service, mock_activation_records):
        """Test getting activation history."""
        response = client.get("/api/monitoring/history")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total"] == 2
        assert len(data["data"]["records"]) == 2

    def test_filters_by_request_id(self, client, mock_monitoring_service):
        """Test filtering history by request ID."""
        mock_monitoring_service.get_history.return_value = [
            {
                "timestamp": "2024-01-15T12:00:01Z",
                "request_id": "req-001",
                "token_position": 0,
                "activations": {1234: 0.82},
                "top_k": [(1234, 0.82)],
            }
        ]

        response = client.get("/api/monitoring/history", params={"request_id": "req-001"})

        assert response.status_code == 200
        mock_monitoring_service.get_history.assert_called_once_with(
            limit=50,
            request_id="req-001",
        )

    def test_limits_history_results(self, client, mock_monitoring_service):
        """Test limiting history results."""
        response = client.get("/api/monitoring/history", params={"limit": 10})

        assert response.status_code == 200
        mock_monitoring_service.get_history.assert_called_once_with(
            limit=10,
            request_id=None,
        )


class TestClearHistory:
    """Tests for DELETE /api/monitoring/history endpoint."""

    def test_clears_history(self, client, mock_monitoring_service):
        """Test clearing activation history."""
        response = client.delete("/api/monitoring/history")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["cleared"] == 50
        assert "Cleared 50" in data["data"]["message"]


class TestGetStatistics:
    """Tests for GET /api/monitoring/statistics endpoint."""

    def test_returns_statistics(self, client, mock_monitoring_service, mock_statistics):
        """Test getting feature statistics."""
        response = client.get("/api/monitoring/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_activations"] == 100
        assert len(data["data"]["features"]) == 2

    def test_filters_by_features(self, client, mock_monitoring_service):
        """Test filtering statistics by feature indices."""
        response = client.get("/api/monitoring/statistics", params={"features": "1234,5678"})

        assert response.status_code == 200
        mock_monitoring_service.get_statistics.assert_called_once_with(
            feature_indices=[1234, 5678],
        )


class TestResetStatistics:
    """Tests for DELETE /api/monitoring/statistics endpoint."""

    def test_resets_statistics(self, client, mock_monitoring_service):
        """Test resetting feature statistics."""
        response = client.delete("/api/monitoring/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["cleared"] == 2
        assert "Reset statistics for 2" in data["data"]["message"]


class TestGetTopFeatures:
    """Tests for POST /api/monitoring/statistics/top endpoint."""

    def test_returns_top_features_by_mean(self, client, mock_monitoring_service):
        """Test getting top features by mean activation."""
        response = client.post(
            "/api/monitoring/statistics/top",
            json={"k": 10, "metric": "mean"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["metric"] == "mean"
        assert data["data"]["k"] == 10

        mock_monitoring_service.get_top_features.assert_called_once_with(
            k=10,
            metric="mean",
        )

    def test_returns_top_features_by_active_ratio(self, client, mock_monitoring_service):
        """Test getting top features by active ratio."""
        response = client.post(
            "/api/monitoring/statistics/top",
            json={"k": 5, "metric": "active_ratio"},
        )

        assert response.status_code == 200
        mock_monitoring_service.get_top_features.assert_called_once_with(
            k=5,
            metric="active_ratio",
        )


class TestMonitoringWorkflow:
    """Integration tests for complete monitoring workflow."""

    def test_full_monitoring_workflow(self, client, mock_monitoring_service, mock_monitoring_state):
        """Test complete workflow: configure -> enable -> get history -> reset."""
        # 1. Configure monitoring
        response = client.post(
            "/api/monitoring/configure",
            json={
                "enabled": True,
                "features": [1234, 5678],
                "history_size": 500,
            },
        )
        assert response.status_code == 200

        # 2. Verify state
        response = client.get("/api/monitoring")
        assert response.status_code == 200
        assert response.json()["data"]["enabled"] is True

        # 3. Get history
        response = client.get("/api/monitoring/history", params={"limit": 100})
        assert response.status_code == 200

        # 4. Get statistics
        response = client.get("/api/monitoring/statistics")
        assert response.status_code == 200

        # 5. Get top features
        response = client.post(
            "/api/monitoring/statistics/top",
            json={"k": 10, "metric": "mean"},
        )
        assert response.status_code == 200

        # 6. Reset statistics
        response = client.delete("/api/monitoring/statistics")
        assert response.status_code == 200

        # 7. Clear history
        response = client.delete("/api/monitoring/history")
        assert response.status_code == 200

        # 8. Disable monitoring
        response = client.post(
            "/api/monitoring/enable",
            json={"enabled": False},
        )
        assert response.status_code == 200

    def test_disabled_monitoring_returns_empty(self, client, mock_monitoring_service):
        """Test that disabled monitoring returns empty state."""
        mock_monitoring_service.get_state.return_value = {
            "enabled": False,
            "sae_attached": False,
            "sae_id": None,
            "monitored_features": [],
            "history_size": 1000,
            "history_count": 0,
        }
        mock_monitoring_service.get_history.return_value = []
        mock_monitoring_service.get_statistics.return_value = {
            "features": [],
            "total_activations": 0,
            "since": "2024-01-15T12:00:00Z",
        }

        # Get state
        response = client.get("/api/monitoring")
        assert response.status_code == 200
        assert response.json()["data"]["enabled"] is False

        # Get history
        response = client.get("/api/monitoring/history")
        assert response.status_code == 200
        assert response.json()["data"]["total"] == 0

        # Get statistics
        response = client.get("/api/monitoring/statistics")
        assert response.status_code == 200
        assert len(response.json()["data"]["features"]) == 0
