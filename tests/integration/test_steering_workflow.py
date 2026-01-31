"""Integration tests for steering workflow.

Tests the complete steering flow through the SAE management API,
including setting steering values, batch updates, and clearing.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from millm.api.dependencies import get_sae_service
from millm.api.routes.management.saes import router


@pytest.fixture
def mock_attachment_status():
    """Create a mock attachment status."""
    status = MagicMock()
    status.is_attached = True
    status.sae_id = "test-sae-123"
    status.layer = 12
    status.memory_usage_mb = 256
    status.steering_enabled = False
    status.monitoring_enabled = False
    return status


@pytest.fixture
def mock_sae_service(mock_attachment_status):
    """Create a mock SAE service."""
    service = MagicMock()

    # Default attachment status
    service.get_attachment_status = MagicMock(return_value=mock_attachment_status)

    # Steering methods
    service.get_steering_values = MagicMock(return_value={})
    service.set_steering = MagicMock()
    service.set_steering_batch = MagicMock()
    service.enable_steering = MagicMock()
    service.clear_steering = MagicMock()

    # Async methods
    service.list_saes = AsyncMock(return_value=[])
    service.get_sae = AsyncMock()
    service.delete_sae = AsyncMock()
    service.start_download = AsyncMock()
    service.attach_sae = AsyncMock()
    service.detach_sae = AsyncMock()
    service.check_compatibility = AsyncMock()

    # Monitoring methods
    service.enable_monitoring = MagicMock()

    return service


@pytest.fixture
def app_with_mock_service(mock_sae_service):
    """Create a test app with mocked service."""
    app = FastAPI()
    app.include_router(router)

    # Override the service dependency
    app.dependency_overrides[get_sae_service] = lambda: mock_sae_service

    return app


@pytest.fixture
def client(app_with_mock_service):
    """Create a test client."""
    return TestClient(app_with_mock_service)


class TestGetSteeringStatus:
    """Tests for GET /api/saes/steering endpoint."""

    def test_returns_disabled_when_no_steering(self, client, mock_sae_service, mock_attachment_status):
        """Test that endpoint returns disabled status when no steering is configured."""
        mock_attachment_status.steering_enabled = False
        mock_sae_service.get_steering_values.return_value = {}

        response = client.get("/api/saes/steering")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enabled"] is False
        assert data["data"]["values"] == {}

    def test_returns_enabled_with_values(self, client, mock_sae_service, mock_attachment_status):
        """Test that endpoint returns steering values when configured."""
        mock_attachment_status.steering_enabled = True
        mock_sae_service.get_steering_values.return_value = {1234: 5.0, 5678: -2.5}

        response = client.get("/api/saes/steering")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enabled"] is True
        assert data["data"]["values"] == {1234: 5.0, 5678: -2.5}


class TestSetSteering:
    """Tests for POST /api/saes/steering endpoint."""

    def test_sets_single_feature_steering(self, client, mock_sae_service):
        """Test setting steering for a single feature."""
        mock_sae_service.get_steering_values.return_value = {1234: 5.0}

        response = client.post(
            "/api/saes/steering",
            json={"feature_idx": 1234, "value": 5.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enabled"] is True

        # Verify service calls
        mock_sae_service.set_steering.assert_called_once_with(1234, 5.0)
        mock_sae_service.enable_steering.assert_called_once_with(True)

    def test_sets_negative_steering_value(self, client, mock_sae_service):
        """Test setting negative steering value (suppression)."""
        mock_sae_service.get_steering_values.return_value = {5678: -3.0}

        response = client.post(
            "/api/saes/steering",
            json={"feature_idx": 5678, "value": -3.0}
        )

        assert response.status_code == 200
        mock_sae_service.set_steering.assert_called_once_with(5678, -3.0)


class TestBatchSteering:
    """Tests for POST /api/saes/steering/batch endpoint."""

    def test_sets_multiple_features(self, client, mock_sae_service):
        """Test setting multiple features at once."""
        mock_sae_service.get_steering_values.return_value = {
            1234: 5.0,
            5678: -2.5,
            9012: 1.0
        }

        response = client.post(
            "/api/saes/steering/batch",
            json={"steering": {1234: 5.0, 5678: -2.5, 9012: 1.0}}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["enabled"] is True
        assert len(data["data"]["values"]) == 3

        # Verify service calls
        mock_sae_service.set_steering_batch.assert_called_once()
        mock_sae_service.enable_steering.assert_called_once_with(True)

    def test_batch_with_empty_dict(self, client, mock_sae_service):
        """Test batch steering with empty dictionary."""
        mock_sae_service.get_steering_values.return_value = {}

        response = client.post(
            "/api/saes/steering/batch",
            json={"steering": {}}
        )

        assert response.status_code == 200
        mock_sae_service.set_steering_batch.assert_called_once_with({})


class TestToggleSteering:
    """Tests for POST /api/saes/steering/enable endpoint."""

    def test_enables_steering(self, client, mock_sae_service, mock_attachment_status):
        """Test enabling steering."""
        mock_attachment_status.steering_enabled = True
        mock_sae_service.get_steering_values.return_value = {1234: 5.0}

        response = client.post(
            "/api/saes/steering/enable",
            params={"enabled": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is True

        mock_sae_service.enable_steering.assert_called_once_with(True)

    def test_disables_steering(self, client, mock_sae_service, mock_attachment_status):
        """Test disabling steering (preserves values)."""
        mock_attachment_status.steering_enabled = False
        mock_sae_service.get_steering_values.return_value = {1234: 5.0}

        response = client.post(
            "/api/saes/steering/enable",
            params={"enabled": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is False
        # Values should be preserved
        assert data["data"]["values"] == {1234: 5.0}


class TestClearSteering:
    """Tests for DELETE /api/saes/steering endpoints."""

    def test_clears_single_feature(self, client, mock_sae_service, mock_attachment_status):
        """Test clearing steering for a single feature."""
        mock_attachment_status.steering_enabled = True
        mock_sae_service.get_steering_values.return_value = {5678: -2.5}

        response = client.delete("/api/saes/steering/1234")

        assert response.status_code == 200
        mock_sae_service.clear_steering.assert_called_once_with(1234)

    def test_clears_all_steering(self, client, mock_sae_service):
        """Test clearing all steering values."""
        response = client.delete("/api/saes/steering")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is False
        assert data["data"]["values"] == {}

        mock_sae_service.clear_steering.assert_called_once_with()


class TestSteeringWorkflow:
    """Integration tests for complete steering workflow."""

    def test_full_steering_workflow(self, client, mock_sae_service, mock_attachment_status):
        """Test complete workflow: set -> enable -> disable -> clear."""
        # 1. Set initial steering
        mock_sae_service.get_steering_values.return_value = {1234: 5.0}
        response = client.post(
            "/api/saes/steering",
            json={"feature_idx": 1234, "value": 5.0}
        )
        assert response.status_code == 200

        # 2. Add more features via batch
        mock_sae_service.get_steering_values.return_value = {1234: 5.0, 5678: -2.0}
        response = client.post(
            "/api/saes/steering/batch",
            json={"steering": {1234: 5.0, 5678: -2.0}}
        )
        assert response.status_code == 200

        # 3. Check status
        mock_attachment_status.steering_enabled = True
        response = client.get("/api/saes/steering")
        assert response.status_code == 200
        assert response.json()["data"]["enabled"] is True

        # 4. Disable steering
        response = client.post(
            "/api/saes/steering/enable",
            params={"enabled": False}
        )
        assert response.status_code == 200

        # 5. Clear all steering
        response = client.delete("/api/saes/steering")
        assert response.status_code == 200
        assert response.json()["data"]["enabled"] is False
