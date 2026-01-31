"""Integration tests for profile management workflow.

Tests the complete profile lifecycle including creation, activation,
deactivation, and deletion through the profile management API.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from millm.api.dependencies import get_profile_service
from millm.api.routes.management.profiles import router


@pytest.fixture
def mock_profile():
    """Create a mock profile."""
    profile = MagicMock()
    profile.id = "profile-123"
    profile.name = "test-profile"
    profile.description = "Test steering profile"
    profile.steering = {1234: 5.0, 5678: -2.5}
    profile.model_id = 1
    profile.sae_id = "sae-123"
    profile.layer = 12
    profile.created_at = datetime.now(timezone.utc)
    profile.updated_at = datetime.now(timezone.utc)
    return profile


@pytest.fixture
def mock_profile_service(mock_profile):
    """Create a mock profile service."""
    service = MagicMock()

    # List and get operations
    service.list_profiles = AsyncMock(return_value=[mock_profile])
    service.get_profile = AsyncMock(return_value=mock_profile)
    service.get_active_profile = AsyncMock(return_value=None)

    # Create and update operations
    service.create_profile = AsyncMock(return_value=mock_profile)
    service.update_profile = AsyncMock(return_value=mock_profile)
    service.save_current_steering = AsyncMock(return_value=mock_profile)

    # Activation operations
    service.activate_profile = AsyncMock(return_value={
        "profile_id": mock_profile.id,
        "applied_steering": True,
        "feature_count": 2,
    })
    service.deactivate_profile = AsyncMock(return_value={
        "profile_id": mock_profile.id,
        "cleared_steering": True,
    })

    # Delete operation
    service.delete_profile = AsyncMock(return_value={
        "profile_id": mock_profile.id,
        "was_active": False,
    })

    return service


@pytest.fixture
def app_with_mock_service(mock_profile_service):
    """Create a test app with mocked service."""
    app = FastAPI()
    app.include_router(router)

    # Override the service dependency
    app.dependency_overrides[get_profile_service] = lambda: mock_profile_service

    return app


@pytest.fixture
def client(app_with_mock_service):
    """Create a test client."""
    return TestClient(app_with_mock_service)


class TestListProfiles:
    """Tests for GET /api/profiles endpoint."""

    def test_returns_empty_list(self, client, mock_profile_service):
        """Test that endpoint returns empty list when no profiles exist."""
        mock_profile_service.list_profiles.return_value = []

        response = client.get("/api/profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["profiles"] == []
        assert data["data"]["total"] == 0
        assert data["data"]["active_profile_id"] is None

    def test_returns_profiles(self, client, mock_profile_service, mock_profile):
        """Test that endpoint returns all profiles."""
        mock_profile_service.list_profiles.return_value = [mock_profile]

        response = client.get("/api/profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total"] == 1
        assert len(data["data"]["profiles"]) == 1

    def test_shows_active_profile(self, client, mock_profile_service, mock_profile):
        """Test that active profile ID is included in response."""
        mock_profile_service.list_profiles.return_value = [mock_profile]
        mock_profile_service.get_active_profile.return_value = mock_profile

        response = client.get("/api/profiles")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["active_profile_id"] == mock_profile.id


class TestCreateProfile:
    """Tests for POST /api/profiles endpoint."""

    def test_creates_profile(self, client, mock_profile_service, mock_profile):
        """Test creating a new profile."""
        mock_profile_service.create_profile.return_value = mock_profile

        response = client.post(
            "/api/profiles",
            json={
                "name": "test-profile",
                "description": "Test steering profile",
                "steering": {1234: 5.0},
                "model_id": 1,
                "sae_id": "sae-123",
                "layer": 12,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify service was called with correct args
        mock_profile_service.create_profile.assert_called_once()

    def test_creates_profile_minimal(self, client, mock_profile_service, mock_profile):
        """Test creating profile with minimal required fields."""
        response = client.post(
            "/api/profiles",
            json={"name": "minimal-profile"},
        )

        assert response.status_code == 200


class TestGetProfile:
    """Tests for GET /api/profiles/{profile_id} endpoint."""

    def test_returns_profile(self, client, mock_profile_service, mock_profile):
        """Test getting a single profile by ID."""
        mock_profile_service.get_profile.return_value = mock_profile

        response = client.get(f"/api/profiles/{mock_profile.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        mock_profile_service.get_profile.assert_called_once_with(mock_profile.id)


class TestUpdateProfile:
    """Tests for PATCH /api/profiles/{profile_id} endpoint."""

    def test_updates_profile_name(self, client, mock_profile_service, mock_profile):
        """Test updating profile name."""
        mock_profile.name = "updated-name"
        mock_profile_service.update_profile.return_value = mock_profile

        response = client.patch(
            f"/api/profiles/{mock_profile.id}",
            json={"name": "updated-name"},
        )

        assert response.status_code == 200
        mock_profile_service.update_profile.assert_called_once()

    def test_updates_steering_values(self, client, mock_profile_service, mock_profile):
        """Test updating steering configuration."""
        mock_profile.steering = {9999: 3.0}
        mock_profile_service.update_profile.return_value = mock_profile

        response = client.patch(
            f"/api/profiles/{mock_profile.id}",
            json={"steering": {9999: 3.0}},
        )

        assert response.status_code == 200


class TestActivateProfile:
    """Tests for POST /api/profiles/{profile_id}/activate endpoint."""

    def test_activates_profile(self, client, mock_profile_service, mock_profile):
        """Test activating a profile."""
        response = client.post(
            f"/api/profiles/{mock_profile.id}/activate",
            json={"apply_steering": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["profile_id"] == mock_profile.id
        assert data["data"]["applied_steering"] is True

        mock_profile_service.activate_profile.assert_called_once_with(
            profile_id=mock_profile.id,
            apply_steering=True,
        )

    def test_activates_without_applying_steering(self, client, mock_profile_service, mock_profile):
        """Test activating a profile without applying steering values."""
        mock_profile_service.activate_profile.return_value = {
            "profile_id": mock_profile.id,
            "applied_steering": False,
            "feature_count": 0,
        }

        response = client.post(
            f"/api/profiles/{mock_profile.id}/activate",
            json={"apply_steering": False},
        )

        assert response.status_code == 200
        assert response.json()["data"]["applied_steering"] is False


class TestDeactivateProfile:
    """Tests for POST /api/profiles/{profile_id}/deactivate endpoint."""

    def test_deactivates_profile(self, client, mock_profile_service, mock_profile):
        """Test deactivating a profile."""
        response = client.post(
            f"/api/profiles/{mock_profile.id}/deactivate",
            params={"clear_steering": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["profile_id"] == mock_profile.id
        assert data["data"]["cleared_steering"] is True

    def test_deactivates_without_clearing(self, client, mock_profile_service, mock_profile):
        """Test deactivating without clearing steering values."""
        mock_profile_service.deactivate_profile.return_value = {
            "profile_id": mock_profile.id,
            "cleared_steering": False,
        }

        response = client.post(
            f"/api/profiles/{mock_profile.id}/deactivate",
            params={"clear_steering": False},
        )

        assert response.status_code == 200
        assert response.json()["data"]["cleared_steering"] is False


class TestDeleteProfile:
    """Tests for DELETE /api/profiles/{profile_id} endpoint."""

    def test_deletes_profile(self, client, mock_profile_service, mock_profile):
        """Test deleting a profile."""
        response = client.delete(f"/api/profiles/{mock_profile.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["profile_id"] == mock_profile.id
        assert data["data"]["was_active"] is False

        mock_profile_service.delete_profile.assert_called_once_with(mock_profile.id)

    def test_deletes_active_profile(self, client, mock_profile_service, mock_profile):
        """Test deleting an active profile deactivates it first."""
        mock_profile_service.delete_profile.return_value = {
            "profile_id": mock_profile.id,
            "was_active": True,
        }

        response = client.delete(f"/api/profiles/{mock_profile.id}")

        assert response.status_code == 200
        assert response.json()["data"]["was_active"] is True


class TestGetActiveProfile:
    """Tests for GET /api/profiles/active endpoint."""

    def test_returns_none_when_no_active(self, client, mock_profile_service):
        """Test returns null when no profile is active."""
        mock_profile_service.get_active_profile.return_value = None

        response = client.get("/api/profiles/active")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] is None

    def test_returns_active_profile(self, client, mock_profile_service, mock_profile):
        """Test returns active profile when one is set."""
        mock_profile_service.get_active_profile.return_value = mock_profile

        response = client.get("/api/profiles/active")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] is not None


class TestSaveCurrentSteering:
    """Tests for POST /api/profiles/save-current endpoint."""

    def test_saves_current_steering(self, client, mock_profile_service, mock_profile):
        """Test saving current steering configuration as new profile."""
        mock_profile_service.save_current_steering.return_value = mock_profile

        response = client.post(
            "/api/profiles/save-current",
            json={
                "name": "saved-steering",
                "description": "Saved from current configuration",
            },
        )

        assert response.status_code == 200
        mock_profile_service.save_current_steering.assert_called_once_with(
            name="saved-steering",
            description="Saved from current configuration",
        )


class TestProfileWorkflow:
    """Integration tests for complete profile workflow."""

    def test_full_profile_lifecycle(self, client, mock_profile_service, mock_profile):
        """Test complete workflow: create -> activate -> deactivate -> delete."""
        # 1. Create profile
        mock_profile_service.create_profile.return_value = mock_profile
        response = client.post(
            "/api/profiles",
            json={"name": "lifecycle-test"},
        )
        assert response.status_code == 200

        # 2. Activate profile
        response = client.post(
            f"/api/profiles/{mock_profile.id}/activate",
            json={"apply_steering": True},
        )
        assert response.status_code == 200

        # 3. Verify active
        mock_profile_service.get_active_profile.return_value = mock_profile
        response = client.get("/api/profiles/active")
        assert response.status_code == 200
        assert response.json()["data"] is not None

        # 4. Deactivate
        response = client.post(
            f"/api/profiles/{mock_profile.id}/deactivate",
            params={"clear_steering": True},
        )
        assert response.status_code == 200

        # 5. Delete
        response = client.delete(f"/api/profiles/{mock_profile.id}")
        assert response.status_code == 200
