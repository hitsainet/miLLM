"""
Unit tests for ProfileService.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import (
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    SAENotAttachedError,
)
from millm.services.profile_service import ProfileService


@pytest.fixture
def mock_repository():
    """Create a mock profile repository."""
    repo = AsyncMock()
    repo.name_exists.return_value = False
    return repo


@pytest.fixture
def mock_sae_service():
    """Create a mock SAE service."""
    service = MagicMock()
    service.get_attachment_status.return_value = MagicMock(
        is_attached=True,
        sae_id="test-sae-123",
        layer=5,
        steering_enabled=False,
    )
    service.get_steering_values.return_value = {0: 1.5, 5: -0.5}
    return service


@pytest.fixture
def profile_service(mock_repository, mock_sae_service):
    """Create a profile service for testing."""
    return ProfileService(
        repository=mock_repository,
        sae_service=mock_sae_service,
    )


class TestProfileServiceList:
    """Tests for listing profiles."""

    @pytest.mark.asyncio
    async def test_list_profiles(self, profile_service, mock_repository):
        """List profiles returns all profiles."""
        mock_profiles = [
            MagicMock(id="prof_1", name="Profile 1"),
            MagicMock(id="prof_2", name="Profile 2"),
        ]
        mock_repository.get_all.return_value = mock_profiles

        result = await profile_service.list_profiles()

        assert len(result) == 2
        mock_repository.get_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_profile_found(self, profile_service, mock_repository):
        """Get profile returns profile when found."""
        mock_profile = MagicMock(id="prof_123", name="Test Profile")
        mock_repository.get.return_value = mock_profile

        result = await profile_service.get_profile("prof_123")

        assert result.id == "prof_123"
        mock_repository.get.assert_called_with("prof_123")

    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, profile_service, mock_repository):
        """Get profile raises error when not found."""
        mock_repository.get.return_value = None

        with pytest.raises(ProfileNotFoundError):
            await profile_service.get_profile("nonexistent")

    @pytest.mark.asyncio
    async def test_get_active_profile(self, profile_service, mock_repository):
        """Get active profile returns active profile."""
        mock_profile = MagicMock(id="prof_active", name="Active Profile")
        mock_repository.get_active.return_value = mock_profile

        result = await profile_service.get_active_profile()

        assert result.id == "prof_active"
        mock_repository.get_active.assert_called_once()


class TestProfileServiceCreate:
    """Tests for creating profiles."""

    @pytest.mark.asyncio
    async def test_create_profile(self, profile_service, mock_repository):
        """Create profile creates a new profile."""
        mock_profile = MagicMock(
            id="prof_new",
            name="New Profile",
            steering={"0": 1.5},
        )
        mock_repository.create.return_value = mock_profile

        result = await profile_service.create_profile(
            name="New Profile",
            steering={0: 1.5},
        )

        assert result.name == "New Profile"
        mock_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_profile_duplicate_name(self, profile_service, mock_repository):
        """Create profile with duplicate name raises error."""
        mock_repository.name_exists.return_value = True

        with pytest.raises(ProfileAlreadyExistsError):
            await profile_service.create_profile(name="Existing Name")

    @pytest.mark.asyncio
    async def test_save_current_steering(
        self, profile_service, mock_repository, mock_sae_service
    ):
        """Save current steering creates profile from current values."""
        mock_profile = MagicMock(
            id="prof_saved",
            name="Saved Profile",
            sae_id="test-sae-123",
            layer=5,
            steering={"0": 1.5, "5": -0.5},
        )
        mock_repository.create.return_value = mock_profile

        result = await profile_service.save_current_steering(
            name="Saved Profile",
            description="Saved from current state",
        )

        assert result.sae_id == "test-sae-123"
        mock_sae_service.get_steering_values.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_current_steering_no_sae(
        self, profile_service, mock_sae_service
    ):
        """Save current steering raises error when no SAE attached."""
        mock_sae_service.get_attachment_status.return_value = MagicMock(
            is_attached=False,
        )

        with pytest.raises(SAENotAttachedError):
            await profile_service.save_current_steering(name="Test")


class TestProfileServiceUpdate:
    """Tests for updating profiles."""

    @pytest.mark.asyncio
    async def test_update_profile(self, profile_service, mock_repository):
        """Update profile updates existing profile."""
        mock_profile = MagicMock(id="prof_123", name="Old Name")
        updated_profile = MagicMock(id="prof_123", name="New Name")
        mock_repository.get.return_value = mock_profile
        mock_repository.update.return_value = updated_profile

        result = await profile_service.update_profile(
            profile_id="prof_123",
            name="New Name",
        )

        assert result.name == "New Name"

    @pytest.mark.asyncio
    async def test_update_profile_not_found(self, profile_service, mock_repository):
        """Update profile raises error when not found."""
        mock_repository.get.return_value = None

        with pytest.raises(ProfileNotFoundError):
            await profile_service.update_profile(
                profile_id="nonexistent",
                name="New Name",
            )

    @pytest.mark.asyncio
    async def test_update_profile_duplicate_name(self, profile_service, mock_repository):
        """Update profile with duplicate name raises error."""
        mock_profile = MagicMock(id="prof_123", name="Old Name")
        mock_repository.get.return_value = mock_profile
        mock_repository.name_exists.return_value = True

        with pytest.raises(ProfileAlreadyExistsError):
            await profile_service.update_profile(
                profile_id="prof_123",
                name="Existing Name",
            )


class TestProfileServiceActivation:
    """Tests for profile activation/deactivation."""

    @pytest.mark.asyncio
    async def test_activate_profile_with_steering(
        self, profile_service, mock_repository, mock_sae_service
    ):
        """Activate profile applies steering values."""
        mock_profile = MagicMock(
            id="prof_123",
            name="Test Profile",
            steering={"0": 1.5, "5": -0.5},
        )
        mock_repository.get.return_value = mock_profile
        mock_repository.set_active.return_value = mock_profile

        result = await profile_service.activate_profile(
            profile_id="prof_123",
            apply_steering=True,
        )

        assert result["applied_steering"] is True
        assert result["feature_count"] == 2
        mock_sae_service.clear_steering.assert_called_once()
        mock_sae_service.set_steering_batch.assert_called_once()
        mock_sae_service.enable_steering.assert_called_with(True)

    @pytest.mark.asyncio
    async def test_activate_profile_without_steering(
        self, profile_service, mock_repository, mock_sae_service
    ):
        """Activate profile without applying steering."""
        mock_profile = MagicMock(
            id="prof_123",
            name="Test Profile",
            steering={"0": 1.5},
        )
        mock_repository.get.return_value = mock_profile
        mock_repository.set_active.return_value = mock_profile

        result = await profile_service.activate_profile(
            profile_id="prof_123",
            apply_steering=False,
        )

        assert result["applied_steering"] is False
        mock_sae_service.set_steering_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_activate_profile_no_sae_attached(
        self, profile_service, mock_repository, mock_sae_service
    ):
        """Activate profile raises error when no SAE attached but steering requested."""
        mock_profile = MagicMock(
            id="prof_123",
            name="Test Profile",
            steering={"0": 1.5},
        )
        mock_repository.get.return_value = mock_profile
        mock_sae_service.get_attachment_status.return_value = MagicMock(
            is_attached=False,
        )

        with pytest.raises(SAENotAttachedError):
            await profile_service.activate_profile(
                profile_id="prof_123",
                apply_steering=True,
            )

    @pytest.mark.asyncio
    async def test_deactivate_profile(
        self, profile_service, mock_repository, mock_sae_service
    ):
        """Deactivate profile clears steering."""
        mock_profile = MagicMock(id="prof_123", name="Test Profile")
        mock_repository.get.return_value = mock_profile
        mock_repository.deactivate.return_value = mock_profile

        result = await profile_service.deactivate_profile(
            profile_id="prof_123",
            clear_steering=True,
        )

        assert result["cleared_steering"] is True
        mock_sae_service.clear_steering.assert_called_once()


class TestProfileServiceDelete:
    """Tests for deleting profiles."""

    @pytest.mark.asyncio
    async def test_delete_profile(self, profile_service, mock_repository):
        """Delete profile removes profile."""
        mock_profile = MagicMock(id="prof_123", name="Test Profile", is_active=False)
        mock_repository.get.return_value = mock_profile
        mock_repository.delete.return_value = True

        result = await profile_service.delete_profile("prof_123")

        assert result["profile_id"] == "prof_123"
        assert result["was_active"] is False
        mock_repository.delete.assert_called_with("prof_123")

    @pytest.mark.asyncio
    async def test_delete_active_profile(self, profile_service, mock_repository):
        """Delete active profile deactivates first."""
        mock_profile = MagicMock(id="prof_123", name="Test Profile", is_active=True)
        mock_repository.get.return_value = mock_profile
        mock_repository.delete.return_value = True

        result = await profile_service.delete_profile("prof_123")

        assert result["was_active"] is True
        mock_repository.deactivate.assert_called_with("prof_123")

    @pytest.mark.asyncio
    async def test_delete_profile_not_found(self, profile_service, mock_repository):
        """Delete profile raises error when not found."""
        mock_repository.get.return_value = None

        with pytest.raises(ProfileNotFoundError):
            await profile_service.delete_profile("nonexistent")
