"""Unit tests for SAEService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import (
    ModelNotLoadedError,
    SAEAlreadyAttachedError,
    SAEIncompatibleError,
    SAENotAttachedError,
    SAENotFoundError,
)
from millm.db.models.sae import SAE, SAEStatus
from millm.services.sae_service import (
    AttachedSAEState,
    AttachmentStatus,
    CompatibilityResult,
    DownloadResult,
    SAEService,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository():
    """Create a mock SAE repository."""
    repo = MagicMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.create_downloading = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    repo.update_downloaded = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.create_attachment = AsyncMock()
    repo.deactivate_attachment = AsyncMock()
    return repo


@pytest.fixture
def mock_emitter():
    """Create a mock progress emitter."""
    emitter = MagicMock()
    emitter.emit_sae_download_progress = AsyncMock()
    emitter.emit_sae_download_complete = AsyncMock()
    emitter.emit_sae_download_error = AsyncMock()
    return emitter


@pytest.fixture
def mock_loaded_sae():
    """Create a mock LoadedSAE instance."""
    sae = MagicMock()
    sae.estimate_memory_mb.return_value = 256.0
    sae.is_steering_enabled = False
    sae.is_monitoring_enabled = False
    sae.set_steering = MagicMock()
    sae.set_steering_batch = MagicMock()
    sae.clear_steering = MagicMock()
    sae.enable_steering = MagicMock()
    sae.get_steering_values = MagicMock(return_value={1234: 5.0})
    sae.enable_monitoring = MagicMock()
    sae.get_last_feature_activations = MagicMock(return_value=None)
    sae.to_cpu = MagicMock()
    return sae


@pytest.fixture
def sample_sae():
    """Create a sample SAE ORM object for testing."""
    sae = MagicMock(spec=SAE)
    sae.id = "jbloom--gemma-2-2b-res-jb"
    sae.repository_id = "jbloom/gemma-2-2b-res-jb"
    sae.revision = "main"
    sae.name = "gemma-2-2b-res-jb"
    sae.format = "saelens"
    sae.d_in = 2304
    sae.d_sae = 16384
    sae.trained_on = "google/gemma-2-2b"
    sae.trained_layer = 12
    sae.file_size_bytes = 268_435_456
    sae.cache_path = "/data/saes/jbloom--gemma-2-2b-res-jb"
    sae.status = SAEStatus.CACHED
    sae.error_message = None
    return sae


@pytest.fixture(autouse=True)
def reset_sae_state():
    """Reset the AttachedSAEState singleton before each test."""
    state = AttachedSAEState()
    # Force clear without calling hook_handle.remove()
    state._attached_sae = None
    state._attached_sae_id = None
    state._attached_layer = None
    state._hook_handle = None
    yield
    # Cleanup after test
    state._attached_sae = None
    state._attached_sae_id = None
    state._attached_layer = None
    state._hook_handle = None


@pytest.fixture
def service(mock_repository, mock_emitter):
    """Create an SAEService with mock dependencies."""
    with patch("millm.services.sae_service.SAEDownloader") as MockDownloader, \
         patch("millm.services.sae_service.SAELoader") as MockLoader, \
         patch("millm.services.sae_service.SAEHooker") as MockHooker:

        mock_downloader_instance = MagicMock()
        mock_downloader_instance.generate_sae_id = MagicMock(
            return_value="jbloom--gemma-2-2b-res-jb"
        )
        mock_downloader_instance.download = AsyncMock(
            return_value="/data/saes/jbloom--gemma-2-2b-res-jb"
        )
        mock_downloader_instance.list_repository_files = AsyncMock(
            return_value={"files": []}
        )
        mock_downloader_instance.delete = AsyncMock(return_value=256.0)
        MockDownloader.return_value = mock_downloader_instance

        mock_loader_instance = MagicMock()
        MockLoader.return_value = mock_loader_instance

        mock_hooker_instance = MagicMock()
        mock_hooker_instance.get_layer_count = MagicMock(return_value=26)
        MockHooker.return_value = mock_hooker_instance

        svc = SAEService(
            repository=mock_repository,
            cache_dir="/data/saes",
            emitter=mock_emitter,
        )

        # Expose internal mocks for assertions
        svc._mock_downloader = mock_downloader_instance
        svc._mock_loader = mock_loader_instance
        svc._mock_hooker = mock_hooker_instance

        yield svc


# =============================================================================
# AttachedSAEState Tests
# =============================================================================


class TestAttachedSAEState:
    """Tests for the AttachedSAEState singleton."""

    def test_singleton_pattern(self, reset_sae_state):
        """Test that AttachedSAEState follows singleton pattern."""
        state1 = AttachedSAEState()
        state2 = AttachedSAEState()
        assert state1 is state2

    def test_initial_state_not_attached(self, reset_sae_state):
        """Test that initial state has no SAE attached."""
        state = AttachedSAEState()
        assert state.is_attached is False
        assert state.attached_sae is None
        assert state.attached_sae_id is None
        assert state.attached_layer is None
        assert state.hook_handle is None

    def test_set_and_read_properties(self, reset_sae_state, mock_loaded_sae):
        """Test setting and reading attached SAE state."""
        state = AttachedSAEState()
        mock_handle = MagicMock()

        state.set(mock_loaded_sae, "test-sae-id", 12, mock_handle)

        assert state.is_attached is True
        assert state.attached_sae is mock_loaded_sae
        assert state.attached_sae_id == "test-sae-id"
        assert state.attached_layer == 12
        assert state.hook_handle is mock_handle

    def test_clear_resets_state(self, reset_sae_state, mock_loaded_sae):
        """Test that clear resets all state to None and removes hook."""
        state = AttachedSAEState()
        mock_handle = MagicMock()
        state.set(mock_loaded_sae, "test-sae-id", 12, mock_handle)

        state.clear()

        assert state.is_attached is False
        assert state.attached_sae is None
        assert state.attached_sae_id is None
        assert state.attached_layer is None
        assert state.hook_handle is None
        mock_handle.remove.assert_called_once()

    def test_clear_handles_hook_remove_error(self, reset_sae_state, mock_loaded_sae):
        """Test that clear gracefully handles errors when removing hook."""
        state = AttachedSAEState()
        mock_handle = MagicMock()
        mock_handle.remove.side_effect = RuntimeError("Hook already removed")
        state.set(mock_loaded_sae, "test-sae-id", 12, mock_handle)

        # Should not raise
        state.clear()

        assert state.is_attached is False


# =============================================================================
# Listing Tests
# =============================================================================


class TestSAEServiceListing:
    """Tests for list_saes and get_sae methods."""

    @pytest.mark.asyncio
    async def test_list_saes_returns_all(self, service, mock_repository, sample_sae):
        """Test that list_saes returns all SAEs from repository."""
        mock_repository.get_all.return_value = [sample_sae]

        result = await service.list_saes()

        assert len(result) == 1
        assert result[0].id == "jbloom--gemma-2-2b-res-jb"
        mock_repository.get_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_saes_returns_empty_list(self, service, mock_repository):
        """Test that list_saes returns empty list when no SAEs exist."""
        mock_repository.get_all.return_value = []

        result = await service.list_saes()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_sae_found(self, service, mock_repository, sample_sae):
        """Test that get_sae returns SAE when found."""
        mock_repository.get.return_value = sample_sae

        result = await service.get_sae("jbloom--gemma-2-2b-res-jb")

        assert result.id == "jbloom--gemma-2-2b-res-jb"
        mock_repository.get.assert_called_once_with("jbloom--gemma-2-2b-res-jb")

    @pytest.mark.asyncio
    async def test_get_sae_not_found_raises(self, service, mock_repository):
        """Test that get_sae raises SAENotFoundError when not found."""
        mock_repository.get.return_value = None

        with pytest.raises(SAENotFoundError) as exc_info:
            await service.get_sae("nonexistent-id")

        assert "nonexistent-id" in str(exc_info.value.message)
        assert exc_info.value.details["sae_id"] == "nonexistent-id"


# =============================================================================
# Compatibility Tests
# =============================================================================


class TestSAEServiceCompatibility:
    """Tests for check_compatibility method."""

    @pytest.mark.asyncio
    async def test_compatible_sae(self, service, mock_repository, sample_sae):
        """Test check_compatibility returns compatible for matching SAE."""
        mock_repository.get.return_value = sample_sae

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2304
        mock_model.config.name_or_path = "google/gemma-2-2b"

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            mock_state.current = mock_loaded
            MockModelState.return_value = mock_state

            result = await service.check_compatibility(sample_sae.id, 12)

        assert result.compatible is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_incompatible_dimensions(self, service, mock_repository, sample_sae):
        """Test check_compatibility detects dimension mismatch."""
        mock_repository.get.return_value = sample_sae

        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096  # Mismatch: SAE d_in is 2304
        mock_model.config.name_or_path = "google/gemma-2-2b"

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            mock_state.current = mock_loaded
            MockModelState.return_value = mock_state

            result = await service.check_compatibility(sample_sae.id, 12)

        assert result.compatible is False
        assert any("Dimension mismatch" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, service, mock_repository, sample_sae):
        """Test check_compatibility detects layer out of range."""
        mock_repository.get.return_value = sample_sae

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2304
        mock_model.config.name_or_path = "google/gemma-2-2b"

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        # Hooker returns 26 layers by default, so layer 30 is out of range
        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            mock_state.current = mock_loaded
            MockModelState.return_value = mock_state

            result = await service.check_compatibility(sample_sae.id, 30)

        assert result.compatible is False
        assert any("out of range" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_compatibility_no_model_loaded(self, service, mock_repository, sample_sae):
        """Test check_compatibility raises when no model is loaded."""
        mock_repository.get.return_value = sample_sae

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = False
            MockModelState.return_value = mock_state

            with pytest.raises(ModelNotLoadedError):
                await service.check_compatibility(sample_sae.id, 12)


# =============================================================================
# Attachment Tests
# =============================================================================


class TestSAEServiceAttachment:
    """Tests for attach_sae method."""

    @pytest.mark.asyncio
    async def test_attach_sae_success(
        self, service, mock_repository, sample_sae, mock_loaded_sae
    ):
        """Test successful SAE attachment to loaded model."""
        mock_repository.get.return_value = sample_sae

        mock_model = MagicMock()
        mock_model.config.hidden_size = 2304
        mock_model.config.name_or_path = "google/gemma-2-2b"
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_handle = MagicMock()
        service._mock_hooker.install.return_value = mock_handle
        service._mock_loader.load.return_value = mock_loaded_sae

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState, \
             patch("millm.services.sae_service.torch") as mock_torch:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            mock_state.current = mock_loaded
            mock_state.loaded_model_id = 1
            MockModelState.return_value = mock_state
            mock_torch.cuda.is_available.return_value = False

            result = await service.attach_sae(sample_sae.id, 12)

        assert result["status"] == "attached"
        assert result["sae_id"] == sample_sae.id
        assert result["layer"] == 12
        assert result["memory_usage_mb"] == 256

        service._mock_loader.load.assert_called_once()
        service._mock_hooker.install.assert_called_once_with(
            mock_model, 12, mock_loaded_sae
        )
        mock_repository.update_status.assert_called_once_with(
            sample_sae.id, SAEStatus.ATTACHED
        )
        mock_repository.create_attachment.assert_called_once()

    @pytest.mark.asyncio
    async def test_attach_sae_no_model_loaded(self, service, mock_repository, sample_sae):
        """Test attach_sae raises when no model is loaded."""
        mock_repository.get.return_value = sample_sae

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = False
            MockModelState.return_value = mock_state

            with pytest.raises(ModelNotLoadedError):
                await service.attach_sae(sample_sae.id, 12)

    @pytest.mark.asyncio
    async def test_attach_sae_already_attached(
        self, service, mock_repository, sample_sae, mock_loaded_sae
    ):
        """Test attach_sae raises when an SAE is already attached."""
        mock_repository.get.return_value = sample_sae

        # Pre-attach an SAE in the singleton state
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "existing-sae-id", 6, MagicMock())

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            MockModelState.return_value = mock_state

            with pytest.raises(SAEAlreadyAttachedError) as exc_info:
                await service.attach_sae(sample_sae.id, 12)

        assert "existing-sae-id" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_attach_sae_incompatible(self, service, mock_repository, sample_sae):
        """Test attach_sae raises when SAE is incompatible with model."""
        mock_repository.get.return_value = sample_sae

        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096  # Dimension mismatch
        mock_model.config.name_or_path = "different/model"
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        with patch("millm.services.sae_service.LoadedModelState") as MockModelState:
            mock_state = MagicMock()
            mock_state.is_loaded = True
            mock_state.current = mock_loaded
            MockModelState.return_value = mock_state

            with pytest.raises(SAEIncompatibleError):
                await service.attach_sae(sample_sae.id, 12)


# =============================================================================
# Detachment Tests
# =============================================================================


class TestSAEServiceDetachment:
    """Tests for detach_sae method."""

    @pytest.mark.asyncio
    async def test_detach_sae_success(
        self, service, mock_repository, sample_sae, mock_loaded_sae
    ):
        """Test successful SAE detachment."""
        mock_repository.get.return_value = sample_sae
        mock_handle = MagicMock()

        # Pre-attach the SAE
        state = AttachedSAEState()
        state.set(mock_loaded_sae, sample_sae.id, 12, mock_handle)

        with patch("millm.services.sae_service.torch") as mock_torch, \
             patch("millm.api.dependencies.get_inference_service", side_effect=Exception("not available")):
            mock_torch.cuda.is_available.return_value = False

            result = await service.detach_sae(sample_sae.id)

        assert result["status"] == "detached"
        assert result["sae_id"] == sample_sae.id
        assert result["memory_freed_mb"] == 256

        service._mock_hooker.remove.assert_called_once_with(mock_handle)
        mock_loaded_sae.to_cpu.assert_called_once()
        mock_repository.update_status.assert_called_once_with(
            sample_sae.id, SAEStatus.CACHED
        )
        mock_repository.deactivate_attachment.assert_called_once_with(sample_sae.id)

        # Verify singleton is cleared
        assert state.is_attached is False

    @pytest.mark.asyncio
    async def test_detach_sae_not_attached_raises(
        self, service, mock_repository, sample_sae
    ):
        """Test detach_sae raises when SAE is not attached."""
        mock_repository.get.return_value = sample_sae

        with pytest.raises(SAENotAttachedError) as exc_info:
            await service.detach_sae(sample_sae.id)

        assert sample_sae.id in str(exc_info.value.message)


# =============================================================================
# Steering Tests
# =============================================================================


class TestSAEServiceSteering:
    """Tests for steering delegation methods."""

    def test_set_steering_delegates(self, service, mock_loaded_sae):
        """Test set_steering delegates to LoadedSAE."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        service.set_steering(1234, 5.0)

        mock_loaded_sae.set_steering.assert_called_once_with(1234, 5.0)

    def test_set_steering_not_attached_raises(self, service):
        """Test set_steering raises when no SAE attached."""
        with pytest.raises(SAENotAttachedError):
            service.set_steering(1234, 5.0)

    def test_set_steering_batch_delegates(self, service, mock_loaded_sae):
        """Test set_steering_batch delegates to LoadedSAE."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        batch = {1234: 5.0, 892: -2.0}
        service.set_steering_batch(batch)

        mock_loaded_sae.set_steering_batch.assert_called_once_with(batch)

    def test_get_steering_values_returns_dict(self, service, mock_loaded_sae):
        """Test get_steering_values returns steering dict from LoadedSAE."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        result = service.get_steering_values()

        assert result == {1234: 5.0}
        mock_loaded_sae.get_steering_values.assert_called_once()

    def test_enable_steering_delegates(self, service, mock_loaded_sae):
        """Test enable_steering delegates to LoadedSAE."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        service.enable_steering(True)
        mock_loaded_sae.enable_steering.assert_called_with(True)

        service.enable_steering(False)
        mock_loaded_sae.enable_steering.assert_called_with(False)

    def test_clear_steering_specific_feature(self, service, mock_loaded_sae):
        """Test clear_steering with specific feature index."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        service.clear_steering(1234)

        mock_loaded_sae.clear_steering.assert_called_once_with(1234)

    def test_clear_steering_all(self, service, mock_loaded_sae):
        """Test clear_steering clears all when no index given."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        service.clear_steering()

        mock_loaded_sae.clear_steering.assert_called_once_with(None)


# =============================================================================
# Download Tests
# =============================================================================


class TestSAEServiceDownload:
    """Tests for start_download and cancel_download methods."""

    @pytest.mark.asyncio
    async def test_start_download_creates_record(self, service, mock_repository):
        """Test start_download creates DB record and starts task."""
        mock_repository.get.return_value = None  # No existing SAE

        result = await service.start_download(
            repository_id="jbloom/gemma-2-2b-res-jb",
            revision="main",
        )

        assert result.status == "downloading"
        assert result.sae_id == "jbloom--gemma-2-2b-res-jb"
        assert "Download started" in result.message
        mock_repository.create_downloading.assert_called_once_with(
            sae_id="jbloom--gemma-2-2b-res-jb",
            repository_id="jbloom/gemma-2-2b-res-jb",
            revision="main",
            cache_path="",
        )
        # Task should be tracked
        assert "jbloom--gemma-2-2b-res-jb" in service._active_downloads

    @pytest.mark.asyncio
    async def test_start_download_returns_cached(self, service, mock_repository, sample_sae):
        """Test start_download returns early for already-cached SAE."""
        sample_sae.status = SAEStatus.CACHED
        mock_repository.get.return_value = sample_sae

        result = await service.start_download(
            repository_id="jbloom/gemma-2-2b-res-jb",
        )

        assert result.status == "cached"
        assert "already downloaded" in result.message
        mock_repository.create_downloading.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_download_returns_already_downloading(
        self, service, mock_repository, sample_sae
    ):
        """Test start_download returns early for in-progress download."""
        sample_sae.status = SAEStatus.DOWNLOADING
        mock_repository.get.return_value = sample_sae

        result = await service.start_download(
            repository_id="jbloom/gemma-2-2b-res-jb",
        )

        assert result.status == "already_downloading"
        mock_repository.create_downloading.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_download_cancels_task(
        self, service, mock_repository, sample_sae
    ):
        """Test cancel_download cancels an active download."""
        sample_sae.status = SAEStatus.DOWNLOADING
        # First call returns downloading, second call returns updated SAE
        cancelled_sae = MagicMock(spec=SAE)
        cancelled_sae.id = sample_sae.id
        cancelled_sae.status = SAEStatus.ERROR
        mock_repository.get.side_effect = [sample_sae, cancelled_sae]

        # Add a mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        service._active_downloads[sample_sae.id] = mock_task

        result = await service.cancel_download(sample_sae.id)

        mock_task.cancel.assert_called_once()
        mock_repository.update_status.assert_called_once_with(
            sae_id=sample_sae.id,
            status=SAEStatus.ERROR,
            error_message="Download cancelled by user",
        )
        assert sample_sae.id in service._cancelled_downloads


# =============================================================================
# Monitoring Tests
# =============================================================================


class TestSAEServiceMonitoring:
    """Tests for monitoring delegation methods."""

    def test_enable_monitoring_delegates(self, service, mock_loaded_sae):
        """Test enable_monitoring delegates to LoadedSAE."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        service.enable_monitoring(True, [1234, 892])

        mock_loaded_sae.enable_monitoring.assert_called_once_with(True, [1234, 892])

    def test_enable_monitoring_not_attached_raises(self, service):
        """Test enable_monitoring raises when no SAE attached."""
        with pytest.raises(SAENotAttachedError):
            service.enable_monitoring(True)


# =============================================================================
# Attachment Status Tests
# =============================================================================


class TestSAEServiceAttachmentStatus:
    """Tests for get_attachment_status method."""

    def test_status_when_not_attached(self, service):
        """Test get_attachment_status when no SAE is attached."""
        status = service.get_attachment_status()

        assert status.is_attached is False
        assert status.sae_id is None
        assert status.layer is None

    def test_status_when_attached(self, service, mock_loaded_sae):
        """Test get_attachment_status when SAE is attached."""
        state = AttachedSAEState()
        state.set(mock_loaded_sae, "sae-id", 12, MagicMock())

        status = service.get_attachment_status()

        assert status.is_attached is True
        assert status.sae_id == "sae-id"
        assert status.layer == 12
        assert status.memory_usage_mb == 256


# =============================================================================
# Parse Path Metadata Tests
# =============================================================================


class TestSAEServiceParsePathMetadata:
    """Tests for _parse_sae_path_metadata helper."""

    def test_parses_standard_path(self, service):
        """Test parsing a standard Gemma-Scope style path."""
        width, average_l0 = service._parse_sae_path_metadata(
            "layer_20/width_16k/average_l0_38/params.npz"
        )

        assert width == "16k"
        assert average_l0 == 38

    def test_parses_path_without_metadata(self, service):
        """Test parsing a path with no width or l0 info."""
        width, average_l0 = service._parse_sae_path_metadata("some/other/path.npz")

        assert width is None
        assert average_l0 is None
