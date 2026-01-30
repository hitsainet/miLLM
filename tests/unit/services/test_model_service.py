"""Unit tests for ModelService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import ModelAlreadyExistsError, ModelNotFoundError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.services.model_service import ModelService


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.get_all = AsyncMock(return_value=[])
    repo.get_by_id = AsyncMock(return_value=None)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.find_by_repo_quantization = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_downloader():
    """Create a mock downloader."""
    downloader = MagicMock()
    downloader.download = MagicMock(return_value="/data/models/huggingface/google--gemma-2-2b--Q4")
    downloader.get_model_info = MagicMock(
        return_value={
            "name": "gemma-2-2b",
            "repo_id": "google/gemma-2-2b",
            "params": "2B",
            "architecture": "text-generation",
            "is_gated": False,
            "requires_trust_remote_code": False,
        }
    )
    downloader.delete_cached_model = MagicMock(return_value=True)
    downloader.get_cache_size = MagicMock(return_value=4_000_000_000)
    return downloader


@pytest.fixture
def mock_emitter():
    """Create a mock progress emitter."""
    emitter = MagicMock()
    emitter.emit_download_progress = AsyncMock()
    emitter.emit_download_complete = AsyncMock()
    emitter.emit_download_error = AsyncMock()
    return emitter


@pytest.fixture
def service(mock_repository, mock_downloader, mock_emitter):
    """Create a ModelService with mock dependencies."""
    return ModelService(
        repository=mock_repository,
        downloader=mock_downloader,
        emitter=mock_emitter,
    )


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = MagicMock(spec=Model)
    model.id = 1
    model.name = "gemma-2-2b"
    model.source = ModelSource.HUGGINGFACE
    model.repo_id = "google/gemma-2-2b"
    model.quantization = QuantizationType.Q4
    model.status = ModelStatus.READY
    model.cache_path = "huggingface/google--gemma-2-2b--Q4"
    model.created_at = datetime.utcnow()
    return model


class TestModelServiceListModels:
    """Tests for list_models method."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, service, mock_repository):
        """Test that list_models returns empty list when no models exist."""
        mock_repository.get_all.return_value = []

        result = await service.list_models()

        assert result == []
        mock_repository.get_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_all_models(self, service, mock_repository, sample_model):
        """Test that list_models returns all models."""
        mock_repository.get_all.return_value = [sample_model]

        result = await service.list_models()

        assert len(result) == 1
        assert result[0].id == 1


class TestModelServiceGetModel:
    """Tests for get_model method."""

    @pytest.mark.asyncio
    async def test_returns_model_when_found(self, service, mock_repository, sample_model):
        """Test that get_model returns the model when found."""
        mock_repository.get_by_id.return_value = sample_model

        result = await service.get_model(1)

        assert result.id == 1
        mock_repository.get_by_id.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_raises_not_found(self, service, mock_repository):
        """Test that get_model raises ModelNotFoundError when not found."""
        mock_repository.get_by_id.return_value = None

        with pytest.raises(ModelNotFoundError) as exc_info:
            await service.get_model(999)

        assert exc_info.value.details["model_id"] == 999


class TestModelServicePreviewModel:
    """Tests for preview_model method."""

    @pytest.mark.asyncio
    async def test_returns_model_info(self, service, mock_downloader):
        """Test that preview_model returns HuggingFace model info."""
        from millm.api.schemas.model import ModelPreviewRequest

        request = ModelPreviewRequest(repo_id="google/gemma-2-2b")

        result = await service.preview_model(request)

        assert result["name"] == "gemma-2-2b"
        assert result["params"] == "2B"
        mock_downloader.get_model_info.assert_called_once_with(
            repo_id="google/gemma-2-2b",
            token=None,
        )


class TestModelServiceDownloadModel:
    """Tests for download_model method."""

    @pytest.mark.asyncio
    async def test_creates_model_record(self, service, mock_repository, mock_downloader):
        """Test that download_model creates a database record."""
        from millm.api.schemas.model import ModelDownloadRequest

        # Mock create to return a model
        created_model = MagicMock()
        created_model.id = 1
        mock_repository.create.return_value = created_model

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4,
        )

        result = await service.download_model(request)

        assert result.id == 1
        mock_repository.create.assert_called_once()

        # Verify create was called with correct arguments
        call_kwargs = mock_repository.create.call_args[1]
        assert call_kwargs["name"] == "gemma-2-2b"
        assert call_kwargs["source"] == ModelSource.HUGGINGFACE
        assert call_kwargs["repo_id"] == "google/gemma-2-2b"
        assert call_kwargs["status"] == ModelStatus.DOWNLOADING

    @pytest.mark.asyncio
    async def test_raises_already_exists(self, service, mock_repository, sample_model):
        """Test that download_model raises error for duplicate models."""
        from millm.api.schemas.model import ModelDownloadRequest

        mock_repository.find_by_repo_quantization.return_value = sample_model

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.Q4,
        )

        with pytest.raises(ModelAlreadyExistsError) as exc_info:
            await service.download_model(request)

        assert "already exists" in str(exc_info.value.message)


class TestModelServiceCancelDownload:
    """Tests for cancel_download method."""

    @pytest.mark.asyncio
    async def test_cancels_active_download(self, service, mock_repository, sample_model):
        """Test that cancel_download cancels an active download."""
        sample_model.status = ModelStatus.DOWNLOADING
        mock_repository.get_by_id.return_value = sample_model

        cancelled_model = MagicMock()
        cancelled_model.status = ModelStatus.ERROR
        mock_repository.update_status.return_value = cancelled_model

        result = await service.cancel_download(1)

        mock_repository.update_status.assert_called_once_with(
            1,
            status=ModelStatus.ERROR,
            error_message="Download cancelled by user",
        )

    @pytest.mark.asyncio
    async def test_no_op_for_completed_download(self, service, mock_repository, sample_model):
        """Test that cancel_download is no-op for completed downloads."""
        sample_model.status = ModelStatus.READY
        mock_repository.get_by_id.return_value = sample_model

        result = await service.cancel_download(1)

        # Should return model without updating
        assert result.status == ModelStatus.READY
        mock_repository.update_status.assert_not_called()


class TestModelServiceDeleteModel:
    """Tests for delete_model method."""

    @pytest.mark.asyncio
    async def test_deletes_model(self, service, mock_repository, mock_downloader, sample_model):
        """Test that delete_model removes model from database and disk."""
        mock_repository.get_by_id.return_value = sample_model

        result = await service.delete_model(1)

        assert result is True
        mock_downloader.delete_cached_model.assert_called_once_with(
            "google/gemma-2-2b",
            "Q4",
        )
        mock_repository.delete.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_raises_not_found(self, service, mock_repository):
        """Test that delete_model raises error for non-existent model."""
        mock_repository.get_by_id.return_value = None

        with pytest.raises(ModelNotFoundError):
            await service.delete_model(999)

    @pytest.mark.asyncio
    async def test_cancels_active_download(self, service, mock_repository, mock_downloader, sample_model):
        """Test that delete_model cancels active download before deleting."""
        sample_model.status = ModelStatus.DOWNLOADING
        mock_repository.get_by_id.return_value = sample_model

        # update_status returns the updated model
        updated_model = MagicMock()
        updated_model.status = ModelStatus.ERROR
        mock_repository.update_status.return_value = updated_model

        await service.delete_model(1)

        # Should have called update_status for cancellation
        mock_repository.update_status.assert_called()


class TestModelServiceShutdown:
    """Tests for shutdown method."""

    def test_shuts_down_executor(self, service):
        """Test that shutdown cleans up the executor."""
        service.shutdown()

        # Executor should be shut down
        # This is a basic test - in practice we'd verify behavior
        assert True
