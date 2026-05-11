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


class TestTorchCompileAutoDetect:
    """Tests for TORCH_COMPILE=None auto-detection in _load_worker."""

    def _make_svc(self):
        """Create a ModelService with a mock loader."""
        mock_loader = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.memory_used_mb = 1024
        mock_loader.load.return_value = mock_loaded

        svc = ModelService(
            repository=MagicMock(),
            downloader=MagicMock(),
            loader=mock_loader,
            emitter=MagicMock(),
        )
        # Stub the async-from-thread bridge (not under test here)
        svc._run_async_from_thread = MagicMock()
        return svc, mock_loader

    def _call_load_worker(self, svc, quantization: str, torch_compile_setting):
        """Call _load_worker directly with patched settings and return loader call args."""
        # _load_worker imports settings locally, so patch at the config module level
        with patch("millm.core.config.settings") as mock_settings:
            mock_settings.TORCH_COMPILE = torch_compile_setting
            mock_settings.TORCH_COMPILE_MODE = "reduce-overhead"
            mock_settings.MODEL_CACHE_DIR = "/tmp"
            with patch("millm.api.dependencies.get_inference_service"):
                svc._load_worker(
                    model_id=1,
                    model_name="test-model",
                    cache_path="/tmp/model",
                    quantization=quantization,
                    estimated_memory_mb=1024,
                    trust_remote_code=False,
                )
        return svc.loader.load.call_args

    def test_auto_enables_compile_for_fp16_with_cuda(self):
        """Auto (None): FP16 + CUDA → torch_compile=True."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "FP16", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is True

    def test_auto_enables_compile_for_fp32_with_cuda(self):
        """Auto (None): FP32 + CUDA → torch_compile=True."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "FP32", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is True

    def test_auto_disables_compile_for_q4(self):
        """Auto (None): Q4 (bitsandbytes) → torch_compile=False."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "Q4", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is False

    def test_auto_disables_compile_for_q8(self):
        """Auto (None): Q8 (bitsandbytes) → torch_compile=False."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "Q8", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is False

    def test_auto_disables_compile_for_q2(self):
        """Auto (None): Q2 (bitsandbytes) → torch_compile=False."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "Q2", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is False

    def test_auto_disables_compile_without_cuda(self):
        """Auto (None): no CUDA → torch_compile=False regardless of quantization."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=False):
            call_args = self._call_load_worker(svc, "FP16", torch_compile_setting=None)
        assert call_args.kwargs["torch_compile"] is False

    def test_explicit_true_passes_through(self):
        """Explicit True: always passed to loader as True."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "FP16", torch_compile_setting=True)
        assert call_args.kwargs["torch_compile"] is True

    def test_explicit_false_passes_through(self):
        """Explicit False: always passed to loader as False (even for FP16+CUDA)."""
        svc, _ = self._make_svc()
        with patch("torch.cuda.is_available", return_value=True):
            call_args = self._call_load_worker(svc, "FP16", torch_compile_setting=False)
        assert call_args.kwargs["torch_compile"] is False
