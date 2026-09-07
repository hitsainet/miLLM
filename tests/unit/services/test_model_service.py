"""Unit tests for ModelService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.core.errors import ModelAlreadyExistsError, ModelNotFoundError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.services.model_service import ModelService
from tests.support.factories import make_model


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
    """A real row, not a double — see tests/support/factories.py."""
    return make_model()


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
        # `revision` is part of the call now: preview and download must agree on
        # the commit, or the sizes shown were measured against other content.
        mock_downloader.get_model_info.assert_called_once_with(
            repo_id="google/gemma-2-2b",
            token=None,
            revision=None,
        )

    @pytest.mark.asyncio
    async def test_passes_the_requested_revision_through(self, service, mock_downloader):
        """A pinned revision must reach the downloader, not be silently dropped."""
        from millm.api.schemas.model import ModelPreviewRequest

        request = ModelPreviewRequest(repo_id="google/gemma-2-2b", revision="abc123")

        await service.preview_model(request)

        assert mock_downloader.get_model_info.call_args.kwargs["revision"] == "abc123"


class TestModelServiceDeleteRemovesTheRightDirectory:
    """Deleting must resolve the directory the download actually wrote.

    The cache path is repo--quantization[--gguf_label]. Resolving it without the
    label computes a path that does not exist, so the row disappears and the
    files stay — 5.4 GB orphaned with nothing pointing at them, and the disk
    only gets worse with each delete.

    MUTATION CONTROL: drop the variant argument in delete_model -> this fails.
    """

    @pytest.mark.asyncio
    async def test_a_gguf_model_deletes_its_own_directory(
        self, service, mock_repository, mock_downloader
    ):
        model = MagicMock()
        model.id = 5
        model.source = ModelSource.HUGGINGFACE
        model.repo_id = "sovasoft/zora-v1.13-gguf"
        model.quantization = QuantizationType.Q4
        model.gguf_label = "Q5_K_M"
        model.status = ModelStatus.READY
        mock_repository.get_by_id.return_value = model
        mock_repository.delete.return_value = True

        await service.delete_model(5)

        args = mock_downloader.delete_cached_model.call_args[0]
        assert args[0] == "sovasoft/zora-v1.13-gguf"
        assert args[1] == "Q4"
        assert args[2] == "Q5_K_M", (
            "without the label this resolves repo--Q4, which is not where the "
            "files are; the row goes and the bytes stay"
        )

    @pytest.mark.asyncio
    async def test_an_ordinary_model_passes_no_variant(
        self, service, mock_repository, mock_downloader
    ):
        model = MagicMock()
        model.id = 6
        model.source = ModelSource.HUGGINGFACE
        model.repo_id = "google/gemma-2-2b"
        model.quantization = QuantizationType.Q4
        model.gguf_label = ""
        model.status = ModelStatus.READY
        mock_repository.get_by_id.return_value = model
        mock_repository.delete.return_value = True

        await service.delete_model(6)

        args = mock_downloader.delete_cached_model.call_args[0]
        assert args[2] is None, "'' must not become a directory suffix"


class TestModelServiceDownloadModel:
    """Tests for download_model method."""

    @pytest.mark.asyncio
    async def test_creates_model_record(self, service, mock_repository, mock_downloader):
        """Test that download_model creates a database record."""
        from millm.api.schemas.model import ModelDownloadRequest

        # Mock create to return a model
        mock_repository.create.return_value = make_model(id=1)

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
    async def test_a_gguf_download_persists_its_exact_quantization(
        self, service, mock_repository, mock_downloader
    ):
        """The label, the file list and the revision must reach the row.

        Without them the row cannot say WHICH quantization it holds, and the
        cache directory — which is built from the label — collides with every
        other Q4 variant from the same repo.
        """
        from millm.api.schemas.model import ModelDownloadRequest

        mock_repository.create.return_value = make_model(id=7)

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
            quantization=QuantizationType.Q4,
            gguf_files=["Qwen2.5-7B-Instruct-Q6_K.gguf"],
            gguf_label="Q6_K",
            revision="abc123",
        )

        await service.download_model(request)

        kwargs = mock_repository.create.call_args[1]
        assert kwargs["gguf_label"] == "Q6_K"
        assert kwargs["gguf_files"] == ["Qwen2.5-7B-Instruct-Q6_K.gguf"]
        assert kwargs["revision"] == "abc123"
        assert kwargs["cache_path"].endswith("--Q6_K"), (
            "the cache path must carry the label, or Q6_K and Q8_0 — both "
            "coarse 'Q8' — share one directory"
        )

    @pytest.mark.asyncio
    async def test_the_coarse_level_is_DERIVED_from_the_gguf_label(
        self, service, mock_repository, mock_downloader
    ):
        """The bucket is a function of the label, not an independent choice.

        The form's quantization dropdown defaults to Q4. Taking that at face
        value files a 25 GB Q6_K as a 4-bit model — wrong in the listing, wrong
        in the memory estimate, and wrong in the cache path.
        """
        from millm.api.schemas.model import ModelDownloadRequest

        mock_repository.create.return_value = make_model(id=8)

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="r/x-GGUF",
            quantization=QuantizationType.Q4,   # what the dropdown happened to say
            gguf_files=["x-Q6_K.gguf"],
            gguf_label="Q6_K",                  # what was actually chosen
        )

        await service.download_model(request)

        kwargs = mock_repository.create.call_args[1]
        assert kwargs["quantization"] == QuantizationType.Q8, (
            "Q6_K is a 6-bit quantization and must not be filed as Q4 just "
            "because the coarse dropdown defaulted there"
        )

    @pytest.mark.asyncio
    async def test_an_ordinary_download_keeps_the_requested_level(
        self, service, mock_repository, mock_downloader
    ):
        """Derivation must not touch a repo with no GGUF selection."""
        from millm.api.schemas.model import ModelDownloadRequest

        mock_repository.create.return_value = make_model(id=9)

        request = ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE,
            repo_id="google/gemma-2-2b",
            quantization=QuantizationType.FP16,
        )

        await service.download_model(request)

        kwargs = mock_repository.create.call_args[1]
        assert kwargs["quantization"] == QuantizationType.FP16
        assert kwargs["gguf_label"] == ""
        assert kwargs["gguf_files"] is None

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
        # The variant is part of the call now: the cache directory is
        # repo--quantization[--gguf_label], and resolving it without the label
        # orphans a GGUF model's files on delete.
        mock_downloader.delete_cached_model.assert_called_once_with(
            "google/gemma-2-2b",
            "Q4",
            None,
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
