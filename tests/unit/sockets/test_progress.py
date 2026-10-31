"""Unit tests for Socket.IO progress emitter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.sockets.progress import (
    ProgressEmitter,
    create_socket_io,
    register_handlers,
)


class TestProgressEmitter:
    """Tests for ProgressEmitter class."""

    @pytest.fixture
    def mock_sio(self):
        """Create a mock Socket.IO server."""
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        """Create a ProgressEmitter with mock Socket.IO."""
        return ProgressEmitter(sio=mock_sio)

    @pytest.fixture
    def emitter_no_sio(self):
        """Create a ProgressEmitter without Socket.IO."""
        return ProgressEmitter(sio=None)


class TestProgressEmitterDownloadProgress:
    """Tests for download progress emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_download_progress(self, emitter, mock_sio):
        """Test that download progress is emitted correctly."""
        await emitter.emit_download_progress(
            model_id=1,
            progress=50.0,
            downloaded_bytes=500_000_000,
            total_bytes=1_000_000_000,
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:progress",
            {
                "modelId": 1,
                "progress": 50.0,
                "downloadedBytes": 500_000_000,
                "totalBytes": 1_000_000_000,
            },
        )

    @pytest.mark.asyncio
    async def test_emit_download_progress_with_speed(self, emitter, mock_sio):
        """Test that download progress includes speed when provided."""
        await emitter.emit_download_progress(
            model_id=1,
            progress=50.0,
            downloaded_bytes=500_000_000,
            total_bytes=1_000_000_000,
            speed_bps=10_000_000,
        )

        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == "model:download:progress"
        assert call_args[0][1]["speedBps"] == 10_000_000

    @pytest.mark.asyncio
    async def test_emit_download_progress_no_sio(self, emitter_no_sio=None):
        """Test that emit does nothing when sio is None."""
        emitter = ProgressEmitter(sio=None)
        # Should not raise
        await emitter.emit_download_progress(
            model_id=1,
            progress=50.0,
            downloaded_bytes=500_000_000,
            total_bytes=1_000_000_000,
        )


class TestProgressEmitterDownloadComplete:
    """Tests for download complete emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_download_complete(self, emitter, mock_sio):
        """Test that download complete is emitted correctly."""
        await emitter.emit_download_complete(
            model_id=1,
            local_path="/data/models/huggingface/google--gemma-2-2b--Q4",
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:complete",
            {
                "modelId": 1,
                "localPath": "/data/models/huggingface/google--gemma-2-2b--Q4",
            },
        )


class TestProgressEmitterDownloadError:
    """Tests for download error emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_download_error(self, emitter, mock_sio):
        """Test that download error is emitted correctly."""
        await emitter.emit_download_error(
            model_id=1,
            error_code="REPO_NOT_FOUND",
            error_message="Repository not found",
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:error",
            {
                "modelId": 1,
                "error": {
                    "code": "REPO_NOT_FOUND",
                    "message": "Repository not found",
                    "details": {},
                },
            },
        )

    @pytest.mark.asyncio
    async def test_emit_download_error_with_details(self, emitter, mock_sio):
        """Test that download error includes details when provided."""
        await emitter.emit_download_error(
            model_id=1,
            error_code="DOWNLOAD_FAILED",
            error_message="Download failed",
            details={"repo_id": "google/gemma-2-2b"},
        )

        call_args = mock_sio.emit.call_args
        assert call_args[0][1]["error"]["details"] == {"repo_id": "google/gemma-2-2b"}


class TestProgressEmitterLoadProgress:
    """Tests for load progress emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_load_progress(self, emitter, mock_sio):
        """Test that load progress is emitted correctly."""
        await emitter.emit_load_progress(
            model_id=1,
            stage="loading_weights",
            progress=75.0,
        )

        mock_sio.emit.assert_called_once_with(
            "model:load:progress",
            {
                "modelId": 1,
                "stage": "loading_weights",
                "progress": 75.0,
            },
        )


class TestProgressEmitterLoadComplete:
    """Tests for load complete emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_load_complete(self, emitter, mock_sio):
        """Test that load complete is emitted correctly."""
        await emitter.emit_load_complete(
            model_id=1,
            memory_used_mb=4096,
        )

        mock_sio.emit.assert_called_once_with(
            "model:load:complete",
            {
                "modelId": 1,
                "memoryUsedMb": 4096,
            },
        )


class TestProgressEmitterLoadError:
    """Tests for load error emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_load_error(self, emitter, mock_sio):
        """Test that load error is emitted correctly."""
        await emitter.emit_load_error(
            model_id=1,
            error_code="INSUFFICIENT_MEMORY",
            error_message="Not enough GPU memory",
        )

        mock_sio.emit.assert_called_once_with(
            "model:load:error",
            {
                "modelId": 1,
                "error": {
                    "code": "INSUFFICIENT_MEMORY",
                    "message": "Not enough GPU memory",
                },
            },
        )


class TestProgressEmitterUnloadComplete:
    """Tests for unload complete emission."""

    @pytest.fixture
    def mock_sio(self):
        sio = MagicMock()
        sio.emit = AsyncMock()
        return sio

    @pytest.fixture
    def emitter(self, mock_sio):
        return ProgressEmitter(sio=mock_sio)

    @pytest.mark.asyncio
    async def test_emit_unload_complete(self, emitter, mock_sio):
        """Test that unload complete is emitted correctly."""
        await emitter.emit_unload_complete(model_id=1)

        mock_sio.emit.assert_called_once_with(
            "model:unload:complete",
            {"modelId": 1},
        )


class TestProgressEmitterSetSio:
    """Tests for setting Socket.IO after initialization."""

    def test_set_sio(self):
        """Test that sio can be set after initialization."""
        emitter = ProgressEmitter(sio=None)
        mock_sio = MagicMock()

        emitter.set_sio(mock_sio)

        assert emitter._sio is mock_sio


class TestCreateSocketIO:
    """Tests for create_socket_io factory function."""

    def test_creates_async_server(self):
        """Test that create_socket_io returns an AsyncServer."""
        import socketio

        sio = create_socket_io()

        assert isinstance(sio, socketio.AsyncServer)

    def test_configures_progress_emitter(self):
        """Test that create_socket_io configures the global emitter."""
        from millm.sockets.progress import progress_emitter

        sio = create_socket_io()

        assert progress_emitter._sio is sio


class TestRegisterHandlers:
    """Tests for register_handlers function."""

    def test_registers_connect_handler(self):
        """Test that connect handler is registered."""
        mock_sio = MagicMock()

        register_handlers(mock_sio)

        # Check that event decorator was called
        assert mock_sio.event.called
