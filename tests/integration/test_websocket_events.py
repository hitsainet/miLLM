"""Integration tests for WebSocket event system.

Tests the Socket.IO progress emitter and event handling for
model downloads, loads, and SAE operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from millm.sockets.progress import (
    ProgressEmitter,
    create_socket_io,
    register_handlers,
)


@pytest.fixture
def mock_sio():
    """Create a mock Socket.IO server."""
    sio = MagicMock()
    sio.emit = AsyncMock()
    return sio


@pytest.fixture
def emitter(mock_sio):
    """Create a progress emitter with mock Socket.IO server."""
    emitter = ProgressEmitter(sio=mock_sio)
    return emitter


class TestProgressEmitterModelEvents:
    """Tests for model-related WebSocket events."""

    @pytest.mark.asyncio
    async def test_emits_download_progress(self, emitter, mock_sio):
        """Test emitting download progress event."""
        await emitter.emit_download_progress(
            model_id=1,
            progress=50.0,
            downloaded_bytes=500000000,
            total_bytes=1000000000,
            speed_bps=10000000,
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:progress",
            {
                "modelId": 1,
                "progress": 50.0,
                "downloadedBytes": 500000000,
                "totalBytes": 1000000000,
                "speedBps": 10000000,
            }
        )

    @pytest.mark.asyncio
    async def test_emits_download_progress_without_speed(self, emitter, mock_sio):
        """Test emitting download progress without speed info."""
        await emitter.emit_download_progress(
            model_id=1,
            progress=75.0,
            downloaded_bytes=750000000,
            total_bytes=1000000000,
        )

        call_args = mock_sio.emit.call_args
        assert "speedBps" not in call_args[0][1]

    @pytest.mark.asyncio
    async def test_emits_download_complete(self, emitter, mock_sio):
        """Test emitting download complete event."""
        await emitter.emit_download_complete(
            model_id=1,
            local_path="/models/gemma-2-2b",
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:complete",
            {
                "modelId": 1,
                "localPath": "/models/gemma-2-2b",
            }
        )

    @pytest.mark.asyncio
    async def test_emits_download_error(self, emitter, mock_sio):
        """Test emitting download error event."""
        await emitter.emit_download_error(
            model_id=1,
            error_code="DOWNLOAD_FAILED",
            error_message="Network error during download",
            details={"retry_count": 3},
        )

        mock_sio.emit.assert_called_once_with(
            "model:download:error",
            {
                "modelId": 1,
                "error": {
                    "code": "DOWNLOAD_FAILED",
                    "message": "Network error during download",
                    "details": {"retry_count": 3},
                }
            }
        )

    @pytest.mark.asyncio
    async def test_emits_load_progress(self, emitter, mock_sio):
        """Test emitting model load progress event."""
        await emitter.emit_load_progress(
            model_id=1,
            stage="loading_weights",
            progress=30.0,
        )

        mock_sio.emit.assert_called_once_with(
            "model:load:progress",
            {
                "modelId": 1,
                "stage": "loading_weights",
                "progress": 30.0,
            }
        )

    @pytest.mark.asyncio
    async def test_emits_load_complete(self, emitter, mock_sio):
        """Test emitting model load complete event."""
        await emitter.emit_load_complete(
            model_id=1,
            memory_used_mb=2048,
        )

        mock_sio.emit.assert_called_once_with(
            "model:load:complete",
            {
                "modelId": 1,
                "memoryUsedMb": 2048,
            }
        )

    @pytest.mark.asyncio
    async def test_emits_load_error(self, emitter, mock_sio):
        """Test emitting model load error event."""
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
                }
            }
        )

    @pytest.mark.asyncio
    async def test_emits_unload_complete(self, emitter, mock_sio):
        """Test emitting model unload complete event."""
        await emitter.emit_unload_complete(model_id=1)

        mock_sio.emit.assert_called_once_with(
            "model:unload:complete",
            {"modelId": 1}
        )


class TestProgressEmitterSAEEvents:
    """Tests for SAE-related WebSocket events."""

    @pytest.mark.asyncio
    async def test_emits_sae_download_progress(self, emitter, mock_sio):
        """Test emitting SAE download progress event."""
        await emitter.emit_sae_download_progress(
            sae_id="sae-123",
            percent=50,
        )

        mock_sio.emit.assert_called_once_with(
            "sae:download:progress",
            {
                "saeId": "sae-123",
                "percent": 50,
            }
        )

    @pytest.mark.asyncio
    async def test_emits_sae_download_complete(self, emitter, mock_sio):
        """Test emitting SAE download complete event."""
        await emitter.emit_sae_download_complete(sae_id="sae-123")

        mock_sio.emit.assert_called_once_with(
            "sae:download:complete",
            {"saeId": "sae-123"}
        )

    @pytest.mark.asyncio
    async def test_emits_sae_download_error(self, emitter, mock_sio):
        """Test emitting SAE download error event."""
        await emitter.emit_sae_download_error(
            sae_id="sae-123",
            error="Repository not found",
        )

        mock_sio.emit.assert_called_once_with(
            "sae:download:error",
            {
                "saeId": "sae-123",
                "error": "Repository not found",
            }
        )

    @pytest.mark.asyncio
    async def test_emits_sae_attached(self, emitter, mock_sio):
        """Test emitting SAE attached event."""
        await emitter.emit_sae_attached(
            sae_id="sae-123",
            layer=12,
            memory_mb=256,
        )

        mock_sio.emit.assert_called_once_with(
            "sae:attached",
            {
                "saeId": "sae-123",
                "layer": 12,
                "memoryMb": 256,
            }
        )

    @pytest.mark.asyncio
    async def test_emits_sae_detached(self, emitter, mock_sio):
        """Test emitting SAE detached event."""
        await emitter.emit_sae_detached(sae_id="sae-123")

        mock_sio.emit.assert_called_once_with(
            "sae:detached",
            {"saeId": "sae-123"}
        )


class TestProgressEmitterMonitoringEvents:
    """Tests for monitoring-related WebSocket events."""

    @pytest.mark.asyncio
    async def test_emits_monitoring_state_changed(self, emitter, mock_sio):
        """Test emitting monitoring state changed event."""
        await emitter.emit_monitoring_state_changed(
            enabled=True,
            monitored_features=[1234, 5678, 9012],
        )

        mock_sio.emit.assert_called_once_with(
            "monitoring:state",
            {
                "enabled": True,
                "monitoredFeatures": [1234, 5678, 9012],
            }
        )


class TestProgressEmitterNoSio:
    """Tests for emitter behavior when Socket.IO is not configured."""

    @pytest.mark.asyncio
    async def test_no_op_when_sio_none(self):
        """Test that emitter does nothing when sio is None."""
        emitter = ProgressEmitter(sio=None)

        # These should not raise any errors
        await emitter.emit_download_progress(1, 50.0, 500, 1000)
        await emitter.emit_download_complete(1, "/path")
        await emitter.emit_download_error(1, "ERR", "msg")
        await emitter.emit_load_progress(1, "stage", 50.0)
        await emitter.emit_load_complete(1, 1024)
        await emitter.emit_load_error(1, "ERR", "msg")
        await emitter.emit_unload_complete(1)
        await emitter.emit_sae_download_progress("sae", 50)
        await emitter.emit_sae_download_complete("sae")
        await emitter.emit_sae_download_error("sae", "error")
        await emitter.emit_sae_attached("sae", 12, 256)
        await emitter.emit_sae_detached("sae")
        await emitter.emit_monitoring_state_changed(True)


class TestProgressEmitterSetSio:
    """Tests for setting Socket.IO server after initialization."""

    @pytest.mark.asyncio
    async def test_set_sio_enables_emission(self, mock_sio):
        """Test that setting sio enables event emission."""
        emitter = ProgressEmitter(sio=None)

        # Should do nothing
        await emitter.emit_download_complete(1, "/path")
        mock_sio.emit.assert_not_called()

        # Set sio
        emitter.set_sio(mock_sio)

        # Now should emit
        await emitter.emit_download_complete(1, "/path")
        mock_sio.emit.assert_called_once()


class TestSocketIOHandlers:
    """Tests for Socket.IO event handlers."""

    def test_registers_connect_handler(self):
        """Test that connect handler is registered."""
        sio = MagicMock()
        sio.event = MagicMock(return_value=lambda f: f)

        register_handlers(sio)

        # Verify event decorator was called
        assert sio.event.call_count >= 1

    def test_create_socket_io_returns_server(self):
        """Test that create_socket_io returns a server instance."""
        with patch("socketio.AsyncServer") as mock_async_server:
            mock_server = MagicMock()
            mock_async_server.return_value = mock_server
            mock_server.event = MagicMock(return_value=lambda f: f)

            result = create_socket_io()

            mock_async_server.assert_called_once()
            assert result == mock_server


class TestSocketIOIntegration:
    """Integration tests for complete Socket.IO event flow."""

    @pytest.mark.asyncio
    async def test_download_event_sequence(self, emitter, mock_sio):
        """Test complete download event sequence."""
        model_id = 1

        # Start download
        await emitter.emit_download_progress(model_id, 0.0, 0, 1000000000)
        await emitter.emit_download_progress(model_id, 25.0, 250000000, 1000000000)
        await emitter.emit_download_progress(model_id, 50.0, 500000000, 1000000000)
        await emitter.emit_download_progress(model_id, 75.0, 750000000, 1000000000)
        await emitter.emit_download_progress(model_id, 100.0, 1000000000, 1000000000)
        await emitter.emit_download_complete(model_id, "/models/test")

        assert mock_sio.emit.call_count == 6

    @pytest.mark.asyncio
    async def test_load_event_sequence(self, emitter, mock_sio):
        """Test complete model load event sequence."""
        model_id = 1

        await emitter.emit_load_progress(model_id, "loading_config", 10.0)
        await emitter.emit_load_progress(model_id, "loading_tokenizer", 30.0)
        await emitter.emit_load_progress(model_id, "loading_weights", 70.0)
        await emitter.emit_load_progress(model_id, "moving_to_device", 90.0)
        await emitter.emit_load_complete(model_id, 2048)

        assert mock_sio.emit.call_count == 5

    @pytest.mark.asyncio
    async def test_error_event_includes_details(self, emitter, mock_sio):
        """Test that error events include proper details."""
        await emitter.emit_download_error(
            model_id=1,
            error_code="GATED_MODEL",
            error_message="Model is gated",
            details={"repo_id": "meta-llama/Llama-2-7b"},
        )

        call_args = mock_sio.emit.call_args
        event_name = call_args[0][0]
        event_data = call_args[0][1]

        assert event_name == "model:download:error"
        assert event_data["error"]["code"] == "GATED_MODEL"
        assert event_data["error"]["details"]["repo_id"] == "meta-llama/Llama-2-7b"
