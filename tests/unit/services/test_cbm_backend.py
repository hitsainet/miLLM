"""Unit tests for ContinuousBatchingBackend."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from millm.services.cbm_backend import ContinuousBatchingBackend


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Create a CBM backend with default config."""
    return ContinuousBatchingBackend(
        max_queue_size=128,
        default_temperature=0.7,
        default_top_p=0.9,
        default_max_tokens=256,
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MagicMock()


@pytest.fixture
def mock_manager():
    """Create a mock ContinuousBatchingManager."""
    manager = MagicMock()
    manager.start = MagicMock()
    manager.stop = MagicMock()
    manager.add_request = MagicMock(return_value="req-001")
    return manager


@pytest.fixture
def running_backend(backend, mock_tokenizer, mock_manager):
    """Create a backend with mocked manager in running state."""
    backend._manager = mock_manager
    backend._started = True
    backend._tokenizer = mock_tokenizer
    return backend


# =============================================================================
# Tests: __init__
# =============================================================================


class TestCBMBackendInit:
    """Tests for ContinuousBatchingBackend initialization."""

    def test_default_values(self):
        """Backend initializes with sensible defaults."""
        backend = ContinuousBatchingBackend()
        assert backend._max_queue_size == 256
        assert backend._default_temperature == 0.7
        assert backend._default_top_p == 0.95
        assert backend._default_max_tokens == 512
        assert backend._manager is None
        assert backend._started is False

    def test_custom_values(self):
        """Backend accepts custom configuration."""
        backend = ContinuousBatchingBackend(
            max_queue_size=64,
            default_temperature=0.5,
            default_top_p=0.8,
            default_max_tokens=1024,
        )
        assert backend._max_queue_size == 64
        assert backend._default_temperature == 0.5
        assert backend._default_top_p == 0.8
        assert backend._default_max_tokens == 1024


# =============================================================================
# Tests: is_running
# =============================================================================


class TestCBMBackendIsRunning:
    """Tests for is_running property."""

    def test_not_running_initially(self, backend):
        """Backend is not running before start()."""
        assert backend.is_running is False

    def test_running_after_start(self, running_backend):
        """Backend is running after start()."""
        assert running_backend.is_running is True

    def test_not_running_when_manager_is_none(self, backend):
        """Backend is not running when manager is None even if started flag is set."""
        backend._started = True
        backend._manager = None
        assert backend.is_running is False

    def test_not_running_when_started_is_false(self, backend):
        """Backend is not running when started is False even if manager exists."""
        backend._started = False
        backend._manager = MagicMock()
        assert backend.is_running is False


# =============================================================================
# Tests: start
# =============================================================================


class TestCBMBackendStart:
    """Tests for start method."""

    def test_creates_manager_and_starts(self, backend, mock_model, mock_tokenizer):
        """Start creates ContinuousBatchingManager and calls start()."""
        mock_mgr = MagicMock()
        MockCBM = MagicMock(return_value=mock_mgr)
        MockGenConfig = MagicMock()

        with patch("transformers.ContinuousBatchingManager", MockCBM):
            with patch("transformers.GenerationConfig", MockGenConfig):
                backend.start(mock_model, mock_tokenizer)

        MockGenConfig.assert_called_once_with(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=2,
            pad_token_id=1,
        )
        MockCBM.assert_called_once_with(
            model=mock_model,
            generation_config=MockGenConfig.return_value,
            max_queue_size=128,
        )
        mock_mgr.start.assert_called_once()
        assert backend._started is True
        assert backend._tokenizer is mock_tokenizer

    def test_uses_eos_as_pad_when_pad_is_none(self, mock_model):
        """Start uses eos_token_id as pad_token_id when pad is None."""
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 50256
        tokenizer.pad_token_id = None

        MockGenConfig = MagicMock()
        backend = ContinuousBatchingBackend()

        with patch("transformers.ContinuousBatchingManager", MagicMock()):
            with patch("transformers.GenerationConfig", MockGenConfig):
                backend.start(mock_model, tokenizer)

        call_kwargs = MockGenConfig.call_args[1]
        assert call_kwargs["pad_token_id"] == 50256
        assert call_kwargs["eos_token_id"] == 50256

    def test_do_sample_false_when_temperature_zero(self, mock_model, mock_tokenizer):
        """Start sets do_sample=False when temperature is 0."""
        MockGenConfig = MagicMock()
        backend = ContinuousBatchingBackend(default_temperature=0.0)

        with patch("transformers.ContinuousBatchingManager", MagicMock()):
            with patch("transformers.GenerationConfig", MockGenConfig):
                backend.start(mock_model, mock_tokenizer)

        call_kwargs = MockGenConfig.call_args[1]
        assert call_kwargs["do_sample"] is False


# =============================================================================
# Tests: stop
# =============================================================================


class TestCBMBackendStop:
    """Tests for stop method."""

    def test_stops_running_manager(self, running_backend, mock_manager):
        """Stop calls manager.stop() and clears state."""
        running_backend.stop()

        mock_manager.stop.assert_called_once_with(block=True, timeout=10)
        assert running_backend._manager is None
        assert running_backend._started is False
        assert running_backend._tokenizer is None

    def test_noop_when_not_started(self, backend):
        """Stop is a no-op when backend was never started."""
        backend.stop()  # Should not raise
        assert backend._started is False

    def test_noop_when_manager_is_none(self, backend):
        """Stop is a no-op when manager is None."""
        backend._started = True
        backend._manager = None
        backend.stop()
        # _started remains True because the guard `if self._manager and self._started` fails
        assert backend._started is True

    def test_handles_stop_exception(self, running_backend, mock_manager):
        """Stop handles exceptions from manager.stop() gracefully."""
        mock_manager.stop.side_effect = RuntimeError("Stop failed")

        running_backend.stop()  # Should not raise

        assert running_backend._manager is None
        assert running_backend._started is False


# =============================================================================
# Tests: generate
# =============================================================================


class TestCBMBackendGenerate:
    """Tests for async generate method."""

    @pytest.mark.asyncio
    async def test_raises_when_not_running(self, backend):
        """Generate raises RuntimeError when backend is not running."""
        with pytest.raises(RuntimeError, match="not running"):
            await backend.generate(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-001",
            )

    @pytest.mark.asyncio
    async def test_successful_generation(self, running_backend, mock_manager):
        """Generate returns token IDs and finish reason on success."""
        mock_result = MagicMock()
        mock_result.generated_tokens = [10, 11, 12]
        mock_result.error = None
        mock_manager.get_result = MagicMock(return_value=mock_result)

        tokens, reason = await running_backend.generate(
            input_ids=[1, 2, 3],
            max_new_tokens=10,
            request_id="test-001",
        )

        assert tokens == [10, 11, 12]
        assert reason == "stop"
        mock_manager.add_request.assert_called_once_with(
            input_ids=[1, 2, 3],
            request_id="test-001",
            max_new_tokens=10,
        )

    @pytest.mark.asyncio
    async def test_finish_reason_length(self, running_backend, mock_manager):
        """Generate returns 'length' when tokens >= max_new_tokens."""
        mock_result = MagicMock()
        mock_result.generated_tokens = [10, 11, 12, 13, 14]
        mock_result.error = None
        mock_manager.get_result = MagicMock(return_value=mock_result)

        tokens, reason = await running_backend.generate(
            input_ids=[1, 2, 3],
            max_new_tokens=5,
            request_id="test-002",
        )

        assert reason == "length"

    @pytest.mark.asyncio
    async def test_finish_reason_stop(self, running_backend, mock_manager):
        """Generate returns 'stop' when tokens < max_new_tokens."""
        mock_result = MagicMock()
        mock_result.generated_tokens = [10, 11]
        mock_result.error = None
        mock_manager.get_result = MagicMock(return_value=mock_result)

        tokens, reason = await running_backend.generate(
            input_ids=[1, 2, 3],
            max_new_tokens=10,
            request_id="test-003",
        )

        assert reason == "stop"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self, running_backend, mock_manager):
        """Generate raises RuntimeError when result is None (timeout)."""
        mock_manager.get_result = MagicMock(return_value=None)

        with pytest.raises(RuntimeError, match="timed out"):
            await running_backend.generate(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-004",
                timeout=5.0,
            )

    @pytest.mark.asyncio
    async def test_raises_on_generation_error(self, running_backend, mock_manager):
        """Generate raises RuntimeError when result has an error."""
        mock_result = MagicMock()
        mock_result.error = "CUDA OOM"
        mock_manager.get_result = MagicMock(return_value=mock_result)

        with pytest.raises(RuntimeError, match="Generation failed: CUDA OOM"):
            await running_backend.generate(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-005",
            )


# =============================================================================
# Tests: generate_stream
# =============================================================================


class TestCBMBackendGenerateStream:
    """Tests for async generate_stream method."""

    @pytest.mark.asyncio
    async def test_raises_when_not_running(self, backend):
        """Stream raises RuntimeError when backend is not running."""
        with pytest.raises(RuntimeError, match="not running"):
            async for _ in backend.generate_stream(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-001",
            ):
                pass

    @pytest.mark.asyncio
    async def test_streams_tokens(self, running_backend, mock_manager):
        """Stream yields token chunks from background thread."""
        # Simulate CBM iterator that produces 3 incremental results
        result1 = MagicMock()
        result1.generated_tokens = [10]
        result1.is_finished = MagicMock(return_value=False)
        result1.error = None

        result2 = MagicMock()
        result2.generated_tokens = [10, 11]
        result2.is_finished = MagicMock(return_value=False)
        result2.error = None

        result3 = MagicMock()
        result3.generated_tokens = [10, 11, 12]
        result3.is_finished = MagicMock(return_value=True)
        result3.error = None

        mock_manager.request_id_iter = MagicMock(
            return_value=iter([result1, result2, result3])
        )

        chunks = []
        async for chunk in running_backend.generate_stream(
            input_ids=[1, 2, 3],
            max_new_tokens=10,
            request_id="test-006",
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == [10]
        assert chunks[1] == [11]
        assert chunks[2] == [12]

        mock_manager.add_request.assert_called_once_with(
            input_ids=[1, 2, 3],
            request_id="test-006",
            max_new_tokens=10,
            streaming=True,
        )

    @pytest.mark.asyncio
    async def test_stream_handles_error(self, running_backend, mock_manager):
        """Stream raises RuntimeError when CBM reports error mid-stream."""
        result1 = MagicMock()
        result1.generated_tokens = [10]
        result1.is_finished = MagicMock(return_value=False)
        result1.error = None

        result2 = MagicMock()
        result2.generated_tokens = [10, 11]
        result2.is_finished = MagicMock(return_value=False)
        result2.error = "CUDA error"

        mock_manager.request_id_iter = MagicMock(
            return_value=iter([result1, result2])
        )

        chunks = []
        with pytest.raises(RuntimeError, match="Generation failed"):
            async for chunk in running_backend.generate_stream(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-007",
            ):
                chunks.append(chunk)

        # Tokens from result2 are enqueued before the error check, so
        # we get both token chunks before the error is raised
        assert len(chunks) == 2
        assert chunks[0] == [10]
        assert chunks[1] == [11]

    @pytest.mark.asyncio
    async def test_stream_handles_iterator_exception(self, running_backend, mock_manager):
        """Stream raises when CBM iterator throws an exception."""

        def _failing_iter(req_id):
            yield MagicMock(
                generated_tokens=[10],
                is_finished=MagicMock(return_value=False),
                error=None,
            )
            raise RuntimeError("Iterator crashed")

        mock_manager.request_id_iter = _failing_iter

        chunks = []
        with pytest.raises(RuntimeError, match="Iterator crashed"):
            async for chunk in running_backend.generate_stream(
                input_ids=[1, 2, 3],
                max_new_tokens=10,
                request_id="test-008",
            ):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_stream_empty_tokens_not_yielded(self, running_backend, mock_manager):
        """Stream does not yield when no new tokens are produced."""
        result1 = MagicMock()
        result1.generated_tokens = [10]
        result1.is_finished = MagicMock(return_value=False)
        result1.error = None

        # Same length as before — no new tokens
        result2 = MagicMock()
        result2.generated_tokens = [10]
        result2.is_finished = MagicMock(return_value=False)
        result2.error = None

        result3 = MagicMock()
        result3.generated_tokens = [10, 11]
        result3.is_finished = MagicMock(return_value=True)
        result3.error = None

        mock_manager.request_id_iter = MagicMock(
            return_value=iter([result1, result2, result3])
        )

        chunks = []
        async for chunk in running_backend.generate_stream(
            input_ids=[1, 2, 3],
            max_new_tokens=10,
            request_id="test-009",
        ):
            chunks.append(chunk)

        # Should only get 2 chunks (result2 had no new tokens)
        assert len(chunks) == 2
        assert chunks[0] == [10]
        assert chunks[1] == [11]
