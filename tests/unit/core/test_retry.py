"""Unit tests for retry decorators in resilience module."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from millm.core.resilience import with_retry, async_with_retry


class TestWithRetry:
    """Tests for synchronous retry decorator."""

    def test_succeeds_on_first_attempt(self):
        """Test function that succeeds on first attempt."""
        mock_func = MagicMock(return_value="success")

        @with_retry(max_attempts=3)
        def wrapped():
            return mock_func()

        result = wrapped()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retries_on_failure(self):
        """Test function is retried on failure."""
        mock_func = MagicMock(side_effect=[ValueError("fail"), "success"])

        @with_retry(max_attempts=3, delay=0.01)
        def wrapped():
            return mock_func()

        result = wrapped()

        assert result == "success"
        assert mock_func.call_count == 2

    def test_raises_after_max_attempts(self):
        """Test exception is raised after max attempts exhausted."""
        mock_func = MagicMock(side_effect=ValueError("always fails"))

        @with_retry(max_attempts=3, delay=0.01)
        def wrapped():
            return mock_func()

        with pytest.raises(ValueError, match="always fails"):
            wrapped()

        assert mock_func.call_count == 3

    def test_only_retries_specified_exceptions(self):
        """Test only specified exception types trigger retry."""
        mock_func = MagicMock(side_effect=TypeError("type error"))

        @with_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def wrapped():
            return mock_func()

        with pytest.raises(TypeError, match="type error"):
            wrapped()

        # Should not retry on TypeError since only ValueError specified
        assert mock_func.call_count == 1

    def test_retries_specified_exception_types(self):
        """Test multiple exception types can be specified for retry."""
        mock_func = MagicMock(
            side_effect=[ValueError("val"), TypeError("type"), "success"]
        )

        @with_retry(max_attempts=3, delay=0.01, exceptions=(ValueError, TypeError))
        def wrapped():
            return mock_func()

        result = wrapped()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_backoff_increases_delay(self):
        """Test exponential backoff increases delay between retries."""
        mock_func = MagicMock(side_effect=[ValueError(), ValueError(), "success"])
        sleep_calls = []

        with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            @with_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
            def wrapped():
                return mock_func()

            result = wrapped()

        assert result == "success"
        # First retry delay: 1.0, second retry delay: 2.0
        assert sleep_calls == [1.0, 2.0]

    def test_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""
        @with_retry(max_attempts=3)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestAsyncWithRetry:
    """Tests for asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        """Test async function that succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")

        @async_with_retry(max_attempts=3)
        async def wrapped():
            return await mock_func()

        result = await wrapped()

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test async function is retried on failure."""
        mock_func = AsyncMock(side_effect=[ValueError("fail"), "success"])

        @async_with_retry(max_attempts=3, delay=0.01)
        async def wrapped():
            return await mock_func()

        result = await wrapped()

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """Test exception is raised after max attempts exhausted."""
        mock_func = AsyncMock(side_effect=ValueError("always fails"))

        @async_with_retry(max_attempts=3, delay=0.01)
        async def wrapped():
            return await mock_func()

        with pytest.raises(ValueError, match="always fails"):
            await wrapped()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self):
        """Test only specified exception types trigger retry."""
        mock_func = AsyncMock(side_effect=TypeError("type error"))

        @async_with_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        async def wrapped():
            return await mock_func()

        with pytest.raises(TypeError, match="type error"):
            await wrapped()

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_increases_delay(self):
        """Test exponential backoff increases delay between retries."""
        mock_func = AsyncMock(side_effect=[ValueError(), ValueError(), "success"])
        sleep_calls = []

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("asyncio.sleep", mock_sleep):
            @async_with_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
            async def wrapped():
                return await mock_func()

            result = await wrapped()

        assert result == "success"
        assert sleep_calls == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""
        @async_with_retry(max_attempts=3)
        async def my_async_function():
            """My async docstring."""
            pass

        assert my_async_function.__name__ == "my_async_function"
        assert my_async_function.__doc__ == "My async docstring."


class TestRetryWithCircuitBreaker:
    """Integration tests for retry with circuit breaker."""

    def test_retry_does_not_open_circuit(self):
        """Test that retried failures within limit don't open circuit."""
        from millm.core.resilience import CircuitBreaker, CircuitBreakerConfig

        # Create a new circuit for this test
        breaker = CircuitBreaker(
            name="test_retry_circuit",
            config=CircuitBreakerConfig(failure_threshold=5),
        )
        breaker.reset()  # Ensure clean state

        call_count = 0

        @breaker
        @with_retry(max_attempts=3, delay=0.01)
        def wrapped():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        result = wrapped()

        assert result == "success"
        assert call_count == 3
        # Circuit should remain closed (only final result matters to breaker)
        assert breaker.state.state.value == "closed"

        # Clean up
        breaker.reset()


class TestRetryEdgeCases:
    """Edge case tests for retry decorators."""

    def test_zero_delay(self):
        """Test retry with zero delay."""
        mock_func = MagicMock(side_effect=[ValueError(), "success"])

        @with_retry(max_attempts=2, delay=0)
        def wrapped():
            return mock_func()

        result = wrapped()
        assert result == "success"

    def test_single_attempt(self):
        """Test retry with max_attempts=1 (no retries)."""
        mock_func = MagicMock(side_effect=ValueError("fail"))

        @with_retry(max_attempts=1, delay=0.01)
        def wrapped():
            return mock_func()

        with pytest.raises(ValueError):
            wrapped()

        assert mock_func.call_count == 1

    def test_no_backoff(self):
        """Test retry with backoff_factor=1 (constant delay)."""
        mock_func = MagicMock(side_effect=[ValueError(), ValueError(), "success"])
        sleep_calls = []

        with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            @with_retry(max_attempts=3, delay=0.5, backoff_factor=1.0)
            def wrapped():
                return mock_func()

            wrapped()

        # All delays should be the same
        assert sleep_calls == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_async_zero_delay(self):
        """Test async retry with zero delay."""
        mock_func = AsyncMock(side_effect=[ValueError(), "success"])

        @async_with_retry(max_attempts=2, delay=0)
        async def wrapped():
            return await mock_func()

        result = await wrapped()
        assert result == "success"
