"""
Resilience patterns for external service calls.

Provides circuit breaker and retry functionality for handling
transient failures when communicating with HuggingFace and other services.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec

import structlog

logger = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 3  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 1  # Successes in half-open to close circuit


@dataclass
class CircuitBreakerState:
    """Mutable state for circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_error: Optional[Exception] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by tracking errors and temporarily
    blocking requests when a service is failing.

    Usage:
        breaker = CircuitBreaker(name="huggingface")

        @breaker
        def download_model(repo_id: str) -> str:
            return snapshot_download(repo_id)
    """

    # Shared state across all instances with same name
    _states: dict[str, CircuitBreakerState] = {}

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (shared across instances)
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # Initialize shared state if not exists
        if name not in self._states:
            self._states[name] = CircuitBreakerState()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state."""
        return self._states[self.name]

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self.state.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.state.last_failure_time >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return False
            return True
        return False

    def _transition_to_open(self, error: Exception) -> None:
        """Transition to open state after failures."""
        self.state.state = CircuitState.OPEN
        self.state.last_failure_time = time.time()
        self.state.last_error = error
        self.state.success_count = 0

        logger.warning(
            "circuit_opened",
            circuit=self.name,
            failure_count=self.state.failure_count,
            error=str(error),
        )

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state to test recovery."""
        self.state.state = CircuitState.HALF_OPEN
        self.state.success_count = 0

        logger.info("circuit_half_open", circuit=self.name)

    def _transition_to_closed(self) -> None:
        """Transition to closed state after recovery."""
        self.state.state = CircuitState.CLOSED
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.last_error = None

        logger.info("circuit_closed", circuit=self.name)

    def _record_success(self) -> None:
        """Record successful call."""
        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1
            if self.state.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.state.failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        self.state.failure_count += 1
        self.state.last_error = error

        if self.state.state == CircuitState.HALF_OPEN:
            self._transition_to_open(error)
        elif self.state.failure_count >= self.config.failure_threshold:
            self._transition_to_open(error)

        logger.warning(
            "circuit_failure",
            circuit=self.name,
            failure_count=self.state.failure_count,
            error=str(error),
        )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._states[self.name] = CircuitBreakerState()
        logger.info("circuit_reset", circuit=self.name)

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to wrap function with circuit breaker."""

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.is_open:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open after {self.state.failure_count} failures. "
                    f"Last error: {self.state.last_error}",
                    circuit_name=self.name,
                    last_error=self.state.last_error,
                )

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise

        return wrapper


class AsyncCircuitBreaker(CircuitBreaker):
    """Async-compatible circuit breaker."""

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to wrap async function with circuit breaker."""

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                if self.is_open:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is open after {self.state.failure_count} failures. "
                        f"Last error: {self.state.last_error}",
                        circuit_name=self.name,
                        last_error=self.state.last_error,
                    )

                try:
                    result = await func(*args, **kwargs)
                    self._record_success()
                    return result
                except Exception as e:
                    self._record_failure(e)
                    raise

            return async_wrapper  # type: ignore
        else:
            return super().__call__(func)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        message: str,
        circuit_name: str,
        last_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.circuit_name = circuit_name
        self.last_error = last_error


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying failed function calls.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on

    Usage:
        @with_retry(max_attempts=3, delay=1.0)
        def fetch_data():
            return api_call()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=current_delay,
                            error=str(e),
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor

            raise last_exception  # type: ignore

        return wrapper

    return decorator


def async_with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Async version of with_retry decorator."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=current_delay,
                            error=str(e),
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


# Pre-configured circuit breakers for common services
huggingface_circuit = CircuitBreaker(
    name="huggingface",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=1,
    ),
)
