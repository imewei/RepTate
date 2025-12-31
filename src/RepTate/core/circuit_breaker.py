"""Circuit breaker pattern for safe fallback to legacy code.

This module implements a circuit breaker that monitors failures in new code
implementations and automatically falls back to legacy code when error rates
exceed thresholds.

Usage:
    from RepTate.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry

    breaker = CircuitBreakerRegistry.get_breaker('jax_optimization')

    @breaker.protected(fallback=legacy_optimize_function)
    def new_jax_optimize(params):
        # New JAX implementation
        return optimized_params
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker.

    CLOSED: Normal operation, calls go to new implementation
    OPEN: Too many failures, calls go to fallback (legacy) implementation
    HALF_OPEN: Testing if new implementation has recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close circuit
        timeout: Seconds to wait before entering half-open state
        failure_rate_threshold: Failure rate (0.0-1.0) to trigger open state
        monitoring_window: Seconds to track for failure rate calculation
        allowed_exceptions: Exceptions that should not trigger the circuit
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    failure_rate_threshold: float = 0.5
    monitoring_window: float = 60.0
    allowed_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)


@dataclass
class CircuitBreakerMetrics:
    """Metrics for a circuit breaker.

    Attributes:
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        fallback_calls: Number of calls that used fallback
        state_changes: List of (timestamp, old_state, new_state)
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_calls: int = 0
    state_changes: list[tuple[datetime, CircuitState, CircuitState]] = field(
        default_factory=list
    )
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None


T = TypeVar('T')


class CircuitBreaker:
    """Circuit breaker for automatic fallback to legacy implementations.

    The circuit breaker monitors the error rate of new implementations and
    automatically switches to legacy implementations when errors exceed
    configured thresholds.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None
    ):
        """Initialize the circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration (uses defaults if not provided)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()
        self._failure_times: deque[float] = deque()
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        with self._lock:
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state.

        Args:
            new_state: The state to transition to
        """
        with self._lock:
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self._metrics.state_changes.append(
                    (datetime.now(), old_state, new_state)
                )
                logger.warning(
                    f"Circuit breaker '{self.name}' state change: "
                    f"{old_state.value} -> {new_state.value}"
                )

                if new_state == CircuitState.OPEN:
                    self._opened_at = time.time()
                    self._consecutive_failures = 0
                elif new_state == CircuitState.CLOSED:
                    self._consecutive_successes = 0
                    self._consecutive_failures = 0
                    self._opened_at = None

    def _check_failure_rate(self) -> bool:
        """Check if failure rate exceeds threshold.

        Returns:
            True if failure rate exceeds threshold
        """
        now = time.time()
        window_start = now - self.config.monitoring_window

        # Remove old failure times
        while self._failure_times and self._failure_times[0] < window_start:
            self._failure_times.popleft()

        if not self._failure_times:
            return False

        # Calculate failure rate over monitoring window
        total_in_window = len(self._failure_times)
        failure_rate = total_in_window / (self._metrics.total_calls + 1)

        return failure_rate > self.config.failure_rate_threshold

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should transition from OPEN to HALF_OPEN.

        Returns:
            True if enough time has passed since opening
        """
        if self._opened_at is None:
            return False

        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = datetime.now()
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call.

        Args:
            exception: The exception that caused the failure
        """
        with self._lock:
            # Check if this exception should be ignored
            if isinstance(exception, self.config.allowed_exceptions):
                logger.debug(
                    f"Circuit breaker '{self.name}' ignoring allowed exception: "
                    f"{type(exception).__name__}"
                )
                return

            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = datetime.now()
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._failure_times.append(time.time())

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                # Check both consecutive failures and failure rate
                if (self._consecutive_failures >= self.config.failure_threshold
                        or self._check_failure_rate()):
                    self._transition_to(CircuitState.OPEN)

    def call(
        self,
        func: Callable[..., T],
        fallback: Callable[..., T] | None = None,
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Call a function with circuit breaker protection.

        Args:
            func: The function to call (new implementation)
            fallback: Fallback function (legacy implementation)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Raises:
            Exception: If both func and fallback fail, or if no fallback provided
        """
        with self._lock:
            self._metrics.total_calls += 1

            # Check if we should attempt reset
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)

            # If circuit is open, use fallback immediately
            if self._state == CircuitState.OPEN:
                if fallback is None:
                    raise RuntimeError(
                        f"Circuit breaker '{self.name}' is OPEN but no fallback provided"
                    )
                self._metrics.fallback_calls += 1
                logger.info(
                    f"Circuit breaker '{self.name}' is OPEN, using fallback"
                )
                return fallback(*args, **kwargs)

        # Try the new implementation
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure(e)

            # Use fallback if available
            if fallback is not None:
                with self._lock:
                    self._metrics.fallback_calls += 1

                logger.warning(
                    f"Circuit breaker '{self.name}' caught exception, "
                    f"using fallback: {type(e).__name__}: {e}"
                )
                try:
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Circuit breaker '{self.name}' fallback also failed: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                    raise

            # No fallback, re-raise original exception
            raise

    def protected(
        self,
        fallback: Callable[..., T] | None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for protecting a function with the circuit breaker.

        Args:
            fallback: Fallback function to use if circuit is open

        Returns:
            Decorator function

        Example:
            @breaker.protected(fallback=legacy_function)
            def new_function(x):
                return x * 2
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return self.call(func, fallback, *args, **kwargs)
            return wrapper
        return decorator

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics for this circuit breaker.

        Returns:
            Dictionary containing current metrics
        """
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'total_calls': self._metrics.total_calls,
                'successful_calls': self._metrics.successful_calls,
                'failed_calls': self._metrics.failed_calls,
                'fallback_calls': self._metrics.fallback_calls,
                'success_rate': (
                    self._metrics.successful_calls / self._metrics.total_calls
                    if self._metrics.total_calls > 0 else 0.0
                ),
                'consecutive_failures': self._consecutive_failures,
                'consecutive_successes': self._consecutive_successes,
                'state_changes': len(self._metrics.state_changes),
                'last_failure_time': (
                    self._metrics.last_failure_time.isoformat()
                    if self._metrics.last_failure_time else None
                ),
                'last_success_time': (
                    self._metrics.last_success_time.isoformat()
                    if self._metrics.last_success_time else None
                ),
            }

    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._failure_times.clear()
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerRegistry:
    """Global registry for circuit breakers.

    This provides a centralized location to manage all circuit breakers
    in the application and collect metrics.
    """

    _breakers: dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    @classmethod
    def get_breaker(
        cls,
        name: str,
        config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Unique name for the circuit breaker
            config: Configuration (only used when creating new breaker)

        Returns:
            CircuitBreaker instance
        """
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(name, config)
            return cls._breakers[name]

    @classmethod
    def get_all_metrics(cls) -> dict[str, Any]:
        """Get metrics for all registered circuit breakers.

        Returns:
            Dictionary mapping breaker names to their metrics
        """
        with cls._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in cls._breakers.items()
            }

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with cls._lock:
            for breaker in cls._breakers.values():
                breaker.reset()
