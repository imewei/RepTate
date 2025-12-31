"""Observability dashboard for migration tracking.

This module provides centralized logging, metrics collection, and monitoring
for the strangler fig migration from legacy to modern implementations.

Usage:
    from RepTate.core.migration_observability import (
        MigrationDashboard,
        get_dashboard
    )

    dashboard = get_dashboard()
    dashboard.log_migration_event('jax_optimization', 'success', duration=0.123)

    # Get current migration status
    status = dashboard.get_status()
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from RepTate.core.circuit_breaker import CircuitBreakerRegistry
from RepTate.core.dual_run import DualRunRegistry
from RepTate.core.feature_flags_enhanced import get_feature_flag_manager

logger = logging.getLogger(__name__)


@dataclass
class MigrationEvent:
    """Single migration event record.

    Attributes:
        component: Component being migrated (e.g., 'jax_optimization')
        event_type: Type of event ('success', 'failure', 'fallback', 'mismatch')
        timestamp: When the event occurred
        duration: Duration of operation in seconds
        metadata: Additional context
    """
    component: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStatus:
    """Status of a single migration component.

    Attributes:
        name: Component name
        total_calls: Total number of calls
        success_count: Number of successful calls
        failure_count: Number of failed calls
        fallback_count: Number of fallbacks to legacy
        avg_duration: Average duration in seconds
        last_success: Timestamp of last success
        last_failure: Timestamp of last failure
        migration_percentage: Estimated migration completion (0-100)
    """
    name: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    avg_duration: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    migration_percentage: float = 0.0


class MigrationDashboard:
    """Centralized dashboard for tracking migration progress.

    This class aggregates data from feature flags, circuit breakers,
    and dual runners to provide a comprehensive view of migration status.
    """

    def __init__(self, log_file: Path | str | None = None):
        """Initialize the migration dashboard.

        Args:
            log_file: Optional file to log migration events (JSON Lines format)
        """
        self._lock = threading.RLock()
        self._events: list[MigrationEvent] = []
        self._component_status: dict[str, ComponentStatus] = {}
        self._duration_samples: dict[str, list[float]] = defaultdict(list)
        self._log_file = Path(log_file) if log_file else None

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_migration_event(
        self,
        component: str,
        event_type: str,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a migration event.

        Args:
            component: Component name (e.g., 'jax_optimization')
            event_type: Event type ('success', 'failure', 'fallback', 'mismatch')
            duration: Duration in seconds
            metadata: Additional context
        """
        event = MigrationEvent(
            component=component,
            event_type=event_type,
            duration=duration,
            metadata=metadata or {}
        )

        with self._lock:
            self._events.append(event)

            # Update component status
            if component not in self._component_status:
                self._component_status[component] = ComponentStatus(name=component)

            status = self._component_status[component]
            status.total_calls += 1

            if event_type == 'success':
                status.success_count += 1
                status.last_success = event.timestamp
            elif event_type == 'failure':
                status.failure_count += 1
                status.last_failure = event.timestamp
            elif event_type == 'fallback':
                status.fallback_count += 1

            # Update average duration
            self._duration_samples[component].append(duration)
            if len(self._duration_samples[component]) > 1000:
                self._duration_samples[component].pop(0)
            status.avg_duration = sum(self._duration_samples[component]) / len(
                self._duration_samples[component]
            )

            # Calculate migration percentage (success rate)
            if status.total_calls > 0:
                status.migration_percentage = (
                    status.success_count / status.total_calls
                ) * 100

        # Write to log file if configured
        if self._log_file:
            self._write_event_to_log(event)

        logger.debug(
            f"Migration event: {component} - {event_type} "
            f"(duration: {duration:.4f}s)"
        )

    def _write_event_to_log(self, event: MigrationEvent) -> None:
        """Write event to JSON Lines log file.

        Args:
            event: Event to write
        """
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                event_dict = {
                    'component': event.component,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.isoformat(),
                    'duration': event.duration,
                    'metadata': event.metadata,
                }
                f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write migration event to log: {e}")

    def get_component_status(self, component: str) -> dict[str, Any] | None:
        """Get status for a specific component.

        Args:
            component: Component name

        Returns:
            Dictionary with component status or None if not found
        """
        with self._lock:
            status = self._component_status.get(component)
            if not status:
                return None

            return {
                'name': status.name,
                'total_calls': status.total_calls,
                'success_count': status.success_count,
                'failure_count': status.failure_count,
                'fallback_count': status.fallback_count,
                'success_rate': (
                    status.success_count / status.total_calls
                    if status.total_calls > 0 else 0.0
                ),
                'avg_duration': status.avg_duration,
                'last_success': (
                    status.last_success.isoformat()
                    if status.last_success else None
                ),
                'last_failure': (
                    status.last_failure.isoformat()
                    if status.last_failure else None
                ),
                'migration_percentage': status.migration_percentage,
            }

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive migration status.

        Returns:
            Dictionary with overall migration status including:
            - Feature flag status
            - Circuit breaker metrics
            - Dual runner metrics
            - Component-level status
            - Recent events
        """
        # Get feature flag status
        flag_manager = get_feature_flag_manager()
        flag_status = flag_manager.get_all_flags()

        # Get circuit breaker metrics
        breaker_metrics = CircuitBreakerRegistry.get_all_metrics()

        # Get dual runner metrics
        runner_metrics = DualRunRegistry.get_all_metrics()

        # Get component status
        with self._lock:
            component_status = {
                name: self.get_component_status(name)
                for name in self._component_status
            }

            # Get recent events (last 100)
            recent_events = [
                {
                    'component': e.component,
                    'event_type': e.event_type,
                    'timestamp': e.timestamp.isoformat(),
                    'duration': e.duration,
                    'metadata': e.metadata,
                }
                for e in self._events[-100:]
            ]

            # Calculate overall migration percentage
            total_components = len(self._component_status)
            if total_components > 0:
                avg_migration = sum(
                    s.migration_percentage
                    for s in self._component_status.values()
                ) / total_components
            else:
                avg_migration = 0.0

        return {
            'timestamp': datetime.now().isoformat(),
            'overall_migration_percentage': avg_migration,
            'total_events': len(self._events),
            'components': component_status,
            'feature_flags': flag_status,
            'circuit_breakers': breaker_metrics,
            'dual_runners': runner_metrics,
            'recent_events': recent_events,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of migration status.

        Returns:
            Formatted string with migration summary
        """
        status = self.get_status()

        lines = [
            "=" * 70,
            "RepTate Migration Dashboard",
            "=" * 70,
            "",
            f"Overall Progress: {status['overall_migration_percentage']:.1f}%",
            f"Total Events: {status['total_events']}",
            "",
            "Component Status:",
            "-" * 70,
        ]

        for name, comp_status in status['components'].items():
            if comp_status:
                lines.append(
                    f"  {name:30} "
                    f"{comp_status['migration_percentage']:6.1f}% "
                    f"({comp_status['success_count']}/{comp_status['total_calls']} success)"
                )

        lines.extend([
            "",
            "Feature Flags:",
            "-" * 70,
        ])

        for flag_name, flag_enabled in status['feature_flags'].items():
            enabled_str = "ENABLED" if flag_enabled else "DISABLED"
            lines.append(f"  {flag_name:30} {enabled_str}")

        lines.extend([
            "",
            "Circuit Breakers:",
            "-" * 70,
        ])

        for breaker_name, breaker_metrics in status['circuit_breakers'].items():
            state = breaker_metrics['state'].upper()
            success_rate = breaker_metrics['success_rate']
            lines.append(
                f"  {breaker_name:30} {state:10} "
                f"(success rate: {success_rate:.1%})"
            )

        lines.append("=" * 70)

        return "\n".join(lines)

    def export_metrics(self, output_file: Path | str) -> None:
        """Export full metrics to JSON file.

        Args:
            output_file: Path to output file
        """
        status = self.get_status()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2)

        logger.info(f"Exported migration metrics to {output_path}")

    def get_events_since(
        self,
        since: datetime,
        component: str | None = None
    ) -> list[dict[str, Any]]:
        """Get events since a specific timestamp.

        Args:
            since: Get events after this timestamp
            component: Optional filter by component name

        Returns:
            List of event dictionaries
        """
        with self._lock:
            filtered_events = [
                e for e in self._events
                if e.timestamp >= since
                and (component is None or e.component == component)
            ]

            return [
                {
                    'component': e.component,
                    'event_type': e.event_type,
                    'timestamp': e.timestamp.isoformat(),
                    'duration': e.duration,
                    'metadata': e.metadata,
                }
                for e in filtered_events
            ]

    def get_health_alerts(self) -> list[dict[str, Any]]:
        """Get health alerts for components with issues.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Check circuit breakers
        breaker_metrics = CircuitBreakerRegistry.get_all_metrics()
        for name, metrics in breaker_metrics.items():
            if metrics['state'] == 'open':
                alerts.append({
                    'severity': 'critical',
                    'component': name,
                    'type': 'circuit_breaker_open',
                    'message': f"Circuit breaker '{name}' is OPEN",
                    'details': metrics,
                })
            elif metrics['success_rate'] < 0.9 and metrics['total_calls'] > 10:
                alerts.append({
                    'severity': 'warning',
                    'component': name,
                    'type': 'low_success_rate',
                    'message': (
                        f"Circuit breaker '{name}' has low success rate: "
                        f"{metrics['success_rate']:.1%}"
                    ),
                    'details': metrics,
                })

        # Check dual runners for mismatches
        runner_metrics = DualRunRegistry.get_all_metrics()
        for name, metrics in runner_metrics.items():
            if metrics['match_rate'] < 0.95 and metrics['total_runs'] > 10:
                alerts.append({
                    'severity': 'warning',
                    'component': name,
                    'type': 'result_mismatch',
                    'message': (
                        f"Dual runner '{name}' has mismatches: "
                        f"{metrics['match_rate']:.1%} match rate"
                    ),
                    'details': metrics,
                })

        # Check component status
        with self._lock:
            for name, status in self._component_status.items():
                if status.total_calls > 10:
                    failure_rate = status.failure_count / status.total_calls
                    if failure_rate > 0.1:
                        alerts.append({
                            'severity': 'warning',
                            'component': name,
                            'type': 'high_failure_rate',
                            'message': (
                                f"Component '{name}' has high failure rate: "
                                f"{failure_rate:.1%}"
                            ),
                            'details': self.get_component_status(name),
                        })

        return alerts


# Global singleton instance
_global_dashboard: MigrationDashboard | None = None
_dashboard_lock = threading.Lock()


def get_dashboard(log_file: Path | str | None = None) -> MigrationDashboard:
    """Get the global migration dashboard instance.

    Args:
        log_file: Optional log file path (only used on first call)

    Returns:
        Global MigrationDashboard instance
    """
    global _global_dashboard

    if _global_dashboard is None:
        with _dashboard_lock:
            if _global_dashboard is None:
                _global_dashboard = MigrationDashboard(log_file=log_file)

    return _global_dashboard


def configure_logging(
    log_level: int = logging.INFO,
    log_file: Path | str | None = None
) -> None:
    """Configure logging for migration tracking.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file to write logs to
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

    logger.info("Migration logging configured")
