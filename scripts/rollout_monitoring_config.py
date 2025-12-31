"""Configuration for progressive rollout monitoring and metrics collection.

This module defines:
- Alert thresholds for automatic rollback
- SLO definitions for each rollout phase
- Metrics to collect and track
- Dashboard configurations
- Runbook triggers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class Phase(Enum):
    """Rollout phases."""
    NONE = 0
    CANARY = 1
    EARLY_ADOPTERS = 2
    GRADUAL = 3
    FULL = 4


class Severity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class Threshold:
    """Alert threshold configuration."""
    name: str
    metric: str
    operator: str  # >, <, >=, <=, ==, !=
    value: float
    duration_seconds: int = 300  # How long threshold must be exceeded
    severity: Severity = Severity.WARNING
    description: str = ""


@dataclass
class SLO:
    """Service Level Objective for a rollout phase."""
    phase: Phase
    error_rate_max: float = 0.01  # 1%
    latency_p99_baseline_multiplier: float = 1.2  # P99 can be 1.2x baseline
    latency_p95_baseline_multiplier: float = 1.15  # P95 can be 1.15x baseline
    memory_baseline_multiplier: float = 1.5  # Memory can be 1.5x baseline
    cpu_utilization_max: float = 0.8  # 80% CPU
    feature_flag_coverage_min: float = 0.95  # 95% of code paths covered
    assertion_failure_rate_max: float = 0.001  # 0.1% of dual-runs
    data_loss_threshold: int = 0  # Zero tolerance


@dataclass
class MetricDefinition:
    """Definition of a metric to track."""
    name: str
    description: str
    unit: str
    metric_type: str  # gauge, counter, histogram, summary
    tags: List[str] = field(default_factory=list)
    rollup_intervals: List[str] = field(default_factory=lambda: ['1m', '5m', '1h', '1d'])


@dataclass
class RolloutMetrics:
    """Complete metrics configuration for rollout monitoring."""

    # Core metrics
    error_rate: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_error_rate',
            description='Request error rate',
            unit='percentage',
            metric_type='gauge',
            tags=['core', 'reliability']
        )
    )

    latency_p50: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_latency_p50',
            description='50th percentile latency',
            unit='milliseconds',
            metric_type='gauge',
            tags=['core', 'performance']
        )
    )

    latency_p95: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_latency_p95',
            description='95th percentile latency',
            unit='milliseconds',
            metric_type='gauge',
            tags=['core', 'performance']
        )
    )

    latency_p99: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_latency_p99',
            description='99th percentile latency',
            unit='milliseconds',
            metric_type='gauge',
            tags=['core', 'performance']
        )
    )

    memory_usage: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_memory_usage',
            description='Memory usage',
            unit='megabytes',
            metric_type='gauge',
            tags=['core', 'resources']
        )
    )

    cpu_usage: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_cpu_usage',
            description='CPU utilization',
            unit='percentage',
            metric_type='gauge',
            tags=['core', 'resources']
        )
    )

    # Migration metrics
    feature_flag_enabled_percentage: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_feature_flag_enabled',
            description='Percentage of traffic using new code',
            unit='percentage',
            metric_type='gauge',
            tags=['migration', 'feature-flags']
        )
    )

    code_path_distribution: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_code_path_distribution',
            description='Distribution between new and legacy code paths',
            unit='count',
            metric_type='counter',
            tags=['migration', 'traffic']
        )
    )

    fallback_count: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_fallback_count',
            description='Number of fallbacks to legacy code',
            unit='count',
            metric_type='counter',
            tags=['migration', 'fallback']
        )
    )

    dual_run_assertions: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_dual_run_assertions',
            description='Dual-run assertion results',
            unit='count',
            metric_type='counter',
            tags=['migration', 'validation']
        )
    )

    mismatch_rate: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_mismatch_rate',
            description='Rate of dual-run mismatches',
            unit='percentage',
            metric_type='gauge',
            tags=['migration', 'validation']
        )
    )

    # Component-specific metrics
    component_error_rate: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_component_error_rate',
            description='Error rate by component',
            unit='percentage',
            metric_type='gauge',
            tags=['component', 'reliability']
        )
    )

    component_latency: MetricDefinition = field(
        default_factory=lambda: MetricDefinition(
            name='reptate_component_latency',
            description='Latency by component',
            unit='milliseconds',
            metric_type='gauge',
            tags=['component', 'performance']
        )
    )


# SLO definitions for each phase
SLO_BY_PHASE = {
    Phase.CANARY: SLO(
        phase=Phase.CANARY,
        error_rate_max=0.001,  # 0.1%
        latency_p99_baseline_multiplier=1.05,  # Very strict
        latency_p95_baseline_multiplier=1.05,
        memory_baseline_multiplier=1.2,
        assertion_failure_rate_max=0.0001,  # Almost zero
    ),
    Phase.EARLY_ADOPTERS: SLO(
        phase=Phase.EARLY_ADOPTERS,
        error_rate_max=0.005,  # 0.5%
        latency_p99_baseline_multiplier=1.1,
        latency_p95_baseline_multiplier=1.1,
        memory_baseline_multiplier=1.3,
        assertion_failure_rate_max=0.001,
    ),
    Phase.GRADUAL: SLO(
        phase=Phase.GRADUAL,
        error_rate_max=0.01,  # 1%
        latency_p99_baseline_multiplier=1.15,
        latency_p95_baseline_multiplier=1.15,
        memory_baseline_multiplier=1.5,
        assertion_failure_rate_max=0.005,
    ),
    Phase.FULL: SLO(
        phase=Phase.FULL,
        error_rate_max=0.01,  # 1% (match or beat baseline)
        latency_p99_baseline_multiplier=1.0,  # Must match baseline
        latency_p95_baseline_multiplier=1.0,
        memory_baseline_multiplier=1.2,  # Some headroom for JIT
    ),
}

# Critical thresholds that trigger immediate rollback
CRITICAL_THRESHOLDS = [
    Threshold(
        name='error_rate_critical',
        metric='error_rate',
        operator='>',
        value=0.05,  # 5%
        duration_seconds=300,  # 5 minutes
        severity=Severity.CRITICAL,
        description='Error rate exceeded 5% for 5 minutes'
    ),
    Threshold(
        name='latency_p99_spike',
        metric='latency_p99',
        operator='>',
        value=2.0,  # 2x baseline (applied as multiplier)
        duration_seconds=600,  # 10 minutes
        severity=Severity.CRITICAL,
        description='P99 latency exceeded 2x baseline for 10 minutes'
    ),
    Threshold(
        name='memory_spike',
        metric='memory_usage',
        operator='>',
        value=2.0,  # 2x baseline
        duration_seconds=300,
        severity=Severity.CRITICAL,
        description='Memory usage exceeded 2x baseline'
    ),
    Threshold(
        name='assertion_failures',
        metric='dual_run_assertions_failure_count',
        operator='>',
        value=100,  # per hour
        duration_seconds=3600,  # 1 hour
        severity=Severity.CRITICAL,
        description='Dual-run assertion failures exceeded 100 per hour'
    ),
    Threshold(
        name='service_crash',
        metric='service_available',
        operator='==',
        value=0,
        duration_seconds=60,
        severity=Severity.CRITICAL,
        description='Service unavailable'
    ),
]

# Warning thresholds that escalate for investigation
WARNING_THRESHOLDS = [
    Threshold(
        name='error_rate_warning',
        metric='error_rate',
        operator='>',
        value=0.02,  # 2%
        duration_seconds=1800,  # 30 minutes
        severity=Severity.WARNING,
        description='Error rate exceeded 2% for 30 minutes'
    ),
    Threshold(
        name='latency_p95_warning',
        metric='latency_p95',
        operator='>',
        value=1.5,  # 1.5x baseline
        duration_seconds=1800,
        severity=Severity.WARNING,
        description='P95 latency exceeded 1.5x baseline for 30 minutes'
    ),
    Threshold(
        name='memory_growth_warning',
        metric='memory_growth_rate',
        operator='>',
        value=0.1,  # 10% per hour
        duration_seconds=3600,
        severity=Severity.WARNING,
        description='Memory growing at 10% per hour'
    ),
    Threshold(
        name='mismatch_rate_warning',
        metric='mismatch_rate',
        operator='>',
        value=0.01,  # 1%
        duration_seconds=1800,
        severity=Severity.WARNING,
        description='Dual-run mismatch rate exceeded 1% for 30 minutes'
    ),
]


# Dashboard configurations
class DashboardConfig:
    """Grafana/monitoring dashboard configurations."""

    @staticmethod
    def get_dashboard_definition() -> Dict[str, Any]:
        """Get dashboard definition for Grafana or similar."""
        return {
            'title': 'RepTate Progressive Rollout',
            'description': 'Monitoring dashboard for modernization migration',
            'tags': ['reptate', 'rollout', 'migration'],
            'panels': [
                {
                    'title': 'Error Rate',
                    'type': 'graph',
                    'metrics': ['reptate_error_rate'],
                    'alert': 'error_rate_critical',
                    'position': 'top-left'
                },
                {
                    'title': 'Latency (p99)',
                    'type': 'graph',
                    'metrics': ['reptate_latency_p99'],
                    'alert': 'latency_p99_spike',
                    'position': 'top-right'
                },
                {
                    'title': 'Memory Usage',
                    'type': 'graph',
                    'metrics': ['reptate_memory_usage'],
                    'alert': 'memory_spike',
                    'position': 'middle-left'
                },
                {
                    'title': 'Feature Flag Distribution',
                    'type': 'pie',
                    'metrics': ['reptate_code_path_distribution'],
                    'position': 'middle-right'
                },
                {
                    'title': 'Dual-Run Status',
                    'type': 'stat',
                    'metrics': [
                        'reptate_dual_run_assertions',
                        'reptate_mismatch_rate'
                    ],
                    'position': 'bottom-left'
                },
                {
                    'title': 'Error Rate by Component',
                    'type': 'heatmap',
                    'metrics': ['reptate_component_error_rate'],
                    'position': 'bottom-right'
                }
            ]
        }


# Metrics collection configuration
class MetricsCollectionConfig:
    """Configuration for metrics collection during rollout."""

    # Collection intervals
    COLLECTION_INTERVALS = {
        'phase_1': 15,  # Canary: every 15 seconds
        'phase_2': 30,  # Early adopters: every 30 seconds
        'phase_3': 60,  # Gradual: every 1 minute
        'phase_4': 300,  # Full: every 5 minutes
    }

    # Retention periods for metrics
    RETENTION_PERIODS = {
        'real_time': '1h',  # Live dashboards
        'short_term': '7d',  # Phase analysis
        'long_term': '90d',  # Historical trends
    }

    # Sampling strategy
    SAMPLING = {
        'requests': 1.0,  # Sample all requests in Phase 1
        'errors': 1.0,  # Sample all errors
        'slow_requests': 0.5,  # Sample 50% of slow requests
        'user_sessions': 0.1,  # Sample 10% of sessions
    }


# Runbook triggers - what events should trigger which runbooks
class RunbookTriggers:
    """Configuration for automatic runbook triggering."""

    TRIGGERS = {
        'error_rate_critical': {
            'runbook': 'EMERGENCY_ROLLBACK',
            'severity': 'critical',
            'auto_execute': True,
            'wait_for_confirmation': False,
        },
        'latency_spike': {
            'runbook': 'INVESTIGATE_LATENCY',
            'severity': 'warning',
            'auto_execute': False,
            'wait_for_confirmation': True,
        },
        'memory_leak': {
            'runbook': 'INVESTIGATE_MEMORY',
            'severity': 'warning',
            'auto_execute': False,
            'wait_for_confirmation': True,
        },
        'assertion_failures': {
            'runbook': 'INVESTIGATE_COMPATIBILITY',
            'severity': 'warning',
            'auto_execute': False,
            'wait_for_confirmation': True,
        },
    }


def get_baseline_metrics() -> Dict[str, float]:
    """Get baseline metrics for comparison.

    These should be established before rollout begins.
    Return actual values from your legacy system.
    """
    return {
        'error_rate': 0.001,  # 0.1%
        'latency_p50': 50.0,  # ms
        'latency_p95': 150.0,  # ms
        'latency_p99': 500.0,  # ms
        'memory_usage': 256.0,  # MB
        'cpu_usage': 30.0,  # %
    }


def get_phase_metrics_checklist(phase: Phase) -> Dict[str, str]:
    """Get metrics checklist for a specific phase.

    Returns dict of metric_name: expected_status
    """
    slo = SLO_BY_PHASE[phase]

    return {
        'error_rate': f"< {slo.error_rate_max * 100}%",
        'latency_p99': f"< {slo.latency_p99_baseline_multiplier}x baseline",
        'latency_p95': f"< {slo.latency_p95_baseline_multiplier}x baseline",
        'memory_usage': f"< {slo.memory_baseline_multiplier}x baseline",
        'cpu_usage': f"< {slo.cpu_utilization_max * 100}%",
        'feature_flag_coverage': f"> {slo.feature_flag_coverage_min * 100}%",
        'assertion_failures': f"< {slo.assertion_failure_rate_max * 100}%",
        'data_loss': f"= {slo.data_loss_threshold}",
    }


def validate_threshold(metric_value: float, threshold: Threshold, baseline: float = 1.0) -> bool:
    """Validate if a metric value violates a threshold.

    Args:
        metric_value: Current metric value
        threshold: Threshold definition
        baseline: Baseline value for multiplier-based thresholds

    Returns:
        True if threshold is violated, False if OK
    """
    # For multiplier-based thresholds (latency, memory)
    if isinstance(threshold.value, float) and threshold.value < 10:
        # Likely a multiplier
        effective_threshold = baseline * threshold.value
    else:
        effective_threshold = threshold.value

    if threshold.operator == '>':
        return metric_value > effective_threshold
    elif threshold.operator == '<':
        return metric_value < effective_threshold
    elif threshold.operator == '>=':
        return metric_value >= effective_threshold
    elif threshold.operator == '<=':
        return metric_value <= effective_threshold
    elif threshold.operator == '==':
        return metric_value == effective_threshold
    elif threshold.operator == '!=':
        return metric_value != effective_threshold
    else:
        raise ValueError(f"Unknown operator: {threshold.operator}")


if __name__ == '__main__':
    # Example usage
    print("Progressive Rollout Monitoring Configuration")
    print("=" * 60)

    print("\nPhase 1 (Canary) SLO:")
    print(f"  Max Error Rate: {SLO_BY_PHASE[Phase.CANARY].error_rate_max * 100}%")
    print(f"  Max P99 Latency: {SLO_BY_PHASE[Phase.CANARY].latency_p99_baseline_multiplier}x baseline")

    print("\nCritical Thresholds:")
    for threshold in CRITICAL_THRESHOLDS:
        print(f"  - {threshold.name}: {threshold.description}")

    print("\nMetrics Checklist for Phase 1:")
    checklist = get_phase_metrics_checklist(Phase.CANARY)
    for metric, expected in checklist.items():
        print(f"  - {metric}: {expected}")

    print("\nDashboard Configuration:")
    dashboard = DashboardConfig.get_dashboard_definition()
    print(f"  Title: {dashboard['title']}")
    print(f"  Panels: {len(dashboard['panels'])}")

    print("\n" + "=" * 60)
    print("Configuration loaded successfully")
