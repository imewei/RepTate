"""Core compute and data processing modules for RepTate."""

from .models import (
    DatasetRecord,
    FitResultRecord,
    ModelRecord,
    ModelRegistry,
    ModelSpec,
    PosteriorResultRecord,
    VisualizationState,
)
from .types import FitDiagnostics, FitProblem, ParameterBounds, UncertaintySummary

__all__ = [
    "DatasetRecord",
    "FitDiagnostics",
    "FitProblem",
    "FitResultRecord",
    "ModelRecord",
    "ModelRegistry",
    "ModelSpec",
    "ParameterBounds",
    "PosteriorResultRecord",
    "UncertaintySummary",
    "VisualizationState",
]

# Strangler fig infrastructure (optional, requires PyYAML for full functionality)
try:
    from .strangler_fig import (
        with_migration,
        migrate_with_validation,
        get_migration_health,
        print_migration_summary,
    )
    from .feature_flags_enhanced import (
        get_feature_flag_manager,
        is_enabled,
    )
    from .circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerRegistry,
    )
    from .dual_run import (
        DualRunner,
        ComparisonStrategy,
        DualRunRegistry,
    )
    from .migration_observability import (
        get_dashboard,
        configure_logging as configure_migration_logging,
    )

    __all__.extend([
        # Strangler fig pattern
        'with_migration',
        'migrate_with_validation',
        'get_migration_health',
        'print_migration_summary',
        # Feature flags
        'get_feature_flag_manager',
        'is_enabled',
        # Circuit breakers
        'CircuitBreaker',
        'CircuitBreakerConfig',
        'CircuitBreakerRegistry',
        # Dual-run validation
        'DualRunner',
        'ComparisonStrategy',
        'DualRunRegistry',
        # Observability
        'get_dashboard',
        'configure_migration_logging',
    ])
except ImportError as e:
    # Strangler fig infrastructure not available (missing dependencies)
    # This is OK - the infrastructure is optional
    import warnings
    warnings.warn(
        f"Strangler fig infrastructure not available: {e}. "
        "This is optional and only needed for migration workflows.",
        ImportWarning,
        stacklevel=2
    )
