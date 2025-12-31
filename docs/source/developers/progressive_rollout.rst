Progressive Rollout Strategy and Monitoring
=============================================

Overview
--------

This document describes RepTate's progressive rollout strategy for the modernization migration (003-reptate-modernization). The strategy uses a strangler fig pattern with feature flags, traffic shaping, and comprehensive monitoring to safely migrate from legacy code to modern JAX-based implementations.

Rollout Phases
--------------

All rollout phases follow the same progression:

.. code-block:: text

    5% Traffic -> 25% Traffic -> 50% Traffic -> 100% Traffic
    (1 hour)      (4 hours)      (8 hours)     (Permanent)

Each phase includes:

1. **Health Checks**: Error rate, latency, and memory usage monitoring
2. **Metrics Collection**: Feature flag usage, dual-run comparisons, fallback frequency
3. **Manual Gate**: Team review before advancing to next phase
4. **Automatic Rollback**: If thresholds are violated

Phase Details
~~~~~~~~~~~~~

**Phase 1: Canary (5% Traffic)**

- Duration: 1 hour minimum
- Scope: Internal testing, limited user subset
- Monitoring: Real-time dashboards
- Rollback: Immediate if any errors
- Success Criteria:
  - Zero critical errors
  - Error rate < 0.1%
  - Latency increase < 5%
  - All assertions pass in dual-run mode

**Phase 2: Early Adopters (25% Traffic)**

- Duration: 4 hours minimum
- Scope: Beta users, opt-in users
- Monitoring: Aggregate metrics, daily reports
- Rollback: Automatic at error rate > 0.5%
- Success Criteria:
  - Error rate < 0.5%
  - Latency increase < 10%
  - Memory usage stable
  - Feature flag coverage > 95%

**Phase 3: Gradual Rollout (50% Traffic)**

- Duration: 8 hours minimum
- Scope: General population, 50% split
- Monitoring: SLO tracking, DORA metrics
- Rollback: Automatic at error rate > 1%
- Success Criteria:
  - Error rate < 1%
  - Latency increase < 15%
  - P95 latency stable
  - No data corruption issues

**Phase 4: Full Rollout (100% Traffic)**

- Duration: Permanent
- Scope: All users
- Monitoring: Production metrics
- Rollback: Manual decision after critical incident
- Success Criteria:
  - All SLOs met
  - Zero regressions vs baseline
  - Feature flag can be hardcoded as enabled

Automatic Rollback Triggers
---------------------------

The system automatically rolls back to the previous phase if:

**Critical Thresholds (Immediate)**

- Error rate > 5% for 5 minutes
- Latency p99 > 2x baseline for 10 minutes
- Memory usage > 2x baseline
- Assertion failures > 100 in dual-run mode
- Service crashes or hangs

**Warning Thresholds (After 30 minutes)**

- Error rate > 2% sustained
- Latency p95 > 1.5x baseline sustained
- CPU usage > 80% sustained
- Persistent feature flag mismatches

**Recovery Process**

.. code-block:: yaml

    Rollback Procedure:
      1. Monitor detects violation
      2. Automatic revert to previous phase (via feature flags)
      3. Alert sent to on-call team
      4. Metrics snapshot captured for analysis
      5. Post-mortem scheduled within 24 hours
      6. New phase only starts after issue resolved + waiting period

Feature Flags and Traffic Shaping
---------------------------------

Traffic distribution is controlled via feature flags:

.. code-block:: python

    # Feature flags for each modernization component
    FEATURE_FLAGS = {
        'USE_SAFE_EVAL': {
            'default': False,
            'enabled_percentage': 5,  # Phase 1: 5%
            'description': 'Use safe expression evaluator'
        },
        'USE_SAFE_SERIALIZATION': {
            'default': False,
            'enabled_percentage': 5,
            'description': 'Use JSON/NPZ instead of pickle'
        },
        'USE_JAX_OPTIMIZATION': {
            'default': False,
            'enabled_percentage': 5,
            'description': 'Use JAX-based optimization'
        }
    }

Traffic Distribution Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import hashlib

    def should_use_new_code(user_id: str, percentage: int) -> bool:
        """Consistent hash-based traffic splitting.

        Args:
            user_id: Consistent identifier (user ID, session ID, etc.)
            percentage: Percentage of traffic to route to new code (0-100)

        Returns:
            True if this user should use the new code
        """
        # Use consistent hashing so same user always gets same code path
        hash_value = int(hashlib.md5(f"{user_id}".encode()).hexdigest(), 16)
        return (hash_value % 100) < percentage

Key Properties:

- **Consistency**: Same user always routes to same code path
- **Gradual**: Increasing percentage smoothly migrates traffic
- **Reversible**: Decreasing percentage immediately rolls back
- **No Session Affinity**: Works across multiple sessions

Monitoring and Observability
-----------------------------

Key Metrics to Track
~~~~~~~~~~~~~~~~~~~~

1. **Error Metrics**

   - Overall error rate (%)
   - Errors by component
   - New code vs legacy code error rates
   - Error types and stack traces

2. **Performance Metrics**

   - Response latency (p50, p95, p99)
   - Throughput (requests/sec)
   - Memory usage (MB, percentile)
   - CPU usage (%)

3. **Migration Metrics**

   - Feature flag adoption rate
   - Code path distribution
   - Fallback frequency
   - Dual-run assertion failures
   - Mismatch detection rate

4. **Business Metrics**

   - User engagement
   - Completion rates
   - Error impact on workflows
   - User satisfaction

Monitoring Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~

The observability system uses:

1. **Feature Flag Dashboard**

   - Current percentage enabled for each flag
   - Traffic distribution over time
   - Manual override controls

2. **Metrics Collection**

   - Application logs (JSON Lines format)
   - Custom metrics via MigrationDashboard
   - Performance baselines

3. **Alerting Rules**

   - Error rate anomalies
   - Latency spike detection
   - Memory leak indicators
   - Custom SLO violations

4. **Runbooks**

   - Automatic response procedures
   - Manual intervention guides
   - Escalation paths

Data Collection Points
~~~~~~~~~~~~~~~~~~~~~~

Each critical component logs:

.. code-block:: python

    from RepTate.core.migration_observability import get_dashboard

    dashboard = get_dashboard()

    # Log successful execution
    dashboard.log_migration_event(
        component='jax_optimization',
        event_type='success',
        duration=0.123,
        metadata={
            'user_id': user_id,
            'data_size': len(data),
            'feature_flag_enabled': True
        }
    )

    # Log failures
    dashboard.log_migration_event(
        component='jax_optimization',
        event_type='failure',
        duration=0.050,
        metadata={
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    )

    # Log fallbacks
    dashboard.log_migration_event(
        component='jax_optimization',
        event_type='fallback',
        metadata={
            'reason': 'legacy_compatibility'
        }
    )

Dual-Run Mode
~~~~~~~~~~~~~

During rollout phases 1-3, the system runs both implementations and compares results:

.. code-block:: python

    from RepTate.core.dual_run import dual_run

    @dual_run('optimization')
    def optimize_polymer_chain(params, data):
        """Runs both legacy and JAX implementations.

        - Compares results for consistency
        - Logs mismatches for investigation
        - Returns new implementation if enabled, legacy otherwise
        """
        pass

Dashboard Configuration
-----------------------

Key Dashboard Views
~~~~~~~~~~~~~~~~~~~

1. **Rollout Status Dashboard**

   - Current traffic percentage
   - Time in current phase
   - Phase progression timeline
   - Manual controls (pause, rollback)

2. **Error Rate Dashboard**

   - Overall error rate trend
   - Error rate by component
   - Error rate by code path (new vs legacy)
   - Error classification

3. **Performance Dashboard**

   - Latency distribution (p50, p95, p99)
   - Latency comparison (new vs legacy)
   - Memory usage trend
   - CPU usage trend

4. **Migration Dashboard**

   - Feature flag adoption
   - Code path distribution
   - Fallback rate
   - Assertion failures
   - Dual-run status

5. **SLO Dashboard**

   - Error rate vs threshold
   - Latency vs threshold
   - Memory vs threshold
   - Overall SLO status

Dashboard Alerts
~~~~~~~~~~~~~~~~

Configure alerts for:

.. code-block:: yaml

    alerts:
      error_rate_critical:
        condition: "error_rate > 5% for 5 minutes"
        action: "IMMEDIATE ROLLBACK"
        notify: "on-call team"

      error_rate_warning:
        condition: "error_rate > 2% for 30 minutes"
        action: "NOTIFY + ESCALATION"
        notify: "engineering team"

      latency_spike:
        condition: "p99_latency > 2x baseline for 10 minutes"
        action: "INVESTIGATION + POSSIBLE ROLLBACK"
        notify: "on-call team"

      memory_leak:
        condition: "memory usage growing 10%/hour"
        action: "NOTIFY + INVESTIGATION"
        notify: "performance team"

      assertion_failures:
        condition: "dual_run assertions fail > 100/hour"
        action: "STOP MIGRATION + INVESTIGATE"
        notify: "engineering team"

Rollback Procedures
-------------------

Manual Rollback (Non-Emergency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Decision**: Team agrees to rollback
2. **Notification**: Alert all stakeholders
3. **Execution**: Reset feature flag percentages to 0%
4. **Verification**: Confirm all traffic on legacy code
5. **Analysis**: Root cause analysis and fix
6. **Retry**: Plan for next attempt after fixes

Automatic Rollback (Emergency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Detection**: Monitoring detects critical threshold violation
2. **Action**: Automatically reset feature flag to 0%
3. **Notification**: Page on-call team immediately
4. **Capture**: Snapshot all metrics and logs
5. **Communication**: Notify users if needed
6. **Investigation**: Full incident investigation

Rollback Runbook
~~~~~~~~~~~~~~~~

.. code-block:: text

    EMERGENCY ROLLBACK RUNBOOK
    ==========================

    Trigger: Error rate > 5% OR P99 latency > 2x baseline

    Step 1: Verify Alert
    - Check dashboard for confirmation
    - Contact on-call engineer

    Step 2: Immediate Action
    - Execute: REPTATE_ROLLBACK_PHASE=0
    - Verify: Monitor traffic distribution
    - Confirm: All users on legacy code

    Step 3: Impact Assessment
    - Count affected users
    - Identify impacted functionality
    - Check for data corruption

    Step 4: Communication
    - Notify stakeholders
    - Prepare status message for users
    - Document incident

    Step 5: Investigation
    - Collect metrics snapshots
    - Review logs for errors
    - Identify root cause
    - Plan remediation

    Step 6: Post-Mortem
    - Schedule within 24 hours
    - Include all relevant teams
    - Document findings
    - Plan prevention

Communication Plan
------------------

Internal Communication
~~~~~~~~~~~~~~~~~~~~~~

- **Daily Standup**: Brief rollout status update
- **Issue Escalation**: Immediate notification for any problems
- **Weekly Review**: Metrics review and next phase planning
- **Post-Mortem**: Incident investigation and lessons learned

User Communication
~~~~~~~~~~~~~~~~~~~

- **Pre-Launch**: Announce modernization improvements
- **During Rollout**: Transparent status updates
- **On Issues**: Clear explanations and ETAs
- **Post-Completion**: Success celebration and thanks

Stakeholder Involvement
~~~~~~~~~~~~~~~~~~~~~~~

- **Data Team**: Monitor metrics, flag anomalies
- **Engineering**: Investigate failures, implement fixes
- **QA**: Verify functionality, test edge cases
- **Product**: Monitor user impact, make go/no-go decisions
- **Ops**: Infrastructure support, incident response

Success Criteria
----------------

Phase Promotion Criteria
~~~~~~~~~~~~~~~~~~~~~~~~

Before advancing to the next phase:

1. **Metrics Meet Targets**

   - Error rate < threshold for entire phase duration
   - Latency within acceptable range
   - Memory usage stable
   - CPU usage acceptable

2. **No Critical Issues**

   - Zero data corruption
   - Zero assertion failures in dual-run
   - No patterns in errors
   - No regressions vs baseline

3. **Team Consensus**

   - Engineering team approves
   - QA confirms testing complete
   - Data team validates metrics
   - Product agrees on user impact

4. **Documentation Complete**

   - Runbooks updated
   - Known issues documented
   - Metrics baseline established
   - Recovery procedures tested

Overall Success Criteria (100% Rollout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Error rate matches or beats legacy baseline
- Latency matches or beats legacy baseline
- Memory usage acceptable
- All users migrated successfully
- Zero data loss or corruption
- Positive user feedback
- Team confidence in modernization

Timeline Example
----------------

.. code-block:: text

    Tuesday 10:00 - Start Phase 1 (5% traffic)
    Tuesday 11:00 - Evaluate metrics, advance if good

    Tuesday 11:00 - Start Phase 2 (25% traffic)
    Tuesday 15:00 - Evaluate metrics, advance if good

    Tuesday 15:00 - Start Phase 3 (50% traffic)
    Tuesday 23:00 - Evaluate metrics, advance if good

    Wednesday 00:00 - Start Phase 4 (100% traffic)
    Wednesday 24:00 - Monitor for stability

    Thursday - Verify success, hardcode flag as enabled
    Thursday - Run full regression tests
    Friday - Celebrate and plan next modernization

This timeline is flexible based on metrics and team confidence.

Testing Strategy
----------------

Pre-Rollout Testing
~~~~~~~~~~~~~~~~~~~

Before starting rollout:

1. **Unit Tests**: All new code has > 90% coverage
2. **Integration Tests**: New and legacy code work together
3. **Performance Tests**: Baseline performance established
4. **Compatibility Tests**: Data formats are compatible
5. **Edge Cases**: Error conditions handled correctly

During-Rollout Testing
~~~~~~~~~~~~~~~~~~~~~~

1. **Continuous Testing**: Automated tests on every commit
2. **Smoke Tests**: Quick validation of critical paths
3. **Load Testing**: Verify performance under traffic
4. **Chaos Testing**: Verify failure modes
5. **Security Testing**: Check for vulnerabilities

Post-Rollout Testing
~~~~~~~~~~~~~~~~~~~~

1. **Regression Tests**: Verify no regressions vs baseline
2. **Stress Tests**: Long-running stability tests
3. **Data Validation**: Verify data integrity
4. **User Acceptance**: Real user scenarios

Contingency Plans
-----------------

If Serious Issues Are Discovered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Immediate**: Pause rollout (stay at current phase)
2. **Investigation**: Root cause analysis
3. **Fix**: Implement and test fix
4. **Restart**: Resume rollout after issue confirmed resolved
5. **Documentation**: Update runbooks and processes

If Rollback Occurs
~~~~~~~~~~~~~~~~~~

1. **Duration**: Minimum 1 week before next attempt
2. **Analysis**: Deep investigation of root cause
3. **Fix Verification**: All tests pass on fix
4. **Simplified Approach**: Consider phased approach or splitting feature
5. **Team Confidence**: Build confidence through testing

Key Contacts
------------

- **On-Call Engineer**: Pages for critical issues (during 5% phase)
- **Engineering Lead**: Approves phase advancement
- **Data Team Lead**: Validates metrics
- **Product Manager**: Makes go/no-go decisions
- **Infrastructure Team**: Infrastructure support

Appendix: Feature Flag Configuration
-------------------------------------

Feature flags are configured in:

.. code-block:: text

    src/RepTate/core/feature_flags_enhanced.py

    Key functions:
    - get_feature_flag_manager(): Get flag manager
    - is_enabled(flag_name): Check if flag is enabled
    - set_percentage(flag_name, percentage): Set traffic percentage
    - get_status(): Get all flag statuses

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Control flags via environment:

.. code-block:: bash

    # Enable a flag for all users
    export REPTATE_USE_SAFE_EVAL=true

    # Disable a flag for all users
    export REPTATE_USE_SAFE_EVAL=false

    # Use default (from configuration)
    unset REPTATE_USE_SAFE_EVAL

Dashboard Access
~~~~~~~~~~~~~~~~

- **Grafana**: https://monitoring.reptate.example.com
- **Logs**: ELK stack at https://logs.reptate.example.com
- **Metrics**: Prometheus at https://metrics.reptate.example.com
- **Alerts**: PagerDuty for critical issues

References
----------

- Feature Flag Implementation: :doc:`../api/feature_flags`
- Migration Observability: :doc:`../api/migration_observability`
- CI/CD Pipeline: :doc:`../ci_cd/github_actions`
- Testing Strategy: :doc:`../testing/strategy`
