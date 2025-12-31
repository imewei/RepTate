"""Contract testing suite for RepTate.

Contract tests validate integration agreements between components:

1. **API Contracts**: Input/output shapes, types, ranges
   - Theory.calculate() contracts
   - Dataset access contracts
   - Application loading contracts

2. **Data Contracts**: Field names, types, null handling
   - Dataset structure contracts
   - Parameter contracts
   - Serialization format contracts

3. **Performance Contracts**: Baseline measurements, regression detection
   - NLSQ curve fitting time budgets
   - NumPyro NUTS inference time budgets
   - Theory calculation performance

4. **Compatibility Contracts**: Migration equivalence guarantees
   - SciPy → JAX numerical equivalence
   - Native library → JAX equivalence
   - Pickle → SafeSerializer round-trip compatibility

These tests are distinct from unit tests (testing components in isolation)
and integration tests (testing workflows). Contract tests focus on the
agreements between components that allow them to work together reliably.
"""
