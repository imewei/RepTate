"""Application integration tests for RepTate.

These tests validate complete application workflows including:
- Application creation and initialization
- Data file loading
- Theory instantiation and calculation
- Fitting/minimization workflows

Tests are marked with:
- @pytest.mark.gui: Requires Qt event loop
- @pytest.mark.slow: May take longer than typical unit tests
- @pytest.mark.requires_data: Requires test data files
"""
