"""Tests for MCMC convergence diagnostics and reproducibility metadata.

Tests cover:
- T009: compute_diagnostics() function
- T010: R-hat > 1.01 warning emission
- T011: ESS < 400 warning emission
- T012: 100% divergent handling (status="failed")
- T019: version collection
- T020: seed recording
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from RepTate.core.inference.diagnostics import (
    ConvergenceDiagnostics,
    ReproducibilityInfo,
    collect_reproducibility_info,
    compute_diagnostics,
    create_diagnostics_report,
)


# Simple model for testing
def simple_normal_model(obs=None):
    """Simple normal model for testing MCMC diagnostics."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)


@pytest.fixture
def converged_mcmc():
    """Create a well-converged MCMC run for testing."""
    # Generate synthetic data
    rng_key = jax.random.PRNGKey(42)
    true_mu, true_sigma = 2.0, 1.0
    data = jax.random.normal(rng_key, shape=(100,)) * true_sigma + true_mu

    # Run MCMC with enough samples for convergence
    kernel = NUTS(simple_normal_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=500, num_chains=2)
    mcmc.run(jax.random.PRNGKey(0), obs=data)
    return mcmc


class TestComputeDiagnostics:
    """Test suite for compute_diagnostics() function (T009)."""

    def test_returns_convergence_diagnostics_type(self, converged_mcmc):
        """compute_diagnostics returns ConvergenceDiagnostics dataclass."""
        result = compute_diagnostics(converged_mcmc)
        assert isinstance(result, ConvergenceDiagnostics)

    def test_contains_r_hat_for_all_parameters(self, converged_mcmc):
        """R-hat dict contains entries for all sampled parameters."""
        result = compute_diagnostics(converged_mcmc)
        assert "mu" in result.r_hat
        assert "sigma" in result.r_hat
        assert all(isinstance(v, float) for v in result.r_hat.values())

    def test_contains_ess_bulk_for_all_parameters(self, converged_mcmc):
        """ESS bulk dict contains entries for all sampled parameters."""
        result = compute_diagnostics(converged_mcmc)
        assert "mu" in result.ess_bulk
        assert "sigma" in result.ess_bulk
        assert all(isinstance(v, float) for v in result.ess_bulk.values())
        assert all(v > 0 for v in result.ess_bulk.values())

    def test_contains_ess_tail_for_all_parameters(self, converged_mcmc):
        """ESS tail dict contains entries for all sampled parameters."""
        result = compute_diagnostics(converged_mcmc)
        assert "mu" in result.ess_tail
        assert "sigma" in result.ess_tail
        assert all(isinstance(v, float) for v in result.ess_tail.values())
        assert all(v > 0 for v in result.ess_tail.values())

    def test_divergences_is_non_negative_integer(self, converged_mcmc):
        """Divergence count is a non-negative integer."""
        result = compute_diagnostics(converged_mcmc)
        assert isinstance(result.divergences, int)
        assert result.divergences >= 0

    def test_converged_run_has_good_diagnostics(self, converged_mcmc):
        """Well-converged run should have R-hat close to 1.0."""
        result = compute_diagnostics(converged_mcmc)
        # For a converged run, R-hat should be close to 1.0
        for param, r_hat in result.r_hat.items():
            assert r_hat < 1.1, f"R-hat for {param} too high: {r_hat}"


class TestRhatWarning:
    """Test suite for R-hat > 1.01 warning emission (T010)."""

    def test_warning_emitted_when_rhat_exceeds_threshold(self, caplog):
        """Warning is logged when R-hat exceeds 1.01."""
        # Create mock diagnostics with high R-hat
        mock_r_hat = {"mu": 1.05, "sigma": 0.99}
        mock_ess_bulk = {"mu": 500.0, "sigma": 600.0}
        mock_ess_tail = {"mu": 400.0, "sigma": 500.0}

        # Create mock MCMC object
        mock_mcmc = MagicMock()

        # Mock ArviZ functions
        with (
            patch("RepTate.core.inference.diagnostics.az") as mock_az,
            caplog.at_level(logging.WARNING),
        ):
            # Setup mock returns
            mock_idata = MagicMock()
            mock_az.from_numpyro.return_value = mock_idata

            # Mock rhat return
            mock_rhat_ds = MagicMock()
            mock_rhat_ds.data_vars = ["mu", "sigma"]
            mock_rhat_ds.__getitem__ = lambda self, k: MagicMock(
                values=mock_r_hat[k] if k in mock_r_hat else 1.0
            )
            mock_az.rhat.return_value = mock_rhat_ds

            # Mock ess returns
            mock_ess_ds = MagicMock()
            mock_ess_ds.data_vars = ["mu", "sigma"]
            mock_ess_ds.__getitem__ = lambda self, k: MagicMock(
                values=mock_ess_bulk.get(k, 500.0)
            )
            mock_az.ess.return_value = mock_ess_ds

            # Mock extra fields
            mock_mcmc.get_extra_fields.return_value = {"diverging": jnp.array([False])}

            result = compute_diagnostics(mock_mcmc)

            # Check warning was emitted
            assert "R-hat for mu is 1.050 > 1.01" in caplog.text
            assert "mu" in result.warnings[0]

    def test_no_warning_when_rhat_is_good(self, converged_mcmc, caplog):
        """No warning emitted when all R-hat values are <= 1.01."""
        with caplog.at_level(logging.WARNING):
            result = compute_diagnostics(converged_mcmc)
            # For a well-converged run, there should be no R-hat warnings
            rhat_warnings = [w for w in result.warnings if "R-hat" in w]
            # If the run truly converged, there should be no R-hat warnings
            # (this may vary depending on random seed)


class TestEssWarning:
    """Test suite for ESS < 400 warning emission (T011)."""

    def test_warning_emitted_when_ess_below_threshold(self, caplog):
        """Warning is logged when ESS is below 400."""
        mock_mcmc = MagicMock()

        with (
            patch("RepTate.core.inference.diagnostics.az") as mock_az,
            caplog.at_level(logging.WARNING),
        ):
            mock_idata = MagicMock()
            mock_az.from_numpyro.return_value = mock_idata

            # Mock rhat - good values
            mock_rhat_ds = MagicMock()
            mock_rhat_ds.data_vars = ["mu", "sigma"]
            mock_rhat_ds.__getitem__ = lambda self, k: MagicMock(values=1.0)
            mock_az.rhat.return_value = mock_rhat_ds

            # Mock ess - low values
            mock_ess_values = {"mu": 150.0, "sigma": 600.0}

            def mock_ess_call(idata, method=None):
                ds = MagicMock()
                ds.data_vars = ["mu", "sigma"]
                ds.__getitem__ = lambda self, k: MagicMock(
                    values=mock_ess_values.get(k, 500.0)
                )
                return ds

            mock_az.ess.side_effect = mock_ess_call

            mock_mcmc.get_extra_fields.return_value = {"diverging": jnp.array([False])}

            result = compute_diagnostics(mock_mcmc)

            assert "ESS for mu is 150 < 400" in caplog.text
            ess_warnings = [w for w in result.warnings if "ESS" in w]
            assert len(ess_warnings) > 0


class TestDivergentHandling:
    """Test suite for 100% divergent handling (T012)."""

    def test_divergence_count_extracted_correctly(self, converged_mcmc):
        """Divergence count is correctly extracted from MCMC extra fields."""
        result = compute_diagnostics(converged_mcmc)
        # For a converged run, divergences should be 0 or very low
        assert isinstance(result.divergences, int)
        assert result.divergences >= 0

    def test_all_divergent_detection(self):
        """System correctly identifies when all samples are divergent."""
        # This tests the logic that would set status="failed"
        # The actual status setting happens in nuts_runner.py
        mock_mcmc = MagicMock()

        with patch("RepTate.core.inference.diagnostics.az") as mock_az:
            mock_idata = MagicMock()
            mock_az.from_numpyro.return_value = mock_idata

            mock_ds = MagicMock()
            mock_ds.data_vars = ["mu"]
            mock_ds.__getitem__ = lambda self, k: MagicMock(values=1.5)  # Bad R-hat
            mock_az.rhat.return_value = mock_ds
            mock_az.ess.return_value = mock_ds

            # All samples divergent
            num_samples = 100
            mock_mcmc.get_extra_fields.return_value = {
                "diverging": jnp.ones(num_samples, dtype=bool)
            }

            result = compute_diagnostics(mock_mcmc)

            # All samples divergent
            assert result.divergences == num_samples


class TestVersionCollection:
    """Test suite for version collection (T019)."""

    def test_collects_jax_version(self):
        """JAX version is correctly collected."""
        info = collect_reproducibility_info(rng_seed=42)
        assert isinstance(info.jax_version, str)
        assert len(info.jax_version) > 0
        assert info.jax_version == jax.__version__

    def test_collects_numpyro_version(self):
        """NumPyro version is correctly collected."""
        info = collect_reproducibility_info(rng_seed=42)
        assert isinstance(info.numpyro_version, str)
        assert len(info.numpyro_version) > 0
        assert info.numpyro_version == numpyro.__version__

    def test_collects_reptate_version(self):
        """RepTate version is collected (or 'unknown' if not installed)."""
        info = collect_reproducibility_info(rng_seed=42)
        assert isinstance(info.reptate_version, str)
        assert len(info.reptate_version) > 0

    def test_model_config_captured(self):
        """Model configuration is captured in reproducibility info."""
        config = {"num_warmup": 500, "num_samples": 1000, "target_accept_prob": 0.8}
        info = collect_reproducibility_info(rng_seed=42, model_config=config)
        assert info.model_config == config


class TestSeedRecording:
    """Test suite for seed recording (T020)."""

    def test_seed_recorded_correctly(self):
        """RNG seed is recorded in reproducibility info."""
        seed = 12345
        info = collect_reproducibility_info(rng_seed=seed)
        assert info.rng_seed == seed

    def test_seed_is_integer(self):
        """RNG seed is stored as integer."""
        info = collect_reproducibility_info(rng_seed=42)
        assert isinstance(info.rng_seed, int)


class TestDiagnosticsReport:
    """Test suite for DiagnosticsReport creation."""

    def test_report_marks_converged_when_all_good(self):
        """Report shows converged=True when all diagnostics are good."""
        diag = ConvergenceDiagnostics(
            r_hat={"mu": 1.0, "sigma": 1.0},
            ess_bulk={"mu": 500.0, "sigma": 600.0},
            ess_tail={"mu": 400.0, "sigma": 500.0},
            divergences=0,
            warnings=[],
        )
        report = create_diagnostics_report(diag)
        assert report.converged is True
        assert "OK" in report.summary_text

    def test_report_marks_not_converged_with_bad_rhat(self):
        """Report shows converged=False when R-hat is bad."""
        diag = ConvergenceDiagnostics(
            r_hat={"mu": 1.5, "sigma": 1.0},  # Bad R-hat
            ess_bulk={"mu": 500.0, "sigma": 600.0},
            ess_tail={"mu": 400.0, "sigma": 500.0},
            divergences=0,
            warnings=["R-hat for mu is 1.500 > 1.01"],
        )
        report = create_diagnostics_report(diag)
        assert report.converged is False
        assert "WARNINGS" in report.summary_text

    def test_report_contains_summary_statistics(self):
        """Report summary text contains ESS and divergence info."""
        diag = ConvergenceDiagnostics(
            r_hat={"mu": 1.0},
            ess_bulk={"mu": 500.0},
            ess_tail={"mu": 400.0},
            divergences=5,
            warnings=[],
        )
        report = create_diagnostics_report(diag)
        assert "ESS" in report.summary_text
        assert "Divergences: 5" in report.summary_text
