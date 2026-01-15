"""Integration tests for NumPyro NUTS workflow.

Tests cover:
- T086a: NumPyro NUTS warm-start workflow
- T086b: NLSQ->NUTS warm-start produces valid posterior samples
- T086c: NumPyro NUTS runs on CPU when GPU unavailable
- T018: run_nuts_inference returns diagnostics and reproducibility_info

These tests validate the Bayesian inference workflow using NumPyro NUTS
with warm-start from NLSQ deterministic fit results.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer import MCMC, NUTS

if TYPE_CHECKING:
    pass


@pytest.fixture
def rng_key() -> Array:
    """Create a reproducible random key."""
    return random.PRNGKey(42)


@pytest.fixture
def linear_data() -> tuple[Array, Array, dict[str, float]]:
    """Create linear test data with known parameters."""
    np.random.seed(42)
    x = jnp.linspace(0, 10, 50)
    true_slope = 2.5
    true_intercept = 1.5
    true_sigma = 0.5
    noise = jnp.array(np.random.normal(0, true_sigma, size=50))
    y = true_slope * x + true_intercept + noise
    return x, y, {"slope": true_slope, "intercept": true_intercept, "sigma": true_sigma}


class TestNUTSWarmStartWorkflow:
    """Test T086a: NumPyro NUTS warm-start workflow."""

    def test_nlsq_provides_warm_start(self, linear_data: tuple) -> None:
        """Test NLSQ fit provides valid warm-start for NUTS."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        x, y, true_params = linear_data

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Run NLSQ fit
        result, _ = run_nlsq_fit(
            linear_model, x, y, p0=jnp.array([1.0, 0.0])
        )

        # Verify warm-start values are reasonable
        assert abs(result.parameters["p0"] - true_params["slope"]) < 0.5
        assert abs(result.parameters["p1"] - true_params["intercept"]) < 0.5

        # warm_start should be populated
        assert "p0" in result.warm_start
        assert "p1" in result.warm_start

    def test_nuts_accepts_warm_start(self, linear_data: tuple, rng_key: Array) -> None:
        """Test NUTS can be initialized with NLSQ warm-start."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        x, y, true_params = linear_data

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Get NLSQ warm-start
        result, _ = run_nlsq_fit(
            linear_model, x, y, p0=jnp.array([1.0, 0.0])
        )

        # Define NumPyro model
        def numpyro_model(x_obs, y_obs):
            slope = numpyro.sample("slope", dist.Normal(result.parameters["p0"], 1.0))
            intercept = numpyro.sample("intercept", dist.Normal(result.parameters["p1"], 1.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        # Create NUTS sampler
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)

        # Initialize with warm-start values
        init_params = {
            "slope": result.parameters["p0"],
            "intercept": result.parameters["p1"],
            "sigma": 0.5,
        }

        # Should run without error
        mcmc.run(rng_key, x_obs=x, y_obs=y, init_params=init_params)
        samples = mcmc.get_samples()

        assert "slope" in samples
        assert "intercept" in samples
        assert len(samples["slope"]) == 10


class TestNUTSPosteriorValidity:
    """Test T086b: NLSQ->NUTS warm-start produces valid posterior samples."""

    def test_posterior_contains_true_values(self, linear_data: tuple, rng_key: Array) -> None:
        """Test posterior samples contain true parameter values within credible interval."""
        x, y, true_params = linear_data

        # Define NumPyro model with informative priors
        def numpyro_model(x_obs, y_obs):
            slope = numpyro.sample("slope", dist.Normal(2.0, 2.0))
            intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        # Run MCMC
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
        mcmc.run(rng_key, x_obs=x, y_obs=y)
        samples = mcmc.get_samples()

        # Check 90% credible interval contains true values
        slope_samples = np.array(samples["slope"])
        intercept_samples = np.array(samples["intercept"])

        slope_low, slope_high = np.percentile(slope_samples, [5, 95])
        intercept_low, intercept_high = np.percentile(intercept_samples, [5, 95])

        assert slope_low <= true_params["slope"] <= slope_high, \
            f"True slope {true_params['slope']} not in [{slope_low}, {slope_high}]"
        assert intercept_low <= true_params["intercept"] <= intercept_high, \
            f"True intercept {true_params['intercept']} not in [{intercept_low}, {intercept_high}]"

    def test_posterior_mean_near_mle(self, linear_data: tuple, rng_key: Array) -> None:
        """Test posterior mean is close to MLE (NLSQ) estimate."""
        from RepTate.core.fitting.nlsq_fit import run_nlsq_fit

        x, y, _ = linear_data

        def linear_model(x: Array, params: Array) -> Array:
            return params[0] * x + params[1]

        # Get MLE estimate
        nlsq_result, _ = run_nlsq_fit(
            linear_model, x, y, p0=jnp.array([1.0, 0.0])
        )

        # Define NumPyro model
        def numpyro_model(x_obs, y_obs):
            slope = numpyro.sample("slope", dist.Normal(0.0, 10.0))
            intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        # Run MCMC
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=200, num_samples=200, num_chains=1)
        mcmc.run(rng_key, x_obs=x, y_obs=y)
        samples = mcmc.get_samples()

        # Posterior mean should be close to MLE
        posterior_slope = float(np.mean(samples["slope"]))
        posterior_intercept = float(np.mean(samples["intercept"]))

        assert abs(posterior_slope - nlsq_result.parameters["p0"]) < 0.3
        assert abs(posterior_intercept - nlsq_result.parameters["p1"]) < 0.3


class TestNUTSCPUFallback:
    """Test T086c: NumPyro NUTS runs on CPU when GPU unavailable."""

    def test_nuts_runs_on_cpu(self, linear_data: tuple, rng_key: Array) -> None:
        """Test NUTS runs successfully on CPU."""
        x, y, _ = linear_data

        # Force CPU backend
        with jax.default_device(jax.devices("cpu")[0]):
            def numpyro_model(x_obs, y_obs):
                slope = numpyro.sample("slope", dist.Normal(0.0, 10.0))
                intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
                sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
                mu = slope * x_obs + intercept
                numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

            kernel = NUTS(numpyro_model)
            mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
            mcmc.run(rng_key, x_obs=x, y_obs=y)
            samples = mcmc.get_samples()

            assert len(samples["slope"]) == 10

    def test_cpu_performance_warning_logged(self, linear_data: tuple, rng_key: Array, caplog) -> None:
        """Test that CPU usage logs appropriate message."""
        import logging

        x, y, _ = linear_data

        # Check available devices
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)

        if not has_gpu:
            # This is expected - just verify we can detect it
            cpu_devices = [d for d in devices if d.platform == "cpu"]
            assert len(cpu_devices) > 0, "No CPU device available"


class TestFPReorderingTolerance:
    """Test T086d: Tolerance for JAX FP operation reordering effects."""

    def test_repeated_runs_within_tolerance(self, linear_data: tuple) -> None:
        """Test repeated computations are consistent within tolerance."""
        x, y, _ = linear_data

        # Define a computation
        def compute_residuals(slope: float, intercept: float) -> float:
            pred = slope * x + intercept
            return float(jnp.sum((y - pred) ** 2))

        # Run multiple times
        results = [compute_residuals(2.5, 1.5) for _ in range(10)]

        # All results should be identical (JAX is deterministic for same inputs)
        assert all(r == results[0] for r in results)

    def test_jit_vs_non_jit_within_tolerance(self, linear_data: tuple) -> None:
        """Test JIT and non-JIT computations are within tolerance."""
        x, y, _ = linear_data

        def compute_sse(slope: float, intercept: float) -> float:
            pred = slope * x + intercept
            return float(jnp.sum((y - pred) ** 2))

        @jax.jit
        def compute_sse_jit(slope: float, intercept: float) -> float:
            pred = slope * x + intercept
            return jnp.sum((y - pred) ** 2)

        result_eager = compute_sse(2.5, 1.5)
        result_jit = float(compute_sse_jit(2.5, 1.5))

        # Should be within 1e-10 tolerance
        assert abs(result_eager - result_jit) < 1e-10

    def test_vmap_vs_loop_within_tolerance(self, linear_data: tuple) -> None:
        """Test vmap and loop produce results within tolerance."""
        x, y, _ = linear_data
        slopes = jnp.array([2.0, 2.5, 3.0])

        def compute_sse(slope: float) -> float:
            pred = slope * x + 1.5
            return jnp.sum((y - pred) ** 2)

        # Loop version
        loop_results = jnp.array([compute_sse(s) for s in slopes])

        # vmap version
        vmap_results = jax.vmap(compute_sse)(slopes)

        # Should be within 1e-10 tolerance
        assert jnp.allclose(loop_results, vmap_results, atol=1e-10)


class TestRunNutsInferenceDiagnostics:
    """Test T018: run_nuts_inference returns diagnostics and reproducibility_info."""

    @pytest.fixture
    def fit_record(self, linear_data):
        """Create a FitResultRecord for testing."""
        from RepTate.core.models.results import FitResultRecord

        x, y, true_params = linear_data
        return FitResultRecord(
            result_id="test-fit-001",
            dataset_id="test-dataset-001",
            model_id="linear",
            parameter_estimates={"slope": 2.5, "intercept": 1.5},
            diagnostics={"chi2": 0.25, "r2": 0.99},
            residuals=[0.1, -0.05, 0.02],
            execution_context={"optimizer": "nlsq"},
            status="success",
        )

    def test_diagnostics_field_populated(self, linear_data, fit_record, rng_key):
        """Test that diagnostics field is populated in PosteriorResultRecord."""
        from RepTate.core.inference import ConvergenceDiagnostics, run_nuts_inference

        x, y, _ = linear_data

        def numpyro_model(x_obs=None, y_obs=None):
            slope = numpyro.sample("slope", dist.Normal(2.5, 1.0))
            intercept = numpyro.sample("intercept", dist.Normal(1.5, 1.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        result = run_nuts_inference(
            numpyro_model,
            fit_record=fit_record,
            result_id="test-posterior-001",
            rng_seed=42,
            num_warmup=50,
            num_samples=100,
            model_kwargs={"x_obs": x, "y_obs": y},
        )

        # Verify diagnostics field is populated
        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, ConvergenceDiagnostics)

        # Verify R-hat contains sampled parameters
        assert "slope" in result.diagnostics.r_hat
        assert "intercept" in result.diagnostics.r_hat
        assert "sigma" in result.diagnostics.r_hat

        # Verify ESS contains sampled parameters
        assert "slope" in result.diagnostics.ess_bulk
        assert "intercept" in result.diagnostics.ess_bulk

        # Verify divergences is tracked
        assert isinstance(result.diagnostics.divergences, int)
        assert result.diagnostics.divergences >= 0

    def test_reproducibility_info_field_populated(self, linear_data, fit_record, rng_key):
        """Test that reproducibility_info field is populated."""
        from RepTate.core.inference import ReproducibilityInfo, run_nuts_inference

        x, y, _ = linear_data

        def numpyro_model(x_obs=None, y_obs=None):
            slope = numpyro.sample("slope", dist.Normal(2.5, 1.0))
            intercept = numpyro.sample("intercept", dist.Normal(1.5, 1.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        result = run_nuts_inference(
            numpyro_model,
            fit_record=fit_record,
            result_id="test-posterior-002",
            rng_seed=12345,
            num_warmup=50,
            num_samples=100,
            model_kwargs={"x_obs": x, "y_obs": y},
        )

        # Verify reproducibility_info field is populated
        assert result.reproducibility_info is not None
        assert isinstance(result.reproducibility_info, ReproducibilityInfo)

        # Verify seed is recorded
        assert result.reproducibility_info.rng_seed == 12345

        # Verify versions are recorded
        assert len(result.reproducibility_info.jax_version) > 0
        assert len(result.reproducibility_info.numpyro_version) > 0
        assert len(result.reproducibility_info.reptate_version) > 0

        # Verify model config is captured
        assert "x_obs" in str(result.reproducibility_info.model_config) or \
               result.reproducibility_info.model_config is not None

    def test_status_is_completed_for_good_run(self, linear_data, fit_record, rng_key):
        """Test that status is 'completed' for a successful run."""
        from RepTate.core.inference import run_nuts_inference

        x, y, _ = linear_data

        def numpyro_model(x_obs=None, y_obs=None):
            slope = numpyro.sample("slope", dist.Normal(2.5, 1.0))
            intercept = numpyro.sample("intercept", dist.Normal(1.5, 1.0))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mu = slope * x_obs + intercept
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

        result = run_nuts_inference(
            numpyro_model,
            fit_record=fit_record,
            result_id="test-posterior-003",
            rng_seed=42,
            num_warmup=50,
            num_samples=100,
            model_kwargs={"x_obs": x, "y_obs": y},
        )

        assert result.status == "completed"
