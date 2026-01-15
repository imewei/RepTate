"""Tests for optax library availability.

Tests cover:
- T030: Verify optax import and basic optimizer creation

Ensures optax library is available in the RepTate environment
for future gradient-based optimization workflows.
"""

from __future__ import annotations

import pytest


class TestOptaxImport:
    """Test suite for optax import verification (T030)."""

    def test_optax_import_succeeds(self):
        """Test that optax can be imported without errors."""
        import optax

        assert hasattr(optax, "__version__")
        assert len(optax.__version__) > 0

    def test_optax_adam_optimizer_available(self):
        """Test that adam optimizer is available."""
        import optax

        optimizer = optax.adam(learning_rate=0.001)
        assert optimizer is not None

    def test_optax_sgd_optimizer_available(self):
        """Test that sgd optimizer is available."""
        import optax

        optimizer = optax.sgd(learning_rate=0.01)
        assert optimizer is not None

    def test_optax_learning_rate_schedules_available(self):
        """Test that learning rate schedules are available."""
        import optax

        # Constant schedule
        schedule = optax.constant_schedule(0.01)
        assert callable(schedule)

        # Exponential decay
        schedule = optax.exponential_decay(
            init_value=0.1, transition_steps=100, decay_rate=0.99
        )
        assert callable(schedule)

    def test_optax_chain_available(self):
        """Test that optimizer chaining is available."""
        import optax

        # Chain multiple transforms
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=0.001),
        )
        assert optimizer is not None

    def test_optax_gradient_clipping_available(self):
        """Test that gradient clipping utilities are available."""
        import optax

        # Global norm clipping
        clipper = optax.clip_by_global_norm(1.0)
        assert clipper is not None

        # Value clipping
        clipper = optax.clip(1.0)
        assert clipper is not None

    def test_optax_works_with_jax_arrays(self):
        """Test that optax works with JAX arrays."""
        import jax
        import jax.numpy as jnp
        import optax

        # Simple parameter update test
        optimizer = optax.adam(learning_rate=0.001)
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)

        # Compute fake gradients
        grads = jnp.array([0.1, 0.2, 0.3])

        # Apply update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert new_params is not None
        assert new_params.shape == params.shape
        # Parameters should have changed
        assert not jnp.allclose(new_params, params)
