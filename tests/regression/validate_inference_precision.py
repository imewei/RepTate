"""Regression check for inference precision consistency."""

from __future__ import annotations

import jax.numpy as jnp

from RepTate.core.inference.nuts_runner import summarize_samples


def main() -> None:
    samples = {
        "param_a": jnp.ones((2, 1000)),
        "param_b": jnp.zeros((2, 1000)),
    }
    summary = summarize_samples(samples)
    assert summary["param_a"]["mean"] == 1.0
    assert summary["param_b"]["mean"] == 0.0


if __name__ == "__main__":
    main()
