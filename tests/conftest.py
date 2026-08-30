"""Shared fixtures for JurisFinanceAI test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@pytest.fixture
def sample_returns(n=200, k=5):
    """Return a (n, k) matrix of random daily returns."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, k)) * 0.01  # ~1% daily vol


@pytest.fixture
def sample_prices(n=200, k=5):
    """Return a (n, k) matrix of cumulative prices starting from 100."""
    rng = np.random.default_rng(42)
    returns = rng.standard_normal((n, k)) * 0.01
    log_prices = np.cumsum(returns, axis=0)
    prices = 100.0 * np.exp(log_prices)
    return prices


@pytest.fixture
def sample_single_returns(n=200):
    """Return a 1-D array of random daily returns."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(n) * 0.01
