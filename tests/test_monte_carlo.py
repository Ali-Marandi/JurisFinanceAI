"""Tests for MonteCarloEngine (src/quant/monte_carlo.py)."""

import numpy as np
import pytest

from quant.monte_carlo import MonteCarloEngine


class TestGBM:
    def test_gbm_basic(self):
        """GBM simulation returns paths and summary statistics."""
        engine = MonteCarloEngine(seed=42)
        result = engine.geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000)

        assert result["method"] == "GBM Monte Carlo"
        assert result["n_paths"] == 1000
        assert len(result["final_prices"]) == 1000
        assert result["mean_final"] > 0
        assert result["percentile_5"] < result["percentile_95"]
        assert len(result["paths"]) == 1000

    def test_gbm_zero_drift(self):
        """GBM with zero drift – mean final price ≈ S0."""
        engine = MonteCarloEngine(seed=42)
        result = engine.geometric_brownian_motion(S0=100, mu=0.0, sigma=0.2, T=1.0, n_steps=252, n_paths=5000)

        # With zero drift, E[S_T] = S0 * exp(-0.5*sigma^2*T) ≈ 98.02
        expected = 100 * np.exp(-0.5 * 0.2**2)
        assert abs(result["mean_final"] - expected) < 5  # generous tolerance


class TestPortfolioSimulation:
    def test_portfolio_simulation(self):
        """Portfolio MC simulation."""
        engine = MonteCarloEngine(seed=42)
        mu = np.array([0.08, 0.10, 0.06])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.025],
        ])
        result = engine.portfolio_simulation(1e6, mu, cov, n_periods=252, n_paths=500)

        assert result["method"] == "Portfolio Monte Carlo"
        assert result["n_paths"] == 500
        assert len(result["final_values"]) == 500
        assert result["mean_final"] > 0
        assert result["var_95"] >= 0
