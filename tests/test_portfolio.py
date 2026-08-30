"""Tests for PortfolioOptimizer (src/quant/portfolio.py)."""

import numpy as np
import pytest

from quant.portfolio import PortfolioOptimizer, FuzzyNumber


class TestMarkowitz:
    """Markowitz Mean-Variance optimization tests."""

    def test_markowitz_basic(self, sample_returns):
        """Normal case with 5 assets."""
        opt = PortfolioOptimizer()
        mu = np.mean(sample_returns, axis=0) * 252  # annualise
        cov = np.cov(sample_returns, rowvar=False) * 252
        result = opt.markowitz_optimize(mu, cov)

        assert result["success"]
        assert result["n_assets"] == 5
        assert result["method"] == "Markowitz Mean-Variance"
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert result["portfolio_risk"] >= 0
        # efficient frontier should have points
        assert len(result["efficient_frontier"]) > 0

    def test_markowitz_2_assets(self):
        """Edge case with 2 assets."""
        opt = PortfolioOptimizer()
        mu = np.array([0.08, 0.12])
        cov = np.array([[0.04, 0.01],
                        [0.01, 0.09]])
        result = opt.markowitz_optimize(mu, cov)

        assert result["success"]
        assert result["n_assets"] == 2
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= -1e-6).all()

    def test_markowitz_single_asset(self):
        """Edge case with 1 asset."""
        opt = PortfolioOptimizer()
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        result = opt.markowitz_optimize(mu, cov)

        assert result["success"]
        assert result["n_assets"] == 1
        np.testing.assert_allclose(result["weights"], [1.0], atol=1e-6)


class TestMaxSharpe:
    def test_max_sharpe(self, sample_returns):
        """Maximize Sharpe ratio."""
        opt = PortfolioOptimizer()
        mu = np.mean(sample_returns, axis=0) * 252
        cov = np.cov(sample_returns, rowvar=False) * 252
        result = opt.maximize_sharpe(mu, cov)

        assert result["success"]
        assert result["n_assets"] == 5
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        # Sharpe should be a finite number
        assert np.isfinite(result["sharpe_ratio"])


class TestRiskParity:
    def test_risk_parity(self, sample_returns):
        """Risk parity allocation – equal risk contribution."""
        opt = PortfolioOptimizer()
        cov = np.cov(sample_returns, rowvar=False) * 252
        result = opt.risk_parity(cov)

        assert result["success"]
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights > 0).all()  # long-only
        # risk contributions should be roughly equal
        rc = np.array(result["risk_contributions"])
        assert np.std(rc) / (np.mean(rc) + 1e-10) < 0.5  # within 50% relative std


class TestMinVariance:
    def test_min_variance(self, sample_returns):
        """Minimum variance portfolio."""
        opt = PortfolioOptimizer()
        cov = np.cov(sample_returns, rowvar=False) * 252
        result = opt.minimum_variance(cov)

        assert result["success"]
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= -1e-6).all()


class TestFuzzy:
    def test_fuzzy_optimize(self, sample_returns):
        """Fuzzy portfolio optimization."""
        opt = PortfolioOptimizer()
        mu = np.mean(sample_returns, axis=0) * 252
        cov = np.cov(sample_returns, rowvar=False) * 252

        # Build triangular fuzzy numbers around expected returns
        fuzzy_returns = [
            FuzzyNumber(m - 0.02, m, m + 0.02) for m in mu
        ]
        result = opt.fuzzy_optimize(fuzzy_returns, cov)

        assert result["method"].startswith("Fuzzy Portfolio Optimization")
        assert result["n_assets"] == 5
        weights = np.array(result["weights"])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert "fuzzy_portfolio_return" in result
        fpr = result["fuzzy_portfolio_return"]
        assert fpr["lower_bound"] <= fpr["peak"] <= fpr["upper_bound"]
