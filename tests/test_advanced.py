"""Tests for advanced modules: Topological, NLP, Explainability, Quantum, GPU."""

import numpy as np
import pytest

from quant.topological import TopologicalAnalyzer
from quant.nlp_engine import FinancialNLPEngine
from quant.explainability import ExplainabilityEngine
from quant.quantum import QuantumFinanceEngine
from quant.gpu_compute import GPUAccelerator


# ---------------------------------------------------------------------------
# Topological
# ---------------------------------------------------------------------------

class TestTopological:
    def test_hurst_exponent(self):
        """Hurst exponent for random walk ≈ 0.5."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        analyzer = TopologicalAnalyzer()
        h = analyzer.hurst_exponent(returns)
        assert 0.1 < h < 1.0  # reasonable range

    def test_fractal_dimension(self):
        """Fractal dimension should be in [0.5, 2.5] range."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        analyzer = TopologicalAnalyzer()
        fd = analyzer.fractal_dimension(returns)
        assert 0.5 <= fd <= 2.5


# ---------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------

class TestNLP:
    def test_sentiment_analysis_positive(self):
        """Positive financial text should yield positive score."""
        engine = FinancialNLPEngine()
        text = "The market showed strong growth and robust profits. Bullish rally with upside."
        result = engine.sentiment_analysis(text)

        assert result["score"] > 0
        assert result["positive_count"] > 0
        assert result["label"] in ("Bullish", "Very Bullish")

    def test_sentiment_analysis_negative(self):
        """Negative financial text should yield negative score."""
        engine = FinancialNLPEngine()
        text = "Severe losses and crash. Bearish decline with risk of default and bankruptcy."
        result = engine.sentiment_analysis(text)

        assert result["score"] < 0
        assert result["negative_count"] > 0
        assert result["label"] in ("Bearish", "Very Bearish")

    def test_ner_basic(self):
        """Named entity recognition finds currencies and indices."""
        engine = FinancialNLPEngine()
        text = "The USD strengthened against EUR. S&P 500 rose 2%. Gold prices increased."
        result = engine.named_entity_recognition(text)

        assert len(result["entities"]) > 0
        entity_types = {e["type"] for e in result["entities"]}
        # Should find at least some entity types
        assert len(entity_types) > 0


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class TestExplainability:
    def test_feature_importance(self):
        """Permutation feature importance returns ranked features."""
        engine = ExplainabilityEngine()
        rng = np.random.default_rng(42)
        n, k = 200, 5
        X = rng.standard_normal((n, k))
        # Make feature 0 important
        y = 3 * X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1

        result = engine.feature_importance(X, y, method="permutation", n_repeats=5)

        assert result["ranking"] is not None
        assert len(result["importance_mean"]) == k
        assert len(result["ranking"]) == k


# ---------------------------------------------------------------------------
# Quantum
# ---------------------------------------------------------------------------

class TestQuantum:
    def test_qaoa_portfolio(self):
        """QAOA portfolio optimization should return solutions."""
        np.random.seed(42)
        engine = QuantumFinanceEngine()
        mu = np.array([0.08, 0.12, 0.06, 0.10, 0.09])
        cov = np.eye(5) * 0.04
        np.fill_diagonal(cov, 0.04)

        result = engine.qaoa_portfolio(mu, cov, budget=2, n_layers=2, n_iterations=30)

        assert "solutions" in result
        assert len(result["solutions"]) > 0
        assert result["n_qubits"] == 5
        assert result["budget"] == 2


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

class TestGPU:
    def test_gpu_monte_carlo(self):
        """GPU-accelerated Monte Carlo should return valid results."""
        gpu = GPUAccelerator()
        result = gpu.accelerated_monte_carlo(
            s0=100, mu=0.05, sigma=0.2, t=1.0,
            n_sims=10000, n_steps=50, antithetic=True
        )

        assert result["mean"] > 0
        assert result["std"] > 0
        assert result["percentile_5"] < result["percentile_95"]
        assert result["antithetic"] is True
        assert "combined_mean" in result  # antithetic should produce this
