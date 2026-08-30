"""Tests for RiskEngine (src/quant/risk_models.py)."""

import numpy as np
import pytest

from quant.risk_models import RiskEngine


class TestValueAtRisk:
    """VaR tests."""

    def test_var_historical(self, sample_single_returns):
        """Historical VaR should be a negative number (loss)."""
        engine = RiskEngine()
        result = engine.value_at_risk(sample_single_returns, confidence=0.95, method="historical")

        assert result["method"] == "VaR (historical)"
        assert result["confidence_level"] == 0.95
        assert result["var_return"] < 0  # loss
        assert result["var_absolute"] > 0
        assert result["var_percentage"] > 0

    def test_var_parametric(self, sample_single_returns):
        """Parametric VaR."""
        engine = RiskEngine()
        result = engine.value_at_risk(sample_single_returns, confidence=0.95, method="parametric")

        assert result["method"] == "VaR (parametric)"
        assert result["var_return"] < 0
        assert result["var_absolute"] > 0

    def test_var_cornish_fisher(self, sample_single_returns):
        """Cornish-Fisher VaR."""
        engine = RiskEngine()
        result = engine.value_at_risk(sample_single_returns, confidence=0.95, method="cornish_fisher")

        assert result["method"] == "VaR (cornish_fisher)"
        assert result["var_return"] < 0


class TestCVaR:
    def test_cvar(self, sample_single_returns):
        """CVaR (Expected Shortfall) should be worse than VaR."""
        engine = RiskEngine()
        result = engine.conditional_var(sample_single_returns, confidence=0.95)

        assert result["method"] == "CVaR (Expected Shortfall)"
        assert result["cvar_return"] < 0
        # CVaR should be at least as bad (more negative) as VaR
        assert result["cvar_return"] <= result["var_return"]
        assert result["cvar_absolute"] > 0
        assert result["tail_losses"] > 0


class TestStressTest:
    def test_stress_test(self, sample_single_returns):
        """Stress testing returns scenario results."""
        engine = RiskEngine()
        result = engine.stress_test(sample_single_returns, portfolio_value=1e6)

        assert result["method"] == "Stress Testing"
        assert result["portfolio_value"] == 1e6
        assert len(result["scenarios"]) > 0
        # Worst case loss should be positive (a loss amount)
        assert result["worst_case_loss"] > 0
        # Each scenario should have severity
        for sc in result["scenarios"]:
            assert "severity" in sc
            assert "portfolio_loss" in sc
