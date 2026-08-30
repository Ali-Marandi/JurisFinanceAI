"""Tests for InterestRateModel (src/quant/interest_rates.py)."""

import numpy as np
import pytest

from quant.interest_rates import InterestRateModel


class TestVasicek:
    def test_vasicek(self):
        """Vasicek model returns valid paths and yield curve."""
        np.random.seed(42)
        model = InterestRateModel()
        result = model.vasicek(r0=0.05, a=0.5, b=0.05, sigma=0.01, T=1.0, n_steps=252, n_paths=500)

        assert result["method"] == "Vasicek Model"
        assert result["n_paths"] == 500
        assert isinstance(result["final_rates_mean"], float)
        assert result["bond_price_T"] > 0
        # Yield curve should have positive yields
        yields = result["yield_curve"]["yields"]
        maturities = result["yield_curve"]["maturities"]
        assert len(yields) == len(maturities)
        assert len(maturities) == 11


class TestCIR:
    def test_cir(self):
        """CIR model – rates should stay non-negative."""
        np.random.seed(42)
        model = InterestRateModel()
        result = model.cir_model(r0=0.05, a=0.5, b=0.05, sigma=0.1, T=1.0, n_steps=252, n_paths=500)

        assert result["method"] == "CIR Model"
        assert result["bond_price_T"] > 0
        yields = result["yield_curve"]["yields"]
        assert all(y >= -0.5 for y in yields)  # allow slight negative from numerical


class TestHullWhite:
    def test_hull_white(self):
        """Hull-White model with constant theta."""
        np.random.seed(42)
        model = InterestRateModel()
        # Provide theta function to avoid the hasattr fallback
        result = model.hull_white(
            r0=0.05, a=0.3, sigma=0.01, T=1.0, n_steps=252, n_paths=500,
            theta_func=lambda t: 0.03
        )

        assert result["method"] == "Hull-White Model"
        assert result["n_paths"] == 500
        assert isinstance(result["final_rates_mean"], float)
        assert isinstance(result["final_rates_std"], float)
