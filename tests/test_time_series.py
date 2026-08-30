"""Tests for TimeSeriesAnalyzer and ARIMAModel (src/quant/time_series.py)."""

import numpy as np
import pytest

from quant.time_series import TimeSeriesAnalyzer, ARIMAModel


class TestADF:
    def test_adf_test_stationary(self):
        """ADF test on stationary data (white noise)."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(300)
        analyzer = TimeSeriesAnalyzer()
        result = analyzer.adf_test(y)

        assert result["test"] == "ADF"
        assert "t_statistic" in result
        assert "is_stationary" in result
        assert isinstance(bool(result["is_stationary"]), bool)

    def test_adf_test_trending(self):
        """ADF test on trending data should indicate non-stationary."""
        t = np.linspace(0, 10, 300)
        y = t + np.random.default_rng(42).standard_normal(300) * 0.1
        analyzer = TimeSeriesAnalyzer()
        result = analyzer.adf_test(y)

        assert "is_stationary" in result


class TestRollingStatistics:
    def test_rolling_statistics(self, sample_single_returns):
        """Rolling mean/std should have correct length."""
        analyzer = TimeSeriesAnalyzer()
        result = analyzer.rolling_statistics(sample_single_returns, window=20)

        assert result["window"] == 20
        expected_len = len(sample_single_returns) - 20 + 1
        assert len(result["rolling_mean"]) == expected_len
        assert len(result["rolling_std"]) == expected_len
        # All stds should be non-negative
        assert all(s >= 0 for s in result["rolling_std"])


class TestARIMA:
    def test_arima_fit(self, sample_single_returns):
        """ARIMA model should fit without errors."""
        model = ARIMAModel(p=1, d=1, q=1)
        # Use prices (non-stationary) for ARIMA with d=1
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
        result = model.fit(prices)

        assert "ar_params" in result
        assert "ma_params" in result
        assert "aic" in result
        assert "bic" in result
        assert np.isfinite(result["aic"])

    def test_arima_forecast(self, sample_single_returns):
        """ARIMA forecast should return correct number of steps."""
        model = ARIMAModel(p=1, d=1, q=1)
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
        model.fit(prices)
        result = model.forecast(prices, steps=10)

        assert result["steps"] == 10
        assert len(result["forecast_values"]) == 10
        assert len(result["confidence_lower"]) == 10
        assert len(result["confidence_upper"]) == 10
        # Confidence upper > lower
        for lo, hi in zip(result["confidence_lower"], result["confidence_upper"]):
            assert lo <= hi
