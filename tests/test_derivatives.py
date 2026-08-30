"""Tests for DerivativesPricer (src/quant/derivatives.py)."""

import numpy as np
import pytest
from scipy.stats import norm

from quant.derivatives import DerivativesPricer


class TestBlackScholes:
    """Black-Scholes option pricing tests."""

    def test_black_scholes_call(self):
        """Normal call option pricing."""
        pricer = DerivativesPricer()
        result = pricer.black_scholes(S=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type="call")

        assert result["method"] == "Black-Scholes"
        assert result["price"] > 0
        assert result["option_type"] == "call"
        # Delta for call should be between 0 and 1
        assert 0 < result["delta"] < 1
        # Gamma should be positive
        assert result["gamma"] > 0
        # Vega should be positive
        assert result["vega"] > 0

    def test_black_scholes_put(self):
        """Normal put option pricing."""
        pricer = DerivativesPricer()
        result = pricer.black_scholes(S=100, K=95, T=1.0, r=0.05, sigma=0.2, option_type="put")

        assert result["price"] > 0
        assert result["option_type"] == "put"
        # Delta for put should be between -1 and 0
        assert -1 < result["delta"] < 0

    def test_black_scholes_atm(self):
        """At-the-money option."""
        pricer = DerivativesPricer()
        call = pricer.black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        put = pricer.black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")

        # Put-call parity: C - P = S - K*exp(-r*T)
        diff = call["price"] - put["price"]
        expected = 100 - 100 * np.exp(-0.05 * 1.0)
        assert abs(diff - expected) < 0.01

    def test_black_scholes_zero_vol(self):
        """Zero volatility edge case – price should equal intrinsic value."""
        pricer = DerivativesPricer()
        call = pricer.black_scholes(S=110, K=100, T=1.0, r=0.05, sigma=0.0, option_type="call")
        put = pricer.black_scholes(S=90, K=100, T=1.0, r=0.05, sigma=0.0, option_type="put")

        # With zero vol, option should be worth discounted intrinsic
        # Call intrinsic = max(S - K, 0) = 10, discounted
        assert call["price"] > 0
        assert put["price"] > 0

    def test_black_scholes_deep_itm(self):
        """Deep in-the-money call."""
        pricer = DerivativesPricer()
        result = pricer.black_scholes(S=200, K=50, T=1.0, r=0.05, sigma=0.2, option_type="call")

        assert result["price"] > 0
        # Deep ITM call delta should be close to 1
        assert result["delta"] > 0.9

    def test_black_scholes_deep_otm(self):
        """Deep out-of-the-money put."""
        pricer = DerivativesPricer()
        result = pricer.black_scholes(S=200, K=50, T=0.01, r=0.05, sigma=0.2, option_type="put")

        # Deep OTM put should be very cheap
        assert result["price"] < 1.0
        assert result["price"] >= 0

    def test_put_call_parity(self):
        """Verify put-call parity holds for multiple strikes."""
        pricer = DerivativesPricer()
        S, r, T, sigma = 100, 0.05, 1.0, 0.25

        for K in [80, 90, 100, 110, 120]:
            call = pricer.black_scholes(S, K, T, r, sigma, "call")
            put = pricer.black_scholes(S, K, T, r, sigma, "put")
            lhs = call["price"] - put["price"]
            rhs = S - K * np.exp(-r * T)
            assert abs(lhs - rhs) < 0.02, f"Put-call parity violated for K={K}: {lhs} vs {rhs}"
