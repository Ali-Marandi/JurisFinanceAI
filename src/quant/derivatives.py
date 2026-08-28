"""JurisFinanceAI - Derivatives Pricing Engine

Implements Black-Scholes, Binomial Tree, Monte Carlo pricing,
Greeks computation, implied volatility, and Fuzzy Black-Scholes.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")


class DerivativesPricer:
    """Comprehensive derivatives pricing engine."""

    def __init__(self):
        self.last_result = None

    def black_scholes(self, S, K, T, r, sigma, option_type="call") -> Dict:
        """Black-Scholes option pricing with full Greeks."""
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return {"price": intrinsic, "method": "Black-Scholes (expired)"}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                      - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        theta = theta_call if option_type == "call" else theta_put
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)) / 100

        self.last_result = {
            "method": "Black-Scholes",
            "option_type": option_type, "spot": S, "strike": K,
            "time_to_maturity": T, "risk_free_rate": r, "volatility": sigma,
            "price": float(price), "d1": float(d1), "d2": float(d2),
            "delta": float(delta), "gamma": float(gamma),
            "theta": float(theta), "vega": float(vega), "rho": float(rho),
            "intrinsic_value": float(max(S - K, 0) if option_type == "call" else max(K - S, 0)),
            "time_value": float(price - max(S - K, 0) if option_type == "call" else price - max(K - S, 0)),
        }
        return self.last_result

    def binomial_tree(self, S, K, T, r, sigma, option_type="call",
                      n_steps=200, american=False) -> Dict:
        """Cox-Ross-Rubinstein Binomial Tree pricing."""
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)

        prices = np.array([S * u ** j * d ** (n_steps - j) for j in range(n_steps + 1)])
        values = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)

        for i in range(n_steps - 1, -1, -1):
            values = np.exp(-r * dt) * (p * values[1:] + (1 - p) * values[:-1])
            if american:
                step_prices = np.array([S * u ** j * d ** (i - j) for j in range(i + 1)])
                exercise = np.maximum(step_prices - K, 0) if option_type == "call" else np.maximum(K - step_prices, 0)
                values = np.maximum(values, exercise)

        bs_price = self.black_scholes(S, K, T, r, sigma, option_type)["price"]

        self.last_result = {
            "method": f"Binomial Tree ({'American' if american else 'European'})",
            "option_type": option_type, "n_steps": n_steps,
            "price": float(values[0]), "bs_price": bs_price,
            "convergence_error": float(abs(values[0] - bs_price)),
            "up_factor": float(u), "down_factor": float(d),
            "risk_neutral_prob": float(p),
        }
        return self.last_result

    def monte_carlo_pricing(self, S, K, T, r, sigma, option_type="call",
                            n_simulations=50000, n_steps=100,
                            antithetic=True) -> Dict:
        """Monte Carlo option pricing with variance reduction."""
        dt = T / n_steps
        drift = (r - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        if antithetic:
            Z = np.random.standard_normal((n_simulations // 2, n_steps))
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.standard_normal((n_simulations, n_steps))

        log_returns = drift + vol * Z
        log_S = np.cumsum(log_returns, axis=1)
        S_T = S * np.exp(log_S[:, -1])

        payoffs = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
        price = float(np.exp(-r * T) * np.mean(payoffs))
        std_error = float(np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs)))

        bs_price = self.black_scholes(S, K, T, r, sigma, option_type)["price"]

        self.last_result = {
            "method": f"Monte Carlo ({n_simulations} paths)",
            "option_type": option_type, "price": price,
            "standard_error": std_error,
            "confidence_interval_95": (price - 1.96 * std_error, price + 1.96 * std_error),
            "bs_price": bs_price,
            "n_simulations": n_simulations, "n_steps": n_steps,
            "antithetic": antithetic,
        }
        return self.last_result

    def implied_volatility(self, S, K, T, r, market_price,
                            option_type="call", max_iter=100, tol=1e-8) -> Dict:
        """Implied volatility via Newton-Raphson."""
        sigma = 0.3
        for i in range(max_iter):
            bs = self.black_scholes(S, K, T, r, sigma, option_type)
            vega = bs["vega"] * 100
            if abs(vega) < 1e-10:
                break
            diff = bs["price"] - market_price
            if abs(diff) < tol:
                break
            sigma = max(0.001, min(sigma - diff / vega, 5.0))

        self.last_result = {
            "method": "Implied Volatility (Newton-Raphson)",
            "implied_volatility": float(sigma),
            "market_price": market_price,
            "bs_price_at_iv": self.black_scholes(S, K, T, r, sigma, option_type)["price"],
            "iterations": i + 1,
            "converged": abs(diff) < tol,
        }
        return self.last_result

    def option_strategy(self, S, K1, K2, T, r, sigma,
                        strategy="bull_call_spread") -> Dict:
        """Common options strategies."""
        strategies = {
            "bull_call_spread": [("call", K1, 1), ("call", K2, -1)],
            "bear_put_spread": [("put", K1, -1), ("put", K2, 1)],
            "straddle": [("call", S, 1), ("put", S, 1)],
            "strangle": [("call", K1, 1), ("put", K2, 1)],
            "butterfly": [("call", K1, 1), ("call", (K1+K2)/2, -2), ("call", K2, 1)],
            "protective_put": [("put", K1, 1)],
            "covered_call": [("call", K1, -1)],
            "iron_condor": [("put", K1, 1), ("put", K1*0.95, -1),
                            ("call", K2, 1), ("call", K2*1.05, -1)],
        }

        legs = strategies.get(strategy, strategies["bull_call_spread"])
        total_cost = 0
        leg_results = []

        for opt_type, strike, quantity in legs:
            bs = self.black_scholes(S, strike, T, r, sigma, opt_type)
            leg_cost = bs["price"] * quantity
            total_cost += leg_cost
            leg_results.append({
                "type": opt_type, "strike": strike, "quantity": quantity,
                "price": bs["price"], "cost": leg_cost, "delta": bs["delta"] * quantity
            })

        spots = np.linspace(S * 0.7, S * 1.3, 30)
        pl_spots = []
        for sp in spots:
            pl = -total_cost
            for opt_type, strike, quantity in legs:
                if opt_type == "call":
                    pl += quantity * max(sp - strike, 0)
                else:
                    pl += quantity * max(strike - sp, 0)
            pl_spots.append(float(pl))

        self.last_result = {
            "method": f"Options Strategy: {strategy}",
            "strategy": strategy, "legs": leg_results,
            "total_cost": float(total_cost),
            "total_delta": float(sum(l["delta"] for l in leg_results)),
            "current_spot": S,
            "payoff_spots": spots.tolist(), "payoff_values": pl_spots,
            "max_profit": float(max(pl_spots)), "max_loss": float(min(pl_spots)),
        }
        return self.last_result
