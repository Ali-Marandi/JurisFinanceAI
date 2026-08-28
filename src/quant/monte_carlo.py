"""JurisFinanceAI - Monte Carlo Simulation Engine

General-purpose Monte Carlo simulation for:
- Portfolio scenario analysis
- Option pricing
- VaR estimation
- Cash flow modeling
- Credit risk simulation
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import warnings
warnings.filterwarnings("ignore")


class MonteCarloEngine:
    """Flexible Monte Carlo simulation framework."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.last_result = None

    def geometric_brownian_motion(self, S0, mu, sigma, T, n_steps,
                                    n_paths=10000, antithetic=True) -> Dict:
        """Simulate Geometric Brownian Motion paths.

        dS = mu*S*dt + sigma*S*dW
        """
        dt = T / n_steps
        if antithetic:
            Z = self.rng.standard_normal((n_paths // 2, n_steps))
            Z = np.vstack([Z, -Z])
        else:
            Z = self.rng.standard_normal((n_paths, n_steps))

        drift = (mu - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        log_returns = drift + vol * Z
        log_S = np.cumsum(log_returns, axis=1)
        S = S0 * np.exp(log_S)

        paths = np.column_stack([np.full(n_paths, S0), S])

        self.last_result = {
            "method": "GBM Monte Carlo",
            "n_paths": n_paths, "n_steps": n_steps,
            "drift": mu, "volatility": sigma,
            "paths": paths.tolist(),
            "final_prices": S[:, -1].tolist(),
            "mean_final": float(np.mean(S[:, -1])),
            "median_final": float(np.median(S[:, -1])),
            "std_final": float(np.std(S[:, -1])),
            "percentile_5": float(np.percentile(S[:, -1], 5)),
            "percentile_95": float(np.percentile(S[:, -1], 95)),
        }
        return self.last_result

    def portfolio_simulation(self, initial_value, expected_returns,
                             cov_matrix, n_periods=252,
                             n_paths=1000) -> Dict:
        """Simulate portfolio value paths.

        Uses multivariate normal returns.
        """
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float)
        n_assets = len(mu)

        # Cholesky decomposition for correlated returns
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma += np.eye(n_assets) * 1e-8
            L = np.linalg.cholesky(Sigma)

        # Simulate returns
        Z = self.rng.standard_normal((n_paths, n_periods, n_assets))
        correlated_Z = Z @ L.T

        # Portfolio daily returns (equal weight or use last optimized weights)
        port_returns = correlated_Z @ mu

        # Cumulative portfolio value
        port_value = initial_value * np.exp(np.cumsum(port_returns, axis=1))
        port_value = np.column_stack([np.full(n_paths, initial_value), port_value])

        final_values = port_value[:, -1]

        self.last_result = {
            "method": "Portfolio Monte Carlo",
            "n_paths": n_paths, "n_periods": n_periods,
            "initial_value": initial_value,
            "final_values": final_values.tolist(),
            "mean_final": float(np.mean(final_values)),
            "median_final": float(np.median(final_values)),
            "std_final": float(np.std(final_values)),
            "percentile_5": float(np.percentile(final_values, 5)),
            "percentile_95": float(np.percentile(final_values, 95)),
            "var_95": float(initial_value - np.percentile(final_values, 5)),
            "cvar_95": float(initial_value - np.mean(final_values[final_values < np.percentile(final_values, 5)])),
            "paths_summary": port_value[:, ::max(1, n_periods // 50)].tolist(),
        }
        return self.last_result

    def credit_risk_simulation(self, n_obligors, pd_vector, lgd_vector,
                                correlation=0.2, n_simulations=10000) -> Dict:
        """Simulate credit portfolio losses using one-factor Gaussian copula.

        Vasicek model: default if Y_i < threshold
        Y_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
        """
        pd = np.array(pd_vector, dtype=float)
        lgd = np.array(lgd_vector, dtype=float)
        exposure = np.ones(len(pd))  # Equal exposure
        n = len(pd)

        thresholds = stats.norm.ppf(pd)  # Use scipy

        Z_systemic = self.rng.standard_normal(n_simulations)
        idiosyncratic = self.rng.standard_normal((n_simulations, n))

        asset_values = np.sqrt(correlation) * Z_systemic[:, None] + np.sqrt(1 - correlation) * idiosyncratic

        defaults = asset_values < thresholds
        losses = defaults @ (exposure * lgd)

        self.last_result = {
            "method": "Credit Risk Monte Carlo (Gaussian Copula)",
            "n_obligors": n,
            "n_simulations": n_simulations,
            "correlation": correlation,
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "percentile_99_loss": float(np.percentile(losses, 99)),
            "percentile_95_loss": float(np.percentile(losses, 95)),
            "expected_shortfall_99": float(np.mean(losses[losses > np.percentile(losses, 99)])),
            "mean_default_rate": float(np.mean(defaults.sum(axis=1) / n)),
            "loss_distribution": np.histogram(losses, bins=50)[0].tolist(),
            "loss_bins": np.histogram(losses, bins=50)[1].tolist(),
        }
        return self.last_result

    def cashflow_simulation(self, initial_cashflow, growth_rate,
                            volatility, n_years=10,
                            n_paths=5000) -> Dict:
        """Simulate future cash flows with uncertainty."""
        dt = 1
        paths = np.zeros((n_paths, n_years + 1))
        paths[:, 0] = initial_cashflow

        for t in range(1, n_years + 1):
            Z = self.rng.standard_normal(n_paths)
            growth = (growth_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z
            paths[:, t] = paths[:, t-1] * np.exp(growth)

        terminal = paths[:, -1]
        discount_rate = 0.08
        pv_paths = np.zeros(n_paths)
        for t in range(1, n_years + 1):
            pv_paths += paths[:, t] / (1 + discount_rate) ** t

        self.last_result = {
            "method": "Cash Flow Monte Carlo",
            "n_paths": n_paths, "n_years": n_years,
            "mean_terminal_cf": float(np.mean(terminal)),
            "npv_mean": float(np.mean(pv_paths)),
            "npv_std": float(np.std(pv_paths)),
            "npv_5pct": float(np.percentile(pv_paths, 5)),
            "npv_95pct": float(np.percentile(pv_paths, 95)),
            "paths_summary": paths[:, ::max(1, n_years // 20)].tolist(),
        }
        return self.last_result


from scipy import stats
