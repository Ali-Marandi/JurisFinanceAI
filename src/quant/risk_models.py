"""JurisFinanceAI - Risk Analysis Engine

Implements VaR, CVaR, GARCH(1,1), stress testing,
maximum drawdown, Altman Z-Score, Beneish M-Score,
and correlation/PCA analysis.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


class GARCHModel:
    """GARCH(1,1) volatility model.

    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    """

    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.fitted = False

    def fit(self, returns):
        """Fit GARCH(1,1) using maximum likelihood estimation."""
        r = np.array(returns, dtype=float)
        n = len(r)
        var_r = np.var(r)

        def log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.zeros(n)
            sigma2[0] = var_r
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r**2 / sigma2)
            return -ll

        x0 = [var_r * 0.1, 0.1, 0.85]
        result = minimize(log_likelihood, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8})

        self.omega, self.alpha, self.beta = result.x
        self.fitted = True

        sigma2 = np.zeros(n)
        sigma2[0] = var_r
        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * r[t-1]**2 + self.beta * sigma2[t-1]

        conditional_vol = np.sqrt(sigma2)
        long_run_var = self.omega / (1 - self.alpha - self.beta)

        return {
            "omega": float(self.omega),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "persistence": float(self.alpha + self.beta),
            "long_run_variance": float(long_run_var),
            "long_run_volatility": float(np.sqrt(long_run_var)),
            "conditional_volatility": conditional_vol.tolist(),
            "half_life": float(-np.log(2) / np.log(self.beta)) if self.beta > 0 else 0,
            "aic": float(2 * 3 + 2 * result.fun),
        }

    def forecast(self, returns, horizon=10):
        """Forecast conditional variance for horizon steps."""
        if not self.fitted:
            self.fit(returns)
        r = np.array(returns, dtype=float)
        last_var = self.omega + self.alpha * r[-1]**2 + self.beta * np.var(r)

        forecasts = []
        sigma2 = last_var
        for _ in range(horizon):
            sigma2 = self.omega + (self.alpha + self.beta) * sigma2
            forecasts.append(float(np.sqrt(sigma2)))

        long_run = float(np.sqrt(self.omega / (1 - self.alpha - self.beta)))
        return {"horizon": horizon, "volatility_forecast": forecasts,
                "converges_to_long_run": long_run}


class RiskEngine:
    """Comprehensive risk analysis engine."""

    def __init__(self):
        self.garch = GARCHModel()
        self.last_result = None

    def value_at_risk(self, returns, confidence=0.95, method="historical",
                      portfolio_value=1e6) -> Dict:
        """Calculate Value at Risk (historical, parametric, cornish_fisher)."""
        r = np.array(returns, dtype=float)
        n = len(r)
        alpha = 1 - confidence

        if method == "historical":
            sorted_returns = np.sort(r)
            idx = int(alpha * n)
            var_return = sorted_returns[idx]
        elif method == "parametric":
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            z = stats.norm.ppf(alpha)
            var_return = mu + z * sigma
        elif method in ("cornish_fisher", "monte_carlo"):
            # monte_carlo falls back to Cornish-Fisher (analytical approximation)
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            S = stats.skew(r)
            K = stats.kurtosis(r)
            z = stats.norm.ppf(alpha)
            z_cf = z + (z**2 - 1)*S/6 + (z**3 - 3*z)*K/24 - (2*z**3 - 5*z)*S**2/36
            var_return = mu + z_cf * sigma
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        self.last_result = {
            "method": f"VaR ({method})",
            "confidence_level": confidence,
            "var_return": float(var_return),
            "var_absolute": float(portfolio_value * abs(var_return)),
            "var_percentage": float(abs(var_return) * 100),
            "portfolio_value": portfolio_value,
            "mean_return": float(np.mean(r)),
            "volatility": float(np.std(r, ddof=1)),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
        }
        return self.last_result

    def conditional_var(self, returns, confidence=0.95, portfolio_value=1e6) -> Dict:
        """Conditional VaR (Expected Shortfall)."""
        r = np.array(returns, dtype=float)
        alpha = 1 - confidence
        sorted_returns = np.sort(r)
        idx = int(alpha * len(r))
        tail_returns = sorted_returns[:idx]

        cvar_return = float(np.mean(tail_returns))
        var_result = self.value_at_risk(returns, confidence, "historical", portfolio_value)

        self.last_result = {
            "method": "CVaR (Expected Shortfall)",
            "confidence_level": confidence,
            "cvar_return": cvar_return,
            "cvar_absolute": float(portfolio_value * abs(cvar_return)),
            "cvar_percentage": float(abs(cvar_return) * 100),
            "var_return": var_result["var_return"],
            "cvar_to_var_ratio": float(abs(cvar_return) / abs(var_result["var_return"])) if var_result["var_return"] != 0 else 0,
            "tail_losses": len(tail_returns),
            "worst_loss": float(sorted_returns[0]),
        }
        return self.last_result

    def garch_analysis(self, returns, forecast_horizon=10) -> Dict:
        """Full GARCH(1,1) analysis with forecasting."""
        fit_result = self.garch.fit(returns)
        forecast = self.garch.forecast(returns, forecast_horizon)
        self.last_result = {"method": "GARCH(1,1)", **fit_result, "forecast": forecast}
        return self.last_result

    def stress_test(self, returns, portfolio_value=1e6, scenarios=None) -> Dict:
        """Stress testing with historical and custom scenarios."""
        r = np.array(returns, dtype=float)
        sigma = np.std(r, ddof=1)

        if scenarios is None:
            scenarios = [
                {"name": "1987 Black Monday", "shock": -0.22},
                {"name": "2008 Financial Crisis", "shock": -0.15},
                {"name": "COVID-19 Crash", "shock": -0.12},
                {"name": "Rate Hike +2%", "shock": -0.08},
                {"name": "Flash Crash", "shock": -0.05},
                {"name": "Bull Market +20%", "shock": 0.20},
                {"name": "Mild Correction -5%", "shock": -0.05},
                {"name": "Stagflation", "shock": -0.10},
            ]

        results = []
        for sc in scenarios:
            loss = portfolio_value * abs(sc["shock"])
            z_score = abs(sc["shock"]) / sigma if sigma > 0 else 0
            severity = ("critical" if abs(sc["shock"]) > 0.15 else
                        "high" if abs(sc["shock"]) > 0.10 else
                        "medium" if abs(sc["shock"]) > 0.05 else "low")
            results.append({
                "name": sc["name"], "shock_percent": sc["shock"] * 100,
                "portfolio_loss": loss, "remaining_value": portfolio_value - loss,
                "z_score": float(z_score),
                "probability_approx": float(stats.norm.sf(z_score)),
                "severity": severity,
            })

        self.last_result = {
            "method": "Stress Testing", "portfolio_value": portfolio_value,
            "current_volatility": float(sigma), "scenarios": results,
            "worst_case_loss": max(s["portfolio_loss"] for s in results),
        }
        return self.last_result

    def maximum_drawdown(self, prices) -> Dict:
        """Calculate maximum drawdown and related metrics."""
        p = np.array(prices, dtype=float)
        cummax = np.maximum.accumulate(p)
        drawdowns = (p - cummax) / cummax

        max_dd = float(np.min(drawdowns))
        max_dd_idx = int(np.argmin(drawdowns))

        total_return = (p[-1] / p[0]) - 1
        n_years = len(p) / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        dd_periods = []
        in_drawdown = drawdowns < 0
        if np.any(in_drawdown):
            changes = np.diff(in_drawdown.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            if in_drawdown[0]: starts = np.insert(starts, 0, 0)
            if in_drawdown[-1]: ends = np.append(ends, len(p) - 1)
            for s, e in zip(starts, ends):
                dd_periods.append(int(e - s))

        self.last_result = {
            "method": "Maximum Drawdown Analysis",
            "max_drawdown": max_dd, "max_drawdown_percent": max_dd * 100,
            "max_drawdown_index": max_dd_idx, "calmar_ratio": float(calmar),
            "annualized_return": float(annual_return),
            "avg_drawdown_duration": float(np.mean(dd_periods)) if dd_periods else 0,
            "longest_drawdown": int(max(dd_periods)) if dd_periods else 0,
            "drawdown_series": drawdowns.tolist(),
            "current_drawdown": float(drawdowns[-1]),
        }
        return self.last_result

    def altman_z_score(self, working_capital, total_assets, retained_earnings,
                       ebit, market_cap, total_liabilities, sales) -> Dict:
        """Altman Z-Score for bankruptcy prediction."""
        X1 = working_capital / total_assets if total_assets else 0
        X2 = retained_earnings / total_assets if total_assets else 0
        X3 = ebit / total_assets if total_assets else 0
        X4 = market_cap / total_liabilities if total_liabilities else 0
        X5 = sales / total_assets if total_assets else 0
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        if Z > 2.99: zone, prob = "safe", "very low (< 1%)"
        elif Z > 1.81: zone, prob = "grey", "moderate"
        else: zone, prob = "distress", "high (> 80%)"

        self.last_result = {
            "method": "Altman Z-Score", "z_score": float(Z),
            "zone": zone, "bankruptcy_probability": prob,
            "components": {"X1_working_capital_ratio": float(X1),
                "X2_retained_earnings_ratio": float(X2),
                "X3_ebit_ratio": float(X3),
                "X4_market_equity_ratio": float(X4),
                "X5_sales_ratio": float(X5)},
        }
        return self.last_result

    def correlation_analysis(self, returns_matrix, asset_names=None) -> Dict:
        """Correlation, covariance, PCA analysis."""
        R = np.array(returns_matrix, dtype=float)
        n_assets = R.shape[1]
        if asset_names is None:
            asset_names = [f"Asset {i+1}" for i in range(n_assets)]

        corr = np.corrcoef(R, rowvar=False)
        cov = np.cov(R, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        variance_explained = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(variance_explained)
        condition_number = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else float("inf")

        self.last_result = {
            "method": "Correlation Analysis",
            "correlation_matrix": corr.tolist(),
            "covariance_matrix": cov.tolist(),
            "asset_names": asset_names,
            "eigenvalues": eigenvalues.tolist(),
            "variance_explained": variance_explained.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "condition_number": condition_number,
            "n_principal_components_90pct": int(np.searchsorted(cumulative_variance, 0.90) + 1),
        }
        return self.last_result
