"""JurisFinanceAI - Time Series Analysis Engine

Implements ARIMA, ADF test, rolling statistics,
return analysis, and full analysis pipeline.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class ARIMAModel:
    """ARIMA(p,d,q) implementation with conditional least squares."""

    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = None
        self.ma_params = None
        self.fitted = False

    def _difference(self, y, d):
        result = y.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def fit(self, y):
        """Fit ARIMA using conditional least squares."""
        y = np.array(y, dtype=float)
        n = len(y)
        y_diff = self._difference(y, self.d)
        n_diff = len(y_diff)
        if n_diff < max(self.p, self.q) + 10:
            return {"error": "Not enough data after differencing"}

        def arima_loss(params):
            ar = params[:self.p]
            ma = params[self.p:self.p + self.q]
            residuals = np.zeros(n_diff)
            for t in range(max(self.p, self.q), n_diff):
                ar_term = sum(ar[j] * y_diff[t - j - 1] for j in range(min(self.p, t)))
                ma_term = sum(ma[j] * residuals[t - j - 1] for j in range(min(self.q, t)))
                residuals[t] = y_diff[t] - ar_term - ma_term
            return np.sum(residuals[max(self.p, self.q):] ** 2)

        x0 = np.zeros(self.p + self.q)
        result = minimize(arima_loss, x0, method="L-BFGS-B", options={"maxiter": 1000})

        self.ar_params = result.x[:self.p] if self.p > 0 else np.array([])
        self.ma_params = result.x[self.p:self.p + self.q] if self.q > 0 else np.array([])
        self.fitted = True

        residuals = np.zeros(n_diff)
        for t in range(max(self.p, self.q), n_diff):
            ar_term = sum(self.ar_params[j] * y_diff[t-j-1] for j in range(min(self.p, t)))
            ma_term = sum(self.ma_params[j] * residuals[t-j-1] for j in range(min(self.q, t)))
            residuals[t] = y_diff[t] - ar_term - ma_term

        sigma2 = np.var(residuals[max(self.p, self.q):])
        aic = n_diff * np.log(sigma2) + 2 * (self.p + self.q)
        bic = n_diff * np.log(sigma2) + np.log(n_diff) * (self.p + self.q)

        return {
            "ar_params": self.ar_params.tolist(),
            "ma_params": self.ma_params.tolist(),
            "residuals_std": float(np.std(residuals[max(self.p, self.q):])),
            "sigma2": float(sigma2), "aic": float(aic), "bic": float(bic),
            "n_observations": n, "n_diff_observations": n_diff,
        }

    def forecast(self, y, steps=10):
        """Forecast future values."""
        if not self.fitted:
            self.fit(y)
        y = np.array(y, dtype=float)
        y_diff = self._difference(y, self.d)
        n_diff = len(y_diff)

        history_diff = list(y_diff)
        history_res = list(np.zeros(n_diff))

        for t in range(max(self.p, self.q), n_diff):
            ar_term = sum(self.ar_params[j] * y_diff[t-j-1] for j in range(self.p))
            ma_term = sum(self.ma_params[j] * history_res[t-j-1] for j in range(self.q))
            history_res[t] = y_diff[t] - ar_term - ma_term

        forecasts_diff = []
        for _ in range(steps):
            t = len(history_diff)
            ar_term = sum(self.ar_params[j] * history_diff[t-j-1] for j in range(self.p))
            ma_term = sum(self.ma_params[j] * history_res[t-j-1] for j in range(self.q))
            pred = ar_term + ma_term
            forecasts_diff.append(pred)
            history_diff.append(pred)
            history_res.append(0)

        full_diff = np.array(list(y_diff) + forecasts_diff)
        forecast_values = []
        for i in range(steps):
            idx = n_diff + i
            val = y[-1] if self.d > 0 else y[-1]
            for dd in range(self.d):
                val = (val + full_diff[idx - dd])
            forecast_values.append(float(val))

        sigma = np.std(y_diff) * np.sqrt(np.arange(1, steps + 1)) * 0.5
        return {
            "forecast_values": forecast_values,
            "confidence_lower": (np.array(forecast_values) - sigma).tolist(),
            "confidence_upper": (np.array(forecast_values) + sigma).tolist(),
            "steps": steps, "method": f"ARIMA({self.p},{self.d},{self.q})",
        }

    def auto_arima(self, y, max_p=3, max_d=2, max_q=3):
        """Automatic ARIMA order selection by AIC."""
        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_fit = None

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p + q == 0: continue
                    try:
                        model = ARIMAModel(p, d, q)
                        fit = model.fit(y)
                        if "error" not in fit and fit["aic"] < best_aic:
                            best_aic = fit["aic"]
                            best_order = (p, d, q)
                            best_fit = fit
                    except Exception:
                        continue

        self.p, self.d, self.q = best_order
        self.fitted = True
        return {"best_order": best_order, "best_aic": float(best_aic), "fit": best_fit}


class TimeSeriesAnalyzer:
    """Comprehensive time series analysis."""

    def __init__(self):
        self.last_result = None

    def adf_test(self, y):
        """Augmented Dickey-Fuller test (simplified)."""
        y = np.array(y, dtype=float)
        dy = np.diff(y)
        y_lag = y[:-1]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        residuals = dy - X @ beta
        sigma2 = np.sum(residuals ** 2) / (len(dy) - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
        t_stat = beta[1] / se_beta
        stationary = t_stat < -2.86

        return {
            "test": "ADF", "t_statistic": float(t_stat),
            "is_stationary": stationary,
            "critical_values": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
            "recommended_differencing": 0 if stationary else 1,
        }

    def rolling_statistics(self, y, window=20):
        """Rolling mean, std, skewness, kurtosis."""
        y = np.array(y, dtype=float)
        n = len(y)
        if n < window: window = max(n // 2, 5)

        rolling_mean = np.convolve(y, np.ones(window)/window, mode="valid").tolist()
        rolling_std, rolling_skew, rolling_kurt = [], [], []
        for i in range(n - window + 1):
            chunk = y[i:i+window]
            rolling_std.append(float(np.std(chunk, ddof=1)))
            rolling_skew.append(float(stats.skew(chunk)))
            rolling_kurt.append(float(stats.kurtosis(chunk)))

        return {"window": window, "rolling_mean": rolling_mean, "rolling_std": rolling_std,
                "rolling_skewness": rolling_skew, "rolling_kurtosis": rolling_kurt}

    def return_analysis(self, prices):
        """Comprehensive return analysis."""
        p = np.array(prices, dtype=float)
        simple_returns = np.diff(p) / p[:-1]
        log_returns = np.log(p[1:] / p[:-1])
        n_years = len(p) / 252
        total_return = (p[-1] / p[0]) - 1
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        annual_vol = float(np.std(log_returns) * np.sqrt(252))
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
        negative_returns = log_returns[log_returns < 0]
        downside_dev = float(np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)) if len(negative_returns) > 0 else 0
        sortino = (annual_return - 0.02) / downside_dev if downside_dev > 0 else 0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annual_return),
            "annualized_volatility": annual_vol,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "best_day_return": float(np.max(simple_returns)),
            "worst_day_return": float(np.min(simple_returns)),
            "win_rate": float(np.sum(simple_returns > 0) / len(simple_returns)),
            "mean_daily_return": float(np.mean(simple_returns)),
            "daily_volatility": float(np.std(simple_returns, ddof=1)),
            "skewness": float(stats.skew(simple_returns)),
            "kurtosis": float(stats.kurtosis(simple_returns)),
            "jarque_bera_stat": float(stats.jarque_bera(simple_returns)[0]),
            "simple_returns": simple_returns.tolist(),
            "log_returns": log_returns.tolist(),
        }

    def full_analysis(self, y, forecast_steps=20):
        """Run complete time series analysis pipeline."""
        y = np.array(y, dtype=float)
        adf = self.adf_test(y)
        returns = self.return_analysis(y)
        rolling = self.rolling_statistics(y, window=min(20, len(y) // 3))

        arima_model = ARIMAModel()
        auto_result = arima_model.auto_arima(y, max_p=3, max_d=2, max_q=3)

        if auto_result.get("best_order"):
            p, d, q = auto_result["best_order"]
            model = ARIMAModel(p, d, q)
            model.fit(y)
            forecast = model.forecast(y, forecast_steps)
        else:
            forecast = {"forecast_values": [], "confidence_lower": [], "confidence_upper": []}

        self.last_result = {
            "adf_test": adf, "return_analysis": returns,
            "rolling_statistics": rolling,
            "arima_order": auto_result["best_order"],
            "arima_aic": auto_result["best_aic"],
            "forecast": forecast, "n_observations": len(y),
        }
        return self.last_result
