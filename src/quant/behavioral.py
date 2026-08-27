"""JurisFinanceAI - Behavioral Finance Analysis

Implements:
- Prospect Theory (Kahneman-Tversky)
- Disposition Effect detection
- Overconfidence bias analysis
- Herd behavior / informational cascades
- Sentiment indicators
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


class ProspectTheory:
    """Prospect Theory value and weighting functions.

    Key insights:
    - Loss aversion: losses hurt ~2.25x more than equivalent gains
    - Diminishing sensitivity: marginal value decreases
    - Probability weighting: people overweight small probabilities
    """

    def __init__(self, lambda_param=2.25, alpha=0.88, beta=0.88,
                 gamma_gain=0.61, gamma_loss=0.69):
        self.lam = lambda_param  # Loss aversion coefficient
        self.alpha = alpha      # Sensitivity to gains
        self.beta = beta        # Sensitivity to losses
        self.gamma_gain = gamma_gain  # Probability weighting (gains)
        self.gamma_loss = gamma_loss  # Probability weighting (losses)

    def value_function(self, x):
        """Prospect Theory value function v(x).

        v(x) = x^alpha           if x >= 0 (gains)
        v(x) = -lambda * |x|^beta  if x < 0 (losses)
        """
        x = np.array(x, dtype=float)
        values = np.where(
            x >= 0,
            np.power(np.abs(x) + 1e-10, self.alpha),
            -self.lam * np.power(np.abs(x) + 1e-10, self.beta)
        )
        return values

    def weighting_function(self, p, domain="gain"):
        """Probability weighting function w(p).

        w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)
        """
        p = np.array(p, dtype=float)
        p = np.clip(p, 0.001, 0.999)
        gamma = self.gamma_gain if domain == "gain" else self.gamma_loss
        w = np.power(p, gamma) / np.power(
            np.power(p, gamma) + np.power(1 - p, gamma), 1/gamma
        )
        return w
    def evaluate_prospect(self, outcomes, probabilities):
        """Evaluate a prospect (gamble) using Prospect Theory.

        V = sum(w(p_i) * v(x_i))
        """
        outcomes = np.array(outcomes, dtype=float)
        probabilities = np.array(probabilities, dtype=float)

        values = self.value_function(outcomes)
        w_gain = self.weighting_function(probabilities, "gain")
        w_loss = self.weighting_function(probabilities, "loss")

        weights = np.where(outcomes >= 0, w_gain, w_loss)
        prospect_value = float(np.sum(weights * values))

        # Expected utility (rational benchmark)
        eu = float(np.sum(probabilities * outcomes))

        return {
            "prospect_value": prospect_value,
            "expected_utility": eu,
            "behavioral_bias": prospect_value - eu,
            "is_risk_seeking": prospect_value > eu,
            "loss_aversion_impact": float(np.sum(
                probabilities[outcomes < 0] * (self.lam - 1) * np.abs(values[outcomes < 0])
            )) if np.any(outcomes < 0) else 0,
        }


class BehavioralAnalyzer:
    """Behavioral finance analysis tools."""

    def __init__(self):
        self.last_result = None
        self.prospect_theory = ProspectTheory()

    def detect_disposition_effect(self, trades, prices=None) -> Dict:
        """Detect disposition effect in trading data.

        Disposition effect: tendency to sell winners too early
        and hold losers too long.

        Args:
            trades: List of dicts with 'entry_price', 'exit_price', 'pnl', 'holding_days'
        """
        if not trades:
            return {"method": "Disposition Effect", "error": "No trades data"}

        pnls = np.array([t.get("pnl", 0) for t in trades])
        holding_days = np.array([t.get("holding_days", 1) for t in trades])

        winners = pnls > 0
        losers = pnls < 0
        n_winners = np.sum(winners)
        n_losers = np.sum(losers)

        # Average holding period
        avg_hold_winners = float(np.mean(holding_days[winners])) if n_winners > 0 else 0
        avg_hold_losers = float(np.mean(holding_days[losers])) if n_losers > 0 else 0

        # Disposition ratio: (winners_sold / total_winners) / (losers_sold / total_losers)
        # Simplified: compare holding periods
        if avg_hold_losers > 0:
            disposition_ratio = avg_hold_winners / avg_hold_losers
        else:
            disposition_ratio = 1.0

        # PnL realization timing
        realized_gains = float(np.sum(pnls[winners])) if n_winners > 0 else 0
        realized_losses = float(np.sum(pnls[losers])) if n_losers > 0 else 0

        has_disposition = disposition_ratio < 0.8  # Selling winners faster

        self.last_result = {
            "method": "Disposition Effect Detection",
            "n_trades": len(trades),
            "n_winners": int(n_winners), "n_losers": int(n_losers),
            "avg_holding_winners": avg_hold_winners,
            "avg_holding_losers": avg_hold_losers,
            "disposition_ratio": float(disposition_ratio),
            "has_disposition_effect": has_disposition,
            "realized_gains": realized_gains,
            "realized_losses": realized_losses,
            "severity": "high" if disposition_ratio < 0.5 else
                        "moderate" if disposition_ratio < 0.8 else "low",
        }
        return self.last_result

    def overconfidence_analysis(self, returns, predicted_returns=None,
                                 benchmark_returns=None) -> Dict:
        """Analyze overconfidence bias.

        Indicators:
        - Trading frequency (excessive)
        - Prediction accuracy vs confidence
        - Risk taking relative to actual skill
        """
        r = np.array(returns, dtype=float)
        n = len(r)

        # Turnover analysis
        avg_abs_return = np.mean(np.abs(r))
        volatility = np.std(r, ddof=1)
        turnover_ratio = avg_abs_return / volatility if volatility > 0 else 0

        # Win rate
        win_rate = float(np.sum(r > 0) / n)

        # Confidence calibration
        if predicted_returns is not None:
            pred = np.array(predicted_returns, dtype=float)
            # Direction accuracy
            direction_correct = np.sum((r > 0) == (pred > 0)) / n
            # Magnitude accuracy
            mae = float(np.mean(np.abs(r - pred)))
            rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
        else:
            direction_correct = win_rate
            mae, rmse = 0, 0

        # Benchmark comparison
        if benchmark_returns is not None:
            bm = np.array(benchmark_returns, dtype=float)
            alpha = float(np.mean(r) - np.mean(bm))
            tracking_error = float(np.std(r - bm, ddof=1))
            info_ratio = alpha / tracking_error if tracking_error > 0 else 0
        else:
            alpha, tracking_error, info_ratio = 0, 0, 0

        # Overconfidence score (0-100)
        overconfidence = min(100, max(0,
            (turnover_ratio * 30 + (1 - direction_correct) * 40 + (1 - win_rate) * 30)
        ))

        self.last_result = {
            "method": "Overconfidence Analysis",
            "overconfidence_score": float(overconfidence),
            "win_rate": win_rate,
            "direction_accuracy": float(direction_correct),
            "turnover_ratio": float(turnover_ratio),
            "alpha_vs_benchmark": alpha,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "mae_vs_predictions": mae,
            "rmse_vs_predictions": rmse,
            "assessment": "high overconfidence" if overconfidence > 60 else
                          "moderate" if overconfidence > 30 else "well-calibrated",
        }
        return self.last_result

    def herd_behavior(self, returns_matrix, market_index=None) -> Dict:
        """Detect herd behavior in a group of assets/traders.

        Herding: tendency to follow the crowd, reducing dispersion
        during stress periods.

        Uses Cross-Sectional Dispersion of Returns (CSAD):
        CSAD_t = (1/N) * sum|r_it - r_mt|
        Low CSAD during high volatility = herding
        """
        R = np.array(returns_matrix, dtype=float)
        n_obs, n_assets = R.shape

        if market_index is None:
            market = R.mean(axis=1)
        else:
            market = np.array(market_index, dtype=float)

        # Cross-sectional absolute deviation
        csad = np.mean(np.abs(R - market.reshape(-1, 1)), axis=1)
        market_return = market
        market_vol = np.std(market_return)

        # Regress CSAD on |market_return| and market_return^2
        abs_mkt = np.abs(market_return)
        mkt_sq = market_return ** 2

        X = np.column_stack([np.ones(n_obs), abs_mkt, mkt_sq])
        y = csad
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Herding exists if beta[2] (coefficient on market_return^2) < 0
        # This means dispersion decreases during extreme market moves
        has_herding = beta[2] < 0

        # Calculate herding intensity during high-vol periods
        vol_threshold = np.percentile(np.abs(market_return), 75)
        high_vol_mask = np.abs(market_return) > vol_threshold
        if np.sum(high_vol_mask) > 5:
            herding_intensity = 1 - (np.mean(csad[high_vol_mask]) / (np.mean(csad) + 1e-10))
            herding_intensity = max(0, min(1, herding_intensity))
        else:
            herding_intensity = 0

        self.last_result = {
            "method": "Herd Behavior Detection",
            "has_herding": has_herding,
            "herding_intensity": float(herding_intensity),
            "csad_mean": float(np.mean(csad)),
            "csad_std": float(np.std(csad)),
            "beta_linear": float(beta[1]),
            "beta_nonlinear": float(beta[2]),
            "assessment": "significant herding" if herding_intensity > 0.3 else
                          "moderate herding" if herding_intensity > 0.1 else "no significant herding",
        }
        return self.last_result

    def sentiment_indicators(self, returns, volume=None) -> Dict:
        """Compute behavioral sentiment indicators from market data."""
        r = np.array(returns, dtype=float)
        n = len(r)

        # Momentum (short-term)
        momentum_5 = float(np.sum(r[-5:])) if n >= 5 else 0
        momentum_20 = float(np.sum(r[-20:])) if n >= 20 else 0

        # Moving average convergence/divergence (simplified)
        ma_short = np.mean(r[-10:]) if n >= 10 else np.mean(r)
        ma_long = np.mean(r[-30:]) if n >= 30 else np.mean(r)
        macd_signal = ma_short - ma_long

        # Relative strength
        up_days = np.sum(r > 0)
        down_days = np.sum(r < 0)
        rs_ratio = up_days / (down_days + 1)

        # Volatility regime
        vol_short = np.std(r[-20:]) if n >= 20 else np.std(r)
        vol_long = np.std(r) if n >= 50 else np.std(r)
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1

        # Fear/Greed index (0-100)
        fear_greed = np.clip(
            50 + momentum_20 * 500 + macd_signal * 200 + (rs_ratio - 1) * 30,
            0, 100
        )

        # Volume confirmation
        if volume is not None:
            v = np.array(volume, dtype=float)
            vol_change = (v[-1] / np.mean(v[-20:]) - 1) if len(v) >= 20 else 0
        else:
            vol_change = 0

        self.last_result = {
            "method": "Sentiment Indicators",
            "momentum_5d": momentum_5,
            "momentum_20d": momentum_20,
            "macd_signal": float(macd_signal),
            "up_down_ratio": float(rs_ratio),
            "volatility_regime": float(vol_ratio),
            "fear_greed_index": float(fear_greed),
            "volume_change_pct": float(vol_change),
            "sentiment": "greed" if fear_greed > 60 else
                        "fear" if fear_greed < 40 else "neutral",
        }
        return self.last_result
