"""JurisFinanceAI - Portfolio Optimization Engine

Implements Markowitz Mean-Variance, Black-Litterman, and Fuzzy Portfolio Optimization.

Mathematical Foundations:
- Markowitz: min(w'Σw) s.t. w'μ = r_target, w'1 = 1
- Black-Litterman: μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1π + P'Ω^-1q]
- Fuzzy: Uses triangular fuzzy numbers for returns
- Risk Parity: Equal risk contribution from each asset
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class FuzzyNumber:
    """Triangular fuzzy number: (a, b, c) where b is peak, a and c are bounds."""

    def __init__(self, a: float, b: float, c: float):
        assert a <= b <= c, f"Invalid fuzzy number ({a}, {b}, {c})"
        self.a = a
        self.b = b
        self.c = c

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """Get alpha-cut interval [L(alpha), U(alpha)]."""
        alpha = np.clip(alpha, 0, 1)
        left = self.a + alpha * (self.b - self.a)
        right = self.c - alpha * (self.c - self.b)
        return left, right

    def expected_value(self) -> float:
        """Defuzzified expected value: (a + 2b + c) / 4."""
        return (self.a + 2 * self.b + self.c) / 4.0

    def defuzzify_centroid(self) -> float:
        """Centroid defuzzification."""
        return (self.a + self.b + self.c) / 3.0

    def __repr__(self):
        return f"FuzzyNumber({self.a:.4f}, {self.b:.4f}, {self.c:.4f})"


class PortfolioOptimizer:
    """Advanced portfolio optimization engine.

    Supports:
    - Markowitz Mean-Variance Optimization (MVO)
    - Black-Litterman model
    - Fuzzy portfolio optimization
    - Sharpe ratio maximization
    - Efficient frontier computation
    - Minimum variance portfolio
    - Risk parity
    """

    def __init__(self):
        self.last_result = None

    def markowitz_optimize(self, expected_returns, cov_matrix,
                            target_return=None,
                            risk_free_rate=0.02,
                            short_selling=False,
                            max_weight=1.0) -> Dict:
        """Classical Markowitz Mean-Variance Optimization.

        Solves: min w'Σw  s.t. w'μ >= r_target, sum(w_i) = 1
        """
        n = len(expected_returns)
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float)
        Sigma = (Sigma + Sigma.T) / 2

        if short_selling:
            bounds = [(-max_weight, max_weight)] * n
        else:
            bounds = [(0.0, max_weight)] * n

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, r=target_return: np.dot(w, mu) - r
            })

        w0 = np.ones(n) / n
        result = minimize(
            fun=lambda w: w @ Sigma @ w,
            x0=w0, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000}
        )

        weights = result.x
        port_return = float(np.dot(weights, mu))
        port_risk = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0

        frontier = self._compute_efficient_frontier(mu, Sigma, risk_free_rate,
                                                      short_selling, max_weight)
        asset_risks = np.sqrt(np.diag(Sigma))
        asset_sharpes = (mu - risk_free_rate) / np.where(asset_risks > 0, asset_risks, 1e-10)

        self.last_result = {
            "method": "Markowitz Mean-Variance",
            "weights": weights.tolist(),
            "portfolio_return": port_return,
            "portfolio_risk": port_risk,
            "sharpe_ratio": float(sharpe),
            "efficient_frontier": frontier,
            "asset_sharpes": asset_sharpes.tolist(),
            "asset_risks": asset_risks.tolist(),
            "n_assets": n,
            "target_return": target_return,
            "success": result.success,
        }
        return self.last_result

    def maximize_sharpe(self, expected_returns, cov_matrix,
                        risk_free_rate=0.02,
                        short_selling=False,
                        max_weight=1.0) -> Dict:
        """Maximize Sharpe Ratio (tangent portfolio)."""
        n = len(expected_returns)
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float)
        Sigma = (Sigma + Sigma.T) / 2

        bounds = [(-max_weight, max_weight) if short_selling else (0.0, max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_sharpe(w):
            ret = w @ mu
            risk = np.sqrt(w @ Sigma @ w)
            return -(ret - risk_free_rate) / risk if risk > 1e-10 else 0

        w0 = np.ones(n) / n
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

        weights = result.x
        port_return = float(np.dot(weights, mu))
        port_risk = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0

        self.last_result = {
            "method": "Sharpe Ratio Maximization",
            "weights": weights.tolist(),
            "portfolio_return": port_return,
            "portfolio_risk": port_risk,
            "sharpe_ratio": float(sharpe),
            "n_assets": n,
            "success": result.success,
        }
        return self.last_result

    def black_litterman(self, market_caps, cov_matrix,
                        risk_aversion=2.5,
                        risk_free_rate=0.02,
                        tau=0.05,
                        views=None) -> Dict:
        """Black-Litterman model combining market equilibrium with investor views.

        pi = delta * Sigma * w_mkt  (implied equilibrium returns)
        mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 [(tau*Sigma)^-1*pi + P'*Omega^-1*q]
        """
        n = len(market_caps)
        Sigma = np.array(cov_matrix, dtype=float)
        Sigma = (Sigma + Sigma.T) / 2

        w_mkt = np.array(market_caps, dtype=float)
        w_mkt = w_mkt / w_mkt.sum()
        pi = risk_aversion * Sigma @ w_mkt

        if views is not None and len(views) > 0:
            k = len(views)
            P_mat = np.zeros((k, n))
            q_vec = np.zeros(k)
            omega_diag = np.zeros(k)

            for i, view in enumerate(views):
                for asset_idx, weight in view.get("assets", {}).items():
                    P_mat[i, int(asset_idx)] = weight
                q_vec[i] = view["value"]
                conf = view.get("confidence", 0.5)
                omega_diag[i] = max((1.0 / conf - 1.0) * tau * (P_mat[i] @ Sigma @ P_mat[i]), 1e-10)

            omega_mat = np.diag(omega_diag)
            tau_sigma = tau * Sigma
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega_mat)
            P_omega_inv_P = P_mat.T @ omega_inv @ P_mat
            M = np.linalg.inv(tau_sigma_inv + P_omega_inv_P)
            bl_returns = M @ (tau_sigma_inv @ pi + P_mat.T @ omega_inv @ q_vec)
            bl_cov = Sigma + M
            delta_sigma_inv = np.linalg.inv(risk_aversion * bl_cov)
            excess_returns = bl_returns - risk_free_rate
            weights = delta_sigma_inv @ excess_returns
            weights = weights / weights.sum()
        else:
            k = 0
            bl_returns = pi
            bl_cov = Sigma
            weights = w_mkt

        port_return = float(np.dot(weights, bl_returns))
        port_risk = float(np.sqrt(weights @ bl_cov @ weights))
        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0

        self.last_result = {
            "method": "Black-Litterman",
            "weights": weights.tolist(),
            "implied_equilibrium_returns": pi.tolist(),
            "bl_returns": bl_returns.tolist(),
            "portfolio_return": port_return,
            "portfolio_risk": port_risk,
            "sharpe_ratio": float(sharpe),
            "market_weights": w_mkt.tolist(),
            "n_assets": n,
            "n_views": k,
            "tau": tau,
            "risk_aversion": risk_aversion,
        }
        return self.last_result

    def fuzzy_optimize(self, fuzzy_returns, cov_matrix,
                       alpha=0.5, risk_free_rate=0.02,
                       short_selling=False, max_weight=1.0,
                       mode="expected") -> Dict:
        """Fuzzy Portfolio Optimization with triangular fuzzy numbers.

        Modes: expected, centroid, possibility, necessity
        """
        n = len(fuzzy_returns)
        Sigma = np.array(cov_matrix, dtype=float)
        Sigma = (Sigma + Sigma.T) / 2

        if mode == "expected":
            mu = np.array([fr.expected_value() for fr in fuzzy_returns])
        elif mode == "centroid":
            mu = np.array([fr.defuzzify_centroid() for fr in fuzzy_returns])
        elif mode == "possibility":
            mu = np.array([fr.alpha_cut(alpha)[1] for fr in fuzzy_returns])
        elif mode == "necessity":
            mu = np.array([fr.alpha_cut(alpha)[0] for fr in fuzzy_returns])
        else:
            mu = np.array([fr.expected_value() for fr in fuzzy_returns])

        fuzzy_intervals = [fr.alpha_cut(alpha) for fr in fuzzy_returns]

        result = self.markowitz_optimize(mu, Sigma, risk_free_rate=risk_free_rate,
                                          short_selling=short_selling, max_weight=max_weight)

        weights = np.array(result["weights"])
        port_lower = sum(w * fr.alpha_cut(alpha)[0] for w, fr in zip(weights, fuzzy_returns))
        port_upper = sum(w * fr.alpha_cut(alpha)[1] for w, fr in zip(weights, fuzzy_returns))
        port_peak = sum(w * fr.b for w, fr in zip(weights, fuzzy_returns))

        self.last_result = {
            "method": f"Fuzzy Portfolio Optimization ({mode})",
            "weights": result["weights"],
            "defuzzified_returns": mu.tolist(),
            "fuzzy_intervals": [(float(l), float(u)) for l, u in fuzzy_intervals],
            "portfolio_return": result["portfolio_return"],
            "portfolio_risk": result["portfolio_risk"],
            "sharpe_ratio": result["sharpe_ratio"],
            "fuzzy_portfolio_return": {
                "lower_bound": float(port_lower),
                "peak": float(port_peak),
                "upper_bound": float(port_upper),
            },
            "alpha_level": alpha,
            "mode": mode,
            "n_assets": n,
        }
        return self.last_result

    def risk_parity(self, cov_matrix,
                    expected_returns=None,
                    risk_free_rate=0.02) -> Dict:
        """Risk Parity - equal risk contribution from each asset."""
        n = len(cov_matrix)
        Sigma = np.array(cov_matrix, dtype=float)
        Sigma = (Sigma + Sigma.T) / 2

        def risk_parity_objective(w):
            sigma_p = np.sqrt(w @ Sigma @ w)
            if sigma_p < 1e-10:
                return 1e10
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / sigma_p
            target_rc = sigma_p / n
            return np.sum((risk_contrib - target_rc) ** 2)

        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n
        result = minimize(risk_parity_objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-15, "maxiter": 2000})

        weights = result.x
        port_risk = float(np.sqrt(weights @ Sigma @ weights))

        if expected_returns is not None:
            mu = np.array(expected_returns, dtype=float)
            port_return = float(np.dot(weights, mu))
            sharpe = (port_return - risk_free_rate) / port_risk
        else:
            port_return = 0.0
            sharpe = 0.0

        marginal_contrib = Sigma @ weights
        risk_contributions = weights * marginal_contrib / port_risk

        self.last_result = {
            "method": "Risk Parity",
            "weights": weights.tolist(),
            "portfolio_return": float(port_return),
            "portfolio_risk": port_risk,
            "sharpe_ratio": float(sharpe),
            "risk_contributions": risk_contributions.tolist(),
            "pct_risk_contributions": (risk_contributions / port_risk * 100).tolist(),
            "n_assets": n,
            "success": result.success,
        }
        return self.last_result

    def minimum_variance(self, cov_matrix,
                         expected_returns=None,
                         risk_free_rate=0.02,
                         short_selling=False) -> Dict:
        """Global Minimum Variance Portfolio."""
        n = len(cov_matrix)
        if expected_returns is None:
            expected_returns = np.zeros(n)
        return self.markowitz_optimize(
            expected_returns, cov_matrix, target_return=None,
            risk_free_rate=risk_free_rate, short_selling=short_selling
        )

    def _compute_efficient_frontier(self, mu, Sigma, risk_free_rate,
                                      short_selling, max_weight, n_points=50):
        """Compute efficient frontier points."""
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier = []
        n = len(mu)
        bounds = [(-max_weight, max_weight) if short_selling else (0.0, max_weight)] * n

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w, r=target: np.dot(w, mu) - r},
            ]
            try:
                result = minimize(
                    fun=lambda w: w @ Sigma @ w, x0=np.ones(n) / n,
                    method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-10, "maxiter": 500}
                )
                if result.success:
                    w = result.x
                    risk = float(np.sqrt(w @ Sigma @ w))
                    ret = float(np.dot(w, mu))
                    frontier.append({"return": ret, "risk": risk,
                                     "sharpe": (ret - risk_free_rate) / risk if risk > 0 else 0})
            except Exception:
                continue
        return frontier
