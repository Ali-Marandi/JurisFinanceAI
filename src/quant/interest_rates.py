"""JurisFinanceAI - Interest Rate Models

Implements short-rate models:
- Vasicek: dr = a(b - r)dt + sigma*dW (mean-reverting)
- CIR (Cox-Ingersoll-Ross): dr = a(b - r)dt + sigma*sqrt(r)*dW
- Hull-White: dr = [theta(t) - a*r]dt + sigma*dW

These are used for:
- Yield curve modeling
- Bond pricing
- Interest rate derivatives
- Duration and convexity analysis
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


class InterestRateModel:
    """Short-rate interest rate model engine."""

    def __init__(self):
        self.last_result = None

    def vasicek(self, r0, a, b, sigma, T=1.0, n_steps=252,
                n_paths=1000) -> Dict:
        """Vasicek mean-reverting short rate model.

        dr = a(b - r)dt + sigma * dW

        Parameters:
            r0: Initial short rate
            a: Mean reversion speed
            b: Long-term mean rate
            sigma: Volatility
            T: Time horizon (years)
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        for t in range(n_steps):
            dr = a * (b - paths[:, t]) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(n_paths)
            paths[:, t + 1] = paths[:, t] + dr

        final_rates = paths[:, -1]
        mean_path = np.mean(paths, axis=0)

        # Zero-coupon bond price: P(t,T) = A(t,T) * exp(-B(t,T) * r_t)
        def vasicek_bond_price(r_t, t, T):
            tau = T - t
            B = (1 - np.exp(-a * tau)) / a
            A = np.exp((b - sigma**2 / (2 * a**2)) * (B - tau) - sigma**2 * B**2 / (4 * a))
            return A * np.exp(-B * r_t)

        bond_price = vasicek_bond_price(r0, 0, T)

        # Yield curve from model
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
        yields = np.zeros(len(maturities))
        for i, tau in enumerate(maturities):
            P = vasicek_bond_price(r0, 0, tau)
            yields[i] = -np.log(P) / tau

        self.last_result = {
            "method": "Vasicek Model",
            "parameters": {"r0": r0, "a": a, "b": b, "sigma": sigma},
            "final_rates_mean": float(np.mean(final_rates)),
            "final_rates_std": float(np.std(final_rates)),
            "mean_path": mean_path.tolist(),
            "bond_price_T": float(bond_price),
            "yield_curve": {
                "maturities": maturities.tolist(),
                "yields": yields.tolist(),
            },
            "n_paths": n_paths,
        }
        return self.last_result

    def cir_model(self, r0, a, b, sigma, T=1.0, n_steps=252,
                  n_paths=1000) -> Dict:
        """Cox-Ingersoll-Ross model.

        dr = a(b - r)dt + sigma * sqrt(r) * dW

        Key difference from Vasicek: rates stay non-negative.
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        for t in range(n_steps):
            r = np.maximum(paths[:, t], 0)
            dr = a * (b - r) * dt + sigma * np.sqrt(r * dt) * np.random.standard_normal(n_paths)
            paths[:, t + 1] = np.maximum(r + dr, 0)

        final_rates = paths[:, -1]
        mean_path = np.mean(paths, axis=0)

        # CIR zero-coupon bond price
        def cir_bond_price(r_t, t, T):
            tau = T - t
            h = np.sqrt(a**2 + 2 * sigma**2)
            B = 2 * (np.exp(h * tau) - 1) / ((h + a) * (np.exp(h * tau) - 1) + 2 * h)
            A = ((2 * h * np.exp((a + h) * tau / 2)) /
                 ((h + a) * (np.exp(h * tau) - 1) + 2 * h)) ** (2 * a * b / sigma**2)
            return A * np.exp(-B * r_t)

        bond_price = cir_bond_price(r0, 0, T)

        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
        yields = np.zeros(len(maturities))
        for i, tau in enumerate(maturities):
            P = cir_bond_price(r0, 0, tau)
            yields[i] = -np.log(max(P, 1e-10)) / tau

        self.last_result = {
            "method": "CIR Model",
            "parameters": {"r0": r0, "a": a, "b": b, "sigma": sigma},
            "final_rates_mean": float(np.mean(final_rates)),
            "final_rates_std": float(np.std(final_rates)),
            "mean_path": mean_path.tolist(),
            "bond_price_T": float(bond_price),
            "yield_curve": {
                "maturities": maturities.tolist(),
                "yields": yields.tolist(),
            },
            "n_paths": n_paths,
        }
        return self.last_result

    def hull_white(self, r0, a, sigma, T=1.0, n_steps=252,
                   n_paths=1000, theta_func=None) -> Dict:
        """Hull-White extended Vasicek model.

        dr = [theta(t) - a*r]dt + sigma*dW

        theta(t) can be calibrated to match the current yield curve.
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        for t in range(n_steps):
            t_val = t * dt
            if theta_func is not None:
                theta = theta_func(t_val)
            else:
                theta = a * b if hasattr(self, '_hw_b') else 0.03
            dr = (theta - a * paths[:, t]) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(n_paths)
            paths[:, t + 1] = paths[:, t] + dr

        self.last_result = {
            "method": "Hull-White Model",
            "final_rates_mean": float(np.mean(paths[:, -1])),
            "final_rates_std": float(np.std(paths[:, -1])),
            "mean_path": np.mean(paths, axis=0).tolist(),
            "n_paths": n_paths,
        }
        return self.last_result

    def duration_convexity(self, cashflows, rates, ytm) -> Dict:
        """Calculate bond duration and convexity.

        Macaulay Duration = sum(t * CF_t / (1+y)^t) / Price
        Modified Duration = Macaulay Duration / (1 + y/n)
        Convexity = sum(t*(t+1) * CF_t / (1+y)^(t+2)) / Price
        """
        cf = np.array(cashflows, dtype=float)
        n = len(cf)
        discount_factors = np.array([(1 + ytm) ** (i + 1) for i in range(n)])
        pv_cf = cf / discount_factors
        price = np.sum(pv_cf)

        # Macaulay duration
        times = np.arange(1, n + 1)
        macaulay_duration = np.sum(times * pv_cf) / price

        # Modified duration
        modified_duration = macaulay_duration / (1 + ytm)

        # Convexity
        convexity = np.sum(times * (times + 1) * pv_cf) / (price * (1 + ytm) ** 2)

        # DV01 (dollar value of one basis point)
        dv01 = modified_duration * price * 0.0001

        self.last_result = {
            "method": "Duration & Convexity",
            "price": float(price),
            "macaulay_duration": float(macaulay_duration),
            "modified_duration": float(modified_duration),
            "convexity": float(convexity),
            "dv01": float(dv01),
            "yield_to_maturity": ytm,
        }
        return self.last_result
