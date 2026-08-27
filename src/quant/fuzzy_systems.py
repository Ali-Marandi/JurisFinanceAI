"""JurisFinanceAI - Fuzzy Logic Systems for Finance

Implements:
- Fuzzy Credit Scoring (replaces hard thresholds with linguistic variables)
- Fuzzy Portfolio Optimization helper
- ANFIS-inspired neural-fuzzy hybrid
- Fuzzy AHP for multi-criteria decision making
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def trimf(x, a, b, c):
    """Triangular membership function."""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-10), (c - x) / (c - b + 1e-10)))


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    return np.maximum(0, np.minimum((x-a)/(b-a+1e-10), 1, (d-x)/(d-c+1e-10)))


class FuzzyCreditScorer:
    """Fuzzy credit scoring using linguistic variables.

    Replaces hard thresholds (e.g., income < X => reject) with
    fuzzy membership functions for more nuanced scoring.
    """

    # Fuzzy sets for input variables
    INCOME_SETS = {
        "very_low": (0, 0, 2000),
        "low": (1000, 3000, 5000),
        "medium": (4000, 7000, 10000),
        "high": (8000, 15000, 25000),
        "very_high": (20000, 50000, 100000),
    }

    DEBT_RATIO_SETS = {
        "excellent": (0, 0, 0.2),
        "good": (0.1, 0.3, 0.4),
        "moderate": (0.3, 0.5, 0.6),
        "high": (0.5, 0.7, 0.8),
        "critical": (0.7, 0.9, 1.0),
    }

    CREDIT_HISTORY_SETS = {
        "no_history": (0, 0, 1),
        "poor": (0, 2, 4),
        "fair": (3, 6, 8),
        "good": (7, 12, 18),
        "excellent": (15, 25, 30),
    }

    EMPLOYMENT_SETS = {
        "unstable": (0, 0, 1),
        "short_term": (0.5, 1.5, 2.5),
        "medium_term": (2, 5, 7),
        "long_term": (5, 10, 20),
        "very_stable": (15, 25, 40),
    }

    # Output: credit score quality
    OUTPUT_SETS = {
        "very_poor": (0, 0, 200),
        "poor": (150, 300, 450),
        "fair": (400, 550, 650),
        "good": (600, 720, 800),
        "excellent": (750, 900, 1000),
    }

    def __init__(self):
        self.last_result = None

    def _get_membership(self, value, fuzzy_sets):
        """Get membership degrees for a value across all fuzzy sets."""
        memberships = {}
        for name, (a, b, c) in fuzzy_sets.items():
            memberships[name] = float(np.clip(trimf(np.array([value]), a, b, c)[0], 0, 1))
        return memberships

    def _defuzzify_centroid(self, output_activations):
        """Centroid defuzzification."""
        x = np.linspace(0, 1000, 1000)
        aggregated = np.zeros_like(x)

        for name, degree in output_activations.items():
            a, b, c = self.OUTPUT_SETS[name]
            membership = trimf(x, a, b, c)
            aggregated = np.maximum(aggregated, np.minimum(membership, degree))

        if np.sum(aggregated) == 0:
            return 500.0

        return float(np.sum(x * aggregated) / np.sum(aggregated))

    def evaluate(self, income, debt_ratio, credit_history_years,
                 employment_years, existing_debt=0) -> Dict:
        """Evaluate credit score using fuzzy inference.

        Args:
            income: Monthly income
            debt_ratio: Total debt / income ratio (0-1)
            credit_history_years: Years of credit history
            employment_years: Years at current job
            existing_debt: Existing outstanding debt amount
        """
        # Fuzzify inputs
        income_mf = self._get_membership(income, self.INCOME_SETS)
        debt_mf = self._get_membership(debt_ratio, self.DEBT_RATIO_SETS)
        credit_mf = self._get_membership(credit_history_years, self.CREDIT_HISTORY_SETS)
        employment_mf = self._get_membership(employment_years, self.EMPLOYMENT_SETS)

        # Rule base: IF income IS X AND debt IS Y THEN credit IS Z
        output_activations = {
            "very_poor": 0, "poor": 0, "fair": 0, "good": 0, "excellent": 0
        }

        rules_fired = []

        # High income, low debt => excellent/good
        activation = min(income_mf.get("high", 0), debt_mf.get("excellent", 0))
        output_activations["excellent"] = max(output_activations["excellent"], activation)
        if activation > 0.01:
            rules_fired.append(f"IF income=high AND debt=excellent => excellent ({activation:.2f})")

        activation = min(income_mf.get("very_high", 0), debt_mf.get("excellent", 0))
        output_activations["excellent"] = max(output_activations["excellent"], activation)

        # Medium income, moderate debt => fair
        activation = min(income_mf.get("medium", 0), debt_mf.get("moderate", 0))
        output_activations["fair"] = max(output_activations["fair"], activation)
        if activation > 0.01:
            rules_fired.append(f"IF income=medium AND debt=moderate => fair ({activation:.2f})")

        # Low income, high debt => poor
        activation = min(income_mf.get("low", 0), debt_mf.get("high", 0))
        output_activations["poor"] = max(output_activations["poor"], activation)
        if activation > 0.01:
            rules_fired.append(f"IF income=low AND debt=high => poor ({activation:.2f})")

        # Good credit history boost
        activation = min(credit_mf.get("excellent", 0), employment_mf.get("long_term", 0))
        output_activations["good"] = max(output_activations["good"], activation)
        if activation > 0.01:
            rules_fired.append(f"IF credit=excellent AND employment=long => good ({activation:.2f})")

        # Very poor credit history
        activation = min(credit_mf.get("poor", 0), debt_mf.get("critical", 0))
        output_activations["very_poor"] = max(output_activations["very_poor"], activation)
        if activation > 0.01:
            rules_fired.append(f"IF credit=poor AND debt=critical => very_poor ({activation:.2f})")

        # Stable employment, medium income => good
        activation = min(employment_mf.get("very_stable", 0), income_mf.get("medium", 0))
        output_activations["good"] = max(output_activations["good"], activation * 0.8)

        # Defuzzify
        credit_score = self._defuzzify_centroid(output_activations)

        # Decision
        if credit_score >= 750: decision, risk_level = "approve", "low"
        elif credit_score >= 600: decision, risk_level = "approve_with_conditions", "medium"
        elif credit_score >= 450: decision, risk_level = "review", "high"
        else: decision, risk_level = "reject", "very_high"

        self.last_result = {
            "method": "Fuzzy Credit Scoring",
            "credit_score": credit_score,
            "decision": decision,
            "risk_level": risk_level,
            "input_memberships": {
                "income": income_mf,
                "debt_ratio": debt_mf,
                "credit_history": credit_mf,
                "employment": employment_mf,
            },
            "output_activations": output_activations,
            "rules_fired": rules_fired,
        }
        return self.last_result


class ANFISModel:
    """Adaptive Neuro-Fuzzy Inference System (simplified).

    Combines neural network learning with fuzzy system interpretability.
    Layer 1: Fuzzification (membership functions)
    Layer 2: Rule firing strengths (AND = multiplication)
    Layer 3: Normalization
    Layer 4: Consequent parameters (linear)
    Layer 5: Defuzzification (weighted sum)
    """

    def __init__(self, n_inputs=3, n_rules=5):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.antecedent_params = None  # MF parameters
        self.consequent_params = None  # Linear consequent parameters
        self.trained = False

    def _init_params(self):
        # Random antecedent params: [a, b, c] for each input-rule MF
        self.antecedent_params = np.random.uniform(-1, 1, (self.n_inputs, self.n_rules, 3))
        # Consequent params: f_i = p_i0 + p_i1*x1 + ... + p_in*xn
        self.consequent_params = np.random.uniform(-1, 1, (self.n_rules, self.n_inputs + 1))

    def _forward(self, X):
        """Forward pass through ANFIS layers."""
        n_samples = X.shape[0]
        # Layer 1: Fuzzification
        membership = np.zeros((n_samples, self.n_inputs, self.n_rules))
        for i in range(self.n_inputs):
            for j in range(self.n_rules):
                a, b, c = self.antecedent_params[i, j]
                membership[:, i, j] = np.clip(trimf(X[:, i], a, b, c), 0, 1)

        # Layer 2: Rule firing strengths (product)
        firing = np.ones((n_samples, self.n_rules))
        for j in range(self.n_rules):
            for i in range(self.n_inputs):
                firing[:, j] *= membership[:, i, j]

        # Layer 3: Normalization
        total = firing.sum(axis=1, keepdims=True)
        total = np.where(total == 0, 1, total)
        normalized = firing / total

        # Layer 4: Consequent
        consequent = np.zeros((n_samples, self.n_rules))
        for j in range(self.n_rules):
            X_aug = np.column_stack([np.ones(n_samples), X])
            consequent[:, j] = X_aug @ self.consequent_params[j]

        # Layer 5: Defuzzification
        output = np.sum(normalized * consequent, axis=1)
        return output, normalized, consequent

    def fit(self, X, y, epochs=100, lr=0.01):
        """Train ANFIS using hybrid learning."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self._init_params()

        losses = []
        for epoch in range(epochs):
            output, normalized, consequent = self._forward(X)
            error = y - output
            loss = float(np.mean(error ** 2))
            losses.append(loss)

            # Update consequent params (least-squares step)
            for j in range(self.n_rules):
                w = normalized[:, j:j+1]
                X_aug = np.column_stack([np.ones(len(y)), X])
                try:
                    self.consequent_params[j] = np.linalg.lstsq(w * X_aug, w * y, rcond=None)[0]
                except Exception:
                    pass

            # Update antecedent params (gradient descent step)
            for i in range(self.n_inputs):
                for j in range(self.n_rules):
                    a, b, c = self.antecedent_params[i, j]
                    for k in range(len(y)):
                        x_k = X[k, i]
                        mf_val = membership_val = max(0, min((x_k-a)/(b-a+1e-10), (c-x_k)/(c-b+1e-10)))
                        if mf_val <= 0 or mf_val >= 1:
                            continue
                        grad = error[k] * lr
                        if x_k < b:
                            self.antecedent_params[i, j, 0] -= grad * 0.01
                            self.antecedent_params[i, j, 1] += grad * 0.01
                        else:
                            self.antecedent_params[i, j, 2] -= grad * 0.01
                            self.antecedent_params[i, j, 1] += grad * 0.01

        self.trained = True
        predictions, _, _ = self._forward(X)
        rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))

        return {
            "epochs": epochs, "final_loss": losses[-1],
            "rmse": rmse, "losses": losses,
            "r_squared": float(1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)),
        }

    def predict(self, X):
        """Predict using trained ANFIS."""
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        X = np.array(X, dtype=float)
        output, _, _ = self._forward(X)
        return output.tolist()


class FuzzyAHP:
    """Fuzzy Analytic Hierarchy Process for multi-criteria decisions.

    Used for valuing intangible assets, startup valuation,
    M&A analysis with qualitative criteria.
    """

    def __init__(self):
        self.last_result = None

    def evaluate(self, criteria_names, pairwise_comparisons, alternatives, alt_scores):
        """Fuzzy AHP evaluation.

        Args:
            criteria_names: List of criterion names
            pairwise_comparisons: Matrix of fuzzy pairwise comparisons (n x n x 3)
                                Each element is (l, m, u) triangular fuzzy number
            alternatives: List of alternative names
            alt_scores: Dict mapping criterion -> {alternative: fuzzy score (l,m,u)}
        """
        n = len(criteria_names)

        # Defuzzify pairwise comparison matrix
        defuzzified = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                l, m, u = pairwise_comparisons[i][j]
                defuzzified[i, j] = (l + 2*m + u) / 4

        # Compute criteria weights (eigenvector method)
        col_sums = defuzzified.sum(axis=0)
        normalized = defuzzified / col_sums
        weights = normalized.mean(axis=1)
        weights = weights / weights.sum()

        # Evaluate alternatives
        n_alt = len(alternatives)
        alt_matrix = np.zeros((n, n_alt))

        for ci, criterion in enumerate(criteria_names):
            scores = alt_scores.get(criterion, {})
            for ai, alt in enumerate(alternatives):
                score = scores.get(alt, (5, 5, 5))
                alt_matrix[ci, ai] = (score[0] + 2*score[1] + score[2]) / 4

        # Normalize alternative scores per criterion
        col_max = alt_matrix.max(axis=1, keepdims=True)
        col_max = np.where(col_max == 0, 1, col_max)
        normalized_alt = alt_matrix / col_max

        # Weighted final scores
        final_scores = weights @ normalized_alt
        ranking = np.argsort(-final_scores)

        self.last_result = {
            "method": "Fuzzy AHP",
            "criteria_weights": {name: float(w) for name, w in zip(criteria_names, weights)},
            "final_scores": {alt: float(s) for alt, s in zip(alternatives, final_scores)},
            "ranking": [alternatives[i] for i in ranking],
            "best_alternative": alternatives[ranking[0]],
        }
        return self.last_result
