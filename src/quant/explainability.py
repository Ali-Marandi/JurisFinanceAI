import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd


class ExplainabilityEngine:
    """Explainable AI (XAI) engine for financial model interpretation.

    Implements SHAP-like feature attribution, LIME-style local explanations,
    counterfactual analysis, feature importance ranking, and model drift detection.
    All methods use numpy/scipy only - no external XAI libraries needed.
    """

    def shap_values(self, X, model_fn, background=None, nsamples=500):
        """Compute SHAP (SHapley Additive exPlanations) values.

        Uses KernelSHAP-like sampling approach for model-agnostic
        feature importance estimation.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        if background is None:
            background = np.mean(X, axis=0)
        else:
            background = np.asarray(background, dtype=float)

        if n_features > 100:
            return np.zeros((n_samples, n_features)), None

        shap_output = np.zeros((n_samples, n_features))
        all_results = []

        for i in range(min(n_samples, 50)):
            x = X[i]
            phi = np.zeros(n_features)
            total_weight = np.zeros(n_features)

            for _ in range(nsamples):
                # Random coalition
                coalition = np.random.randint(0, 2, n_features).astype(float)
                x_coal = background * (1 - coalition) + x * coalition
                f_coal = model_fn(x_coal)

                for j in range(n_features):
                    x_no_j = x_coal.copy()
                    x_no_j[j] = background[j]
                    f_no_j = model_fn(x_no_j)

                    marginal = f_coal - f_no_j
                    weight = 1.0 / (np.sum(coalition) * (n_features - np.sum(coalition)) + 1e-10)
                    phi[j] += marginal * weight
                    total_weight[j] += weight

            phi = np.where(total_weight > 0, phi / total_weight, 0)
            shap_output[i] = phi

            # Expected value
            base_value = model_fn(background)
            all_results.append({
                'shap_values': phi,
                'base_value': base_value,
                'prediction': model_fn(x),
                'feature_names': [f'F{j+1}' for j in range(n_features)]
            })

        return shap_output, all_results

    def lime_explanation(self, x, model_fn, n_perturbations=1000, kernel_width=1.0):
        """LIME (Local Interpretable Model-agnostic Explanations).

        Fits a local linear model around the prediction point
        using weighted perturbations.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n_features = x.shape[1]

        # Generate perturbations
        X_pert = x + np.random.randn(n_perturbations, n_features) * kernel_width
        y_pert = np.array([model_fn(xi) for xi in X_pert])

        # Compute distances and weights
        distances = np.sqrt(np.sum((X_pert - x) ** 2, axis=1))
        weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

        # Weighted linear regression
        W = np.diag(weights)
        X_aug = np.column_stack([np.ones(n_perturbations), X_pert])
        try:
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y_pert
            coeffs = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_features + 1)

        intercept = coeffs[0]
        local_importance = coeffs[1:n_features + 1]

        # R-squared of local model
        y_pred = X_aug @ coeffs
        ss_res = np.sum(weights * (y_pert - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_pert - np.average(y_pert, weights=weights)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            'local_importance': local_importance,
            'intercept': intercept,
            'r_squared': float(r_squared),
            'prediction': float(model_fn(x.flatten())),
            'feature_names': [f'F{j+1}' for j in range(n_features)]
        }

    def counterfactual_analysis(self, x, model_fn, target=None, max_changes=3,
                                 n_attempts=1000, step_size=0.1):
        """Generate counterfactual explanations.

        Finds minimal changes to input that change the prediction
        to the target value.
        """
        x = np.asarray(x, dtype=float).copy()
        original_pred = model_fn(x)

        if target is None:
            target = -original_pred  # Flip the prediction

        n_features = len(x)
        best_cf = None
        best_dist = np.inf

        for _ in range(n_attempts):
            x_cf = x.copy()
            # Randomly select features to modify
            n_changes = np.random.randint(1, max_changes + 1)
            selected = np.random.choice(n_features, min(n_changes, n_features), replace=False)

            for _ in range(50):  # Gradient-free optimization steps
                for j in selected:
                    # Try perturbation
                    direction = 1 if np.random.rand() > 0.5 else -1
                    x_cf[j] += direction * step_size * np.std(x)

                    new_pred = model_fn(x_cf)
                    if abs(new_pred - target) < abs(best_dist):
                        best_cf = x_cf.copy()
                        best_dist = abs(new_pred - target)

                    if abs(new_pred - target) < 1e-6:
                        break

                if abs(model_fn(x_cf) - target) < 1e-6:
                    break

        if best_cf is None:
            return {
                'counterfactual': x,
                'changed_features': [],
                'change_magnitudes': [],
                'original_prediction': original_pred,
                'new_prediction': original_pred,
                'distance': 0.0
            }

        changes = best_cf - x
        changed_mask = np.abs(changes) > 1e-10

        return {
            'counterfactual': best_cf,
            'changed_features': np.where(changed_mask)[0].tolist(),
            'change_magnitudes': changes[changed_mask].tolist(),
            'original_prediction': float(original_pred),
            'new_prediction': float(model_fn(best_cf)),
            'distance': float(np.linalg.norm(changes))
        }

    def feature_importance(self, X, y, method='permutation', n_repeats=10):
        """Compute global feature importance using permutation importance.

        Shuffles each feature and measures the increase in error.
        Higher importance = feature is more critical for the model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape

        # Fit a simple linear model as proxy
        X_aug = np.column_stack([np.ones(n_samples), X])
        try:
            coeffs = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_features + 1)

        def model(x):
            return np.dot(x, coeffs[1:]) + coeffs[0]

        # Baseline error
        y_pred = np.array([model(xi) for xi in X])
        baseline_error = np.mean((y - y_pred) ** 2)

        if method == 'permutation':
            importances = np.zeros((n_features, n_repeats))
            for j in range(n_features):
                for r in range(n_repeats):
                    X_perm = X.copy()
                    np.random.shuffle(X_perm[:, j])
                    y_perm = np.array([model(xi) for xi in X_perm])
                    perm_error = np.mean((y - y_perm) ** 2)
                    importances[j, r] = perm_error - baseline_error

            mean_importance = np.mean(importances, axis=1)
            std_importance = np.std(importances, axis=1)
        else:
            # Coefficient-based importance
            mean_importance = np.abs(coeffs[1:n_features + 1])
            std_importance = np.zeros(n_features)

        # Rank features
        ranking = np.argsort(-mean_importance)

        return {
            'importance_mean': mean_importance,
            'importance_std': std_importance,
            'ranking': ranking.tolist(),
            'baseline_error': float(baseline_error)
        }

    def model_drift_detection(self, reference_data, current_data, threshold=2.0,
                                n_bins=30):
        """Detect concept drift between reference and current data distributions.

        Uses PSI (Population Stability Index), KS test approximation,
        and distribution comparison metrics.
        """
        ref = np.asarray(reference_data, dtype=float)
        ref = ref[~np.isnan(ref)]
        cur = np.asarray(current_data, dtype=float)
        cur = cur[~np.isnan(cur)]

        # PSI (Population Stability Index)
        all_data = np.concatenate([ref, cur])
        bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

        ref_hist, _ = np.histogram(ref, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur, bins=bins, density=True)

        ref_hist = np.clip(ref_hist, 1e-10, None)
        cur_hist = np.clip(cur_hist, 1e-10, None)

        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

        # KS statistic (two-sample)
        all_sorted = np.sort(np.concatenate([ref, cur]))
        ref_cdf = np.searchsorted(np.sort(ref), all_sorted) / len(ref)
        cur_cdf = np.searchsorted(np.sort(cur), all_sorted) / len(cur)
        ks_stat = np.max(np.abs(ref_cdf - cur_cdf))

        # Wasserstein distance (Earth Mover's Distance)
        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)
        min_len = min(len(ref_sorted), len(cur_sorted))
        wasserstein = np.mean(np.abs(ref_sorted[:min_len] - cur_sorted[:min_len]))

        # Moment comparison
        ref_mean, cur_mean = np.mean(ref), np.mean(cur)
        ref_std, cur_std = np.std(ref, ddof=1), np.std(cur, ddof=1)
        ref_skew = np.mean(((ref - ref_mean) / (ref_std + 1e-10)) ** 3)
        cur_skew = np.mean(((cur - cur_mean) / (cur_std + 1e-10)) ** 3)

        # Drift assessment
        drift_detected = psi > threshold
        drift_level = (
            'No Drift' if psi < threshold * 0.5 else
            'Minor Drift' if psi < threshold else
            'Moderate Drift' if psi < threshold * 1.5 else
            'Severe Drift'
        )

        return {
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'wasserstein_distance': float(wasserstein),
            'drift_detected': drift_detected,
            'drift_level': drift_level,
            'reference_stats': {'mean': float(ref_mean), 'std': float(ref_std), 'skew': float(ref_skew)},
            'current_stats': {'mean': float(cur_mean), 'std': float(cur_std), 'skew': float(cur_skew)},
        }

    def attention_weights(self, sequence, model_fn=None):
        """Compute attention-like weights for a financial time series.

        Identifies which time points have the most influence on
        future outcomes using self-attention mechanism.
        """
        seq = np.asarray(sequence, dtype=float)
        n = len(seq)
        if n < 3:
            return np.ones(n) / n

        # Compute self-similarity matrix
        X = seq.reshape(-1, 1)
        # Normalize
        X_norm = (X - np.mean(X)) / (np.std(X) + 1e-10)

        # Attention scores (scaled dot-product)
        scores = X_norm @ X_norm.T / np.sqrt(n)

        # Softmax
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

        # Aggregate: how much each position is attended to
        importance = np.mean(attention, axis=0)
        importance = importance / np.sum(importance)

        return importance
