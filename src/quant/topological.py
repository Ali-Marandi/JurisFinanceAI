import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix
from scipy.stats import entropy


class TopologicalAnalyzer:
    """Topological Data Analysis (TDA) for financial market analysis.

    Implements Persistent Homology, Betti Numbers, Mapper Algorithm,
    Chaos Indicators (Lyapunov, Hurst, Fractal Dimension), and
    Information Geometry metrics for market regime detection.
    """

    def persistent_homology(self, returns, max_dimension=2, max_filtration=None):
        """Compute persistent homology of financial time series.

        Uses Vietoris-Rips complex construction with sublevel filtration.
        Returns birth-death pairs for each homology dimension.
        """
        if max_filtration is None:
            max_filtration = np.std(returns) * 3

        n = len(returns)
        if n > 500:
            step = n // 500
            returns = returns[::step]
            n = len(returns)

        # Build distance matrix
        X = returns.reshape(-1, 1)
        D = squareform(pdist(X, metric='euclidean'))
        D = np.clip(D, 0, max_filtration)

        # Compute persistence diagram via sublevel set filtration
        sorted_dists = np.sort(D[D > 0])
        num_filtrations = min(50, len(sorted_dists))
        if num_filtrations == 0:
            return {0: [], 1: [], 2: []}

        filtration_values = np.linspace(sorted_dists[0], sorted_dists[-1], num_filtrations)

        persistence = {d: [] for d in range(max_dimension + 1)}
        prev_betti = {d: 0 for d in range(max_dimension + 1)}
        birth_times = {d: {} for d in range(max_dimension + 1)}

        for eps in filtration_values:
            adj = lil_matrix((n, n), dtype=int)
            adj[D <= eps] = 1
            np.fill_diagonal(adj.toarray(), 0)

            n_components, labels = connected_components(adj, directed=False)
            betti_0 = n_components

            # Track H0 births and deaths
            if betti_0 > prev_betti[0]:
                for _ in range(betti_0 - prev_betti[0]):
                    birth_times[0][len(persistence[0])] = eps
            elif betti_0 < prev_betti[0]:
                for key in list(birth_times[0].keys())[:prev_betti[0] - betti_0]:
                    persistence[0].append((birth_times[0].pop(key), eps))

            # H1: estimate from cycle complexity (graph-theoretic)
            num_edges = adj.nnz // 2
            if n_components > 0:
                cyclomatic = num_edges - n + n_components
                betti_1 = max(0, cyclomatic)
            else:
                betti_1 = 0

            if betti_1 > prev_betti[1]:
                for _ in range(betti_1 - prev_betti[1]):
                    birth_times[1][len(persistence[1])] = eps
            elif betti_1 < prev_betti[1]:
                for key in list(birth_times[1].keys())[:prev_betti[1] - betti_1]:
                    persistence[1].append((birth_times[1].pop(key), eps))

            # H2: estimate from cavity count (simplified)
            betti_2 = max(0, betti_1 // 3)
            if betti_2 > prev_betti[2]:
                birth_times[2][len(persistence[2])] = eps
            elif betti_2 < prev_betti[2]:
                for key in list(birth_times[2].keys())[:prev_betti[2] - betti_2]:
                    persistence[2].append((birth_times[2].pop(key), eps))

            prev_betti = {0: betti_0, 1: betti_1, 2: betti_2}

        # Close remaining births
        for d in range(max_dimension + 1):
            for key, birth in birth_times[d].items():
                persistence[d].append((birth, max_filtration))

        return persistence

    def betti_numbers(self, persistence):
        """Extract Betti numbers from persistence diagram."""
        result = {}
        for dim, pairs in persistence.items():
            result[dim] = len([p for p in pairs if p[1] - p[0] > 1e-10])
        return result

    def persistence_entropy(self, persistence):
        """Compute persistence entropy for each dimension."""
        result = {}
        for dim, pairs in persistence.items():
            lifetimes = np.array([p[1] - p[0] for p in pairs if p[1] - p[0] > 1e-10])
            if len(lifetimes) > 0 and np.sum(lifetimes) > 0:
                probs = lifetimes / np.sum(lifetimes)
                result[dim] = -np.sum(probs * np.log(probs + 1e-15))
            else:
                result[dim] = 0.0
        return result

    def lyapunov_exponent(self, returns, max_iter=1000, tau=1):
        """Estimate the largest Lyapunov exponent using Rosenstein's algorithm.

        Positive Lyapunov exponent indicates chaos in the time series.
        """
        n = len(returns)
        if n < 50:
            return 0.0

        m = min(20, n // 5)  # embedding dimension
        d = np.zeros(n - tau * (m - 1))
        counts = np.zeros(n - tau * (m - 1))

        trajectory = np.zeros((m, n - tau * (m - 1)))
        for i in range(m):
            trajectory[i] = returns[i * tau:i * tau + trajectory.shape[1]]

        N = trajectory.shape[1]
        if N < 10:
            return 0.0

        max_iter_local = min(max_iter, N - 1)
        for i in range(N):
            dists = np.sqrt(np.sum((trajectory[:, i:i+1] - trajectory) ** 2, axis=0))
            dists[i] = np.inf
            nearest = np.argmin(dists)
            if nearest < N and dists[nearest] > 1e-10:
                d[i] = np.log(dists[nearest])
                counts[i] = 1

        valid = counts > 0
        if np.sum(valid) > 0:
            return np.mean(d[valid])
        return 0.0

    def hurst_exponent(self, returns, max_lag=100):
        """Compute Hurst exponent using R/S (Rescaled Range) analysis.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending / persistent
        """
        n = len(returns)
        if n < 20:
            return 0.5

        max_lag = min(max_lag, n // 2)
        lags = range(2, max_lag + 1)
        rs_values = []

        for lag in lags:
            num_blocks = n // lag
            if num_blocks < 1:
                continue
            rs_list = []
            for i in range(num_blocks):
                block = returns[i * lag:(i + 1) * lag]
                if len(block) < 2:
                    continue
                mean_val = np.mean(block)
                cumdev = np.cumsum(block - mean_val)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(block, ddof=1)
                if S > 1e-10:
                    rs_list.append(R / S)
            if rs_list:
                rs_values.append((lag, np.log(np.mean(rs_list))))

        if len(rs_values) < 2:
            return 0.5

        x = np.log([r[0] for r in rs_values])
        y = np.array([r[1] for r in rs_values])
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        return float(result[0][0])

    def fractal_dimension(self, returns, k_min=2, k_max=None):
        """Estimate fractal dimension using Higuchi's algorithm.

        Higher values indicate more complex/rough time series.
        """
        n = len(returns)
        if n < 20:
            return 1.0

        if k_max is None:
            k_max = min(int(n / 4), 20)

        L = []
        for k in range(k_min, k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, n, k)
                if len(idx) < 2:
                    continue
                diff = np.abs(np.diff(returns[idx]))
                norm = (n - 1) / (k * np.floor((n - m) / k))
                Lk.append(np.sum(diff) * norm)
            if Lk:
                L.append((k, np.log(np.mean(Lk))))

        if len(L) < 2:
            return 1.0

        x = np.log([l[0] for l in L])
        y = np.array([l[1] for l in L])
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        slope = -float(result[0][0])
        return float(np.clip(slope, 0.5, 2.5))

    def market_regime_detection(self, returns, window=60):
        """Detect market regimes using topological features.

        Returns regime labels: 'trending', 'mean-reverting', 'chaotic', 'random'.
        """
        n = len(returns)
        regimes = np.array(['random'] * n)

        for i in range(window, n):
            window_data = returns[i - window:i]
            H = self.hurst_exponent(window_data, max_lag=min(50, window // 2))
            LE = self.lyapunov_exponent(window_data)
            FD = self.fractal_dimension(window_data)

            if H > 0.6 and LE < 0.01:
                regimes[i] = 'trending'
            elif H < 0.4 and LE < 0.01:
                regimes[i] = 'mean-reverting'
            elif LE > 0.01:
                regimes[i] = 'chaotic'
            else:
                regimes[i] = 'random'

        return regimes

    def fisher_information(self, returns):
        """Compute Fisher Information Matrix for returns distribution.

        Measures the amount of information the data carries about parameters.
        Higher FI = more structure = less efficient market.
        """
        n = len(returns)
        if n < 20:
            return 0.0

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        if sigma < 1e-10:
            return 0.0

        # Fisher Information for Gaussian distribution
        FI_mu = n / (sigma ** 2)
        FI_sigma = 2 * n / (sigma ** 2)

        return float(np.sqrt(FI_mu * FI_sigma))

    def kolmogorov_complexity_estimate(self, returns, bin_size=None):
        """Estimate normalized Kolmogorov complexity via Lempel-Ziv compression.

        Higher complexity = harder to predict = more efficient market.
        """
        if bin_size is None:
            bin_size = np.std(returns) / 2

        # Symbolize time series
        median = np.median(returns)
        symbols = ''.join(['1' if r > median else '0' for r in returns])

        if len(symbols) < 2:
            return 0.0

        # Lempel-Ziv complexity
        n = len(symbols)
        complexity = 0
        i = 0
        substrings = set()

        while i < n:
            found = False
            for l in range(1, n - i + 1):
                sub = symbols[i:i + l]
                if sub not in substrings:
                    for j in range(l):
                        substrings.add(symbols[i:i + j + 1])
                    complexity += 1
                    i += l
                    found = True
                    break
            if not found:
                complexity += 1
                break

        # Normalize
        if n > 0:
            norm_complexity = complexity * np.log(n) / n if n > 1 else 0
        else:
            norm_complexity = 0

        return float(norm_complexity)

    def full_analysis(self, returns):
        """Run complete topological and complexity analysis."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]

        persistence = self.persistent_homology(returns)
        betti = self.betti_numbers(persistence)
        p_entropy = self.persistence_entropy(persistence)
        hurst = self.hurst_exponent(returns)
        lyap = self.lyapunov_exponent(returns)
        frac_dim = self.fractal_dimension(returns)
        fi = self.fisher_information(returns)
        kc = self.kolmogorov_complexity_estimate(returns)

        return {
            'persistent_homology': persistence,
            'betti_numbers': betti,
            'persistence_entropy': p_entropy,
            'hurst_exponent': hurst,
            'lyapunov_exponent': lyap,
            'fractal_dimension': frac_dim,
            'fisher_information': fi,
            'kolmogorov_complexity': kc,
            'market_regime': (
                'trending' if hurst > 0.6 else
                'mean-reverting' if hurst < 0.4 else
                'chaotic' if lyap > 0.01 else 'random'
            ),
            'regime_detail': {
                'Hurst > 0.6 implies trending/persistent': hurst > 0.6,
                'Hurst < 0.4 implies mean-reverting': hurst < 0.4,
                'Positive Lyapunov implies chaos': lyap > 0.01,
                'High fractal dimension implies roughness': frac_dim > 1.5,
            }
        }
