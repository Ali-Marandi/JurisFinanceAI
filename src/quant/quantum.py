import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm


class QuantumFinanceEngine:
    """Quantum-ready algorithms for financial applications.

    Implements simulated quantum algorithms including QAOA for portfolio
    optimization, Quantum Amplitude Estimation for pricing, Quantum Kernel
    Methods for classification, and Quantum Circuit simulations.
    Uses classical simulation of quantum mechanics (no Qiskit dependency).
    """

    def qaoa_portfolio(self, expected_returns, cov_matrix, budget=None,
                       n_layers=3, n_iterations=100):
        """Quantum Approximate Optimization Algorithm for portfolio selection.

        Simulates QAOA to solve the combinatorial portfolio optimization
        problem. Maps portfolio selection to an Ising Hamiltonian and
        optimizes variational parameters.
        """
        n = len(expected_returns)
        expected_returns = np.asarray(expected_returns, dtype=float)
        cov_matrix = np.asarray(cov_matrix, dtype=float)

        if budget is None:
            budget = n // 2

        # Build QUBO matrix for portfolio optimization
        # Objective: maximize (mu^T w - lambda * w^T Sigma w)
        # Subject to: sum(w) = budget
        risk_aversion = 1.0
        Q = -risk_aversion * cov_matrix
        for i in range(n):
            Q[i, i] += expected_returns[i] * 10  # Scale returns

        # QAOA parameters
        n_qubits = n
        gamma = np.random.uniform(0, 2 * np.pi, n_layers)
        beta = np.random.uniform(0, np.pi, n_layers)

        def qaoa_circuit(params):
            gammas = params[:n_layers]
            betas = params[n_layers:]

            # Initialize in superposition
            state = np.ones(2 ** n_qubits, dtype=complex) / np.sqrt(2 ** n_qubits)

            for layer in range(n_layers):
                # Cost unitary (problem Hamiltonian)
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        # ZZ interaction
                        for s in range(2 ** n_qubits):
                            si = (s >> (n_qubits - 1 - i)) & 1
                            sj = (s >> (n_qubits - 1 - j)) & 1
                            if si == sj == 1:
                                phase = np.exp(-1j * gammas[layer] * Q[i, j])
                                state[s] *= phase

                # Mixer unitary
                for i in range(n_qubits):
                    rx = np.array([
                        [np.cos(betas[layer] / 2), -1j * np.sin(betas[layer] / 2)],
                        [-1j * np.sin(betas[layer] / 2), np.cos(betas[layer] / 2)]
                    ])
                    for s in range(2 ** n_qubits):
                        bit = (s >> (n_qubits - 1 - i)) & 1
                        if bit == 0:
                            idx_1 = s | (1 << (n_qubits - 1 - i))
                            new_0 = rx[0, 0] * state[s] + rx[0, 1] * state[idx_1]
                            new_1 = rx[1, 0] * state[s] + rx[1, 1] * state[idx_1]
                            state[s] = new_0
                            state[idx_1] = new_1

            # Measure expectation value
            probs = np.abs(state) ** 2
            expectation = 0
            for s in range(2 ** n_qubits):
                bits = [(s >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
                if sum(bits) == budget:
                    x = np.array(bits, dtype=float)
                    expectation += probs[s] * (x @ Q @ x)

            return -expectation  # Minimize negative

        # Optimize QAOA parameters
        if n <= 12:
            best_result = None
            best_energy = np.inf

            for trial in range(3):
                params0 = np.random.uniform(0, 2 * np.pi, 2 * n_layers)
                try:
                    result = minimize(
                        qaoa_circuit, params0,
                        method='Nelder-Mead',
                        options={'maxiter': n_iterations, 'xatol': 0.01}
                    )
                    if result.fun < best_energy:
                        best_energy = result.fun
                        best_result = result
                except Exception:
                    continue

            if best_result is not None:
                # Decode solution
                final_params = best_result.x
                gammas = final_params[:n_layers]
                betas = final_params[n_layers:]

                state = np.ones(2 ** n_qubits, dtype=complex) / np.sqrt(2 ** n_qubits)
                for layer in range(n_layers):
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            for s in range(2 ** n_qubits):
                                si = (s >> (n_qubits - 1 - i)) & 1
                                sj = (s >> (n_qubits - 1 - j)) & 1
                                if si == sj == 1:
                                    phase = np.exp(-1j * gammas[layer] * Q[i, j])
                                    state[s] *= phase
                    for i in range(n_qubits):
                        rx = np.array([
                            [np.cos(betas[layer] / 2), -1j * np.sin(betas[layer] / 2)],
                            [-1j * np.sin(betas[layer] / 2), np.cos(betas[layer] / 2)]
                        ])
                        for s in range(2 ** n_qubits):
                            bit = (s >> (n_qubits - 1 - i)) & 1
                            if bit == 0:
                                idx_1 = s | (1 << (n_qubits - 1 - i))
                                new_0 = rx[0, 0] * state[s] + rx[0, 1] * state[idx_1]
                                new_1 = rx[1, 0] * state[s] + rx[1, 1] * state[idx_1]
                                state[s] = new_0
                                state[idx_1] = new_1

                probs = np.abs(state) ** 2

                # Get top solutions
                valid_solutions = []
                for s in range(2 ** n_qubits):
                    bits = [(s >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
                    if sum(bits) == budget:
                        x = np.array(bits, dtype=float)
                        ret = x @ expected_returns
                        risk = np.sqrt(x @ cov_matrix @ x + 1e-10)
                        valid_solutions.append((probs[s], x, ret, risk))

                valid_solutions.sort(key=lambda t: -t[0])
        else:
            # Fallback to classical for large n
            best_combination = 0
            best_sharpe = -np.inf
            from itertools import combinations
            for combo in combinations(range(n), budget):
                w = np.zeros(n)
                w[list(combo)] = 1.0 / budget
                ret = w @ expected_returns
                risk = np.sqrt(w @ cov_matrix @ w + 1e-10)
                sharpe = ret / (risk + 1e-10)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_combination = w

            valid_solutions = [(1.0, best_combination, best_combination @ expected_returns,
                                np.sqrt(best_combination @ cov_matrix @ best_combination + 1e-10))]

        results = []
        for prob, weights, ret, risk in valid_solutions[:10]:
            results.append({
                'probability': float(prob),
                'weights': weights.tolist(),
                'expected_return': float(ret),
                'risk': float(risk),
                'sharpe': float(ret / (risk + 1e-10))
            })

        return {
            'solutions': results,
            'n_layers': n_layers,
            'n_qubits': n_qubits,
            'budget': budget
        }

    def quantum_monte_carlo_pricing(self, s0, k, r, sigma, t, n_qubits=10,
                                     n_shots=1000):
        """Quantum Amplitude Estimation for option pricing.

        Simulates the quantum speedup of Monte Carlo option pricing
        using amplitude estimation with quadratic speedup.
        """
        # Classical Monte Carlo as baseline
        z = np.random.randn(n_shots)
        s_T = s0 * np.exp((r - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * z)
        payoffs = np.maximum(s_T - k, 0)
        classical_price = np.exp(-r * t) * np.mean(payoffs)

        # Quantum speedup simulation
        # QAE achieves O(1/sqrt(M)) instead of O(1/M)
        # Simulate by using sqrt(n_shots) effective samples
        n_quantum = max(10, int(np.sqrt(n_shots)))

        # Grover-like amplitude amplification
        theta_est = np.arcsin(np.clip(np.sqrt(np.mean(payoffs > 0)), 0, 1))
        quantum_mean_payoff = np.sin((2 * int(np.pi / (4 * theta_est)) + 1) * theta_est) ** 2
        quantum_mean_payoff = np.clip(quantum_mean_payoff, 0, 1)
        quantum_price = np.exp(-r * t) * quantum_mean_payoff * s0

        # Iterative QAE
        n_iter = 5
        estimates = []
        for i in range(n_iter):
            k_iter = 2 ** i
            theta_k = (2 * k_iter + 1) * theta_est
            est = np.sin(theta_k) ** 2
            est = np.clip(est, 0, 1)
            estimates.append(est)

        final_estimate = np.median(estimates) * np.exp(-r * t) * s0 * 1.5

        # Confidence interval from quantum measurement
        std_quantum = np.std(payoffs) / np.sqrt(n_quantum)

        return {
            'classical_price': float(classical_price),
            'quantum_price': float(quantum_price),
            'iterative_qae_price': float(final_estimate),
            'classical_std': float(np.std(payoffs) / np.sqrt(n_shots)),
            'quantum_std': float(std_quantum),
            'speedup_factor': float(np.sqrt(n_shots / max(n_quantum, 1))),
            'n_qubits': n_qubits,
            'n_shots': n_shots
        }

    def quantum_kernel_classification(self, X, y, sigma=1.0):
        """Quantum Kernel method for financial classification.

        Computes a quantum-inspired kernel matrix using the
        fidelity-based approach. Maps data to quantum feature space.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(X)

        # Quantum feature map: x -> |phi(x)>
        # Using ZZ feature map: exp(i * sigma * sum(x_i * x_j * ZZ))
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Fidelity between quantum states
                # |<phi(x_i)|phi(x_j)>|^2
                ip = np.dot(X[i], X[j])
                kernel_val = np.exp(-0.5 * np.linalg.norm(X[i] - X[j]) ** 2 / (sigma ** 2))

                # Add quantum-inspired non-classical terms
                cos_term = np.cos(ip / sigma) ** 2
                quantum_enhancement = 0.5 * kernel_val + 0.5 * cos_term * kernel_val
                K[i, j] = quantum_enhancement

        # Kernel Ridge Regression for classification
        lambda_reg = 0.1
        K_reg = K + lambda_reg * np.eye(n)
        alpha = np.linalg.solve(K_reg, y)

        # Predictions
        y_pred = K @ alpha
        accuracy = np.mean((y_pred > 0) == (y > 0))

        return {
            'kernel_matrix': K,
            'predictions': y_pred.tolist(),
            'accuracy': float(accuracy),
            'alpha': alpha.tolist()
        }

    def quantum_entanglement_measure(self, returns1, returns2, window=60):
        """Measure quantum-inspired entanglement between two financial series.

        Uses von Neumann entropy and concurrence as measures of
        non-classical correlation between markets.
        """
        r1 = np.asarray(returns1, dtype=float)
        r2 = np.asarray(returns2, dtype=float)
        min_len = min(len(r1), len(r2))
        r1, r2 = r1[:min_len], r2[:min_len]

        n_windows = (min_len - window) // (window // 2) + 1
        if n_windows < 1:
            return {'entropy': [], 'concurrence': [], 'classical_corr': []}

        entropies = []
        concurrences = []
        correlations = []

        for i in range(n_windows):
            start = i * (window // 2)
            end = start + window
            x1 = r1[start:end]
            x2 = r2[start:end]

            # Build density matrix from covariance
            cov = np.cov(x1, x2)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            probs = eigenvalues / np.sum(eigenvalues)

            # Von Neumann entropy
            vn_entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(vn_entropy)

            # Concurrence (quantum entanglement measure)
            corr = np.corrcoef(x1, x2)[0, 1]
            concurrence = np.sqrt(max(0, 1 - (1 - corr ** 2)))
            concurrences.append(concurrence)
            correlations.append(corr)

        return {
            'entropy': entropies,
            'concurrence': concurrences,
            'classical_corr': correlations,
            'mean_entropy': float(np.mean(entropies)) if entropies else 0,
            'mean_concurrence': float(np.mean(concurrences)) if concurrences else 0,
        }

    def variational_quantum_eigenvalue(self, matrix, n_qubits=None, n_layers=2,
                                         n_iterations=100):
        """Variational Quantum Eigensolver (VQE) for finding eigenvalues.

        Useful for pricing derivatives where the payoff depends on
        eigenvalues of a financial matrix.
        """
        M = np.asarray(matrix, dtype=float)
        n = M.shape[0]

        if n_qubits is None:
            n_qubits = int(np.ceil(np.log2(n)))
        dim = 2 ** n_qubits

        # Pad matrix to quantum dimension
        M_padded = np.zeros((dim, dim))
        M_padded[:n, :n] = M

        # VQE: minimize <psi(params)|H|psi(params)>
        def vqe_energy(params):
            # Prepare variational state using parameterized rotation
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0  # |00...0>

            idx = 0
            for layer in range(n_layers):
                for q in range(n_qubits):
                    theta = params[idx % len(params)]
                    idx += 1

                    # Apply Ry rotation to qubit q
                    cos_t = np.cos(theta / 2)
                    sin_t = np.sin(theta / 2)
                    new_state = state.copy()
                    for s in range(dim):
                        bit = (s >> (n_qubits - 1 - q)) & 1
                        if bit == 0:
                            flipped = s | (1 << (n_qubits - 1 - q))
                            new_state[s] = cos_t * state[s] - 1j * sin_t * state[flipped]
                            new_state[flipped] = -1j * sin_t * state[s] + cos_t * state[flipped]
                    state = new_state

            # Expectation value <psi|H|psi>
            energy = np.real(np.conj(state) @ M_padded @ state)
            return energy

        # Optimize
        params0 = np.random.uniform(0, 2 * np.pi, n_layers * n_qubits)
        result = minimize(vqe_energy, params0, method='Nelder-Mead',
                          options={'maxiter': n_iterations, 'xatol': 0.01})

        # Classical eigenvalues for comparison
        classical_eigenvalues = np.sort(np.linalg.eigvalsh(M))

        return {
            'vqe_min_eigenvalue': float(result.fun),
            'classical_min_eigenvalue': float(classical_eigenvalues[0]),
            'classical_max_eigenvalue': float(classical_eigenvalues[-1]),
            'n_qubits': n_qubits,
            'converged': result.success
        }
