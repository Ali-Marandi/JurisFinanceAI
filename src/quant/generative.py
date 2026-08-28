import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, svd


class GenerativeModel:
    """Generative AI models for financial scenario simulation.

    Implements Score-Based Diffusion, Wasserstein GAN-like synthetic data
    generation, Variational Autoencoder, and adversarial stress scenarios.
    """

    def diffusion_scenarios(self, historical_returns, n_scenarios=1000,
                            n_steps=100, noise_scale=0.1):
        """Generate financial scenarios using score-based diffusion model.

        Forward process: gradually add noise to data.
        Reverse process: learn to denoise and generate new samples.
        Uses Euler-Maruyama discretization of the reverse SDE.
        """
        returns = np.asarray(historical_returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        if sigma < 1e-10:
            sigma = 0.01

        # Forward process: compute noise schedule
        betas = np.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        # Reverse process (Euler-Maruyama)
        dt = 1.0 / n_steps
        scenarios = np.zeros((n_scenarios, len(returns)))

        for s in range(n_scenarios):
            # Start from pure noise
            x = np.random.randn(len(returns)) * sigma * np.sqrt(alpha_bars[-1])

            # Reverse diffusion
            for t in reversed(range(n_steps)):
                # Score function (approximation: gradient of log density)
                # Using Tweedie's formula: score ≈ (mu - x_t * alpha_bar) / (sigma^2 * alpha_bar)
                score = (mu - x * alpha_bars[t]) / (sigma ** 2 * alpha_bars[t] + 1e-10)
                score = np.clip(score, -10, 10)

                # Conditional mean for the reverse step
                f_t = -0.5 * betas[t] * x
                g_t = np.sqrt(betas[t])

                x = x + (f_t - 0.5 * g_t ** 2 * score) * dt + g_t * np.sqrt(dt) * np.random.randn(len(x))

            scenarios[s] = x

        return scenarios

    def wgan_synthetic_data(self, real_data, n_samples=1000, n_iterations=200,
                            latent_dim=16, hidden_dim=32):
        """Generate synthetic financial data using a Wasserstein GAN-like approach.

        Uses gradient penalty and spectral normalization concepts adapted
        for numpy-only implementation with moment matching.
        """
        data = np.asarray(real_data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < 20:
            return np.random.randn(n_samples) * np.std(data) + np.mean(data)

        # Extract statistical features (real distribution moments)
        real_mean = np.mean(data)
        real_std = np.std(data, ddof=1)
        real_skew = self._skewness(data)
        real_kurt = self._kurtosis(data)
        real_acf1 = self._acf(data, 1)
        real_acf2 = self._acf(data, 2)

        # Generator: learn a mapping from latent space to data space
        # Using iterative moment matching (simplified GAN training)
        W = np.random.randn(hidden_dim, latent_dim) * 0.02
        b = np.zeros(hidden_dim)
        W2 = np.random.randn(1, hidden_dim) * 0.02
        b2 = np.zeros(1)

        lr = 0.01
        for iteration in range(n_iterations):
            z = np.random.randn(n, latent_dim)

            # Forward pass (generator)
            h = np.maximum(0, z @ W.T + b)  # ReLU
            fake = h @ W2.T + b2
            fake = fake.flatten()

            # Compute moment losses (Wasserstein-like)
            fake_mean = np.mean(fake)
            fake_std = np.std(fake, ddof=1) + 1e-10
            fake_skew = self._skewness(fake)
            fake_kurt = self._kurtosis(fake)
            fake_acf1 = self._acf(fake, 1)
            fake_acf2 = self._acf(fake, 2)

            # Gradient of moment matching loss
            loss = (
                (fake_mean - real_mean) ** 2 +
                (fake_std - real_std) ** 2 +
                0.5 * (fake_skew - real_skew) ** 2 +
                0.1 * (fake_kurt - real_kurt) ** 2 +
                0.5 * (fake_acf1 - real_acf1) ** 2 +
                0.3 * (fake_acf2 - real_acf2) ** 2
            )

            # Backpropagation (manual)
            d_fake = np.sign(fake - real_mean) * 0.01
            d_h = (d_fake @ W2) * (h > 0)
            d_W2 = d_fake.reshape(-1, 1).T @ h
            d_b2 = np.sum(d_fake)
            d_W = d_h.T @ z
            d_b = np.sum(d_h, axis=0)

            W2 -= lr * np.clip(d_W2, -1, 1) / n
            b2 -= lr * np.clip(d_b2, -1, 1) / n
            W -= lr * np.clip(d_W, -1, 1) / n
            b -= lr * np.clip(d_b, -1, 1) / n

        # Generate final synthetic samples
        z_final = np.random.randn(n_samples, latent_dim)
        h_final = np.maximum(0, z_final @ W.T + b)
        synthetic = (h_final @ W2.T + b2).flatten()

        # Post-hoc correction to match target distribution
        synthetic = (synthetic - np.mean(synthetic)) / (np.std(synthetic) + 1e-10)
        synthetic = synthetic * real_std + real_mean

        return synthetic

    def variational_autoencoder(self, data, latent_dim=4, n_epochs=100,
                                 encoding_dim=8):
        """Compress and reconstruct financial data using VAE-like architecture.

        Encoder maps data to latent distribution, decoder reconstructs.
        Uses ELBO-like objective with reparameterization trick.
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < 20:
            return data.copy(), data.copy(), {'loss': [0]}

        # Normalize
        mu_data = np.mean(data)
        std_data = np.std(data, ddof=1) + 1e-10
        normalized = (data - mu_data) / std_data

        # Initialize weights
        W_enc = np.random.randn(encoding_dim, 1) * 0.1
        b_enc = np.zeros(encoding_dim)
        W_mu = np.random.randn(latent_dim, encoding_dim) * 0.1
        b_mu = np.zeros(latent_dim)
        W_logvar = np.random.randn(latent_dim, encoding_dim) * 0.1
        b_logvar = np.zeros(latent_dim)
        W_dec = np.random.randn(encoding_dim, latent_dim) * 0.1
        b_dec = np.zeros(encoding_dim)
        W_out = np.random.randn(1, encoding_dim) * 0.1
        b_out = np.zeros(1)

        lr = 0.005
        losses = []

        for epoch in range(n_epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            # Mini-batch
            idx = np.random.permutation(n)
            batch_size = min(32, n)

            for start in range(0, n, batch_size):
                batch = normalized[idx[start:start + batch_size]].reshape(-1, 1)
                bs = len(batch)

                # Encoder
                h = np.tanh(batch @ W_enc.T + b_enc)
                z_mu = h @ W_mu.T + b_mu
                z_logvar = h @ W_logvar.T + b_logvar
                z_logvar = np.clip(z_logvar, -5, 5)

                # Reparameterization
                eps = np.random.randn(bs, latent_dim)
                z = z_mu + np.exp(0.5 * z_logvar) * eps

                # Decoder
                h_dec = np.tanh(z @ W_dec.T + b_dec)
                x_recon = h_dec @ W_out.T + b_out

                # Losses
                recon_loss = np.mean((x_recon - batch) ** 2)
                kl_loss = -0.5 * np.mean(1 + z_logvar - z_mu ** 2 - np.exp(z_logvar))
                loss = recon_loss + 0.01 * kl_loss

                # Backprop
                d_recon = 2 * (x_recon - batch) / bs
                d_W_out = d_recon.T @ h_dec
                d_b_out = np.sum(d_recon)
                d_h_dec = d_recon @ W_out * (1 - h_dec ** 2)
                d_W_dec = d_h_dec.T @ z
                d_b_dec = np.sum(d_h_dec, axis=0)
                d_z = d_h_dec @ W_dec

                d_mu = d_z * np.exp(0.5 * z_logvar)
                d_logvar = d_z * eps * 0.5 * np.exp(0.5 * z_logvar)
                d_logvar += -0.005 * 0.5 * (1 - np.exp(z_logvar)) / bs
                d_mu += -0.005 * z_mu / bs

                d_W_mu = d_mu.T @ h
                d_b_mu = np.sum(d_mu, axis=0)
                d_W_logvar = d_logvar.T @ h
                d_b_logvar = np.sum(d_logvar, axis=0)
                d_h = d_mu @ W_mu + d_logvar @ W_logvar
                d_h *= (1 - h ** 2)

                d_W_enc = d_h.T @ batch
                d_b_enc = np.sum(d_h, axis=0)

                # Update
                for param, grad in [(W_enc, d_W_enc), (W_mu, d_W_mu),
                                    (W_logvar, d_W_logvar), (W_dec, d_W_dec),
                                    (W_out, d_W_out)]:
                    param -= lr * np.clip(grad, -5, 5)
                for param, grad in [(b_enc, d_b_enc), (b_mu, d_b_mu),
                                    (b_logvar, d_b_logvar), (b_dec, d_b_dec),
                                    (b_out, d_b_out)]:
                    param -= lr * np.clip(grad, -5, 5)

                total_loss += loss
                total_recon += recon_loss
                total_kl += kl_loss

            losses.append(total_loss / (n // batch_size + 1))

        # Encode full dataset
        h_all = np.tanh(normalized.reshape(-1, 1) @ W_enc.T + b_enc)
        z_all_mu = h_all @ W_mu.T + b_mu
        latent = z_all_mu

        # Reconstruct
        h_dec_all = np.tanh(latent @ W_dec.T + b_dec)
        reconstructed = (h_dec_all @ W_out.T + b_out).flatten() * std_data + mu_data

        return latent, reconstructed, {'loss': losses}

    def stress_scenarios(self, returns, n_scenarios=50, severity=None):
        """Generate adversarial stress scenarios using perturbation analysis.

        Creates extreme but plausible market scenarios by combining
        historical shocks with worst-case perturbations.
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        n = len(returns)

        if severity is None:
            severity = 'extreme'

        severity_multipliers = {
            'mild': 1.5,
            'moderate': 2.5,
            'severe': 4.0,
            'extreme': 6.0,
        }
        mult = severity_multipliers.get(severity, 2.5)

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = self._skewness(returns)
        kurt = self._kurtosis(returns)

        scenarios = []
        scenario_names = []

        # 1. Historical worst replay
        if n > 20:
            rolling_max = np.maximum.accumulate(returns)
            drawdowns = (returns - rolling_max) / (np.abs(rolling_max) + 1e-10)
            worst_period = returns[np.argmin(drawdowns):np.argmin(drawdowns) + min(20, n - np.argmin(drawdowns))]
            if len(worst_period) > 0:
                scenarios.append(worst_period * mult)
                scenario_names.append(f'Historical Worst x{mult}')

        # 2. Fat-tail shock
        if kurt > 3:
            shock = np.random.standard_t(df=max(3, 10 - kurt / 2), size=n) * sigma * mult
        else:
            shock = np.random.randn(n) * sigma * mult
        scenarios.append(mu + shock)
        scenario_names.append('Fat-Tail Shock')

        # 3. Mean-reversion crisis
        mr_crisis = -np.abs(returns - mu) * mult + mu
        scenarios.append(mr_crisis)
        scenario_names.append('Mean-Reversion Crisis')

        # 4. Volatility clustering spike
        garch_shock = np.zeros(n)
        var = sigma ** 2
        for t in range(n):
            var = 0.1 * sigma ** 2 + 0.85 * var + 0.05 * returns[max(0, t-1)] ** 2
            if t > n * 0.8:
                var *= mult  # Spike at end
            garch_shock[t] = np.random.randn() * np.sqrt(var)
        scenarios.append(garch_shock)
        scenario_names.append('GARCH Volatility Spike')

        # 5. Correlation breakdown
        decorrelated = returns * (1 + np.random.randn(n) * (mult - 1) * 0.3)
        scenarios.append(decorrelated)
        scenario_names.append('Correlation Breakdown')

        # 6. Liquidity freeze
        liquidity = np.zeros(n)
        for t in range(n):
            if t > n * 0.7:
                liquidity[t] = np.random.randn() * sigma * mult * 0.5 - sigma * mult * 0.3
            else:
                liquidity[t] = np.random.randn() * sigma * 0.5 + mu
        scenarios.append(liquidity)
        scenario_names.append('Liquidity Freeze')

        # 7. Structural break
        structural = returns.copy()
        break_point = n // 2
        structural[break_point:] = returns[break_point:] * (-mult) + mu * 2
        scenarios.append(structural)
        scenario_names.append('Structural Break')

        # 8. Black swan
        black_swan = returns.copy()
        event_start = np.random.randint(n // 4, 3 * n // 4)
        event_length = min(10, n - event_start)
        black_swan[event_start:event_start + event_length] = (
            np.random.randn(event_length) * sigma * mult * 1.5 - sigma * mult
        )
        scenarios.append(black_swan)
        scenario_names.append('Black Swan Event')

        return scenarios, scenario_names

    def _skewness(self, x):
        n = len(x)
        if n < 3:
            return 0.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    def _kurtosis(self, x):
        n = len(x)
        if n < 4:
            return 3.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10:
            return 3.0
        return float(np.mean(((x - m) / s) ** 4))

    def _acf(self, x, lag):
        n = len(x)
        if lag >= n or n < 2:
            return 0.0
        m = np.mean(x)
        c0 = np.sum((x - m) ** 2) / (n - 1)
        if c0 < 1e-10:
            return 0.0
        cl = np.sum((x[:n - lag] - m) * (x[lag:] - m)) / (n - 1)
        return float(cl / c0)
