import math
from dataclasses import dataclass

import numpy as np

from mnist_mlp import reshape_heatmap, top_k_indices


@dataclass
class DMMMENConfig:
    n_components: int = 8
    n_regimes: int = 3
    burn_in: int = 150
    n_samples: int = 200
    thin: int = 2
    alpha_shape: float = 2.0
    alpha_rate: float = 1.0
    sigma_shape: float = 2.0
    sigma_rate: float = 1.0
    lambda_shape: float = 2.0
    lambda_rate: float = 1.0
    mh_step_alpha: float = 0.15
    mh_step_lambda: float = 0.12
    ridge_jitter: float = 1e-4
    eps: float = 1e-6
    random_state: int = 7
    feature_shape: tuple[int, int] = (28, 28)

    @property
    def total_iterations(self):
        return self.burn_in + self.n_samples * self.thin


def _logsumexp(values, axis=None):
    values = np.asarray(values, dtype=float)
    max_value = np.max(values, axis=axis, keepdims=True)
    stabilized = np.exp(values - max_value)
    reduced = np.log(np.sum(stabilized, axis=axis, keepdims=True)) + max_value
    if axis is None:
        return reduced.reshape(())
    return np.squeeze(reduced, axis=axis)


def _normal_logpdf(y, mean, variance, eps):
    variance = np.maximum(variance, eps)
    return -0.5 * (np.log(2.0 * np.pi * variance) + ((y - mean) ** 2) / variance)


def _stick_breaking(u):
    weights = np.empty(u.size + 1, dtype=float)
    remaining = 1.0
    for idx, value in enumerate(u):
        weights[idx] = remaining * value
        remaining *= 1.0 - value
    weights[-1] = remaining
    return weights / np.sum(weights)


def _sample_categorical(log_probabilities, rng):
    normalized = log_probabilities - _logsumexp(log_probabilities, axis=1)[:, None]
    probabilities = np.exp(normalized)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(size=probabilities.shape[0])[:, None]
    return np.argmax(draws <= cumulative, axis=1)


class DMMMEN:
    """Truncated DP regression mixture with multiple elastic-net regimes."""

    def __init__(self, config=None):
        self.config = config or DMMMENConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        self.is_fitted_ = False

    def _initialize_state(self, x, y):
        n_samples, n_features = x.shape
        variance = float(np.var(y) + self.config.eps)
        beta = self.rng.normal(0.0, 0.05, size=(self.config.n_components, n_features))
        sigma2 = np.full(self.config.n_components, variance, dtype=float)
        z = self.rng.integers(0, self.config.n_components, size=n_samples)
        c = self.rng.integers(0, self.config.n_regimes, size=self.config.n_components)
        tau = np.ones((self.config.n_components, n_features), dtype=float)
        lambda1 = np.linspace(0.8, 1.4, self.config.n_regimes)
        lambda2 = np.linspace(0.3, 0.9, self.config.n_regimes)
        u = self.rng.beta(1.0, 1.0, size=self.config.n_components - 1)
        return {
            "beta": beta,
            "sigma2": sigma2,
            "z": z,
            "c": c,
            "tau": tau,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "w": np.full(self.config.n_regimes, 1.0 / self.config.n_regimes, dtype=float),
            "u": u,
            "pi": _stick_breaking(u),
            "alpha": 1.0,
        }

    def _sample_assignments(self, x, y, state):
        component_means = x @ state["beta"].T
        log_probabilities = np.log(np.maximum(state["pi"], self.config.eps))[None, :] + _normal_logpdf(
            y[:, None],
            component_means,
            state["sigma2"][None, :],
            self.config.eps,
        )
        state["z"] = _sample_categorical(log_probabilities, self.rng)

    def _sample_stick_breaking(self, state):
        counts = np.bincount(state["z"], minlength=self.config.n_components)
        u = np.empty(self.config.n_components - 1, dtype=float)
        for idx in range(self.config.n_components - 1):
            remaining_count = int(np.sum(counts[(idx + 1) :]))
            u[idx] = self.rng.beta(1.0 + counts[idx], state["alpha"] + remaining_count)
        state["u"] = u
        state["pi"] = _stick_breaking(u)

    def _log_alpha_posterior(self, alpha, u):
        if alpha <= 0.0:
            return -np.inf
        prior = (self.config.alpha_shape - 1.0) * math.log(alpha) - self.config.alpha_rate * alpha
        log_likelihood = u.size * math.log(alpha) + (alpha - 1.0) * float(
            np.sum(np.log(np.maximum(1.0 - u, self.config.eps)))
        )
        return prior + log_likelihood

    def _sample_alpha(self, state, stats):
        current = float(state["alpha"])
        proposal = float(np.exp(np.log(current) + self.rng.normal(0.0, self.config.mh_step_alpha)))
        log_current = self._log_alpha_posterior(current, state["u"])
        log_proposal = self._log_alpha_posterior(proposal, state["u"])
        log_acceptance = log_proposal - log_current + np.log(proposal) - np.log(current)
        if np.log(self.rng.random()) < log_acceptance:
            state["alpha"] = proposal
            stats["alpha_accepts"] += 1

    def _sample_regime_weights(self, state):
        counts = np.bincount(state["c"], minlength=self.config.n_regimes)
        concentration = np.full(self.config.n_regimes, 1.0 / self.config.n_regimes) + counts
        state["w"] = self.rng.dirichlet(concentration)

    def _log_lambda_posterior(self, parameter, assigned_beta, mode):
        if parameter <= 0.0:
            return -np.inf
        prior = (self.config.lambda_shape - 1.0) * math.log(parameter) - self.config.lambda_rate * parameter
        if assigned_beta.size == 0:
            return prior
        if mode == "lasso":
            penalty = -parameter * float(np.sum(np.abs(assigned_beta)))
            normalizer = assigned_beta.shape[1] * assigned_beta.shape[0] * math.log(parameter)
            return prior + penalty + normalizer
        penalty = -0.5 * parameter * float(np.sum(assigned_beta ** 2))
        normalizer = 0.5 * assigned_beta.shape[1] * assigned_beta.shape[0] * math.log(parameter)
        return prior + penalty + normalizer

    def _sample_lambdas(self, state, stats):
        for regime in range(self.config.n_regimes):
            assigned = state["beta"][state["c"] == regime]
            current_l1 = float(state["lambda1"][regime])
            proposal_l1 = float(np.exp(np.log(current_l1) + self.rng.normal(0.0, self.config.mh_step_lambda)))
            log_current = self._log_lambda_posterior(current_l1, assigned, "lasso")
            log_proposal = self._log_lambda_posterior(proposal_l1, assigned, "lasso")
            log_acceptance = log_proposal - log_current + np.log(proposal_l1) - np.log(current_l1)
            if np.log(self.rng.random()) < log_acceptance:
                state["lambda1"][regime] = proposal_l1
                stats["lambda1_accepts"] += 1

            current_l2 = float(state["lambda2"][regime])
            proposal_l2 = float(np.exp(np.log(current_l2) + self.rng.normal(0.0, self.config.mh_step_lambda)))
            log_current = self._log_lambda_posterior(current_l2, assigned, "ridge")
            log_proposal = self._log_lambda_posterior(proposal_l2, assigned, "ridge")
            log_acceptance = log_proposal - log_current + np.log(proposal_l2) - np.log(current_l2)
            if np.log(self.rng.random()) < log_acceptance:
                state["lambda2"][regime] = proposal_l2
                stats["lambda2_accepts"] += 1

    def _sample_regimes(self, state):
        scores = np.empty(self.config.n_regimes, dtype=float)
        for component in range(self.config.n_components):
            beta = state["beta"][component]
            for regime in range(self.config.n_regimes):
                scores[regime] = (
                    np.log(np.maximum(state["w"][regime], self.config.eps))
                    - state["lambda1"][regime] * np.sum(np.abs(beta))
                    - 0.5 * state["lambda2"][regime] * np.sum(beta ** 2)
                )
            normalized = scores - _logsumexp(scores)
            state["c"][component] = _sample_categorical(normalized[None, :], self.rng)[0]

    def _sample_component_parameters(self, x, y, state):
        n_features = x.shape[1]
        identity_cache = {}
        for component in range(self.config.n_components):
            regime = int(state["c"][component])
            lambda1 = float(state["lambda1"][regime])
            lambda2 = float(state["lambda2"][regime])
            beta_current = state["beta"][component]
            sigma2_current = float(state["sigma2"][component])

            mean_param = lambda1 * np.sqrt(max(sigma2_current, self.config.eps)) / (
                np.abs(beta_current) + self.config.eps
            )
            inv_tau = self.rng.wald(mean=np.maximum(mean_param, self.config.eps), scale=lambda1 ** 2 + self.config.eps)
            state["tau"][component] = 1.0 / np.maximum(inv_tau, self.config.eps)

            member_indices = np.flatnonzero(state["z"] == component)
            x_component = x[member_indices]
            y_component = y[member_indices]
            diag_precision = 1.0 / state["tau"][component] + lambda2 + self.config.ridge_jitter
            diag_variance = 1.0 / np.maximum(diag_precision, self.config.eps)

            if x_component.size == 0:
                state["beta"][component] = self.rng.normal(0.0, np.sqrt(diag_variance))
                state["sigma2"][component] = 1.0 / self.rng.gamma(self.config.sigma_shape, 1.0 / self.config.sigma_rate)
                continue

            x_weighted = x_component * diag_variance[None, :]
            n_component = x_component.shape[0]
            if n_component not in identity_cache:
                identity_cache[n_component] = np.eye(n_component)
            covariance_dual = (
                sigma2_current * identity_cache[n_component]
                + x_weighted @ x_component.T
                + self.config.ridge_jitter * identity_cache[n_component]
            )
            try:
                rhs = np.linalg.solve(covariance_dual, y_component)
            except np.linalg.LinAlgError:
                rhs = np.linalg.pinv(covariance_dual) @ y_component
            posterior_mean = diag_variance * (x_component.T @ rhs)
            approx_precision = diag_precision + np.sum(x_component ** 2, axis=0) / max(sigma2_current, self.config.eps)
            approx_variance = 1.0 / np.maximum(approx_precision, self.config.eps)
            state["beta"][component] = posterior_mean + self.rng.normal(0.0, np.sqrt(approx_variance))

            residual = y_component - x_component @ state["beta"][component]
            penalty = np.sum((state["beta"][component] ** 2) * (1.0 / state["tau"][component] + lambda2))
            shape = self.config.sigma_shape + 0.5 * (n_component + n_features)
            rate = self.config.sigma_rate + 0.5 * (np.sum(residual ** 2) + penalty)
            state["sigma2"][component] = 1.0 / self.rng.gamma(shape, 1.0 / max(rate, self.config.eps))

    def _relabel_order(self, state):
        counts = np.bincount(state["z"], minlength=self.config.n_components)
        return np.lexsort((-state["pi"], -counts))

    def _compute_component_membership(self, x):
        mixture_mean = self.logit_offset_ + np.sum((x @ self.beta_mean_.T) * self.pi_mean_[None, :], axis=1)
        component_means = x @ self.beta_mean_.T + self.logit_offset_
        scores = np.log(np.maximum(self.pi_mean_, self.config.eps))[None, :] + _normal_logpdf(
            mixture_mean[:, None],
            component_means,
            self.sigma2_mean_[None, :],
            self.config.eps,
        )
        return np.argmax(scores, axis=1)

    def fit(self, x, logits):
        x = np.asarray(x, dtype=float)
        logits = np.asarray(logits, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if x.shape[0] != logits.shape[0]:
            raise ValueError("X and logits must have the same number of rows.")

        self.logit_offset_ = float(np.mean(logits))
        centered_logits = logits - self.logit_offset_
        state = self._initialize_state(x, centered_logits)

        beta_trace = []
        sigma2_trace = []
        pi_trace = []
        c_trace = []
        stats = {"alpha_accepts": 0, "lambda1_accepts": 0, "lambda2_accepts": 0}

        for iteration in range(self.config.total_iterations):
            self._sample_assignments(x, centered_logits, state)
            self._sample_stick_breaking(state)
            self._sample_alpha(state, stats)
            self._sample_regimes(state)
            self._sample_regime_weights(state)
            self._sample_lambdas(state, stats)
            self._sample_component_parameters(x, centered_logits, state)

            if iteration >= self.config.burn_in and (iteration - self.config.burn_in) % self.config.thin == 0:
                order = self._relabel_order(state)
                beta_trace.append(state["beta"][order].copy())
                sigma2_trace.append(state["sigma2"][order].copy())
                pi_trace.append(state["pi"][order].copy())
                c_trace.append(state["c"][order].copy())

        if not beta_trace:
            raise RuntimeError("No posterior samples were collected. Increase n_samples or reduce burn_in.")

        self.beta_mean_ = np.mean(np.asarray(beta_trace), axis=0)
        self.sigma2_mean_ = np.mean(np.asarray(sigma2_trace), axis=0)
        self.pi_mean_ = np.mean(np.asarray(pi_trace), axis=0)
        self.pi_mean_ = self.pi_mean_ / np.sum(self.pi_mean_)
        c_trace = np.asarray(c_trace, dtype=int)
        self.regime_mode_ = np.apply_along_axis(lambda col: np.bincount(col).argmax(), 0, c_trace)

        memberships = self._compute_component_membership(x)
        self.component_centroids_ = np.zeros((self.config.n_components, x.shape[1]), dtype=float)
        self.component_scales_ = np.ones(self.config.n_components, dtype=float)
        for component in range(self.config.n_components):
            component_x = x[memberships == component]
            if component_x.size == 0:
                continue
            centroid = component_x.mean(axis=0)
            self.component_centroids_[component] = centroid
            distances = np.linalg.norm(component_x - centroid[None, :], axis=1)
            self.component_scales_[component] = float(np.mean(distances ** 2) + self.config.eps)

        self.is_fitted_ = True
        return self

    def predict_logit(self, x):
        if not self.is_fitted_:
            raise RuntimeError("DMMMEN must be fitted before prediction.")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.logit_offset_ + np.sum((x @ self.beta_mean_.T) * self.pi_mean_[None, :], axis=1)

    def _select_component(self, x):
        if not self.is_fitted_:
            raise RuntimeError("DMMMEN must be fitted before explanation.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        distances = np.linalg.norm(self.component_centroids_ - x, axis=1) ** 2
        scores = np.log(np.maximum(self.pi_mean_, self.config.eps)) - distances / (2.0 * self.component_scales_)
        return int(np.argmax(scores))

    def explain_instance(self, x, top_k=20):
        component = self._select_component(x)
        coefficients = self.beta_mean_[component]
        contributions = np.asarray(x, dtype=float) * coefficients
        top_indices = top_k_indices(contributions, top_k)
        return {
            "dominant_component": component,
            "coefficients": coefficients.copy(),
            "contributions": contributions,
            "top_indices": top_indices,
            "top_weights": coefficients[top_indices],
            "top_contributions": contributions[top_indices],
            "heatmap": reshape_heatmap(coefficients, self.config.feature_shape),
        }

    def global_patterns(self, top_k=150):
        if not self.is_fitted_:
            raise RuntimeError("DMMMEN must be fitted before computing global patterns.")
        patterns = []
        for component in range(self.config.n_components):
            coefficients = self.beta_mean_[component]
            top_indices = top_k_indices(coefficients, top_k)
            patterns.append(
                {
                    "component": component,
                    "mixture_weight": float(self.pi_mean_[component]),
                    "regime": int(self.regime_mode_[component]),
                    "top_indices": top_indices,
                    "top_weights": coefficients[top_indices],
                    "heatmap": reshape_heatmap(coefficients, self.config.feature_shape),
                }
            )
        return patterns


def fit_dmm_men(x, logits, config=None):
    model = DMMMEN(config=config)
    return model.fit(x, logits)
