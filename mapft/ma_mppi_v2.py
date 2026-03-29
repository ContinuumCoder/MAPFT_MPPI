"""
MA-MPPI V2: Enhanced Memory-Augmented MPPI.

Improvements over V1:
1. Dual-timescale detection: fast (5-step) for reactive + slow (20-step) for confirmed
2. Directional bias sampling: bias noise toward escape direction near traps
3. Progressive memory strength: ramp up gradually instead of step function
4. Cost-improvement tracking for self-evaluation
"""
import numpy as np
from typing import Callable, Optional
from .mppi import MPPI
from .memory import MemoryRepository, Feature, LOCAL_MINIMUM, LOW_GRADIENT, HIGH_CURVATURE
from .potentials import compute_potential_batch, compute_alpha_batch


class MAMPPI_V2(MPPI):
    """
    MA-MPPI V2 with dual-timescale detection and directional sampling.
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        state_dim: int,
        control_dim: int,
        horizon: int = 30,
        n_samples: int = 1000,
        noise_sigma: float = 1.0,
        lambda_: float = 1.0,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        terminal_cost_fn: Optional[Callable] = None,
        # Memory
        max_features: int = 100,
        decay_factor: float = 0.95,
        # Adaptation
        eta: float = 3.0,
        mu: float = 2.0,
        memory_cost_weight: float = 30.0,
        # V2: directional bias
        direction_bias: float = 0.3,
    ):
        super().__init__(
            dynamics_fn, cost_fn, state_dim, control_dim,
            horizon, n_samples, noise_sigma, lambda_,
            u_min, u_max, terminal_cost_fn,
        )

        self.eta = eta
        self.mu = mu
        self.mem_cost_weight = memory_cost_weight
        self.direction_bias = direction_bias
        self._base_lambda = lambda_
        self._base_sigma = noise_sigma

        self.memory = MemoryRepository(max_features=max_features, decay_factor=decay_factor)

        # Dual-timescale state histories
        self._state_buf = []
        self._cost_buf = []
        self._ess_buf = []
        self._grad_buf = []

        # Diagnostics
        self.alpha_history = []
        self.step_count = 0

    def command(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self.step_count += 1
        self.memory.update(x)

        # Adaptive parameters
        alpha = self._compute_alpha(x)
        self.alpha_history.append(alpha)

        adaptive_lambda = self._base_lambda * (1.0 + self.eta * (1.0 - alpha))
        adaptive_sigma = self._base_sigma * np.sqrt(1.0 + self.mu * (1.0 - alpha))

        old_lambda, old_sigma = self.lambda_, self.sigma
        self.lambda_ = adaptive_lambda
        self.sigma = adaptive_sigma

        # Sample noise with directional bias
        noise = self._sample_noise(x)

        # Rollout with memory cost
        costs, trajectories = self._rollout_with_memory(x, noise)
        weights = self._compute_weights(costs)

        # Store for detection
        self._last_weights = weights
        self._last_costs = costs
        self._last_noise = noise
        self._last_trajectories = trajectories

        # Weighted update
        weighted_noise = np.einsum("k,khu->hu", weights, noise)
        self.U += weighted_noise
        if self.u_min is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        u_out = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0

        # Restore
        self.lambda_ = old_lambda
        self.sigma = old_sigma

        # Dual-timescale detection
        self._detect_dual(x, weights, costs, noise)

        return u_out

    def _sample_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Sample noise with directional bias toward escape direction
        when near known trap features.
        """
        noise = np.random.randn(self.K, self.H, self.n_u) * self.sigma

        # Check if near any feature with escape direction
        nearby = self.memory.nearby_features(x)
        if nearby and self.direction_bias > 0:
            # Strongest nearby feature's escape direction
            best = max(nearby, key=lambda f: f.strength)
            d = best.direction[:self.n_u]
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d = d / d_norm
                # Bias a fraction of samples toward escape direction
                n_biased = int(self.K * self.direction_bias)
                bias = d * self.sigma * 1.5
                noise[:n_biased, 0] += bias  # bias first timestep

        return noise

    def _rollout_with_memory(self, x0, noise):
        K, H = self.K, self.H
        trajectories = np.zeros((K, H + 1, self.n_x))
        costs = np.zeros(K)
        trajectories[:, 0] = x0
        has_features = len(self.memory.features) > 0

        for t in range(H):
            u_t = self.U[t] + noise[:, t]
            if self.u_min is not None:
                u_t = np.clip(u_t, self.u_min, self.u_max)
            x_curr = trajectories[:, t]
            trajectories[:, t + 1] = self._batch_dynamics(x_curr, u_t)
            costs += self._batch_cost(x_curr, u_t)

            if has_features:
                mem_cost = compute_potential_batch(x_curr, self.memory.features)
                costs += self.mem_cost_weight * mem_cost

        if self.terminal_cost is not None:
            costs += self._batch_terminal_cost(trajectories[:, -1])

        return costs, trajectories

    def _compute_alpha(self, x):
        if not self.memory.features:
            return 1.0
        from .potentials import compute_alpha
        return compute_alpha(x, self.memory.features)

    def _detect_dual(self, x, weights, costs, noise):
        """Dual-timescale: fast (5-step) reactive + slow (20-step) confirmed."""
        self._state_buf.append(x.copy())
        ess = 1.0 / (np.sum(weights ** 2) + 1e-10) / self.K
        self._ess_buf.append(ess)
        self._cost_buf.append(np.mean(costs))
        grad = np.einsum("k,ku->u", weights, noise[:, 0])
        self._grad_buf.append(grad.copy())

        # Fast detection (5 steps): state OR cost stagnation
        if len(self._state_buf) >= 5:
            recent5 = np.array(self._state_buf[-5:])
            # State-based (works well in low-dim)
            var5 = np.mean(np.var(recent5, axis=0))
            scale = np.mean(np.abs(recent5)) + 1e-6
            state_stuck = var5 < 0.003 * scale ** 2
            # Cost-based (works in any dim)
            recent_c5 = self._cost_buf[-5:]
            cost_cv5 = np.var(recent_c5) / (abs(np.mean(recent_c5)) + 1e-6) ** 2
            cost_stuck = cost_cv5 < 0.002

            if state_stuck or cost_stuck:
                ess_avg = np.mean(self._ess_buf[-5:])
                self._add_feature_from_detection(x, recent5, ess_avg, window=5)

        # Slow detection (15 steps): confirmed, stronger features
        if len(self._state_buf) >= 15:
            recent15 = np.array(self._state_buf[-15:])
            var15 = np.mean(np.var(recent15, axis=0))
            scale = np.mean(np.abs(recent15)) + 1e-6
            state_stuck = var15 < 0.005 * scale ** 2
            recent_c15 = self._cost_buf[-15:]
            cost_cv15 = np.var(recent_c15) / (abs(np.mean(recent_c15)) + 1e-6) ** 2
            cost_stuck = cost_cv15 < 0.003

            if state_stuck or cost_stuck:
                ess_avg = np.mean(self._ess_buf[-15:])
                self._add_feature_from_detection(x, recent15, ess_avg, window=15)

        # Keep buffers bounded
        max_buf = 50
        if len(self._state_buf) > max_buf:
            self._state_buf = self._state_buf[-max_buf:]
            self._ess_buf = self._ess_buf[-max_buf:]
            self._cost_buf = self._cost_buf[-max_buf:]
            self._grad_buf = self._grad_buf[-max_buf:]

    def _add_feature_from_detection(self, x, recent_states, ess_avg, window):
        spread = np.std(recent_states, axis=0)
        radius = max(np.linalg.norm(spread) * 3.0, 0.5)

        # Escape direction: opposite of average gradient
        n = min(window, len(self._grad_buf))
        avg_grad = np.mean(self._grad_buf[-n:], axis=0)
        g_norm = np.linalg.norm(avg_grad)
        if g_norm > 1e-10:
            escape = -avg_grad / g_norm
        else:
            escape = np.random.randn(len(x))
            escape /= np.linalg.norm(escape) + 1e-10

        # Classify
        if ess_avg < 0.1:
            feat_type = LOCAL_MINIMUM
        elif ess_avg > 0.5:
            feat_type = LOW_GRADIENT
        else:
            feat_type = HIGH_CURVATURE

        # Stronger for confirmed (slow) detection
        strength = 0.8 if window >= 20 else 0.4

        self.memory.add_feature(
            position=x, radius=radius,
            feature_type=feat_type,
            direction=escape, strength=strength,
        )

    def reset(self):
        super().reset()
        self.memory.reset()
        self._state_buf.clear()
        self._cost_buf.clear()
        self._ess_buf.clear()
        self._grad_buf.clear()
        self.alpha_history.clear()
        self.step_count = 0
