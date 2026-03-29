"""
Standard Model Predictive Path Integral (MPPI) controller.

Reference: Williams et al., "Model Predictive Path Integral Control:
From Theory to Parallel Computation", JGCD 2017.
"""
import numpy as np
from typing import Callable, Optional, Tuple


class MPPI:
    """
    Standard MPPI controller for arbitrary dynamics and cost functions.

    Usage:
        dynamics = lambda x, u: x + u * dt
        cost = lambda x, u: np.sum((x - goal)**2)

        controller = MPPI(dynamics, cost, state_dim=4, control_dim=2,
                          horizon=30, n_samples=1000, noise_sigma=0.5)
        u = controller.command(x)
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
    ):
        self.dynamics = dynamics_fn
        self.cost = cost_fn
        self.terminal_cost = terminal_cost_fn
        self.n_x = state_dim
        self.n_u = control_dim
        self.H = horizon
        self.K = n_samples
        self.lambda_ = lambda_
        self.sigma = noise_sigma

        self.u_min = np.asarray(u_min) if u_min is not None else None
        self.u_max = np.asarray(u_max) if u_max is not None else None

        # Nominal control sequence (warm-started across calls)
        self.U = np.zeros((self.H, self.n_u))

        # Diagnostics from last call (available to subclasses)
        self._last_weights = None
        self._last_costs = None
        self._last_noise = None
        self._last_trajectories = None

    def command(self, x: np.ndarray) -> np.ndarray:
        """Compute optimal control for current state x. Returns u (n_u,)."""
        x = np.asarray(x, dtype=np.float64)

        # Sample noise: (K, H, n_u)
        noise = np.random.randn(self.K, self.H, self.n_u) * self.sigma

        # Rollout all samples
        costs, trajectories = self._rollout(x, noise)

        # Compute weights via softmax
        weights = self._compute_weights(costs)

        # Store diagnostics
        self._last_weights = weights
        self._last_costs = costs
        self._last_noise = noise
        self._last_trajectories = trajectories

        # Weighted average of noise perturbations
        # U_new = U_nominal + sum(w_k * eps_k)
        weighted_noise = np.einsum("k,khu->hu", weights, noise)
        self.U += weighted_noise

        # Clamp controls
        if self.u_min is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # Return first control
        u_out = self.U[0].copy()

        # Shift nominal sequence (warm start)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0

        return u_out

    def _rollout(
        self, x0: np.ndarray, noise: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rollout K trajectories from x0.
        Returns: costs (K,), trajectories (K, H+1, n_x)
        """
        K, H = self.K, self.H
        trajectories = np.zeros((K, H + 1, self.n_x))
        costs = np.zeros(K)

        trajectories[:, 0] = x0

        for t in range(H):
            u_t = self.U[t] + noise[:, t]  # (K, n_u)
            if self.u_min is not None:
                u_t = np.clip(u_t, self.u_min, self.u_max)

            x_curr = trajectories[:, t]  # (K, n_x)

            # Vectorized dynamics: try batch call first, fall back to loop
            x_next = self._batch_dynamics(x_curr, u_t)
            trajectories[:, t + 1] = x_next

            # Accumulate running cost
            costs += self._batch_cost(x_curr, u_t)

        # Terminal cost
        if self.terminal_cost is not None:
            costs += self._batch_terminal_cost(trajectories[:, -1])

        return costs, trajectories

    def _batch_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Apply dynamics to batch (K, n_x), (K, n_u) -> (K, n_x)."""
        try:
            return self.dynamics(x, u)
        except (ValueError, TypeError):
            # Fallback: per-sample
            return np.array([self.dynamics(x[k], u[k]) for k in range(len(x))])

    def _batch_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute cost for batch (K, n_x), (K, n_u) -> (K,)."""
        try:
            result = self.cost(x, u)
            if np.ndim(result) == 0:
                raise ValueError
            return result
        except (ValueError, TypeError):
            return np.array([self.cost(x[k], u[k]) for k in range(len(x))])

    def _batch_terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Terminal cost for batch (K, n_x) -> (K,)."""
        try:
            result = self.terminal_cost(x)
            if np.ndim(result) == 0:
                raise ValueError
            return result
        except (ValueError, TypeError):
            return np.array([self.terminal_cost(x[k]) for k in range(len(x))])

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """Softmax weights from trajectory costs. Returns (K,)."""
        shifted = costs - np.min(costs)
        w = np.exp(-shifted / self.lambda_)
        return w / (np.sum(w) + 1e-10)

    def reset(self):
        """Reset nominal control sequence."""
        self.U = np.zeros((self.H, self.n_u))

    # --- Sampling statistics (used by MA-MPPI subclass) ---

    @property
    def weight_entropy(self) -> float:
        """Shannon entropy of weight distribution. High = flat landscape."""
        if self._last_weights is None:
            return 0.0
        w = self._last_weights
        w = w[w > 1e-10]
        return -np.sum(w * np.log(w))

    @property
    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_k^2). Low = peaked weights = sharp minimum."""
        if self._last_weights is None:
            return 0.0
        return 1.0 / (np.sum(self._last_weights ** 2) + 1e-10)

    @property
    def cost_statistics(self) -> dict:
        """Mean, std, min, max of trajectory costs."""
        if self._last_costs is None:
            return {}
        c = self._last_costs
        return {
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
            "min": float(np.min(c)),
            "max": float(np.max(c)),
        }

    @property
    def gradient_direction(self) -> Optional[np.ndarray]:
        """
        Estimated gradient direction from weighted perturbations.
        The weighted mean of noise at t=0 approximates -∇V direction.
        """
        if self._last_weights is None or self._last_noise is None:
            return None
        return np.einsum("k,ku->u", self._last_weights, self._last_noise[:, 0])
