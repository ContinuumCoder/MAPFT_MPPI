"""
Memory-Augmented Model Predictive Path Integral (MA-MPPI) controller.

Drop-in replacement for MPPI that automatically detects topological features
of the value function landscape from its own sampling statistics — no extra
observations, gradients, or cost function calls required.

GPU-accelerated via PyTorch when available.
"""
import numpy as np
from typing import Callable, Optional
from .mppi import MPPI
from .memory import (
    MemoryRepository, SamplingFeatureDetector,
    LOCAL_MINIMUM, LOW_GRADIENT, HIGH_CURVATURE,
)
from .potentials import compute_potential_batch, compute_alpha_batch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MAMPPI(MPPI):
    """
    Memory-Augmented MPPI controller.

    Extends standard MPPI with:
    1. Automatic landscape feature detection from sampling statistics
    2. Persistent memory of problematic regions
    3. Adaptive potential field that reshapes the cost landscape
    4. Temperature adaptation for enhanced exploration near traps

    Same interface as MPPI — just swap the class name.

    Usage:
        controller = MAMPPI(dynamics_fn, cost_fn,
                            state_dim=4, control_dim=2,
                            horizon=30, n_samples=1000)
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
        # Memory parameters
        max_features: int = 100,
        decay_factor: float = 0.95,
        # Adaptation parameters
        eta: float = 3.0,
        mu: float = 2.0,
        memory_cost_weight: float = 30.0,
        # GPU acceleration
        device: str = "auto",
    ):
        super().__init__(
            dynamics_fn, cost_fn, state_dim, control_dim,
            horizon, n_samples, noise_sigma, lambda_,
            u_min, u_max, terminal_cost_fn,
        )

        self.eta = eta              # temperature enhancement coefficient
        self.mu = mu                # covariance inflation coefficient
        self.mem_cost_weight = memory_cost_weight

        # Memory system
        self.memory = MemoryRepository(
            max_features=max_features,
            decay_factor=decay_factor,
        )
        self.detector = SamplingFeatureDetector()

        # Adaptive state
        self._base_lambda = lambda_
        self._base_sigma = noise_sigma

        # GPU setup: only use GPU path when CUDA is available
        self._device = self._resolve_device(device)
        self._use_gpu = (self._device is not None and HAS_TORCH
                         and self._device.type == "cuda")

        # Diagnostics
        self.alpha_history = []
        self.temperature_history = []
        self.feature_count_history = []

    def _resolve_device(self, device: str):
        if not HAS_TORCH:
            return None
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def command(self, x: np.ndarray) -> np.ndarray:
        """Compute optimal control with memory augmentation."""
        x = np.asarray(x, dtype=np.float64)

        # Update memory proximity
        self.memory.update(x)

        # Adaptive temperature and covariance
        alpha = self._compute_alpha(x)
        adaptive_lambda = self._base_lambda * (1.0 + self.eta * (1.0 - alpha))
        adaptive_sigma = self._base_sigma * np.sqrt(1.0 + self.mu * (1.0 - alpha))

        # Store for diagnostics
        self.alpha_history.append(alpha)
        self.temperature_history.append(adaptive_lambda)
        self.feature_count_history.append(len(self.memory.features))

        # Temporarily adjust parameters
        old_lambda, old_sigma = self.lambda_, self.sigma
        self.lambda_ = adaptive_lambda
        self.sigma = adaptive_sigma

        if self._use_gpu:
            u_out = self._command_gpu(x)
        else:
            u_out = self._command_cpu(x)

        # Restore parameters
        self.lambda_ = old_lambda
        self.sigma = old_sigma

        # Detect features from last sampling statistics
        self._detect_features(x)

        return u_out

    def _command_cpu(self, x: np.ndarray) -> np.ndarray:
        """CPU path: standard MPPI + memory cost injection."""
        noise = np.random.randn(self.K, self.H, self.n_u) * self.sigma

        # Rollout with memory-augmented cost
        costs, trajectories = self._rollout_with_memory(x, noise)

        weights = self._compute_weights(costs)

        # Store diagnostics for feature detection
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
        return u_out

    def _command_gpu(self, x: np.ndarray) -> np.ndarray:
        """GPU-accelerated path using PyTorch."""
        dev = self._device

        x_t = torch.from_numpy(x).float().to(dev)
        U_t = torch.from_numpy(self.U).float().to(dev)
        noise_t = torch.randn(self.K, self.H, self.n_u, device=dev) * self.sigma

        # Rollout on GPU
        costs_t = torch.zeros(self.K, device=dev)
        trajectories = torch.zeros(self.K, self.H + 1, self.n_x, device=dev)
        trajectories[:, 0] = x_t

        u_min_t = torch.from_numpy(self.u_min).float().to(dev) if self.u_min is not None else None
        u_max_t = torch.from_numpy(self.u_max).float().to(dev) if self.u_max is not None else None

        for t in range(self.H):
            u_t = U_t[t] + noise_t[:, t]
            if u_min_t is not None:
                u_t = torch.clamp(u_t, u_min_t, u_max_t)

            x_curr = trajectories[:, t]

            # Dynamics (numpy fallback if not tensorized)
            x_np = x_curr.cpu().numpy()
            u_np = u_t.cpu().numpy()
            x_next = self._batch_dynamics(x_np, u_np)
            trajectories[:, t + 1] = torch.from_numpy(x_next).float().to(dev)

            # Running cost
            c = self._batch_cost(x_np, u_np)
            costs_t += torch.from_numpy(c).float().to(dev)

        # Memory cost on GPU
        if self.memory.features:
            for t in range(self.H):
                x_np = trajectories[:, t].cpu().numpy()
                mem_cost = compute_potential_batch(x_np, self.memory.features)
                costs_t += torch.from_numpy(
                    mem_cost * self.mem_cost_weight
                ).float().to(dev)

        # Terminal cost
        if self.terminal_cost is not None:
            x_T = trajectories[:, -1].cpu().numpy()
            tc = self._batch_terminal_cost(x_T)
            costs_t += torch.from_numpy(tc).float().to(dev)

        # Weights on GPU
        shifted = costs_t - costs_t.min()
        w_t = torch.exp(-shifted / self.lambda_)
        w_t = w_t / (w_t.sum() + 1e-10)

        # Weighted update on GPU
        weighted = torch.einsum("k,khu->hu", w_t, noise_t)
        U_t += weighted

        if u_min_t is not None:
            U_t = torch.clamp(U_t, u_min_t, u_max_t)

        # Back to numpy
        self.U = U_t.cpu().numpy()
        costs_np = costs_t.cpu().numpy()
        weights_np = w_t.cpu().numpy()
        noise_np = noise_t.cpu().numpy()
        traj_np = trajectories.cpu().numpy()

        self._last_weights = weights_np
        self._last_costs = costs_np
        self._last_noise = noise_np
        self._last_trajectories = traj_np

        u_out = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0
        return u_out

    def _rollout_with_memory(self, x0, noise):
        """Rollout with memory-augmented cost (CPU path)."""
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
            x_next = self._batch_dynamics(x_curr, u_t)
            trajectories[:, t + 1] = x_next

            # Running cost
            costs += self._batch_cost(x_curr, u_t)

            # Memory potential cost (repel from known traps)
            if has_features:
                mem_cost = compute_potential_batch(x_curr, self.memory.features)
                costs += self.mem_cost_weight * mem_cost

        if self.terminal_cost is not None:
            costs += self._batch_terminal_cost(trajectories[:, -1])

        return costs, trajectories

    def _compute_alpha(self, x: np.ndarray) -> float:
        """Adaptive weight α(x, M). 1=base dominates, 0=memory dominates."""
        if not self.memory.features:
            return 1.0
        from .potentials import compute_alpha
        return compute_alpha(x, self.memory.features)

    def _detect_features(self, x: np.ndarray):
        """Detect landscape features from last MPPI sampling."""
        if self._last_weights is None:
            return

        result = self.detector.observe(
            state=x,
            weights=self._last_weights,
            costs=self._last_costs,
            noise=self._last_noise,
            n_samples=self.K,
        )

        if result is not None:
            feat_type, radius, direction = result
            self.memory.add_feature(
                position=x,
                radius=max(radius, 0.5),
                feature_type=feat_type,
                direction=direction,
            )

    def reset(self):
        """Reset controller, memory, and detector."""
        super().reset()
        self.memory.reset()
        self.detector.reset()
        self.alpha_history.clear()
        self.temperature_history.clear()
        self.feature_count_history.clear()

    @property
    def diagnostics(self) -> dict:
        """Current state of the memory-augmented controller."""
        return {
            "n_features": len(self.memory.features),
            "alpha": self.alpha_history[-1] if self.alpha_history else 1.0,
            "temperature": self.temperature_history[-1] if self.temperature_history else self._base_lambda,
            "ess": self.effective_sample_size,
            "weight_entropy": self.weight_entropy,
            "features": [
                {
                    "position": f.position.tolist(),
                    "radius": f.radius,
                    "strength": f.strength,
                    "type": f.feature_type,
                    "encounters": f.encounter_count,
                }
                for f in self.memory.features
            ],
        }
