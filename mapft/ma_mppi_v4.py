"""
MA-MPPI V3: Production-grade Memory-Augmented MPPI.

Key improvements over V2:
1. Progress-aware detection: only detect traps when cost stops improving
2. Aggressive memory aging: features have max lifetime, fast inter-episode decay
3. Integrated adaptive network: online-learned temperature/weight/exploration
4. Vectorized memory cost with spatial indexing
"""
import numpy as np
from typing import Callable, Optional
from .mppi import MPPI
from .memory import MemoryRepository, Feature, LOCAL_MINIMUM, LOW_GRADIENT, HIGH_CURVATURE
from .potentials import compute_potential_batch

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _ParamNet(torch.nn.Module):
    """Tiny MLP: 6 features → 3 params (temp_scale, mem_weight, sigma_scale)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 24), nn.ReLU(),
            nn.Linear(24, 3), nn.Softplus(),
        )
        with torch.no_grad():
            self.net[-2].bias.fill_(0.5)
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-3)
        self.buf = []

    def forward(self, x):
        return self.net(x)

    def predict_np(self, feats):
        with torch.no_grad():
            out = self.forward(torch.FloatTensor(feats).unsqueeze(0)).squeeze(0).numpy()
        return np.clip(out[0], 0.5, 4.0), np.clip(out[1], 5.0, 100.0), np.clip(out[2], 0.5, 3.0)

    def record_and_learn(self, feats, reward):
        self.buf.append((feats.copy(), reward))
        if len(self.buf) > 200:
            self.buf.pop(0)
        if len(self.buf) >= 32:
            idx = np.random.choice(len(self.buf), 32, replace=False)
            f = torch.FloatTensor(np.array([self.buf[i][0] for i in idx]))
            r = torch.FloatTensor([self.buf[i][1] for i in idx])
            r = (r - r.mean()) / (r.std() + 1e-8)
            params = self.forward(f)
            loss = -(torch.log(params + 1e-8).mean(dim=1) * r).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()


class MAMPPI_V4(MPPI):
    """
    MA-MPPI V3: progress-aware detection + adaptive network + aging memory.
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
        max_features: int = 50,
        feature_lifetime: int = 150,
        eta: float = 3.0,
        mu: float = 2.0,
        direction_bias: float = 0.25,
    ):
        super().__init__(
            dynamics_fn, cost_fn, state_dim, control_dim,
            horizon, n_samples, noise_sigma, lambda_,
            u_min, u_max, terminal_cost_fn,
        )
        self.eta = eta
        self.mu = mu
        self._base_lambda = lambda_
        self._base_sigma = noise_sigma
        self.direction_bias = direction_bias
        self.feature_lifetime = feature_lifetime

        # Memory with faster decay
        self.memory = MemoryRepository(
            max_features=max_features,
            decay_factor=0.97,
            decay_after=50,
            min_strength=0.15,
        )

        # Adaptive network
        self._param_net = _ParamNet() if HAS_TORCH else None
        self._mem_weight = 30.0
        self._temp_scale = 1.0
        self._sigma_scale = 1.0

        # Progress tracking
        self._state_buf = []
        self._cost_buf = []
        self._ess_buf = []
        self._grad_buf = []
        self._prev_best_cost = None
        self.step_count = 0

    def command(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self.step_count += 1
        self.memory.update(x)
        self._age_features()

        # Adaptive params from network
        alpha = self._compute_alpha(x)
        if self._param_net is not None and len(self._cost_buf) > 10:
            feats = self._build_net_features(alpha)
            self._temp_scale, self._mem_weight, self._sigma_scale = self._param_net.predict_np(feats)

        adaptive_lambda = self._base_lambda * (1.0 + self.eta * (1.0 - alpha) * self._temp_scale)
        adaptive_sigma = self._base_sigma * np.sqrt(1.0 + self.mu * (1.0 - alpha)) * self._sigma_scale

        old_lambda, old_sigma = self.lambda_, self.sigma
        self.lambda_ = adaptive_lambda
        self.sigma = adaptive_sigma

        # Sample with directional bias
        noise = self._sample_noise(x)
        costs, trajectories = self._rollout_mem(x, noise)
        weights = self._compute_weights(costs)

        self._last_weights = weights
        self._last_costs = costs
        self._last_noise = noise

        # Weighted update
        weighted_noise = np.einsum("k,khu->hu", weights, noise)
        self.U += weighted_noise
        if self.u_min is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        u_out = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0

        self.lambda_ = old_lambda
        self.sigma = old_sigma

        # Train adaptive network
        best_cost = float(np.min(costs))
        if self._prev_best_cost is not None and self._param_net is not None:
            improvement = self._prev_best_cost - best_cost
            feats = self._build_net_features(alpha)
            self._param_net.record_and_learn(feats, improvement)
        self._prev_best_cost = best_cost

        # Progress-aware detection
        self._detect_progressive(x, weights, costs, noise)

        return u_out

    def _sample_noise(self, x):
        noise = np.random.randn(self.K, self.H, self.n_u) * self.sigma
        nearby = self.memory.nearby_features(x)
        if nearby and self.direction_bias > 0:
            best = max(nearby, key=lambda f: f.strength)
            d = best.direction[:self.n_u]
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d = d / d_norm
                n_biased = int(self.K * self.direction_bias)
                noise[:n_biased, 0] += d * self.sigma * 1.5
        return noise

    def _rollout_mem(self, x0, noise):
        K, H = self.K, self.H
        traj = np.zeros((K, H + 1, self.n_x))
        costs = np.zeros(K)
        traj[:, 0] = x0
        has_feat = len(self.memory.features) > 0

        for t in range(H):
            u_t = self.U[t] + noise[:, t]
            if self.u_min is not None:
                u_t = np.clip(u_t, self.u_min, self.u_max)
            x_c = traj[:, t]
            traj[:, t+1] = self._batch_dynamics(x_c, u_t)
            costs += self._batch_cost(x_c, u_t)
            if has_feat:
                costs += self._mem_weight * compute_potential_batch(x_c, self.memory.features)

        if self.terminal_cost is not None:
            costs += self._batch_terminal_cost(traj[:, -1])
        return costs, traj

    def _compute_alpha(self, x):
        if not self.memory.features:
            return 1.0
        delta = 0.0
        for f in self.memory.features:
            dist = np.linalg.norm(x - f.position)
            if dist < f.radius:
                delta += f.strength * max(0.0, 1.0 - dist / f.radius)
        return min(1.0, 0.5 / (delta + 1e-6))

    def _detect_progressive(self, x, weights, costs, noise):
        """Dual-timescale detection with progress gate."""
        ess = 1.0 / (np.sum(weights**2) + 1e-10) / self.K
        cost_mean = float(np.mean(costs))
        grad = np.einsum("k,ku->u", weights, noise[:, 0])

        self._state_buf.append(x.copy())
        self._cost_buf.append(cost_mean)
        self._ess_buf.append(ess)
        self._grad_buf.append(grad.copy())

        max_buf = 40
        for buf in [self._state_buf, self._cost_buf, self._ess_buf, self._grad_buf]:
            while len(buf) > max_buf:
                buf.pop(0)

        # Progress gate: only suppress if strongly improving (>10%)
        strongly_improving = False
        if len(self._cost_buf) >= 10:
            early = np.mean(self._cost_buf[-10:-5])
            late = np.mean(self._cost_buf[-5:])
            strongly_improving = late < early * 0.90

        if strongly_improving:
            return

        # Fast detection (5 steps): react quickly
        if len(self._state_buf) >= 5:
            recent = np.array(self._state_buf[-5:])
            var = np.mean(np.var(recent, axis=0))
            scale = np.mean(np.abs(recent)) + 1e-6
            if var < 0.003 * scale**2:
                self._add_feature(x, 5, strength=0.5)

        # Confirmed detection (15 steps): stronger feature
        if len(self._state_buf) >= 15:
            recent = np.array(self._state_buf[-15:])
            var = np.mean(np.var(recent, axis=0))
            scale = np.mean(np.abs(recent)) + 1e-6
            if var < 0.005 * scale**2:
                self._add_feature(x, 15, strength=0.8)

    def _add_feature(self, x, window, strength=0.5):
        """Add a feature from detection window."""
        recent = np.array(self._state_buf[-window:])
        ess_avg = np.mean(self._ess_buf[-window:])
        spread = np.std(recent, axis=0)
        radius = max(np.linalg.norm(spread) * 3.0, 0.5)

        n = min(window, len(self._grad_buf))
        avg_g = np.mean(self._grad_buf[-n:], axis=0)
        g_n = np.linalg.norm(avg_g)
        escape = -avg_g / g_n if g_n > 1e-10 else np.random.randn(len(x))
        escape /= np.linalg.norm(escape) + 1e-10

        feat_type = LOCAL_MINIMUM if ess_avg < 0.1 else (LOW_GRADIENT if ess_avg > 0.5 else HIGH_CURVATURE)
        self.memory.add_feature(position=x, radius=radius, feature_type=feat_type,
                                direction=escape, strength=strength)

    def _age_features(self):
        """Remove features that exceeded their lifetime."""
        to_remove = []
        for i, f in enumerate(self.memory.features):
            age = self.memory.step_count - f.last_seen
            if age > self.feature_lifetime:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.memory.features.pop(i)

    def _build_net_features(self, alpha):
        """6-dim feature vector for adaptive network."""
        ess = np.mean(self._ess_buf[-5:]) if self._ess_buf else 0.5
        cost_cv = 0.0
        if len(self._cost_buf) > 2:
            cm = np.mean(self._cost_buf[-5:])
            cs = np.std(self._cost_buf[-5:])
            cost_cv = cs / (abs(cm) + 1e-6)
        return np.array([
            ess,
            alpha,
            cost_cv,
            min(len(self.memory.features) / 20.0, 1.0),
            self._temp_scale / 4.0,
            self._mem_weight / 100.0,
        ], dtype=np.float32)

    def new_episode(self):
        """Call between episodes: decay memory, keep learned features."""
        for f in self.memory.features:
            f.strength *= 0.7  # inter-episode decay
        self.memory.features = [f for f in self.memory.features if f.strength > 0.15]
        self.U = np.zeros((self.H, self.n_u))
        self._state_buf.clear()
        self._cost_buf.clear()
        self._ess_buf.clear()
        self._grad_buf.clear()
        self._prev_best_cost = None

    def reset(self):
        super().reset()
        self.memory.reset()
        self._state_buf.clear()
        self._cost_buf.clear()
        self._ess_buf.clear()
        self._grad_buf.clear()
        self._prev_best_cost = None
        self.step_count = 0
        if self._param_net is not None:
            self._param_net = _ParamNet()
