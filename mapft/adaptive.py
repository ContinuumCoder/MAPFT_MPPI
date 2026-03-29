"""
Lightweight adaptive parameter network for MA-MPPI.

A small MLP that learns to predict optimal hyperparameters (temperature,
memory weight, exploration coefficient) from sampling statistics features,
trained online via simple gradient descent.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class AdaptiveParamNet(nn.Module):
        """
        Predicts (temperature_scale, memory_weight, exploration_scale)
        from sampling statistics features.

        Input features (8-dim):
            [ess_ratio, weight_entropy, cost_mean, cost_std,
             grad_magnitude, alpha, n_features_norm, stagnation_score]

        Output (3-dim, all positive via softplus):
            [temperature_multiplier, memory_cost_weight, sigma_multiplier]
        """

        def __init__(self, hidden: int = 32, lr: float = 1e-3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
                nn.Softplus(),
            )
            # Initialize near identity: outputs ≈ 1.0
            with torch.no_grad():
                self.net[-2].bias.fill_(0.5)

            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.buffer = []  # (features, cost_improvement) pairs
            self.buffer_size = 256
            self.batch_size = 32
            self.min_samples = 64

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features)

        def predict(self, stats: dict) -> dict:
            """Predict adaptive parameters from sampling statistics."""
            feat = self._extract_features(stats)
            feat_t = torch.FloatTensor(feat).unsqueeze(0)
            with torch.no_grad():
                out = self.forward(feat_t).squeeze(0).numpy()
            return {
                "temperature_scale": float(np.clip(out[0], 0.5, 5.0)),
                "memory_weight": float(np.clip(out[1], 1.0, 200.0)),
                "sigma_scale": float(np.clip(out[2], 0.5, 3.0)),
            }

        def record(self, stats: dict, cost_improvement: float):
            """Record a (features, reward) pair for online learning."""
            feat = self._extract_features(stats)
            self.buffer.append((feat, cost_improvement))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

            if len(self.buffer) >= self.min_samples:
                self._train_step()

        def _train_step(self):
            """One gradient step: maximize cost improvement."""
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            feats = torch.FloatTensor(np.array([self.buffer[i][0] for i in indices]))
            rewards = torch.FloatTensor([self.buffer[i][1] for i in indices])

            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Forward
            params = self.forward(feats)
            # Loss: encourage parameters that correlate with cost improvement
            # Use log-params as policy, rewards as signal (REINFORCE-like)
            log_params = torch.log(params + 1e-8)
            loss = -(log_params.mean(dim=1) * rewards).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

        def _extract_features(self, stats: dict) -> np.ndarray:
            """Extract 8-dim feature vector from controller diagnostics."""
            return np.array([
                stats.get("ess_ratio", 0.5),
                stats.get("weight_entropy", 1.0),
                stats.get("cost_mean", 0.0) / (abs(stats.get("cost_max", 1.0)) + 1e-6),
                stats.get("cost_std", 0.0) / (abs(stats.get("cost_mean", 1.0)) + 1e-6),
                stats.get("grad_magnitude", 0.0),
                stats.get("alpha", 1.0),
                min(stats.get("n_features", 0) / 20.0, 1.0),
                stats.get("stagnation_score", 0.0),
            ], dtype=np.float32)

else:
    # Fallback: no PyTorch, use fixed heuristic
    class AdaptiveParamNet:
        def predict(self, stats: dict) -> dict:
            return {
                "temperature_scale": 1.0,
                "memory_weight": 50.0,
                "sigma_scale": 1.0,
            }
        def record(self, stats: dict, cost_improvement: float):
            pass
