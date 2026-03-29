"""
Memory module for MA-MPPI: topological feature detection and storage.

Features are detected from MPPI sampling statistics (no external gradient needed).
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# Feature types
LOCAL_MINIMUM = 1
LOW_GRADIENT = 2
HIGH_CURVATURE = 3


@dataclass
class Feature:
    """A detected topological feature in the value function landscape."""
    position: np.ndarray       # m_i: center in state space
    radius: float              # r_i: influence radius
    strength: float            # γ_i: current importance
    feature_type: int          # κ_i: 1=local_min, 2=low_grad, 3=high_curv
    direction: np.ndarray      # d_i: escape/guidance direction
    encounter_count: int = 1
    last_seen: int = 0


class MemoryRepository:
    """
    Stores and manages topological features detected during control.

    Features are added when the controller detects challenging landscape
    regions, merged when spatially close, and decayed over time.
    """

    def __init__(
        self,
        max_features: int = 100,
        merge_threshold: float = 1.5,
        decay_factor: float = 0.99,
        strength_increment: float = 0.1,
        max_strength: float = 5.0,
        min_strength: float = 0.1,
        decay_after: int = 100,
    ):
        self.max_features = max_features
        self.merge_threshold = merge_threshold
        self.decay_factor = decay_factor
        self.strength_increment = strength_increment
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.decay_after = decay_after

        self.features: List[Feature] = []
        self.step_count = 0

    def add_feature(
        self,
        position: np.ndarray,
        radius: float,
        feature_type: int,
        direction: np.ndarray,
        strength: float = 0.5,
    ):
        """Add or merge a new feature."""
        position = np.asarray(position, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        d_norm = np.linalg.norm(direction)
        if d_norm > 1e-10:
            direction = direction / d_norm

        # Try to merge with existing nearby feature of same type
        for f in self.features:
            if f.feature_type != feature_type:
                continue
            dist = np.linalg.norm(f.position - position)
            if dist < self.merge_threshold * f.radius:
                self._merge(f, position, radius, strength, direction)
                return

        # Add new feature
        feat = Feature(
            position=position.copy(),
            radius=radius,
            strength=strength,
            feature_type=feature_type,
            direction=direction.copy(),
            last_seen=self.step_count,
        )
        self.features.append(feat)

        # Evict weakest if at capacity
        if len(self.features) > self.max_features:
            self.features.sort(key=lambda f: f.strength)
            self.features.pop(0)

    def _merge(self, existing: Feature, pos: np.ndarray, radius: float,
               strength: float, direction: np.ndarray):
        """Merge new detection into existing feature (Eq. 51-54)."""
        g_old, g_new = existing.strength, strength
        total = g_old + g_new

        existing.position = (g_old * existing.position + g_new * pos) / total
        existing.radius = max(
            existing.radius, radius,
            np.linalg.norm(existing.position - pos) / 2 + min(existing.radius, radius)
        )
        existing.strength = min(total, self.max_strength)
        existing.encounter_count += 1
        existing.last_seen = self.step_count

        # Merge direction
        d_merged = g_old * existing.direction + g_new * direction
        d_norm = np.linalg.norm(d_merged)
        if d_norm > 1e-10:
            existing.direction = d_merged / d_norm

    def update(self, current_pos: np.ndarray):
        """
        Decay features and reinforce nearby ones.
        Called every control step.
        """
        self.step_count += 1
        current_pos = np.asarray(current_pos)

        to_remove = []
        for i, f in enumerate(self.features):
            dist = np.linalg.norm(f.position - current_pos)

            if dist <= f.radius:
                # Reinforce: agent is near this feature
                f.strength = min(f.strength + self.strength_increment,
                                 self.max_strength)
                f.last_seen = self.step_count
            else:
                # Decay if not seen for a while
                steps_since = self.step_count - f.last_seen
                if steps_since > self.decay_after:
                    f.strength *= self.decay_factor

            if f.strength < self.min_strength:
                to_remove.append(i)

        for i in reversed(to_remove):
            self.features.pop(i)

    def nearby_features(self, pos: np.ndarray) -> List[Feature]:
        """Get features whose influence region contains pos."""
        pos = np.asarray(pos)
        return [f for f in self.features
                if np.linalg.norm(f.position - pos) <= f.radius]

    def proximity(self, pos: np.ndarray) -> float:
        """
        Weighted proximity to memory features δ(x, M_t) (Eq. 61).
        Returns 0 if far from all features, >0 if near.
        """
        pos = np.asarray(pos)
        total = 0.0
        for f in self.features:
            dist = np.linalg.norm(pos - f.position)
            if dist < f.radius:
                total += f.strength * max(0.0, 1.0 - dist / f.radius)
        return total

    def reset(self):
        self.features.clear()
        self.step_count = 0


class SamplingFeatureDetector:
    """
    Detects topological features purely from MPPI sampling statistics.

    No external gradient or Hessian computation needed — everything is
    extracted from the trajectory samples that MPPI already computes.
    """

    def __init__(
        self,
        stagnation_window: int = 15,
        ess_low_threshold: float = 0.1,
        ess_high_threshold: float = 0.8,
        cost_plateau_threshold: float = 0.05,
        radius_scale: float = 3.0,
    ):
        self.stagnation_window = stagnation_window
        self.ess_low_thresh = ess_low_threshold
        self.ess_high_thresh = ess_high_threshold
        self.cost_plateau_thresh = cost_plateau_threshold
        self.radius_scale = radius_scale

        self.state_history: List[np.ndarray] = []
        self.ess_history: List[float] = []
        self.cost_mean_history: List[float] = []
        self.cost_std_history: List[float] = []
        self.grad_history: List[np.ndarray] = []

    def observe(
        self,
        state: np.ndarray,
        weights: np.ndarray,
        costs: np.ndarray,
        noise: np.ndarray,
        n_samples: int,
    ) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Analyze sampling statistics and detect features.

        Returns: (feature_type, radius, escape_direction) or None
        """
        state = np.asarray(state)
        self.state_history.append(state.copy())

        # Compute sampling statistics
        ess = 1.0 / (np.sum(weights ** 2) + 1e-10)
        ess_ratio = ess / n_samples
        cost_mean = np.mean(costs)
        cost_std = np.std(costs)
        cost_cv = cost_std / (abs(cost_mean) + 1e-10)

        self.ess_history.append(ess_ratio)
        self.cost_mean_history.append(cost_mean)
        self.cost_std_history.append(cost_std)

        # Gradient direction from weighted perturbations
        grad_dir = np.einsum("k,ku->u", weights, noise[:, 0])
        self.grad_history.append(grad_dir.copy())

        # Need enough history for stagnation detection
        if len(self.state_history) < self.stagnation_window:
            return None

        # --- Detection logic ---

        # State stagnation: position variance over window
        recent = np.array(self.state_history[-self.stagnation_window:])
        state_var = np.mean(np.var(recent, axis=0))
        is_stagnant = state_var < self._adaptive_var_threshold(recent)

        if not is_stagnant:
            return None

        # Classify the type of challenging region from sampling statistics
        recent_ess = np.mean(self.ess_history[-self.stagnation_window:])

        # Estimate radius from trajectory spread
        spread = np.std(recent, axis=0)
        radius = np.linalg.norm(spread) * self.radius_scale

        # Escape direction: opposite of recent average gradient
        recent_grads = np.array(self.grad_history[-self.stagnation_window:])
        avg_grad = np.mean(recent_grads, axis=0)
        grad_norm = np.linalg.norm(avg_grad)

        if grad_norm > 1e-10:
            escape_dir = -avg_grad / grad_norm
        else:
            # Random escape if no gradient signal
            escape_dir = np.random.randn(len(state))
            escape_dir /= np.linalg.norm(escape_dir) + 1e-10

        if recent_ess < self.ess_low_thresh:
            # Low ESS + stagnant = local minimum (sharp basin, few good samples)
            return (LOCAL_MINIMUM, radius, escape_dir)
        elif cost_cv < self.cost_plateau_thresh:
            # Flat cost landscape = low-gradient plateau
            return (LOW_GRADIENT, radius, escape_dir)
        else:
            # Stagnant but ESS moderate, cost varies = high-curvature region
            return (HIGH_CURVATURE, radius, escape_dir)

    def _adaptive_var_threshold(self, recent_states: np.ndarray) -> float:
        """Adaptive stagnation threshold based on state scale."""
        scale = np.mean(np.abs(recent_states)) + 1e-6
        return 0.005 * scale ** 2

    def reset(self):
        self.state_history.clear()
        self.ess_history.clear()
        self.cost_mean_history.clear()
        self.cost_std_history.clear()
        self.grad_history.clear()
