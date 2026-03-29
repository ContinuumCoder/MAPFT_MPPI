"""
Potential field synthesis for MA-MPPI.

Constructs memory-augmented potential fields from detected features,
following the type-specific basis functions (Eq. 56-58 in the paper).
"""
import numpy as np
from typing import List
from .memory import Feature, LOCAL_MINIMUM, LOW_GRADIENT, HIGH_CURVATURE


def _match_dim(d: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate direction vector to match target dimension."""
    if len(d) == target_dim:
        return d
    elif len(d) > target_dim:
        return d[:target_dim]
    else:
        out = np.zeros(target_dim)
        out[:len(d)] = d
        return out


def compute_potential(x: np.ndarray, features: List[Feature]) -> float:
    """
    Compute total memory potential V_mem(x, M) = Σ γ_i · φ(x, f_i).
    x: (n_x,) state
    """
    total = 0.0
    for f in features:
        total += f.strength * _basis_potential(x, f)
    return total


def compute_potential_batch(x: np.ndarray, features: List[Feature]) -> np.ndarray:
    """
    Batch potential for (K, n_x) states. Returns (K,).
    """
    K = x.shape[0]
    result = np.zeros(K)
    for f in features:
        diff = x - f.position  # (K, n_x)
        dist_sq = np.sum(diff ** 2, axis=1)  # (K,)
        r_sq = f.radius ** 2
        mask = dist_sq < r_sq
        if not np.any(mask):
            continue
        result[mask] += f.strength * _basis_potential_batch(
            diff[mask], dist_sq[mask], r_sq, f
        )
    return result


def compute_alpha(x: np.ndarray, features: List[Feature],
                  delta_0: float = 0.5, epsilon: float = 1e-6) -> float:
    """
    Adaptive weight α(x, M) ∈ [0, 1] (Eq. 60).
    α → 1 far from features (base dominates), α → 0 near features (memory dominates).
    """
    if not features:
        return 1.0
    delta = 0.0
    for f in features:
        dist = np.linalg.norm(x - f.position)
        if dist < f.radius:
            delta += f.strength * max(0.0, 1.0 - dist / f.radius)
    return min(1.0, delta_0 / (delta + epsilon))


def compute_alpha_batch(x: np.ndarray, features: List[Feature],
                        delta_0: float = 0.5, epsilon: float = 1e-6) -> np.ndarray:
    """Batch α computation for (K, n_x) states. Returns (K,)."""
    K = x.shape[0]
    if not features:
        return np.ones(K)
    delta = np.zeros(K)
    for f in features:
        dist = np.linalg.norm(x - f.position, axis=1)  # (K,)
        mask = dist < f.radius
        delta[mask] += f.strength * np.maximum(0.0, 1.0 - dist[mask] / f.radius)
    return np.minimum(1.0, delta_0 / (delta + epsilon))


def _basis_potential(x: np.ndarray, f: Feature) -> float:
    """Type-specific basis potential φ(x, m, r, κ, d)."""
    diff = x - f.position
    dist_sq = np.sum(diff ** 2)
    r_sq = f.radius ** 2

    if dist_sq >= r_sq:
        return 0.0

    ratio = dist_sq / r_sq

    if f.feature_type == LOCAL_MINIMUM:
        # Repulsive: φ₁ = max(0, (1 - ||x-m||²/r²)²)  (Eq. 56)
        return (1.0 - ratio) ** 2

    elif f.feature_type == LOW_GRADIENT:
        # Directional guide: φ₂ = max(0, 1 - ||x-m||²/r²) · (d·(x-m))  (Eq. 57)
        d = _match_dim(f.direction, len(diff))
        return max(0.0, 1.0 - ratio) * np.dot(d, diff)

    elif f.feature_type == HIGH_CURVATURE:
        # Saddle: φ₃ (Eq. 58)
        d = _match_dim(f.direction, len(diff))
        radial = max(0.0, 1.0 - ratio)
        d_dot = np.dot(d, diff)
        perp_sq = max(0.0, dist_sq - d_dot ** 2)
        beta = 0.5
        return radial * (d_dot ** 2 - beta * perp_sq)

    return 0.0


def _basis_potential_batch(diff: np.ndarray, dist_sq: np.ndarray,
                           r_sq: float, f: Feature) -> np.ndarray:
    """Batch basis potential for masked samples."""
    ratio = dist_sq / r_sq

    if f.feature_type == LOCAL_MINIMUM:
        return (1.0 - ratio) ** 2

    elif f.feature_type == LOW_GRADIENT:
        d = _match_dim(f.direction, diff.shape[1])
        d_dot = diff @ d
        return np.maximum(0.0, 1.0 - ratio) * d_dot

    elif f.feature_type == HIGH_CURVATURE:
        d = _match_dim(f.direction, diff.shape[1])
        radial = np.maximum(0.0, 1.0 - ratio)
        d_dot = diff @ d
        perp_sq = dist_sq - d_dot ** 2
        return radial * (d_dot ** 2 - 0.5 * np.maximum(0.0, perp_sq))

    return np.zeros(len(dist_sq))
