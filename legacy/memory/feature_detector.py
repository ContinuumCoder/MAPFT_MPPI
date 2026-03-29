"""
Detector for topological features in the state space.
"""
import numpy as np

class FeatureDetector:
    """Detects topological features in state space"""
    
    def __init__(self, state_dim=2, feature_types=3, thresholds=None):
        """
        Initialize feature detector.
        
        Args:
            state_dim: Dimension of state space
            feature_types: Number of feature types
            thresholds: Detection thresholds
        """
        self.state_dim = state_dim
        self.feature_types = feature_types
        
        # Default thresholds
        self.thresholds = {
            'var': 0.01,         # State stagnation threshold
            'grad': 0.01,        # Gradient magnitude threshold
            'curv': 0.5,         # Curvature threshold
            'dist': 0.5,         # Feature distance threshold
            'merge': 1.5         # Feature merging threshold
        }
        
        # Update thresholds if provided
        if thresholds is not None:
            self.thresholds.update(thresholds)
    
    def detect_state_stagnation(self, state_history, window_size=5):
        """
        Detect state stagnation by analyzing variance of recent states.
        
        Args:
            state_history: History of states
            window_size: Window size for variance calculation
            
        Returns:
            is_stagnant: Boolean indicating stagnation
            variance: State variance
        """
        if len(state_history) < window_size:
            return False, float('inf')
        
        recent_states = np.array(state_history[-window_size:])[:, :self.state_dim]
        variance = np.var(recent_states, axis=0).sum()
        
        return variance < self.thresholds['var'], variance
    
    def detect_low_gradient(self, grad_magnitude):
        """
        Detect low gradient regions.
        
        Args:
            grad_magnitude: Gradient magnitude
            
        Returns:
            is_low_gradient: Boolean indicating low gradient
        """
        return grad_magnitude < self.thresholds['grad']
    
    def detect_high_curvature(self, curvature):
        """
        Detect high curvature regions.
        
        Args:
            curvature: Curvature measure
            
        Returns:
            is_high_curvature: Boolean indicating high curvature
        """
        return curvature > self.thresholds['curv']
    
    def should_merge_features(self, feature1, feature2):
        """
        Determine if features should be merged.
        
        Args:
            feature1: First feature
            feature2: Second feature
            
        Returns:
            should_merge: Boolean indicating whether to merge
        """
        # Extract positions and radii
        pos1, radius1 = feature1['position'], feature1['radius']
        pos2, radius2 = feature2['position'], feature2['radius']
        
        # Check if same type
        if feature1['type'] != feature2['type']:
            return False
        
        # Calculate normalized distance
        distance = np.linalg.norm(pos1 - pos2)
        normalized_distance = distance / min(radius1, radius2)
        
        return normalized_distance < self.thresholds['merge']
