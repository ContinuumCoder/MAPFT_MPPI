"""
Memory-based potential field implementations.
"""
import numpy as np

class MemoryPotential:
    """Memory-based potential field generator"""
    
    def __init__(self, memory_repository):
        """
        Initialize memory potential field.
        
        Args:
            memory_repository: Repository of memory features
        """
        self.memory = memory_repository
    
    def compute_potential(self, position, goal=None):
        """
        Compute memory potential at given position.
        
        Args:
            position: Position to evaluate
            goal: Optional goal position
            
        Returns:
            potential: Potential value
            gradient: Potential gradient (if requested)
        """
        # Get nearby features
        nearby_features = self.memory.get_nearby_features(position)
        
        # If no features, return zero potential
        if not nearby_features:
            return 0.0, np.zeros_like(position)
        
        # Calculate total potential and gradient
        total_potential = 0.0
        gradient = np.zeros_like(position)
        
        for feature in nearby_features:
            # Calculate potential based on feature type
            if feature['type'] == 1:  # Local minimum
                pot, grad = self._local_minimum_potential(position, feature)
            elif feature['type'] == 2:  # Low gradient
                pot, grad = self._low_gradient_potential(position, feature, goal)
            elif feature['type'] == 3:  # High curvature
                pot, grad = self._high_curvature_potential(position, feature)
            else:
                continue
                
            # Add to total
            total_potential += pot
            gradient += grad
        
        return total_potential, gradient
    
    def _local_minimum_potential(self, position, feature):
        """
        Compute potential for local minimum feature.
        
        Args:
            position: Position to evaluate
            feature: Feature data
            
        Returns:
            potential: Potential value
            gradient: Potential gradient
        """
        # Extract feature parameters
        center = feature['position']
        radius = feature['radius']
        strength = feature['strength']
        
        # Calculate distance
        dist = np.linalg.norm(position - center)
        
        # If outside influence radius, no effect
        if dist >= radius:
            return 0.0, np.zeros_like(position)
        
        # Calculate repulsive potential - quadratic with zero on boundary
        potential = strength * (1.0 - dist**2 / radius**2)**2
        
        # Calculate gradient - derivative of the potential
        if dist < 1e-6:
            # If at center point, random direction
            direction = np.random.randn(*position.shape)
            direction = direction / np.linalg.norm(direction)
        else:
            direction = (position - center) / dist
            
        gradient_magnitude = -4.0 * strength * (1.0 - dist**2 / radius**2) * dist / radius**2
        gradient = gradient_magnitude * direction
        
        return potential, gradient
    
    def _low_gradient_potential(self, position, feature, goal=None):
        """
        Compute potential for low gradient feature.
        
        Args:
            position: Position to evaluate
            feature: Feature data
            goal: Goal position
            
        Returns:
            potential: Potential value
            gradient: Potential gradient
        """
        # Extract feature parameters
        center = feature['position']
        radius = feature['radius']
        strength = feature['strength']
        direction = feature['direction']
        
        # Calculate distance
        dist = np.linalg.norm(position - center)
        
        # If outside influence radius, no effect
        if dist >= radius:
            return 0.0, np.zeros_like(position)
        
        # If direction not specified, use direction to goal if available
        if np.linalg.norm(direction) < 1e-6 and goal is not None:
            dir_to_goal = goal - center
            if np.linalg.norm(dir_to_goal) > 1e-6:
                direction = dir_to_goal / np.linalg.norm(dir_to_goal)
        
        # Normalize direction
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            # If no direction, use random
            direction = np.random.randn(*position.shape)
            direction = direction / np.linalg.norm(direction)
        
        # Calculate directional potential - higher in preferred direction
        rel_pos = position - center
        projection = np.dot(rel_pos, direction)
        
        # Potential increases in preferred direction
        # and decreases in radius
        scale = (1.0 - dist**2 / radius**2)
        potential = strength * scale * projection
        
        # Gradient calculation
        gradient = strength * (
            direction * scale - 
            2.0 * dist / radius**2 * projection * rel_pos / dist
        )
        
        return potential, gradient
    
    def _high_curvature_potential(self, position, feature):
        """
        Compute potential for high curvature feature.
        
        Args:
            position: Position to evaluate
            feature: Feature data
            
        Returns:
            potential: Potential value
            gradient: Potential gradient
        """
        # Extract feature parameters
        center = feature['position']
        radius = feature['radius']
        strength = feature['strength']
        direction = feature['direction']
        
        # Calculate distance
        dist = np.linalg.norm(position - center)
        
        # If outside influence radius, no effect
        if dist >= radius:
            return 0.0, np.zeros_like(position)
        
        # Normalize direction
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            # If no direction, use random
            direction = np.random.randn(*position.shape)
            direction = direction / np.linalg.norm(direction)
        
        # Calculate saddle-like potential
        rel_pos = position - center
        
        # Projection onto principal direction
        proj = np.dot(rel_pos, direction)
        
        # Perpendicular component
        perp = rel_pos - proj * direction
        perp_norm = np.linalg.norm(perp)
        
        # Saddle shape: increase along direction, decrease perpendicular
        saddle = proj**2 - perp_norm**2
        
        # Scale by distance from center
        scale = (1.0 - dist**2 / radius**2)
        potential = strength * scale * saddle
        
        # Gradient calculation (simplified)
        # d/dx[proj^2 - perp^2] = 2*proj*direction - 2*perp
        gradient_saddle = 2.0 * proj * direction - 2.0 * perp
        
        # Apply scaling
        gradient = strength * (
            gradient_saddle * scale - 
            2.0 * dist / radius**2 * saddle * rel_pos / dist
        )
        
        return potential, gradient
