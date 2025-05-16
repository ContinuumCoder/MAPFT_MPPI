"""
Memory repository for storing topological features.
"""
import numpy as np

class MemoryRepository:
    """Repository for storing and managing memory of topological features"""
    
    def __init__(self, max_features=100, decay_factor=0.95):
        """
        Initialize memory repository.
        
        Args:
            max_features: Maximum number of features to store
            decay_factor: Strength decay factor for features
        """
        self.max_features = max_features
        self.decay_factor = decay_factor
        
        # Feature lists
        self.positions = []     # Feature positions
        self.radii = []         # Feature influence radii
        self.strengths = []     # Feature strengths
        self.types = []         # Feature types (1=local min, 2=low grad, 3=high curv)
        self.directions = []    # Feature direction vectors (for types 2 and 3)
        
        # Auxiliary structures
        self.last_access = []   # Last access timestamps
        self.creation_time = [] # Creation timestamps
        self.update_count = []  # Number of times feature has been updated
        
        # Step counter
        self.step_count = 0
    
    def add_feature(self, position, radius, strength, feature_type, direction=None):
        """
        Add a new feature to memory.
        
        Args:
            position: Feature position
            radius: Feature influence radius
            strength: Feature strength
            feature_type: Feature type
            direction: Direction vector
            
        Returns:
            success: Boolean indicating success
        """
        # Check if similar feature already exists
        for i, pos in enumerate(self.positions):
            if self.types[i] == feature_type and np.linalg.norm(pos - position) < radius:
                # Update existing feature
                self._update_feature(i, position, radius, strength, direction)
                return True
        
        # If at capacity, remove least important feature
        if len(self.positions) >= self.max_features:
            self._remove_least_important()
        
        # Add new feature
        self.positions.append(position.copy())
        self.radii.append(radius)
        self.strengths.append(strength)
        self.types.append(feature_type)
        
        if direction is not None:
            self.directions.append(direction.copy())
        else:
            self.directions.append(np.zeros(len(position)))
        
        # Set metadata
        self.last_access.append(self.step_count)
        self.creation_time.append(self.step_count)
        self.update_count.append(1)
        
        return True
    
    def _update_feature(self, index, position, radius, strength, direction):
        """
        Update an existing feature.
        
        Args:
            index: Feature index
            position: New position
            radius: New radius
            strength: New strength
            direction: New direction
        """
        # Update strength and radius
        self.strengths[index] = min(10.0, self.strengths[index] + strength)
        self.radii[index] = max(self.radii[index], radius)
        
        # Update position (weighted average)
        weight = 0.2
        self.positions[index] = (1-weight)*self.positions[index] + weight*position
        
        # Update direction if provided and meaningful
        if direction is not None and np.linalg.norm(direction) > 0.001:
            if np.linalg.norm(self.directions[index]) > 0.001:
                # Blend directions
                d1 = self.directions[index] / np.linalg.norm(self.directions[index])
                d2 = direction / np.linalg.norm(direction)
                blend = (d1 + d2) / 2
                self.directions[index] = blend / np.linalg.norm(blend)
            else:
                # Set direction
                self.directions[index] = direction / np.linalg.norm(direction)
        
        # Update metadata
        self.last_access[index] = self.step_count
        self.update_count[index] += 1
    
    def _remove_least_important(self):
        """Remove least important feature from memory"""
        if not self.positions:
            return
        
        # Calculate importance scores
        importance = []
        for i in range(len(self.positions)):
            # Factors: strength, recency, update frequency
            strength_factor = self.strengths[i]
            recency_factor = 1.0 / (1.0 + (self.step_count - self.last_access[i]) / 100.0)
            frequency_factor = min(1.0, self.update_count[i] / 10.0)
            
            score = strength_factor * recency_factor * frequency_factor
            importance.append(score)
        
        # Remove least important feature
        min_idx = np.argmin(importance)
        self._remove_feature(min_idx)
    
    def _remove_feature(self, index):
        """
        Remove feature at specified index.
        
        Args:
            index: Feature index to remove
        """
        self.positions.pop(index)
        self.radii.pop(index)
        self.strengths.pop(index)
        self.types.pop(index)
        self.directions.pop(index)
        self.last_access.pop(index)
        self.creation_time.pop(index)
        self.update_count.pop(index)
    
    def decay_strengths(self):
        """Decay the strength of all features"""
        for i in range(len(self.strengths)):
            # Only decay if not recently accessed
            time_since_access = self.step_count - self.last_access[i]
            if time_since_access > 50:
                self.strengths[i] *= self.decay_factor
                
                # Remove if strength becomes too small
                if self.strengths[i] < 0.1:
                    self._remove_feature(i)
                    break  # Only remove one per call to avoid indexing issues
    
    def get_nearby_features(self, position, radius=float('inf')):
        """
        Get features near the specified position.
        
        Args:
            position: Query position
            radius: Maximum distance
            
        Returns:
            features: List of nearby features
        """
        nearby = []
        for i, pos in enumerate(self.positions):
            dist = np.linalg.norm(pos - position)
            if dist <= radius:
                feature = {
                    'position': self.positions[i],
                    'radius': self.radii[i],
                    'strength': self.strengths[i],
                    'type': self.types[i],
                    'direction': self.directions[i],
                    'distance': dist
                }
                nearby.append(feature)
        
        return nearby
    
    def step(self):
        """Update step counter and perform maintenance"""
        self.step_count += 1
        
        # Periodically decay strengths
        if self.step_count % 10 == 0:
            self.decay_strengths()
    
    def __len__(self):
        """Get number of features in memory"""
        return len(self.positions)
