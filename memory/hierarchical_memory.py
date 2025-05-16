"""
Hierarchical memory representation with feature abstraction capabilities.
"""
import numpy as np
import sklearn.cluster as cluster
from memory.memory_repository import MemoryRepository

class HierarchicalMemory(MemoryRepository):
    """Hierarchical memory with feature abstraction and generalization"""
    
    def __init__(self, max_features=100, decay_factor=0.95, 
                 generalization_threshold=0.7, abstraction_interval=100):
        """
        Initialize hierarchical memory.
        
        Args:
            max_features: Maximum number of features to store
            decay_factor: Strength decay factor for features
            generalization_threshold: Similarity threshold for generalization
            abstraction_interval: Number of steps between abstractions
        """
        super().__init__(max_features, decay_factor)
        
        self.generalization_threshold = generalization_threshold
        self.abstraction_interval = abstraction_interval
        
        # Additional structures for hierarchical memory
        self.abstract_features = []  # List of abstract/generalized features
        self.feature_clusters = []   # Cluster assignments for features
        self.feature_embeddings = [] # Learned embeddings for features
    
    def add_feature(self, position, radius, strength, feature_type, direction=None):
        """
        Add a new feature with generalization capabilities.
        
        Args:
            position: Feature position
            radius: Feature influence radius
            strength: Feature strength
            feature_type: Feature type
            direction: Direction vector
            
        Returns:
            success: Boolean indicating success
        """
        # Check for similarity with existing abstract features
        for i, abstract in enumerate(self.abstract_features):
            similarity = self._calculate_similarity(
                position, radius, feature_type, direction,
                abstract['position'], abstract['radius'], abstract['type'], abstract['direction']
            )
            
            if similarity > self.generalization_threshold:
                # Update the abstract feature
                self._update_abstract_feature(i, position, radius, strength, feature_type, direction)
                return True
        
        # If no similar abstract feature, add to base repository
        return super().add_feature(position, radius, strength, feature_type, direction)
    
    def step(self):
        """Update step counter and perform maintenance"""
        super().step()
        
        # Periodically perform abstraction
        if self.step_count % self.abstraction_interval == 0 and len(self.positions) > 5:
            self._perform_abstraction()
    
    def _calculate_similarity(self, pos1, rad1, type1, dir1, pos2, rad2, type2, dir2):
        """Calculate similarity between features"""
        # Only compare features of same type
        if type1 != type2:
            return 0.0
        
        # Calculate positional similarity (distance relative to radii)
        dist = np.linalg.norm(pos1 - pos2)
        pos_sim = max(0.0, 1.0 - dist / (rad1 + rad2))
        
        # Calculate directional similarity if applicable
        dir_sim = 1.0
        if dir1 is not None and dir2 is not None and np.linalg.norm(dir1) > 0.001 and np.linalg.norm(dir2) > 0.001:
            dir1_norm = dir1 / np.linalg.norm(dir1)
            dir2_norm = dir2 / np.linalg.norm(dir2)
            dir_sim = max(0.0, 0.5 * (1.0 + np.dot(dir1_norm, dir2_norm)))
        
        # Calculate radius similarity
        rad_sim = min(rad1, rad2) / max(rad1, rad2)
        
        # Combine similarities
        return 0.6 * pos_sim + 0.2 * dir_sim + 0.2 * rad_sim
    
    def _update_abstract_feature(self, index, position, radius, strength, feature_type, direction):
        """Update an abstract feature with new information"""
        abstract = self.abstract_features[index]
        
        # Update position (weighted average)
        weight = 0.2
        abstract['position'] = (1 - weight) * abstract['position'] + weight * position
        
        # Update radius (expand to cover new feature)
        abstract['radius'] = max(abstract['radius'], 
                                radius + np.linalg.norm(abstract['position'] - position))
        
        # Update strength
        abstract['strength'] = max(abstract['strength'], strength)
        
        # Update direction if provided and meaningful
        if direction is not None and np.linalg.norm(direction) > 0.001:
            if np.linalg.norm(abstract['direction']) > 0.001:
                # Blend directions
                d1 = abstract['direction'] / np.linalg.norm(abstract['direction'])
                d2 = direction / np.linalg.norm(direction)
                blend = 0.8 * d1 + 0.2 * d2
                abstract['direction'] = blend / np.linalg.norm(blend)
            else:
                # Set direction
                abstract['direction'] = direction / np.linalg.norm(direction)
        
        # Update metadata
        abstract['last_access'] = self.step_count
        abstract['update_count'] += 1
    
    def _perform_abstraction(self):
        """Perform feature abstraction to identify general patterns"""
        if len(self.positions) < 5:
            return  # Not enough data
        
        # Extract feature data
        feature_data = []
        for i in range(len(self.positions)):
            # Create feature vector: [position, radius, type, strength]
            feature_vec = np.concatenate([
                self.positions[i], 
                [self.radii[i], self.types[i], self.strengths[i]]
            ])
            feature_data.append(feature_vec)
        
        feature_data = np.array(feature_data)
        
        # Perform clustering to find similar features
        n_clusters = min(3, len(feature_data) // 2)  # At most 3 clusters, at least 2 per cluster
        if n_clusters < 1:
            return
        
        # Use DBSCAN for clustering (finds clusters of arbitrary shape)
        # epsilon parameter is the max distance between samples for them to be considered neighbors
        clustering = cluster.DBSCAN(eps=2.0, min_samples=2).fit(feature_data)
        
        labels = clustering.labels_
        self.feature_clusters = labels
        
        # Create or update abstract features based on clusters
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue
            
            # Get features in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue  # Need at least 2 features for a meaningful abstraction
            
            # Calculate cluster center and spread
            cluster_features = [feature_data[i] for i in cluster_indices]
            cluster_center = np.mean(cluster_features, axis=0)
            
            # Extract abstract feature properties
            abs_position = cluster_center[:2]  # First 2 dimensions are position
            abs_radius = max([self.radii[i] + np.linalg.norm(abs_position - self.positions[i]) 
                             for i in cluster_indices])
            abs_type = int(round(cluster_center[3]))  # 4th dimension is type
            abs_strength = np.max([self.strengths[i] for i in cluster_indices])
            
            # Calculate direction as average of directions
            directions = [self.directions[i] for i in cluster_indices]
            valid_dirs = [d for d in directions if np.linalg.norm(d) > 0.001]
            
            if valid_dirs:
                # Normalize and average
                norm_dirs = [d / np.linalg.norm(d) for d in valid_dirs]
                avg_dir = np.mean(norm_dirs, axis=0)
                abs_direction = avg_dir / np.linalg.norm(avg_dir) if np.linalg.norm(avg_dir) > 0.001 else np.zeros_like(avg_dir)
            else:
                abs_direction = np.zeros_like(self.directions[0])
            
            # Check if this abstract feature already exists
            exists = False
            for i, abstract in enumerate(self.abstract_features):
                if abstract['type'] == abs_type and np.linalg.norm(abstract['position'] - abs_position) < abs_radius:
                    # Update existing abstract feature
                    self._update_abstract_feature(i, abs_position, abs_radius, abs_strength, abs_type, abs_direction)
                    exists = True
                    break
            
            # Create new abstract feature if it doesn't exist
            if not exists:
                self.abstract_features.append({
                    'position': abs_position,
                    'radius': abs_radius,
                    'strength': abs_strength,
                    'type': abs_type,
                    'direction': abs_direction,
                    'last_access': self.step_count,
                    'creation_time': self.step_count,
                    'update_count': 1,
                    'member_indices': cluster_indices.tolist()
                })
    
    def get_nearby_features(self, position, radius=float('inf')):
        """
        Get features near the specified position, including abstract features.
        
        Args:
            position: Query position
            radius: Maximum distance
            
        Returns:
            features: List of nearby features
        """
        # Get features from base repository
        nearby = super().get_nearby_features(position, radius)
        
        # Add abstract features if they're closer
        for i, abstract in enumerate(self.abstract_features):
            dist = np.linalg.norm(abstract['position'] - position)
            if dist <= radius:
                # Create feature dict
                feature = {
                    'position': abstract['position'],
                    'radius': abstract['radius'],
                    'strength': abstract['strength'],
                    'type': abstract['type'],
                    'direction': abstract['direction'],
                    'distance': dist,
                    'is_abstract': True
                }
                nearby.append(feature)
        
        return nearby
