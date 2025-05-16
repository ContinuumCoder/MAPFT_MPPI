"""
Dynamic obstacles scenario for testing controllers in changing environments.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class DynamicObstaclesScenario:
    """Scenario with moving obstacles requiring continuous adaptation"""
    
    def __init__(self, num_obstacles=5, world_size=10.0, obstacle_speed=0.05, obstacle_radius=0.3):
        """
        Initialize dynamic obstacles scenario.
        
        Args:
            num_obstacles: Number of moving obstacles
            world_size: Size of square world
            obstacle_speed: Maximum speed of obstacles
            obstacle_radius: Radius of obstacles
        """
        self.num_obstacles = num_obstacles
        self.world_size = world_size
        self.obstacle_speed = obstacle_speed
        self.obstacle_radius = obstacle_radius
        
        # Initialize obstacles
        self.obstacles = []
        self.obstacle_velocities = []
        
        for _ in range(num_obstacles):
            # Random position
            position = np.random.uniform(1.0, world_size-1.0, 2)
            
            # Random velocity
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(0.5*obstacle_speed, obstacle_speed)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            
            self.obstacles.append({
                'position': position,
                'radius': obstacle_radius
            })
            self.obstacle_velocities.append(velocity)
        
        # Set start and goal positions (opposite corners)
        self.start = np.array([1.0, 1.0, 0.0, 0.0])
        self.goal = np.array([world_size-1.0, world_size-1.0, 0.0, 0.0])
        
        # Time tracking for simulation
        self.last_update = time.time()
    
    def update_obstacles(self, dt=None):
        """
        Update obstacle positions based on velocities.
        
        Args:
            dt: Time step (if None, use actual elapsed time)
        """
        # Use elapsed time if dt not provided
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update
            self.last_update = current_time
        
        # Update each obstacle
        for i in range(self.num_obstacles):
            # Get current position and velocity
            position = self.obstacles[i]['position']
            velocity = self.obstacle_velocities[i]
            
            # Update position
            new_position = position + velocity * dt
            
            # Bounce off walls
            for j in range(2):
                if new_position[j] < self.obstacle_radius or new_position[j] > self.world_size - self.obstacle_radius:
                    # Reverse velocity component and bounce
                    velocity[j] = -velocity[j]
                    # Adjust position to be within bounds
                    if new_position[j] < self.obstacle_radius:
                        new_position[j] = 2*self.obstacle_radius - new_position[j]
                    else:
                        new_position[j] = 2*(self.world_size - self.obstacle_radius) - new_position[j]
            
            # Update obstacle position and velocity
            self.obstacles[i]['position'] = new_position
            self.obstacle_velocities[i] = velocity
            
            # Occasionally change direction randomly
            if np.random.random() < 0.01:
                angle = np.random.uniform(0, 2*np.pi)
                speed = np.linalg.norm(velocity)
                self.obstacle_velocities[i] = speed * np.array([np.cos(angle), np.sin(angle)])
    
    def visualize(self, ax=None, show=True):
        """
        Visualize dynamic obstacles scenario.
        
        Args:
            ax: Matplotlib axis (optional)
            show: Whether to show plot
            
        Returns:
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set limits
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        
        # Draw grid for reference
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Draw obstacles
        for i, obs in enumerate(self.obstacles):
            circle = Circle(
                obs['position'],
                obs['radius'],
                facecolor='blue',
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(circle)
            
            # Draw velocity vector
            velocity = self.obstacle_velocities[i]
            ax.arrow(
                obs['position'][0],
                obs['position'][1],
                velocity[0],
                velocity[1],
                head_width=0.1,
                head_length=0.1,
                fc='red',
                ec='red',
                alpha=0.7
            )
        
        # Draw start and goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        ax.set_title('Dynamic Obstacles Scenario')
        ax.legend()
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def get_scenario_dict(self):
        """
        Get scenario dictionary for controllers.
        
        Returns:
            scenario: Scenario dictionary
        """
        return {
            'name': 'Dynamic-Obstacles',
            'start': self.start.copy(),
            'goal': self.goal.copy(),
            'obstacles': self.obstacles,
            'max_steps': 300,
            'is_dynamic': True,
            'update_func': self.update_obstacles
        }
