"""
Complex maze scenario for testing memory-augmented controllers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MazeScenario:
    """Complex maze scenario with multiple local minima"""
    
    def __init__(self):
        """Initialize maze scenario"""
        # Maze dimensions (grid cells)
        self.width = 20
        self.height = 20
        self.cell_size = 0.5
        
        # Generate maze grid (0 = wall, 1 = free space)
        self.grid = self._generate_maze()
        
        # Convert to obstacle list for controllers
        self._generate_obstacles()
        
        # Set start and goal positions
        self.start = np.array([1.0, 1.0, 0.0, 0.0])
        self.goal = np.array([9.0, 9.0, 0.0, 0.0])
    
    def _generate_maze(self):
        """Generate random maze grid using modified DFS algorithm"""
        # Initialize grid with walls (0)
        grid = np.zeros((self.height, self.width), dtype=int)
        
        # Define directions: (dx, dy)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Function to check if a cell is valid and unvisited
        def is_valid(x, y):
            return (0 <= x < self.width and 0 <= y < self.height and grid[y, x] == 0)
        
        # Start from the middle
        start_x, start_y = self.width // 2, self.height // 2
        grid[start_y, start_x] = 1  # Mark as path
        
        # Stack for DFS
        stack = [(start_x, start_y)]
        
        while stack:
            current_x, current_y = stack[-1]
            
            # Get valid neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = current_x + 2*dx, current_y + 2*dy
                if is_valid(nx, ny):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose a random neighbor
                nx, ny, dx, dy = neighbors[np.random.randint(len(neighbors))]
                
                # Mark the connecting path
                grid[current_y + dy, current_x + dx] = 1
                
                # Mark the neighbor as path
                grid[ny, nx] = 1
                
                # Add neighbor to stack
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        # Add some cycles to make it more interesting (remove some walls)
        for _ in range(self.width * self.height // 10):
            x = np.random.randint(1, self.width-1)
            y = np.random.randint(1, self.height-1)
            if grid[y, x] == 0:
                grid[y, x] = 1
        
        # Ensure border walls
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        
        # Ensure start and goal areas are paths
        grid[1:3, 1:3] = 1  # Start area
        grid[-3:-1, -3:-1] = 1  # Goal area
        
        return grid
    
    def _generate_obstacles(self):
        """Convert grid to obstacle list"""
        self.obstacles = []
        
        # For each wall cell, create an obstacle
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:  # Wall
                    self.obstacles.append({
                        'position': np.array([x * self.cell_size, y * self.cell_size]),
                        'width': self.cell_size,
                        'height': self.cell_size
                    })
    
    def is_collision(self, position, radius=0.1):
        """
        Check if position collides with any obstacle.
        
        Args:
            position: [x, y] position
            radius: Agent radius
            
        Returns:
            collision: Boolean indicating collision
        """
        for obs in self.obstacles:
            # Check collision with rectangular obstacle
            rect_x, rect_y = obs['position']
            rect_w = obs['width']
            rect_h = obs['height']
            
            # Find closest point on rectangle to position
            closest_x = max(rect_x, min(position[0], rect_x + rect_w))
            closest_y = max(rect_y, min(position[1], rect_y + rect_h))
            
            # Check distance to closest point
            dist = np.sqrt((position[0] - closest_x)**2 + (position[1] - closest_y)**2)
            
            if dist < radius:
                return True
        
        return False
    
    def visualize(self, ax=None, show=True):
        """
        Visualize maze scenario.
        
        Args:
            ax: Matplotlib axis (optional)
            show: Whether to show plot
            
        Returns:
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set limits
        max_x = self.width * self.cell_size
        max_y = self.height * self.cell_size
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        
        # Draw grid for reference
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Draw obstacles (walls)
        for obs in self.obstacles:
            rect = Rectangle(
                obs['position'],
                obs['width'],
                obs['height'],
                facecolor='black',
                edgecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Draw start and goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        ax.set_title('Maze Scenario')
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
            'name': 'Complex-Maze',
            'start': self.start.copy(),
            'goal': self.goal.copy(),
            'obstacles': self.obstacles,
            'max_steps': 500
        }
