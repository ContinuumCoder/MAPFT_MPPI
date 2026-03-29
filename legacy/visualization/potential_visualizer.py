"""
Visualization tools for potential fields and trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import os

class PotentialVisualizer:
    """Visualizer for memory-augmented potential fields"""
    
    def __init__(self, resolution=0.1, bounds=None):
        """
        Initialize visualizer.
        
        Args:
            resolution: Grid resolution
            bounds: Visualization bounds (xmin, xmax, ymin, ymax)
        """
        self.resolution = resolution
        self.bounds = bounds if bounds is not None else (0, 10, 0, 10)
        
        # Create directory for output
        os.makedirs("results/visualizations", exist_ok=True)
    
    def visualize_potential_field(self, memory_potential, goal=None, obstacles=None, 
                                  trajectory=None, filename=None):
        """
        Visualize potential field.
        
        Args:
            memory_potential: Memory potential object
            goal: Goal position
            obstacles: List of obstacles
            trajectory: Robot trajectory
            filename: Output filename
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid
        x_min, x_max, y_min, y_max = self.bounds
        x_grid = np.arange(x_min, x_max, self.resolution)
        y_grid = np.arange(y_min, y_max, self.resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Calculate potential at each point
        Z = np.zeros_like(X)
        U = np.zeros_like(X)  # x-component of gradient
        V = np.zeros_like(Y)  # y-component of gradient
        
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                position = np.array([X[j, i], Y[j, i]])
                potential, gradient = memory_potential.compute_potential(position, goal)
                Z[j, i] = potential
                
                if len(gradient) >= 2:
                    U[j, i] = gradient[0]
                    V[j, i] = gradient[1]
        
        # Apply Gaussian smoothing for visualization
        Z_smooth = gaussian_filter(Z, sigma=1.0)
        
        # Plot potential field
        contour = ax.contourf(X, Y, Z_smooth, levels=20, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z_smooth, levels=10, colors='k', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Potential Value')
        
        # Plot gradient field (downsampled)
        step = 8  # Downsample for clarity
        scale = 20  # Scale for better visualization
        ax.quiver(X[::step, ::step], Y[::step, ::step], 
                 -U[::step, ::step], -V[::step, ::step],  # Negative for descent direction
                 color='r', alpha=0.5, scale=scale, width=0.003)
        
        # Add obstacles
        if obstacles:
            for obs in obstacles:
                if 'position' in obs and 'radius' in obs:
                    circle = Circle(obs['position'], obs['radius'], color='blue', alpha=0.7)
                    ax.add_patch(circle)
        
        # Add goal
        if goal is not None:
            ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        # Add trajectory
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, label='Trajectory')
        
        # Set chart properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.set_title('Memory-Augmented Potential Field', fontsize=14)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend()
        
        # Save or show
        if filename:
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"Potential field visualization saved to {filename}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_weight_evolution(self, weight_history, stuck_periods=None, filename=None):
        """
        Visualize weight evolution over time.
        
        Args:
            weight_history: History of weights
            stuck_periods: List of (start, end) periods of being stuck
            filename: Output filename
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Extract data
        steps = len(weight_history)
        time_steps = np.arange(steps)
        
        # Extract weight components
        goal_weights = [w['goal'] for w in weight_history]
        memory_weights = [w['memory'] for w in weight_history]
        obstacle_weights = [w.get('obstacle', 0) for w in weight_history]
        
        # Plot weights
        ax.plot(time_steps, goal_weights, 'g-', linewidth=2, label='Goal Weight')
        ax.plot(time_steps, memory_weights, 'b-', linewidth=2, label='Memory Weight')
        
        if any(w > 0 for w in obstacle_weights):
            ax.plot(time_steps, obstacle_weights, 'r-', linewidth=2, label='Obstacle Weight')
        
        # Highlight stuck periods
        if stuck_periods:
            for start, end in stuck_periods:
                ax.axvspan(start, end, color='red', alpha=0.15)
                
                # Add annotation if period is long enough
                if end - start > 5:
                    mid = (start + end) // 2
                    ax.annotate('Stuck', xy=(mid, 0.8), ha='center', 
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Set chart properties
        ax.set_xlim(0, steps-1)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_title('Potential Weight Evolution', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Weight Value', fontsize=12)
        ax.legend(loc='upper right')
        
        # Save or show
        if filename:
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"Weight evolution visualization saved to {filename}")
        else:
            plt.tight_layout()
            plt.show()
