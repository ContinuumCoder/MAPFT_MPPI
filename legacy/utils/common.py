"""
Common utility functions for the MA-MPPI framework.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def create_directory(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_experiment_results(results, filename):
    """
    Save experiment results.
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    # Create directory if needed
    directory = os.path.dirname(filename)
    create_directory(directory)
    
    # Save results
    np.save(filename, results)
    
    print(f"Results saved to {filename}")

def generate_noisy_trajectory(start, goal, duration, noise_level=0.1, dt=0.1):
    """
    Generate a simple noisy trajectory between start and goal.
    
    Args:
        start: Start position
        goal: Goal position
        duration: Trajectory duration
        noise_level: Noise level
        dt: Time step
        
    Returns:
        trajectory: Trajectory array
    """
    steps = int(duration / dt)
    dimension = len(start)
    
    # Linear interpolation
    trajectory = np.zeros((steps, dimension))
    for t in range(steps):
        alpha = t / (steps - 1)
        trajectory[t] = (1 - alpha) * start + alpha * goal
        
        # Add noise
        if noise_level > 0:
            trajectory[t] += np.random.randn(dimension) * noise_level
    
    return trajectory

def create_trap_scenario():
    """
    Create a simple trap scenario with obstacles.
    
    Returns:
        scenario: Scenario dictionary
    """
    return {
        'name': 'Trap-Scenario',
        'start': np.array([5.0, 2.0, 0.0, 0.0]),
        'goal': np.array([5.0, 9.0, 0.0, 0.0]),
        'obstacles': [
            {'position': np.array([3.5, 6.0]), 'radius': 0.5},
            {'position': np.array([4.5, 6.0]), 'radius': 0.5},
            {'position': np.array([5.5, 6.0]), 'radius': 0.5},
            {'position': np.array([6.5, 6.0]), 'radius': 0.5},
            {'position': np.array([4.0, 5.1]), 'radius': 0.5},
            {'position': np.array([6.0, 5.1]), 'radius': 0.5},
        ],
        'max_steps': 300
    }
