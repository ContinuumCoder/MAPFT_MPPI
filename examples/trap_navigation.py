"""
Example demonstrating trap navigation using MA-MPPI.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import imageio
import argparse

# Import MA-MPPI components
from controllers.mppi import MPPI
from controllers.ma_mppi import MAMPPI
from utils.common import create_directory, save_experiment_results, create_trap_scenario
from visualization.potential_visualizer import PotentialVisualizer

def dynamics_2d(x, u, dt=0.1):
    """
    Simple 2D double integrator dynamics.
    
    Args:
        x: State [x, y, vx, vy]
        u: Control [ax, ay]
        dt: Time step
        
    Returns:
        next_state: Next state
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 0.95, 0],
        [0, 0, 0, 0.95]
    ])
    B = np.array([
        [0.5*dt**2, 0],
        [0, 0.5*dt**2],
        [dt, 0],
        [0, dt]
    ])
    
    next_state = A @ x + B @ u
    next_state[2:4] = np.clip(next_state[2:4], -3.0, 3.0)  # Velocity limit
    
    return next_state

def default_running_cost(x, u, goal, obstacles=None):
    """
    Default running cost function.
    
    Args:
        x: State
        u: Control
        goal: Goal state
        obstacles: List of obstacles
        
    Returns:
        cost: Cost value
    """
    # Position cost
    position_error = x[:2] - goal[:2]
    position_cost = 10.0 * np.sum(position_error**2)
    
    # Velocity cost
    velocity_cost = 1.0 * np.sum(x[2:]**2)
    
    # Control cost
    control_cost = 0.1 * np.sum(u**2)
    
    # Obstacle cost
    obstacle_cost = 0
    if obstacles:
        for obs in obstacles:
            if 'position' in obs and 'radius' in obs:
                dist = np.linalg.norm(x[:2] - obs['position']) - obs['radius']
                
                if dist < 0:  # Collision
                    obstacle_cost += 1000.0
                elif dist < 0.5:  # Extremely dangerous
                    obstacle_cost += 200.0 * (1 - dist/0.5)
                elif dist < 1.5:  # Danger zone
                    obstacle_cost += 50.0 * (1 - dist/1.5)
    
    return position_cost + velocity_cost + control_cost + obstacle_cost

def default_terminal_cost(x, goal):
    """
    Default terminal cost function.
    
    Args:
        x: State
        goal: Goal state
        
    Returns:
        cost: Cost value
    """
    position_error = x[:2] - goal[:2]
    position_cost = 50.0 * np.sum(position_error**2)
    velocity_cost = 10.0 * np.sum(x[2:]**2)
    
    return position_cost + velocity_cost

def run_experiment(controller, controller_name, scenario, max_steps=None, visualize=True):
    """
    Run experiment with given controller.
    
    Args:
        controller: Controller object
        controller_name: Controller name
        scenario: Scenario dictionary
        max_steps: Maximum number of steps
        visualize: Whether to visualize
        
    Returns:
        result: Experiment result
    """
    print(f"\n=== Running {controller_name} on {scenario['name']} ===")
    
    # Extract scenario information
    start = scenario['start'].copy()
    goal = scenario['goal'].copy()
    obstacles = scenario['obstacles']
    
    # Set maximum steps
    if max_steps is None:
        max_steps = scenario['max_steps']
    
    # Initialize
    x = start.copy()
    states_history = [x.copy()]
    controls_history = []
    frames = []
    
    # Create persistent window if visualizing
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.show(block=False)
    
    # Reset controller
    controller.reset()
    
    # Timing
    start_time = time.time()
    
    # Simulation loop
    collision = False
    goal_reached = False
    stuck_counter = 0
    
    for step in range(max_steps):
        print(f"\nStep {step+1}/{max_steps}")
        
        # Calculate control
        u = controller.compute_control(x, goal, obstacles)
        controls_history.append(u)
        
        # Apply control
        x_next = dynamics_2d(x, u)
        x = x_next.copy()
        states_history.append(x.copy())
        
        # Visualization
        if visualize:
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in obstacles:
                circle = plt.Circle(obs['position'], obs['radius'], color='blue', alpha=0.7)
                ax.add_patch(circle)
            
            # Draw trajectory
            states_array = np.array(states_history)
            ax.plot(states_array[:, 0], states_array[:, 1], 'g-', linewidth=2)
            
            # Draw predicted trajectory if available
            if hasattr(controller, 'best_trajectory'):
                ax.plot(controller.best_trajectory[:, 0], controller.best_trajectory[:, 1], 
                       'y--', linewidth=1.5, alpha=0.7)
            
            # Draw start and goal
            ax.plot(start[0], start[1], 'go', markersize=10)
            ax.plot(goal[0], goal[1], 'ro', markersize=10)
            
            # Draw current position
            ax.plot(x[0], x[1], 'ko', markersize=8)
            
            # If MA-MPPI, draw memory features
            if hasattr(controller, 'memory'):
                # Draw trap areas
                for i in range(len(controller.memory.positions)):
                    pos = controller.memory.positions[i]
                    radius = controller.memory.radii[i]
                    strength = controller.memory.strengths[i]
                    circle = plt.Circle(pos, radius, color='red', alpha=0.2)
                    ax.add_patch(circle)
                    ax.text(pos[0], pos[1], f"{strength:.1f}", 
                           color='black', fontsize=8, ha='center', va='center')
            
            # Display status
            status_text = ""
            
            if hasattr(controller, 'stuck'):
                if controller.stuck['is_stuck']:
                    status_text += f"Stuck! ({controller.stuck['duration']} steps)\n"
                
                if hasattr(controller, 'weights'):
                    w = controller.weights
                    status_text += f"Weights: Goal={w['goal']:.1f}, Memory={w['memory']:.1f}\n"
                
                status_text += f"Temperature: {controller.temperature:.2f}\n"
            
            if status_text:
                ax.text(0.02, 0.02, status_text, transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.7))
            
            goal_dist = np.linalg.norm(x[:2] - goal[:2])
            ax.set_title(f'{scenario["name"]} - {controller_name} - Step: {step} - Distance: {goal_dist:.2f}')
            
            fig.canvas.draw()
            plt.pause(0.01)
            
            # Capture frame
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        # Check goal reached
        goal_dist = np.linalg.norm(x[:2] - goal[:2])
        if goal_dist < 0.3:
            print(f"\nSuccess! {controller_name} reached goal at step {step+1}!")
            goal_reached = True
            break
            
        # Check collision
        for obs in obstacles:
            if np.linalg.norm(x[:2] - obs['position']) <= obs['radius']:
                print(f"\nCollision! {controller_name} collided at step {step+1}!")
                collision = True
                break
        
        if collision:
            break
        
        # Check stuck
        if len(states_history) >= 10:
            recent_positions = np.array([s[:2] for s in states_history[-10:]])
            position_var = np.var(recent_positions, axis=0).sum()
            if position_var < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            if stuck_counter >= 50 and not hasattr(controller, 'memory'):
                print(f"\nEarly termination: {controller_name} stuck for too long!")
                break
    
    # Close visualization
    if visualize:
        plt.ioff()
        plt.close(fig)
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    total_steps = step + 1
    states_array = np.array(states_history)
    controls_array = np.array(controls_history)
    
    # Calculate path length
    path_length = 0
    for i in range(1, len(states_history)):
        path_length += np.linalg.norm(states_history[i][:2] - states_history[i-1][:2])
    
    # Calculate control energy
    control_energy = np.mean(np.sum(controls_array**2, axis=1)) if len(controls_array) > 0 else 0
    
    # Calculate final distance
    final_dist = np.linalg.norm(x[:2] - goal[:2])
    
    # Organize results
    result = {
        'success': goal_reached,
        'collision': collision,
        'stuck': stuck_counter >= 50,
        'steps': total_steps,
        'time': elapsed_time,
        'path_length': path_length,
        'control_energy': control_energy,
        'final_dist': final_dist,
        'states': states_array,
        'controls': controls_array,
        'frames': frames
    }
    
    # Save animation
    if visualize and len(frames) > 5:
        save_animation(frames, f"results/{scenario['name']}_{controller_name}.gif")
    
    print(f"\n{controller_name} Results:")
    print(f"Success: {goal_reached}")
    print(f"Collision: {collision}")
    print(f"Steps: {total_steps}")
    print(f"Path Length: {path_length:.2f}")
    print(f"Control Energy: {control_energy:.2f}")
    print(f"Final Distance: {final_dist:.2f}")
    
    return result

def save_animation(frames, filename, fps=10):
    """
    Save animation from frames.
    
    Args:
        frames: List of frames
        filename: Output filename
        fps: Frames per second
    """
    # Create directory if needed
    directory = os.path.dirname(filename)
    create_directory(directory)
    
    # Take every few frames to avoid large files
    step = max(1, len(frames) // 100)
    frames_subset = frames[::step]
    
    # Limit number of frames
    if len(frames_subset) > 100:
        frames_subset = frames_subset[:100]
    
    # Save as GIF
    imageio.mimsave(filename, frames_subset, fps=fps)
    print(f"Animation saved to {filename}")

def compare_results(standard_result, ma_result, scenario):
    """
    Compare and visualize results.
    
    Args:
        standard_result: Standard MPPI result
        ma_result: MA-MPPI result
        scenario: Scenario dictionary
    """
    print("\nResult Comparison:")
    print(f"{'Metric':<15} {'Standard MPPI':<15} {'MA-MPPI':<15}")
    print("-" * 45)
    
    # Success status
    standard_status = "Success" if standard_result['success'] else ("Collision" if standard_result['collision'] else ("Stuck" if standard_result['stuck'] else "Incomplete"))
    ma_status = "Success" if ma_result['success'] else ("Collision" if ma_result['collision'] else ("Stuck" if ma_result['stuck'] else "Incomplete"))
    
    print(f"{'Status':<15} {standard_status:<15} {ma_status:<15}")
    print(f"{'Steps':<15} {standard_result['steps']:<15} {ma_result['steps']:<15}")
    print(f"{'Path Length':<15} {standard_result['path_length']:<15.2f} {ma_result['path_length']:<15.2f}")
    print(f"{'Control Energy':<15} {standard_result['control_energy']:<15.2f} {ma_result['control_energy']:<15.2f}")
    print(f"{'Final Distance':<15} {standard_result['final_dist']:<15.2f} {ma_result['final_dist']:<15.2f}")
    
    # Create comparison visualization
    create_directory("results/comparisons")
    
    plt.figure(figsize=(12, 6))
    
    # Draw scenario and obstacles
    obstacles = scenario['obstacles']
    
    for i, title in enumerate(["Standard MPPI", "Memory-Augmented MPPI"]):
        plt.subplot(1, 2, i+1)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obs in obstacles:
            circle = plt.Circle(obs['position'], obs['radius'], color='blue', alpha=0.7)
            plt.gca().add_patch(circle)
        
        # Draw start and goal
        start = scenario['start']
        goal = scenario['goal']
        
        plt.plot(start[0], start[1], 'go', markersize=10)
        plt.plot(goal[0], goal[1], 'ro', markersize=10)
        
        # Draw trajectory
        result = standard_result if i == 0 else ma_result
        status = "Success" if result['success'] else ("Collision" if result['collision'] else ("Stuck" if result['stuck'] else "Incomplete"))
        
        plt.plot(result['states'][:, 0], result['states'][:, 1], 'g-', linewidth=2)
        plt.title(f"{title}: {status}, steps={result['steps']}")
    
    plt.tight_layout()
    plt.savefig(f"results/comparisons/{scenario['name']}_comparison.png", dpi=150)
    plt.close()
    
    print("\nComparison visualization saved to results/comparisons/")

def run_comparison():
    """Run comparison between standard MPPI and MA-MPPI"""
    # Create scenario
    scenario = create_trap_scenario()
    
    # Control limits
    u_min = np.array([-2.0, -2.0])
    u_max = np.array([2.0, 2.0])
    
    # Create standard MPPI controller
    standard_mppi = MPPI(
        dynamics_model=dynamics_2d,
        running_cost=default_running_cost,
        terminal_cost=default_terminal_cost,
        state_dim=4,
        control_dim=2,
        horizon=30,
        num_samples=200,
        noise_sigma=np.diag([3.0, 3.0]),
        lambda_=1.0,
        temperature=0.5,
        u_min=u_min,
        u_max=u_max
    )
    
    # Create MA-MPPI controller
    ma_mppi = MAMPPI(
        dynamics_model=dynamics_2d,
        running_cost=default_running_cost,
        terminal_cost=default_terminal_cost,
        state_dim=4,
        control_dim=2,
        horizon=30,
        num_samples=200,
        noise_sigma=np.diag([3.0, 3.0]),
        lambda_=1.0,
        temperature=0.5,
        u_min=u_min,
        u_max=u_max,
        max_features=100,
        detection_thresholds={
            'var': 0.01,
            'grad': 0.01,
            'curv': 0.5,
            'dist': 0.5,
            'merge': 1.5
        }
    )
    
    # Run standard MPPI
    standard_result = run_experiment(standard_mppi, "Standard MPPI", scenario)
    
    # Run MA-MPPI
    ma_result = run_experiment(ma_mppi, "MA-MPPI", scenario)
    
    # Compare results
    compare_results(standard_result, ma_result, scenario)
    
    # Save results
    save_experiment_results({
        'standard': standard_result,
        'ma': ma_result,
        'scenario': scenario
    }, "results/trap_comparison.npy")
    
    return standard_result, ma_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trap navigation experiment")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--steps", type=int, default=None, help="Maximum steps")
    args = parser.parse_args()
    
    # Create results directory
    create_directory("results")
    
    # Run comparison
    standard_result, ma_result = run_comparison()
