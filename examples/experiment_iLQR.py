"""
Full experiment comparing multiple controllers on multiple scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from multiprocessing import Pool
import itertools

# Import controllers
from controllers.mppi import MPPI
from controllers.ma_mppi import MAMPPI
from controllers.ilqr import iLQR

# Import scenarios
from scenarios.maze_scenario import MazeScenario
from scenarios.dynamic_obstacles import DynamicObstaclesScenario
from utils.common import create_trap_scenario

# Import benchmark utilities
from benchmark.performance_metrics import PerformanceMetrics

# Import utilities
from utils.common import create_directory, save_experiment_results, dynamics_2d
from visualization.potential_visualizer import PotentialVisualizer

def default_cost_func(state, control, goal, obstacles, terminal=False):
    """
    Default cost function for controllers.
    
    Args:
        state: State
        control: Control
        goal: Goal state
        obstacles: List of obstacles
        terminal: Whether this is terminal cost
        
    Returns:
        cost: Cost value
    """
    # Position cost
    pos_error = state[:2] - goal[:2]
    pos_cost = 10.0 * np.sum(pos_error**2)
    
    # Velocity cost
    vel_cost = 1.0 * np.sum(state[2:]**2)
    
    # Control cost (only for non-terminal)
    control_cost = 0.0
    if not terminal and control is not None:
        control_cost = 0.1 * np.sum(control**2)
    
    # Obstacle cost
    obstacle_cost = 0.0
    if obstacles is not None:
        for obs in obstacles:
            if 'position' in obs and 'radius' in obs:
                dist = np.linalg.norm(state[:2] - obs['position']) - obs['radius']
                if dist < 0:  # Collision
                    obstacle_cost += 1000.0
                elif dist < 0.5:  # Extremely dangerous
                    obstacle_cost += 200.0 * (1 - dist/0.5)
                elif dist < 1.5:  # Danger zone
                    obstacle_cost += 50.0 * (1 - dist/1.5)
            elif 'position' in obs and 'width' in obs:
                # Rectangular obstacle
                rect_x, rect_y = obs['position']
                rect_w = obs['width']
                rect_h = obs['height']
                
                # Find closest point on rectangle to position
                closest_x = max(rect_x, min(state[0], rect_x + rect_w))
                closest_y = max(rect_y, min(state[1], rect_y + rect_h))
                
                # Check distance to closest point
                dist = np.sqrt((state[0] - closest_x)**2 + (state[1] - closest_y)**2)
                
                if dist < 0.1:  # Collision
                    obstacle_cost += 1000.0
                elif dist < 0.5:  # Extremely dangerous
                    obstacle_cost += 200.0 * (1 - dist/0.5)
                elif dist < 1.0:  # Danger zone
                    obstacle_cost += 50.0 * (1 - dist/1.0)
    
    return pos_cost + vel_cost + control_cost + obstacle_cost

def create_controller(controller_type, state_dim=4, control_dim=2, **kwargs):
    """
    Create controller instance.
    
    Args:
        controller_type: Controller type string
        state_dim: State dimension
        control_dim: Control dimension
        **kwargs: Additional controller parameters
        
    Returns:
        controller: Controller instance
    """
    # Control limits
    u_min = kwargs.get('u_min', np.array([-2.0, -2.0]))
    u_max = kwargs.get('u_max', np.array([2.0, 2.0]))
    
    if controller_type.lower() == 'mppi':
        return MPPI(
            dynamics_model=dynamics_2d,
            running_cost=default_cost_func,
            terminal_cost=None,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=kwargs.get('horizon', 20),
            num_samples=kwargs.get('num_samples', 200),
            noise_sigma=kwargs.get('noise_sigma', np.diag([2.0, 2.0])),
            lambda_=kwargs.get('lambda_', 1.0),
            temperature=kwargs.get('temperature', 0.5),
            u_min=u_min,
            u_max=u_max
        )
    elif controller_type.lower() == 'ma_mppi':
        return MAMPPI(
            dynamics_model=dynamics_2d,
            running_cost=default_cost_func,
            terminal_cost=None,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=kwargs.get('horizon', 20),
            num_samples=kwargs.get('num_samples', 200),
            noise_sigma=kwargs.get('noise_sigma', np.diag([2.0, 2.0])),
            lambda_=kwargs.get('lambda_', 1.0),
            temperature=kwargs.get('temperature', 0.5),
            u_min=u_min,
            u_max=u_max,
            max_features=kwargs.get('max_features', 100),
            detection_thresholds=kwargs.get('detection_thresholds', {
                'var': 0.01,
                'grad': 0.01,
                'curv': 0.5,
                'dist': 0.5,
                'merge': 1.5
            })
        )
    elif controller_type.lower() == 'ilqr':
        return iLQR(
            dynamics_model=dynamics_2d,
            cost_function=default_cost_func,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=kwargs.get('horizon', 20),
            max_iterations=kwargs.get('max_iterations', 10),
            stopping_threshold=kwargs.get('stopping_threshold', 1e-4),
            regularization=kwargs.get('regularization', 1e-6),
            line_search_factor=kwargs.get('line_search_factor', 0.5),
            u_min=u_min,
            u_max=u_max
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

def create_scenario(scenario_type):
    """
    Create scenario instance.
    
    Args:
        scenario_type: Scenario type string
        
    Returns:
        scenario: Scenario dictionary
    """
    if scenario_type.lower() == 'trap':
        return create_trap_scenario()
    elif scenario_type.lower() == 'maze':
        maze = MazeScenario()
        return maze.get_scenario_dict()
    elif scenario_type.lower() == 'dynamic':
        dynamic = DynamicObstaclesScenario()
        return dynamic.get_scenario_dict()
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

def run_experiment(controller_type, scenario_type, visualize=False, max_steps=None, **kwargs):
    """
    Run experiment with specified controller and scenario.
    
    Args:
        controller_type: Controller type string
        scenario_type: Scenario type string
        visualize: Whether to visualize
        max_steps: Maximum steps
        **kwargs: Additional controller parameters
        
    Returns:
        result: Experiment result
    """
    print(f"\n=== Running {controller_type} on {scenario_type} scenario ===")
    
    # Create controller
    controller = create_controller(controller_type, **kwargs)
    
    # Create scenario
    scenario = create_scenario(scenario_type)
    
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
    
    # Create visualization if needed
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
    computation_times = []
    
    for step in range(max_steps):
        print(f"\rStep {step+1}/{max_steps}", end="")
        
        # Update dynamic obstacles if needed
        if 'is_dynamic' in scenario and scenario['is_dynamic'] and 'update_func' in scenario:
            scenario['update_func']()
        
        # Start computation timer
        comp_start = time.time()
        
        # Calculate control
        u = controller.compute_control(x, goal, obstacles)
        
        # Record computation time
        computation_times.append(time.time() - comp_start)
        
        controls_history.append(u)
        
        # Apply control
        x_next = dynamics_2d(x, u)
        x = x_next.copy()
        states_history.append(x.copy())
        
        # Visualization
        if visualize:
            ax.clear()
            
            # Set limits based on scenario
            if scenario_type.lower() == 'maze':
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 20)
            else:
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
            
            ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in obstacles:
                if 'position' in obs and 'radius' in obs:
                    # Circular obstacle
                    circle = plt.Circle(
                        obs['position'], obs['radius'], 
                        color='blue', alpha=0.7
                    )
                    ax.add_patch(circle)
                elif 'position' in obs and 'width' in obs:
                    # Rectangular obstacle
                    rect = plt.Rectangle(
                        obs['position'],
                        obs['width'],
                        obs['height'],
                        color='black',
                        alpha=0.7
                    )
                    ax.add_patch(rect)
            
            # Draw trajectory
            states_array = np.array(states_history)
            ax.plot(states_array[:, 0], states_array[:, 1], 'g-', linewidth=2)
            
            # Draw predicted trajectory if available
            if hasattr(controller, 'best_trajectory') and controller.best_trajectory is not None:
                ax.plot(controller.best_trajectory[:, 0], controller.best_trajectory[:, 1], 
                       'y--', linewidth=1.5, alpha=0.7)
            
            # Draw start and goal
            ax.plot(start[0], start[1], 'go', markersize=10)
            ax.plot(goal[0], goal[1], 'ro', markersize=10)
            
            # Draw current position
            ax.plot(x[0], x[1], 'ko', markersize=8)
            
            # Display status
            goal_dist = np.linalg.norm(x[:2] - goal[:2])
            ax.set_title(f'{scenario["name"]} - {controller_type} - Step: {step} - Distance: {goal_dist:.2f}')
            
            fig.canvas.draw()
            plt.pause(0.01)
        
        # Check goal reached
        goal_dist = np.linalg.norm(x[:2] - goal[:2])
        if goal_dist < 0.3:
            print(f"\nSuccess! {controller_type} reached goal at step {step+1}!")
            goal_reached = True
            break
            
        # Check collision
        for obs in obstacles:
            if 'position' in obs and 'radius' in obs:
                dist = np.linalg.norm(x[:2] - obs['position'])
                if dist <= obs['radius']:
                    print(f"\nCollision! {controller_type} collided at step {step+1}!")
                    collision = True
                    break
            elif 'position' in obs and 'width' in obs:
                # Rectangular obstacle
                rect_x, rect_y = obs['position']
                rect_w = obs['width']
                rect_h = obs['height']
                
                # Check if inside rectangle
                if (rect_x <= x[0] <= rect_x + rect_w and 
                    rect_y <= x[1] <= rect_y + rect_h):
                    print(f"\nCollision! {controller_type} collided at step {step+1}!")
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
                print(f"\nEarly termination: {controller_type} stuck for too long!")
                break
    
    # Print newline to end "Step X/Y" print
    print()
    
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
    
    # Calculate average computation time
    avg_comp_time = np.mean(computation_times)
    
    # Organize results
    result = {
        'controller': controller_type,
        'scenario': scenario_type,
        'success': goal_reached,
        'collision': collision,
        'stuck': stuck_counter >= 50,
        'steps': total_steps,
        'time': elapsed_time,
        'path_length': path_length,
        'control_energy': control_energy,
        'final_dist': final_dist,
        'avg_comp_time': avg_comp_time,
        'states': states_array,
        'controls': controls_array
    }
    
    print(f"\n{controller_type} Results on {scenario_type}:")
    print(f"Success: {goal_reached}")
    print(f"Collision: {collision}")
    print(f"Steps: {total_steps}")
    print(f"Path Length: {path_length:.2f}")
    print(f"Control Energy: {control_energy:.2f}")
    print(f"Final Distance: {final_dist:.2f}")
    print(f"Avg Computation Time: {avg_comp_time*1000:.2f} ms")
    
    return result

def parallel_experiment(params):
    """
    Run experiment function for parallel processing.
    
    Args:
        params: Tuple of (controller_type, scenario_type, visualize, max_steps, kwargs)
        
    Returns:
        result: Experiment result
    """
    controller_type, scenario_type, visualize, max_steps, kwargs = params
    return run_experiment(controller_type, scenario_type, visualize, max_steps, **kwargs)

def run_full_benchmark():
    """Run full benchmark comparing all controllers on all scenarios"""
    # Define controllers to benchmark
    controllers = ['MPPI', 'MA_MPPI', 'iLQR']
    
    # Define scenarios to benchmark
    scenarios = ['trap', 'maze', 'dynamic']
    
    # Number of runs per configuration for statistical significance
    runs_per_config = 10
    
    # Create results directory
    create_directory("results/benchmarks")
    
    # Create benchmark metrics tracker
    metrics = PerformanceMetrics()
    
    # List to collect all experiment parameters
    all_params = []
    
    for run in range(runs_per_config):
        for controller in controllers:
            for scenario in scenarios:
                # Special parameters for different controllers
                kwargs = {}
                
                if controller == 'MPPI':
                    kwargs['num_samples'] = 200
                    kwargs['temperature'] = 0.5
                elif controller == 'MA_MPPI':
                    kwargs['num_samples'] = 200
                    kwargs['temperature'] = 0.5
                    kwargs['max_features'] = 100
                elif controller == 'iLQR':
                    kwargs['max_iterations'] = 10
                
                # Add randomized seed
                kwargs['seed'] = run + 42
                
                # Disable visualization for batch runs
                visualize = False
                
                # Add parameters to list
                all_params.append((controller, scenario, visualize, None, kwargs))
    
    # Run experiments in parallel
    print(f"Running {len(all_params)} experiments in parallel...")
    
    with Pool(processes=min(os.cpu_count(), 8)) as pool:
        results = pool.map(parallel_experiment, all_params)
    
    # Record all results
    for result in results:
        metrics.record_experiment(result['controller'], result['scenario'], result)
    
    # Generate comparison report
    metrics.generate_comparison_report(controllers, scenarios)
    
    # Visualize comparison
    metrics.visualize_comparison(controllers)
    
    # Save all results
    metrics.save_results()
    
    print("\nBenchmark complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run experiments with different controllers and scenarios")
    parser.add_argument("--controller", type=str, default="MA_MPPI", choices=["MPPI", "MA_MPPI", "iLQR"],
                        help="Controller to use")
    parser.add_argument("--scenario", type=str, default="trap", choices=["trap", "maze", "dynamic"],
                        help="Scenario to use")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--steps", type=int, default=None, help="Maximum steps")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create results directory
    create_directory("results")
    
    if args.benchmark:
        run_full_benchmark()
    else:
        # Run single experiment
        result = run_experiment(
            args.controller, 
            args.scenario, 
            not args.no_vis, 
            args.steps
        )
        
        # Save result
        save_experiment_results(result, f"results/{args.scenario}_{args.controller}.npy")

if __name__ == "__main__":
    main()
