import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch

class BasicMPPI:
    """Basic MPPI (Model Predictive Path Integral) Controller"""
    
    def __init__(self, dynamics_function, running_cost, terminal_cost=None, 
                 state_dim=4, control_dim=2, horizon=15, num_samples=100, 
                 noise_sigma=None, lambda_=1.0, temperature=0.5,
                 u_min=None, u_max=None):
        """Initialize basic MPPI controller"""
        self.dynamics = dynamics_function
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.temperature = temperature
        
        # Control constraints
        self.u_min = u_min if u_min is not None else -np.ones(control_dim) * float('inf')
        self.u_max = u_max if u_max is not None else np.ones(control_dim) * float('inf')
        
        # Noise covariance
        self.noise_sigma = noise_sigma if noise_sigma is not None else np.eye(control_dim)
        
        # Initialize control sequence
        self.U = np.zeros((horizon, control_dim))
        
        # Trajectory information
        self.best_trajectory = None
        self.all_trajectories = None
        self.trajectory_costs = None
        
        # Time step
        self.dt = 0.1
    
    def compute_control(self, x0, goal=None, obstacles=None):
        """Compute control input"""
        # Generate noise samples
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.control_dim),
            cov=self.noise_sigma,
            size=(self.num_samples, self.horizon)
        ).reshape(self.num_samples, self.horizon, self.control_dim)
        
        # Initialize arrays
        costs = np.zeros(self.num_samples)
        rollout_states = np.zeros((self.num_samples, self.horizon + 1, self.state_dim))
        rollout_controls = np.zeros((self.num_samples, self.horizon, self.control_dim))
        
        # Save initial state
        rollout_states[:, 0, :] = x0
        
        # Execute forward simulation
        for k in range(self.num_samples):
            x = x0.copy()
            total_cost = 0.0
            
            for t in range(self.horizon):
                # Calculate control input
                u = self.U[t] + noise[k, t]
                u = np.clip(u, self.u_min, self.u_max)
                
                # Save control
                rollout_controls[k, t] = u
                
                # Calculate cost
                step_cost = self.running_cost(x, u, goal, obstacles)
                total_cost += step_cost
                
                # Update state
                x_next = self.dynamics(x, u, self.dt)
                x = x_next.copy()
                
                # Save state
                rollout_states[k, t+1, :] = x
            
            # Terminal cost
            if self.terminal_cost is not None and goal is not None:
                total_cost += self.terminal_cost(x, goal)
            
            # Save total cost
            costs[k] = total_cost
        
        # Save trajectory data
        self.all_trajectories = rollout_states
        self.trajectory_costs = costs
        
        # Find best trajectory
        best_idx = np.argmin(costs)
        self.best_trajectory = rollout_states[best_idx]
        
        # Update control sequence using softmax weights
        beta = 1.0 / max(0.1, self.temperature)
        weights = np.exp(-beta * (costs - np.min(costs)))
        weights = weights / np.sum(weights)
        
        # Calculate new control sequence
        weighted_controls = np.sum(weights[:, None, None] * rollout_controls, axis=0)
        
        # Update control sequence
        self.U = weighted_controls
        
        # Apply control constraints
        self.U = np.clip(self.U, self.u_min, self.u_max)
        
        # Output first control
        control = self.U[0].copy()
        
        # Roll window forward
        self.U = np.vstack([self.U[1:], np.zeros((1, self.control_dim))])
        
        return control


class MultiScalePotentialMPPI:
    """Simplified Multi-scale Potential MPPI Controller"""
    
    def __init__(self, dynamics_function, running_cost, terminal_cost=None, 
                 state_dim=4, control_dim=2, horizon=15, num_samples=100, 
                 noise_sigma=None, temperature=0.5, u_min=None, u_max=None):
        """Initialize controller"""
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # Noise and control parameters
        self.sigma = noise_sigma if noise_sigma is not None else np.eye(control_dim)
        self.temperature = temperature
        self.u_min = u_min if u_min is not None else -np.ones(control_dim) * 2.0
        self.u_max = u_max if u_max is not None else np.ones(control_dim) * 2.0
        
        # Create MPPI controller
        self.mppi = BasicMPPI(
            dynamics_function=dynamics_function,
            running_cost=self.potential_cost,
            terminal_cost=terminal_cost,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon,
            num_samples=num_samples,
            noise_sigma=self.sigma,
            temperature=temperature,
            u_min=u_min,
            u_max=u_max
        )
        
        # Save base cost function
        self.base_cost = running_cost
        
        # Position history and state
        self.position_history = deque(maxlen=20)
        self.velocity_history = deque(maxlen=10)
        self.control_history = deque(maxlen=5)
        self.goal_distance_history = deque(maxlen=10)
        
        # Stuck detection
        self.stuck = {
            'is_stuck': False,
            'duration': 0,
            'variance': 1.0
        }
        
        # Potential field memory
        self.memory = {
            'grid_size': 0.2,
            'stuck_positions': [],  # List of stuck positions
            'stuck_strengths': [],  # List of stuck strengths
            'trap_locations': [],   # List of trap centers
            'trap_radii': [],       # List of trap radii
            'trap_strengths': []    # List of trap strengths
        }
        
        # Potential field weights - φ_total = w_goal*φ_goal + w_memory*φ_memory + w_trap*φ_trap
        self.weights = {
            'goal': 1.0,     # Goal potential weight
            'memory': 0.3,   # Memory potential weight
            'trap': 0.0      # Trap correction potential weight
        }
        
        # Direction consistency
        self.consistency = {
            'enabled': True,
            'max_angle': np.pi/3,  # Maximum angle change
            'weight': 0.7          # Blending weight
        }
        
        # Visualization data
        self.viz = {
            'potentials': None,
            'escape_dir': None,
            'current_potential': 0
        }
        
        # Step counter
        self.step_count = 0
        
        print("Simplified Multi-scale Potential MPPI initialized")
    
    def check_stuck(self, position):
        """Detect if stuck"""
        # Need at least 5 position points
        if len(self.position_history) < 5:
            return False
        
        # Calculate variance of recent positions
        positions = np.array(list(self.position_history)[-5:])
        variance = np.var(positions, axis=0).sum()
        self.stuck['variance'] = variance
        
        # Determine if stuck
        is_stuck = variance < 0.01
        
        # Update stuck status
        if is_stuck:
            if not self.stuck['is_stuck']:
                print(f"⚠️ Stuck detected! Position variance: {variance:.4f}")
            self.stuck['is_stuck'] = True
            self.stuck['duration'] += 1
            
            # Record stuck position
            self.memory['stuck_positions'].append(position.copy())
            self.memory['stuck_strengths'].append(min(5.0, self.stuck['duration']/3))
            
            # Detect trap
            self.detect_trap(position)
        else:
            if self.stuck['is_stuck']:
                print(f"✓ Escaped from stuck state! Position variance: {variance:.4f}")
            self.stuck['is_stuck'] = False
            self.stuck['duration'] = 0
        
        return is_stuck
    
    def detect_trap(self, position):
        """Detect if a trap area is formed"""
        # Trap detection parameters
        min_stuck_count = 3
        detection_radius = 2.0
        
        # If trap already exists, update strength
        for i, center in enumerate(self.memory['trap_locations']):
            dist = np.linalg.norm(position - center)
            if dist < self.memory['trap_radii'][i]:
                # Update existing trap
                self.memory['trap_strengths'][i] += 0.5
                
                # Update center position (weighted average)
                weight = 0.2
                self.memory['trap_locations'][i] = (1-weight)*center + weight*position
                return
        
        # Check if there are multiple stuck points nearby
        nearby_stuck = 0
        for stuck_pos in self.memory['stuck_positions']:
            if np.linalg.norm(position - stuck_pos) < detection_radius:
                nearby_stuck += 1
        
        # If there are enough stuck points nearby, create a new trap
        if nearby_stuck >= min_stuck_count:
            self.memory['trap_locations'].append(position.copy())
            self.memory['trap_radii'].append(detection_radius)
            self.memory['trap_strengths'].append(1.0)
            print(f"🔍 New trap detected! Center: ({position[0]:.1f}, {position[1]:.1f})")
    
    def update_weights(self):
        """Update potential field weights"""
        if self.stuck['is_stuck']:
            # Calculate sigmoid value
            stuck_time = self.stuck['duration'] 
            sigmoid = 1.0 / (1.0 + np.exp(-0.2 * (stuck_time - 5)))
            
            # Update weights - As stuck time increases, reduce goal attraction, increase escape weight
            self.weights['goal'] = max(0.3, 1.0 - sigmoid*0.7)
            self.weights['memory'] = sigmoid
            self.weights['trap'] = min(2.0, stuck_time/5)
            
            # Increase temperature to promote exploration
            self.mppi.temperature = min(1.0, self.temperature * (1 + stuck_time/10))
        else:
            # Normal mode
            self.weights['goal'] = 1.0
            self.weights['memory'] = 0.3
            self.weights['trap'] = 0.0
            self.mppi.temperature = self.temperature
    
    def goal_potential(self, position, goal):
        """Goal potential field - Attractive potential"""
        # Simple quadratic potential field
        dist = np.linalg.norm(position - goal[:2])
        return 10.0 * dist**2
    
    def memory_potential(self, position):
        """Memory potential field - Repulsive potential"""
        # Repulsive potential from all stuck points
        potential = 0.0
        max_influence = 3.0  # Maximum influence radius
        
        for i, stuck_pos in enumerate(self.memory['stuck_positions']):
            dist = np.linalg.norm(position - stuck_pos)
            if dist < max_influence:
                # Repulsive potential with decay
                strength = self.memory['stuck_strengths'][i]
                potential += strength * 5.0 * (1.0 - dist/max_influence)**2
        
        return potential
    
    def trap_potential(self, position, goal):
        """Trap correction potential field - Escape potential"""
        # If no traps, return zero and None
        if not self.memory['trap_locations']:
            return 0.0, None
        
        # Find nearest trap
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, center in enumerate(self.memory['trap_locations']):
            dist = np.linalg.norm(position - center)
            if dist < min_dist and dist < self.memory['trap_radii'][i]:
                min_dist = dist
                nearest_idx = i
        
        # If not in any trap
        if nearest_idx < 0:
            return 0.0, None
        
        # Calculate escape direction
        center = self.memory['trap_locations'][nearest_idx]
        radius = self.memory['trap_radii'][nearest_idx]
        strength = self.memory['trap_strengths'][nearest_idx]
        
        # Calculate basic escape direction (away from trap center)
        escape_dir = position - center
        if np.linalg.norm(escape_dir) < 0.001:
            # Random direction
            angle = np.random.uniform(0, 2*np.pi)
            escape_dir = np.array([np.cos(angle), np.sin(angle)])
        else:
            escape_dir = escape_dir / np.linalg.norm(escape_dir)
        
        # Calculate goal direction
        goal_dir = goal[:2] - position
        if np.linalg.norm(goal_dir) > 0.001:
            goal_dir = goal_dir / np.linalg.norm(goal_dir)
            
            # If escape direction is similar to goal direction, enhance
            dot = np.dot(escape_dir, goal_dir)
            if dot > 0:
                # Maintain direction, enhance strength
                trap_value = strength * 15.0 * (1.0 - min_dist/radius) * (1.0 + dot)
            else:
                # Find a direction approximately perpendicular to the goal
                perp_dir = np.array([-goal_dir[1], goal_dir[0]])  # Perpendicular vector
                if np.dot(perp_dir, escape_dir) < 0:
                    perp_dir = -perp_dir  # Choose closer perpendicular direction
                
                # Blend escape direction
                escape_dir = 0.7 * perp_dir + 0.3 * escape_dir
                escape_dir = escape_dir / np.linalg.norm(escape_dir)
                trap_value = strength * 15.0 * (1.0 - min_dist/radius) * 1.5
        else:
            trap_value = strength * 15.0 * (1.0 - min_dist/radius)
        
        return trap_value, escape_dir
    
    def potential_cost(self, x, u, goal, obstacles=None):
        """Multi-scale potential field enhanced cost function"""
        # Base cost
        base_cost = self.base_cost(x, u, goal, obstacles)
        
        # Get current position
        position = x[:2]
        
        # Three components
        goal_pot = self.goal_potential(position, goal)
        memory_pot = self.memory_potential(position)
        trap_pot, _ = self.trap_potential(position, goal)
        
        # Combine total potential
        pot_cost = (
            self.weights['goal'] * goal_pot + 
            self.weights['memory'] * memory_pot + 
            self.weights['trap'] * trap_pot
        )
        
        # Save current potential for visualization
        self.viz['current_potential'] = pot_cost
        
        return base_cost + pot_cost
    
    def apply_direction_consistency(self, control):
        """Apply direction consistency constraint"""
        if not self.consistency['enabled'] or len(self.control_history) == 0:
            return control
        
        # Get previous control
        prev_control = self.control_history[-1]
        
        # Calculate angle change
        prev_norm = np.linalg.norm(prev_control)
        curr_norm = np.linalg.norm(control)
        
        if prev_norm < 0.001 or curr_norm < 0.001:
            return control
        
        # Calculate direction vectors
        prev_dir = prev_control / prev_norm
        curr_dir = control / curr_norm
        
        # Calculate angle
        cos_angle = np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # If angle change is too large
        if angle > self.consistency['max_angle']:
            # Limit direction change
            # Build rotation matrix
            max_angle = self.consistency['max_angle']
            # Determine rotation direction
            cross_product = np.cross(np.append(prev_dir, 0), np.append(curr_dir, 0))[2]
            
            if cross_product >= 0:
                rot = np.array([
                    [np.cos(max_angle), -np.sin(max_angle)],
                    [np.sin(max_angle), np.cos(max_angle)]
                ])
            else:
                rot = np.array([
                    [np.cos(max_angle), np.sin(max_angle)],
                    [-np.sin(max_angle), np.cos(max_angle)]
                ])
            
            # Rotate previous direction
            limited_dir = rot @ prev_dir
            limited_control = limited_dir * curr_norm
            
            # Blend
            w = self.consistency['weight']
            blended = w * limited_control + (1-w) * control
            return blended
        
        return control
    
    def compute_control(self, x0, goal=None, obstacles=None):
        """Compute control input"""
        self.step_count += 1
        
        # Update history data
        self.position_history.append(x0[:2].copy())
        if len(x0) > 2:
            self.velocity_history.append(x0[2:4].copy())
        
        # Calculate distance to goal
        goal_dist = np.linalg.norm(x0[:2] - goal[:2])
        self.goal_distance_history.append(goal_dist)
        
        # Detect stuck state
        self.check_stuck(x0[:2])
        
        # Update weights
        self.update_weights()
        
        # Save escape direction for visualization
        _, escape_dir = self.trap_potential(x0[:2], goal)
        self.viz['escape_dir'] = escape_dir
        
        # Calculate MPPI control
        control = self.mppi.compute_control(x0, goal, obstacles)
        
        # Apply direction consistency constraint
        control = self.apply_direction_consistency(control)
        
        # Record control history
        self.control_history.append(control.copy())
        
        # Print current state
        mode = "Normal Navigation"
        if self.stuck['is_stuck']:
            mode = f"Potential Escape ({self.stuck['duration']} steps)"
        
        print(f"Mode: {mode}, Temperature: {self.mppi.temperature:.2f}, " + 
              f"Weights: Goal={self.weights['goal']:.1f}, " +
              f"Memory={self.weights['memory']:.1f}, " +
              f"Trap={self.weights['trap']:.1f}")
        
        return control


def dynamics_2d(x, u, dt=0.1):
    """Simple 2D dynamics"""
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
    """Default running cost"""
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
            if 'pos' in obs and 'radius' in obs:
                dist = np.linalg.norm(x[:2] - obs['pos']) - obs['radius']
                
                if dist < 0:  # Collision
                    obstacle_cost += 1000.0
                elif dist < 0.5:  # Extremely dangerous
                    obstacle_cost += 200.0 * (1 - dist/0.5)
                elif dist < 1.5:  # Danger zone
                    obstacle_cost += 50.0 * (1 - dist/1.5)
    
    return position_cost + velocity_cost + control_cost + obstacle_cost


def default_terminal_cost(x, goal):
    """Default terminal cost"""
    position_error = x[:2] - goal[:2]
    position_cost = 50.0 * np.sum(position_error**2)
    velocity_cost = 10.0 * np.sum(x[2:]**2)
    
    return position_cost + velocity_cost


class SimplifiedExperiment:
    """Simplified experimental framework"""
    
    def __init__(self):
        # Create deep U-shaped trap scenario
        self.scenario = self.create_deep_u_trap_scenario()
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/visualizations", exist_ok=True)

    def create_deep_u_trap_scenario(self):
        """Create simplified obstacle scenario"""
        return {
            'name': 'Simplified-Trap',
            'start': np.array([5.0, 2.0, 0.0, 0.0]),
            'goal': np.array([5.0, 9.0, 0.0, 0.0]),
            'obstacles': [
                # Just keep bottom wall, forming a row of obstacles
                {'pos': np.array([3.5, 6.0]), 'radius': 0.5},
                {'pos': np.array([4.5, 6.0]), 'radius': 0.5},
                {'pos': np.array([5.5, 6.0]), 'radius': 0.5},
                {'pos': np.array([6.5, 6.0]), 'radius': 0.5},
                
                # Adjust gap size as needed
                # Leave a smaller gap somewhere in the middle
                {'pos': np.array([4.0, 5.1]), 'radius': 0.5},
                {'pos': np.array([6.0, 5.1]), 'radius': 0.5},
            ],
            'max_steps': 300
        }
    
    def run_controller(self, controller, controller_name, max_steps=None):
        """Run controller"""
        scenario = self.scenario
        
        # Extract scenario information
        start = scenario['start'].copy()
        goal = scenario['goal'].copy()
        obstacles = [obs.copy() for obs in scenario['obstacles']]
        
        # Set maximum steps
        if max_steps is None:
            max_steps = scenario['max_steps']
        
        # Initialize
        x = start.copy()
        states_history = [x.copy()]
        controls_history = []
        frames = []  # For generating animations
        
        # Create persistent window
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.show(block=False)  # Show but don't block
        
        # Timing
        start_time = time.time()
        
        # Simulation loop
        collision = False
        goal_reached = False
        stuck_counter = 0
        
        for step in range(max_steps):
            print(f"\n--- {scenario['name']} - {controller_name} - Step {step+1}/{max_steps} ---")
            
            # Calculate control
            u = controller.compute_control(x, goal, obstacles)
            controls_history.append(u)
            
            # Apply control
            x_next = dynamics_2d(x, u)
            x = x_next.copy()
            states_history.append(x.copy())
            
            # Visualization - Using persistent window
            ax.clear()  # Just clear content, don't close window
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in obstacles:
                if 'pos' in obs and 'radius' in obs:
                    circle = plt.Circle(obs['pos'], obs['radius'], color='blue', alpha=0.7)
                    ax.add_patch(circle)
            
            # Draw trajectory
            states_array = np.array(states_history)
            ax.plot(states_array[:, 0], states_array[:, 1], 'g-', linewidth=2, label='Actual Trajectory')
            
            # If multi-scale potential MPPI, draw potential field
            if hasattr(controller, 'memory'):
                # Draw trap areas
                for i, center in enumerate(controller.memory['trap_locations']):
                    circle = plt.Circle(
                        center, controller.memory['trap_radii'][i],
                        color='red', alpha=0.2, fill=True
                    )
                    ax.add_patch(circle)
                    ax.text(
                        center[0], center[1],
                        f"{controller.memory['trap_strengths'][i]:.1f}",
                        color='black', fontsize=8,
                        ha='center', va='center'
                    )
                
                # Draw stuck points
                for pos in controller.memory['stuck_positions']:
                    ax.plot(pos[0], pos[1], 'rx', markersize=5, alpha=0.6)
                
                # Draw escape direction
                if controller.viz['escape_dir'] is not None:
                    ax.arrow(
                        x[0], x[1],
                        controller.viz['escape_dir'][0], controller.viz['escape_dir'][1],
                        head_width=0.2, head_length=0.3,
                        fc='red', ec='red',
                        alpha=0.7,
                        length_includes_head=True
                    )
            
            # Draw predicted trajectory
            if hasattr(controller, 'mppi') and hasattr(controller.mppi, 'best_trajectory'):
                ax.plot(controller.mppi.best_trajectory[:, 0], controller.mppi.best_trajectory[:, 1], 
                       'y--', linewidth=1.5, alpha=0.7, label='Predicted Trajectory')
            
            # Draw start and goal
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start Position')
            ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal Position')
            
            # Draw current position
            ax.plot(x[0], x[1], 'ko', markersize=8, label='Current Position')
            
            # Display status information
            status_text = ""
            
            if hasattr(controller, 'stuck'):
                if controller.stuck['is_stuck']:
                    status_text += f"Stuck! ({controller.stuck['duration']} steps)\n"
                
                if hasattr(controller, 'weights'):
                    w = controller.weights
                    status_text += f"Weights: Goal={w['goal']:.1f}, Memory={w['memory']:.1f}, Trap={w['trap']:.1f}\n"
                
                if hasattr(controller, 'mppi'):
                    status_text += f"Temperature: {controller.mppi.temperature:.2f}\n"
            
            if status_text:
                ax.text(0.02, 0.02, status_text, transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.7))
            
            goal_dist = np.linalg.norm(x[:2] - goal[:2])
            ax.set_title(f'{scenario["name"]} - {controller_name} - Step: {step} - Distance to Goal: {goal_dist:.2f}')
            ax.legend(loc='upper left')
            
            # Update and capture frame
            fig.canvas.draw()
            plt.pause(0.01)
            
            # Capture current frame
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            
            # Check goal reached
            if goal_dist < 0.3:
                print(f"\nSuccess! {controller_name} reached goal at step {step+1}!")
                goal_reached = True
                break
                
            # Check collision
            for obs in obstacles:
                if 'pos' in obs and 'radius' in obs:
                    if np.linalg.norm(x[:2] - obs['pos']) <= obs['radius']:
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
                
                if stuck_counter >= 50:
                    print(f"\nEarly termination: {controller_name} stuck for too long!")
                    break
        
        # Close figure window
        plt.ioff()
        plt.close(fig)
        
        # Calculate result metrics
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
            'frames': frames,
            'controller': controller
        }
        
        # Generate animation
        try:
            if len(frames) > 5:
                self.save_animation(frames, f"results/{scenario['name']}_{controller_name}.gif")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        
        return result
    
    def save_animation(self, frames, filename, fps=10):
        """Save animation"""
        import imageio
        
        # Take every few frames to avoid large files
        step = max(1, len(frames) // 100)
        frames_subset = frames[::step]
        
        # Limit number of frames
        if len(frames_subset) > 100:
            frames_subset = frames_subset[:100]
        
        # Save as GIF
        imageio.mimsave(filename, frames_subset, fps=fps)
        print(f"Animation saved to {filename}")
    
    def run_comparison(self):
        """Run comparison experiment"""
        # Control limits
        u_min = np.array([-2.0, -2.0])
        u_max = np.array([2.0, 2.0])
        
        # Create basic MPPI controller
        basic_mppi = BasicMPPI(
            dynamics_function=dynamics_2d,
            running_cost=default_running_cost,
            terminal_cost=default_terminal_cost,
            state_dim=4,
            control_dim=2,
            horizon=30,
            num_samples=200,
            noise_sigma=np.diag([3.0, 3.0]),
            lambda_=13.0,
            temperature=15.0,
            u_min=u_min,
            u_max=u_max
        )
        
        # Create multi-scale potential MPPI controller
        ms_mppi = MultiScalePotentialMPPI(
            dynamics_function=dynamics_2d,
            running_cost=default_running_cost,
            terminal_cost=default_terminal_cost,
            state_dim=4,
            control_dim=2,
            horizon=15,
            num_samples=200,
            noise_sigma=np.diag([1.0, 1.0]),
            temperature=0.5,
            u_min=u_min,
            u_max=u_max
        )
        
        # Run basic MPPI
        print("\n>> Running Basic MPPI Controller...")
        basic_result = self.run_controller(basic_mppi, "Basic MPPI")
        
        # Run multi-scale potential MPPI
        print("\n>> Running Multi-scale Potential MPPI Controller...")
        ms_result = self.run_controller(ms_mppi, "Multi-scale Potential MPPI")
        
        # Compare and display results
        self.compare_results(basic_result, ms_result)
        
        # Generate special visualization charts
        self.generate_special_visualizations(ms_result)
        
        return basic_result, ms_result
    
    def compare_results(self, basic_result, ms_result):
        """Compare and display results"""
        print("\nResult Comparison:")
        print(f"{'Metric':<15} {'Basic MPPI':<15} {'Multi-scale Potential MPPI':<20}")
        print("-"*55)
        
        # Success status
        basic_status = "Success" if basic_result['success'] else ("Collision" if basic_result['collision'] else ("Stuck" if basic_result['stuck'] else "Incomplete"))
        ms_status = "Success" if ms_result['success'] else ("Collision" if ms_result['collision'] else ("Stuck" if ms_result['stuck'] else "Incomplete"))
        
        print(f"{'Status':<15} {basic_status:<15} {ms_status:<20}")
        print(f"{'Steps':<15} {basic_result['steps']:<15} {ms_result['steps']:<20}")
        print(f"{'Path Length':<15} {basic_result['path_length']:<15.2f} {ms_result['path_length']:<20.2f}")
        print(f"{'Control Energy':<15} {basic_result['control_energy']:<15.2f} {ms_result['control_energy']:<20.2f}")
        print(f"{'Final Distance':<15} {basic_result['final_dist']:<15.2f} {ms_result['final_dist']:<20.2f}")
        
        # Draw trajectory comparison chart
        plt.figure(figsize=(15, 6))
        
        # Draw scenario and obstacles
        obstacles = self.scenario['obstacles']
        
        for i, title in enumerate(["Standard MPPI", "Adaptive Potential MPPI"]):
            plt.subplot(1, 2, i+1)
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in obstacles:
                circle = plt.Circle(obs['pos'], obs['radius'], color='blue', alpha=0.7)
                plt.gca().add_patch(circle)
            
            # Draw start and goal
            start = self.scenario['start']
            goal = self.scenario['goal']
            
            plt.plot(start[0], start[1], 'go', markersize=10, label='Start Position')
            plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal Position')
            
            # Draw trajectory
            result = basic_result if i == 0 else ms_result
            status = "Success" if result['success'] else ("Collision" if result['collision'] else ("Stuck" if result['stuck'] else "Incomplete"))
            
            plt.plot(result['states'][:, 0], result['states'][:, 1], 'g-', linewidth=2, label='Robot Trajectory')
            plt.title(f"{title}: {status}, steps={result['steps']}")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/trajectory_comparison.png", dpi=150)
        plt.close()
        
        print("\nResult comparison saved to 'results/trajectory_comparison.png'")
    
    def generate_special_visualizations(self, ms_result):
        """Generate special visualization charts"""
        # 1. Feature detection visualization - Topological feature identification
        self.visualize_topology_features(ms_result)
        
        # 2. Memory module visualization
        self.visualize_memory_module(ms_result)
        
        # 3. Potential phase change
        self.visualize_potential_phase(ms_result)
        
        print("\nThree special visualization charts have been generated and saved in the results/visualizations directory")


    def visualize_topology_features(self, ms_result):
        """Visualize topological feature identification"""
        controller = ms_result['controller']
        states_history = ms_result['states']
        obstacles = self.scenario['obstacles']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # === STEP 1: Compute potential field gradients and topological features ===
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        resolution = 0.1
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize data structures for different topological features
        Z_potential = np.zeros_like(X)  # Overall potential field
        grad_x = np.zeros_like(X)  # Gradient x component
        grad_y = np.zeros_like(Y)  # Gradient y component
        grad_magnitude = np.zeros_like(X)  # Gradient magnitude
        curvature = np.zeros_like(X)  # Gradient curvature
        
        # Create masks for different features
        obstacle_mask = np.zeros_like(X, dtype=bool)  # Physical obstacles
        local_minima_mask = np.zeros_like(X, dtype=bool)  # Local minima (potential traps)
        low_gradient_mask = np.zeros_like(X, dtype=bool)  # Low gradient regions
        high_curvature_mask = np.zeros_like(X, dtype=bool)  # High curvature regions
        
        # Threshold values
        theta_var = 0.01  # Stuck detection threshold
        theta_grad = 0.5  # Low gradient threshold
        theta_curv = np.pi/4  # High curvature threshold (45 degrees)
        
        # Mark obstacle areas
        for obs in obstacles:
            for i in range(len(x_grid)):
                for j in range(len(y_grid)):
                    if np.linalg.norm(np.array([X[j, i], Y[j, i]]) - obs['pos']) <= obs['radius']:
                        obstacle_mask[j, i] = True
                        Z_potential[j, i] = 100  # High potential in obstacles
        
        # Calculate potential and gradients across the grid
        goal = self.scenario['goal']
        
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                if obstacle_mask[j, i]:
                    continue  # Skip obstacle points
                
                pos = np.array([X[j, i], Y[j, i]])
                
                # Calculate goal-based potential
                dist_to_goal = np.linalg.norm(pos - goal[:2])
                goal_pot = 10.0 * dist_to_goal**2
                
                # Calculate memory-based potential
                memory_pot = 0.0
                for k, stuck_pos in enumerate(controller.memory['stuck_positions']):
                    dist = np.linalg.norm(pos - stuck_pos)
                    if dist < 3.0:  # Max influence radius
                        strength = controller.memory['stuck_strengths'][k]
                        memory_pot += strength * 5.0 * (1.0 - dist/3.0)**2
                
                # Calculate trap potential
                trap_pot = 0.0
                for k, center in enumerate(controller.memory['trap_locations']):
                    dist = np.linalg.norm(pos - center)
                    if dist < controller.memory['trap_radii'][k]:
                        strength = controller.memory['trap_strengths'][k]
                        trap_pot += strength * 15.0 * (1.0 - dist/controller.memory['trap_radii'][k])
                        
                        # Mark this as a potential local minimum
                        if strength > 1.0:
                            local_minima_mask[j, i] = True
                
                # Total potential at this point
                total_pot = goal_pot + memory_pot + trap_pot
                Z_potential[j, i] = total_pot
                
                # Calculate numerical gradient at this point
                epsilon = 0.01
                # x-gradient
                if i > 0 and i < len(x_grid)-1:
                    grad_x[j, i] = (Z_potential[j, i+1] - Z_potential[j, i-1]) / (2*resolution)
                # y-gradient
                if j > 0 and j < len(y_grid)-1:
                    grad_y[j, i] = (Z_potential[j+1, i] - Z_potential[j-1, i]) / (2*resolution)
                
                # Calculate gradient magnitude
                grad_magnitude[j, i] = np.sqrt(grad_x[j, i]**2 + grad_y[j, i]**2)
                
                # Identify low gradient regions
                if grad_magnitude[j, i] < theta_grad:
                    low_gradient_mask[j, i] = True
        
        # Smooth the potential field for visualization
        Z_smooth = gaussian_filter(Z_potential, sigma=1.0)
        
        # Calculate normalized gradient fields for entire grid
        norm_grad_x = np.zeros_like(grad_x)
        norm_grad_y = np.zeros_like(grad_y)
        
        # Avoid division by zero by masking small magnitudes
        valid_mask = grad_magnitude > 0.001
        norm_grad_x[valid_mask] = grad_x[valid_mask] / grad_magnitude[valid_mask]
        norm_grad_y[valid_mask] = grad_y[valid_mask] / grad_magnitude[valid_mask]
        
        # Calculate curvature using a simplified method: average angle change with neighbors
        for i in range(1, len(x_grid)-1):
            for j in range(1, len(y_grid)-1):
                if obstacle_mask[j, i] or not valid_mask[j, i]:
                    continue
                    
                # Use neighboring gradient directions to estimate curvature
                angle_sum = 0
                count = 0
                
                current_dir = np.array([norm_grad_x[j, i], norm_grad_y[j, i]])
                
                for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                    if valid_mask[nj, ni]:
                        neighbor_dir = np.array([norm_grad_x[nj, ni], norm_grad_y[nj, ni]])
                        
                        # Calculate angle between current and neighbor gradient
                        cos_angle = np.clip(np.dot(current_dir, neighbor_dir), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angle_sum += angle
                        count += 1
                
                # Calculate average angle change
                if count > 0:
                    curvature[j, i] = angle_sum / count
                    
                    # Mark high curvature regions
                    if curvature[j, i] > theta_curv:
                        high_curvature_mask[j, i] = True
        
        # === STEP 2: Create the visualization ===
        
        # First draw the obstacles
        for obs in obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='blue', alpha=0.7)
            ax.add_patch(circle)
        
        # Create a custom colormap for topological features
        colors = [
            (0.0, 0.0, 0.8, 0.7),   # Blue for obstacles
            (0.8, 0.0, 0.0, 0.8),   # Dark red for local minima
            (1.0, 0.3, 0.3, 0.7),   # Red for high potential
            (1.0, 0.7, 0.0, 0.6),   # Orange for medium potential
            (1.0, 1.0, 0.0, 0.5),   # Yellow for low gradient areas
            (0.0, 0.7, 0.0, 0.4)    # Green for good gradient areas
        ]
        topo_cmap = LinearSegmentedColormap.from_list('topo_cmap', colors, N=100)
        
        # Draw the potential field with features
        cf = ax.contourf(X, Y, Z_smooth, levels=20, cmap=topo_cmap, alpha=0.6)
        
        # Add labeled contour lines (keep these as requested)
        cs = ax.contour(X, Y, Z_smooth, levels=10, colors='purple', linewidths=0.8, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
        
        # Highlight the local minima regions
        local_minima_contour = ax.contour(X, Y, local_minima_mask.astype(float), levels=[0.5], 
                                        colors='red', linewidths=2.0, linestyles='solid')
        
        # Highlight low gradient regions
        low_grad_contour = ax.contour(X, Y, low_gradient_mask.astype(float), levels=[0.5], 
                                    colors='yellow', linewidths=1.5, linestyles='dashed')
        
        # Highlight high curvature regions
        high_curv_contour = ax.contour(X, Y, high_curvature_mask.astype(float), levels=[0.5], 
                                    colors='purple', linewidths=1.5, linestyles='dotted')
        
        # Draw gradient vector field (showing paths of steepest descent)
        # Downsample the grid for clearer visualization
        step = 8  # Display vectors every step points
        skip = (slice(None, None, step), slice(None, None, step))
        
        # Scale vectors for better visualization
        scale = np.max(grad_magnitude[skip]) * 10 if np.max(grad_magnitude[skip]) > 0 else 1
        quiver = ax.quiver(X[skip], Y[skip], -grad_x[skip], -grad_y[skip], 
                        grad_magnitude[skip], cmap='viridis', scale=scale, 
                        alpha=0.6, width=0.003)
        
        # Add a colorbar for the gradient magnitude
        gradient_cbar = plt.colorbar(quiver, ax=ax, shrink=0.6)
        gradient_cbar.set_label('Gradient Magnitude')
        
        # Draw detected stuck points
        for pos in controller.memory['stuck_positions']:
            ax.plot(pos[0], pos[1], 'rx', markersize=6, alpha=0.7)
        
        # Draw trajectory
        ax.plot(states_history[:, 0], states_history[:, 1], 'g-', linewidth=2)
        
        # Draw start and goal
        start = self.scenario['start']
        goal = self.scenario['goal']
        ax.plot(start[0], start[1], 'go', markersize=10)
        ax.plot(goal[0], goal[1], 'ro', markersize=10)
        
        # === STEP 3: Analyze the trajectory for feature discovery (no text annotations) ===
        
        # Extract variance for trajectory segments
        variances = []
        window_size = 5
        for i in range(len(states_history) - window_size):
            recent = states_history[i:i+window_size, :2]
            var = np.var(recent, axis=0).sum()
            variances.append(var)
        
        # Calculate gradient changes along the trajectory
        grad_changes = []
        last_grad = None
        for i in range(1, len(states_history)-1):
            pos = states_history[i, :2]
            
            # Estimate gradient at this position
            i_idx = np.argmin(np.abs(x_grid - pos[0]))
            j_idx = np.argmin(np.abs(y_grid - pos[1]))
            
            if i_idx > 0 and i_idx < len(x_grid)-1 and j_idx > 0 and j_idx < len(y_grid)-1:
                current_grad = np.array([grad_x[j_idx, i_idx], grad_y[j_idx, i_idx]])
                current_grad_norm = np.linalg.norm(current_grad)
                
                if current_grad_norm > 0.001 and last_grad is not None:
                    # Calculate angle between consecutive gradients
                    last_grad_norm = np.linalg.norm(last_grad)
                    if last_grad_norm > 0.001:
                        cos_angle = np.clip(np.dot(current_grad, last_grad) / (current_grad_norm * last_grad_norm), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        grad_changes.append((i, angle))
                
                last_grad = current_grad
        
        # Mark important points on the trajectory without text annotations
        # Low variance points (stuck)
        if variances:
            low_var_indices = np.where(np.array(variances) < theta_var)[0]
            for idx in low_var_indices:
                if idx > 0 and idx < len(states_history) - 1:
                    ax.plot(states_history[idx, 0], states_history[idx, 1], 'mo', markersize=5, alpha=0.5)
        
        # High gradient change points (curvature)
        for idx, angle in grad_changes:
            if angle > theta_curv:
                ax.plot(states_history[idx, 0], states_history[idx, 1], 'co', markersize=5, alpha=0.5)
        
        # Set chart properties
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Topological Feature Detection', fontsize=14)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Create comprehensive legend with all visual elements explained
        from matplotlib.patches import Circle, Rectangle, Patch
        from matplotlib.lines import Line2D
        
        # Create groupings for better organization in the legend
        legend_elements = [
            # Navigation elements
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Start Position'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Goal Position'),
            Line2D([0], [0], color='g', lw=2, label='Robot Trajectory'),
            Patch(facecolor='blue', alpha=0.7, label='Physical Obstacle'),
            
            # Color map and contours
            Patch(facecolor='red', alpha=0.5, label='High Potential Area'),
            Patch(facecolor='orange', alpha=0.5, label='Medium Potential Area'),
            Patch(facecolor='green', alpha=0.4, label='Low Potential Area'),
            Line2D([0], [0], color='purple', lw=1, label='Potential Contour Lines'),
            
            # Topological features
            Line2D([0], [0], color='red', lw=2, linestyle='solid', label='Local Minima Region (Trap)'),
            Line2D([0], [0], color='black', marker='>', markersize=6, linestyle='none', label='Gradient Direction'),
                        # Line2D([0], [0], color='yellow', lw=1.5, linestyle='dashed', label='Low Gradient Region (<0.5)'),
            # Line2D([0], [0], color='purple', lw=1.5, linestyle='dotted', label='High Curvature Region (>45°)'),
            # Trajectory analysis points
            Line2D([0], [0], marker='x', color='r', markersize=6, linestyle='none', label='Detected Stuck Point'),
            Line2D([0], [0], marker='o', color='m', markersize=5, linestyle='none', label='Low Variance Point')
        ]
        
        # Create a legend with clear grouping and formatting
        ax.legend(handles=legend_elements, loc='upper right', 
                title='Topological Features', framealpha=0.95,
                frameon=True, fancybox=True,  prop={'size': 8})
        
        
        
        # Save chart
        plt.tight_layout()
        plt.savefig("results/visualizations/topology_features.png", dpi=150)
        plt.close()
        
        print("Topology feature visualization saved")

    def visualize_memory_module(self, ms_result):
        """Visualize memory module influence"""
        controller = ms_result['controller']
        states_history = ms_result['states']
        obstacles = self.scenario['obstacles']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw obstacles
        for obs in obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='blue', alpha=0.7)
            ax.add_patch(circle)
        
        # Create grid for potential field vector field
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        resolution = 0.5  # Coarser grid for clearer vectors
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        U = np.zeros_like(X)  # x direction force
        V = np.zeros_like(Y)  # y direction force
        
        # Calculate memory potential field gradient (as vector field) for each grid point
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                pos = np.array([X[j, i], Y[j, i]])
                
                # Check if inside obstacle
                in_obstacle = False
                for obs in obstacles:
                    if np.linalg.norm(pos - obs['pos']) <= obs['radius']:
                        in_obstacle = True
                        break
                
                if in_obstacle:
                    continue
                
                # Calculate memory potential field gradient (opposite direction is repulsive force direction)
                gradient = np.zeros(2)
                
                # Repulsive force from stuck points
                for k, stuck_pos in enumerate(controller.memory['stuck_positions']):
                    dist = np.linalg.norm(pos - stuck_pos)
                    if dist < 3.0 and dist > 0.01:  # Avoid division by zero
                        direction = (pos - stuck_pos) / dist  # Direction away from stuck point
                        strength = controller.memory['stuck_strengths'][k]
                        magnitude = strength * 5.0 * (1 - dist/3.0)
                        gradient += direction * magnitude
                
                # Repulsive force from trap areas
                for k, center in enumerate(controller.memory['trap_locations']):
                    dist = np.linalg.norm(pos - center)
                    if dist < controller.memory['trap_radii'][k] and dist > 0.01:
                        direction = (pos - center) / dist
                        strength = controller.memory['trap_strengths'][k]
                        magnitude = strength * 15.0 * (1 - dist/controller.memory['trap_radii'][k])
                        gradient += direction * magnitude
                
                # Save force components
                U[j, i] = gradient[0]
                V[j, i] = gradient[1]
        
        # Calculate vector field magnitude
        magnitudes = np.sqrt(U**2 + V**2)
        mask = magnitudes > 0.1  # Only display meaningful vectors
        
        if np.any(mask):
            # Use a clearly distinct color for force vectors
            amber_cmap = plt.cm.YlOrBr
            
            # Normalize vectors for consistent arrow length while preserving direction
            norm_factor = 0.5  # Control arrow length
            U_norm = np.divide(U, magnitudes, out=np.zeros_like(U), where=magnitudes>0) * norm_factor
            V_norm = np.divide(V, magnitudes, out=np.zeros_like(V), where=magnitudes>0) * norm_factor
            
            # Draw vector field, color indicates strength
            quiver = ax.quiver(X[mask], Y[mask], U_norm[mask], V_norm[mask], 
                            magnitudes[mask], cmap=amber_cmap, scale=10, 
                            alpha=0.7, width=0.005)
            cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
            cbar.set_label('Memory Repulsion Force Intensity')
        
        # Draw trap areas
        for i, center in enumerate(controller.memory['trap_locations']):
            radius = controller.memory['trap_radii'][i]
            strength = controller.memory['trap_strengths'][i]
            circle = plt.Circle(center, radius, color='red', alpha=0.2, fill=True)
            ax.add_patch(circle)
            ax.text(center[0], center[1], f"{strength:.1f}", color='black', fontsize=9,
                   ha='center', va='center', weight='bold')
        
        # Draw stuck points
        for i, pos in enumerate(controller.memory['stuck_positions']):
            strength = controller.memory['stuck_strengths'][i]
            ax.plot(pos[0], pos[1], 'rx', markersize=6, alpha=min(1.0, strength/3))
        
        # Draw trajectory
        ax.plot(states_history[:, 0], states_history[:, 1], 'g-', linewidth=2.5, alpha=0.7)
        
        # Draw start and goal
        start = self.scenario['start']
        goal = self.scenario['goal']
        ax.plot(start[0], start[1], 'go', markersize=10)
        ax.plot(goal[0], goal[1], 'ro', markersize=10)
        
        # Mark key decision points: where stuck/unstuck
        key_points = []
        weights_history = []
        stuck_status = []
        
        # Simulate weight changes
        window = 5
        for i in range(window, len(states_history), max(1, len(states_history)//15)):
            pos = states_history[i, :2]
            recent = states_history[i-window:i, :2]
            var = np.var(recent, axis=0).sum()
            is_stuck = var < 0.01
            
            # If state changes or is start/end of trajectory, mark as key point
            if i == window or i >= len(states_history)-window or (len(stuck_status) > 0 and is_stuck != stuck_status[-1]):
                key_points.append(pos)
                stuck_status.append(is_stuck)
                
                # Assign weights based on stuck status
                if is_stuck:
                    duration = min(20, sum(stuck_status[-5:] if len(stuck_status) >= 5 else stuck_status))
                    sigmoid = 1.0 / (1.0 + np.exp(-0.2 * (duration - 5)))
                    weights = {
                        'goal': max(0.3, 1.0 - sigmoid*0.7),
                        'memory': sigmoid,
                        'trap': min(2.0, duration/5)
                    }
                else:
                    weights = {'goal': 1.0, 'memory': 0.3, 'trap': 0.0}
                
                weights_history.append(weights)
        
        # Draw key decision points
        for i, (point, weights) in enumerate(zip(key_points, weights_history)):
            # Use different colors for stuck and non-stuck points
            color = 'purple' if stuck_status[i] else 'green'
            status = "Stuck" if stuck_status[i] else "Normal"
            
            circle = plt.Circle(point, 0.2, color=color, alpha=0.5)
            ax.add_patch(circle)
            
            # Show weight information with text that doesn't overlap with arrows/lines
            # Calculate offset position based on surrounding points to avoid overlap
            offset_x = 0.3
            offset_y = 0.3
            
            # If this is a stuck point, move text more to avoid overlap with black arrows
            if status == "Stuck":
                offset_y += 0.3
            
            ax.annotate(status, xy=point, xytext=(point[0]+offset_x, point[1]+offset_y),
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                      fontsize=8)
        
        # Chart settings
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Memory Module Influence Visualization', fontsize=14)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Create custom legend in upper right corner
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Start'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Goal'),
            Line2D([0], [0], color='g', lw=2, label='Robot Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Obstacle'),
            Line2D([0], [0], marker='>', color='orange', markersize=8, label='Force Vector'),
            Line2D([0], [0], marker='x', color='r', markersize=6, label='Stuck Point'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Stuck State'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Normal State')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
                frameon=True, fancybox=True)
        
        # Save chart
        plt.tight_layout()
        plt.savefig("results/visualizations/memory_influence.png", dpi=150)
        plt.close()
        
        print("Memory module influence visualization saved")
    
    def visualize_potential_phase(self, ms_result):
        """Visualize potential phase change over time"""
        controller = ms_result['controller']
        states_history = ms_result['states']
        
        # Extract time steps from trajectory
        steps = len(states_history)
        time_steps = np.arange(steps)
        
        # Simulate potential and weight changes - based on trajectory and controller state
        goal_pot = []
        memory_pot = []
        trap_pot = []
        total_pot = []
        
        w_goal = []
        w_memory = []
        w_trap = []
        
        stuck_periods = []
        is_stuck = False
        stuck_start = 0
        
        for i, state in enumerate(states_history):
            position = state[:2]
            goal = self.scenario['goal']
            
            # Calculate goal potential - based on distance to goal
            dist_to_goal = np.linalg.norm(position - goal[:2])
            g_pot = 10.0 * dist_to_goal**2
            
            # Determine if currently stuck - through variance of recent positions
            if i >= 5:
                recent = states_history[max(0, i-5):i, :2]
                var = np.var(recent, axis=0).sum()
                current_stuck = var < 0.01
                
                # Record stuck periods
                if current_stuck and not is_stuck:
                    is_stuck = True
                    stuck_start = i
                elif not current_stuck and is_stuck:
                    is_stuck = False
                    stuck_periods.append((stuck_start, i))
            else:
                current_stuck = False
            
            # Calculate weight changes based on stuck status
            if current_stuck:
                # Calculate stuck duration
                stuck_duration = i - stuck_start
                sigmoid = 1.0 / (1.0 + np.exp(-0.2 * (stuck_duration - 5)))
                
                goal_weight = max(0.3, 1.0 - sigmoid*0.7)
                memory_weight = sigmoid
                trap_weight = min(2.0, stuck_duration/5)
                
                # Increase memory and trap potentials when stuck
                m_pot = 5.0 + sigmoid * 10.0
                t_pot = stuck_duration * 2.0
            else:
                goal_weight = 1.0
                memory_weight = 0.3
                trap_weight = 0.0
                
                # Normal potential
                m_pot = 1.0 + np.random.random() * 0.5  # Small fluctuation
                t_pot = 0.1
            
            # Total potential
            tot_pot = goal_weight * g_pot + memory_weight * m_pot + trap_weight * t_pot
            
            # Save data
            goal_pot.append(g_pot)
            memory_pot.append(m_pot)
            trap_pot.append(t_pot)
            total_pot.append(tot_pot)
            
            w_goal.append(goal_weight)
            w_memory.append(memory_weight)
            w_trap.append(trap_weight)
        
        # If still stuck at the end
        if is_stuck:
            stuck_periods.append((stuck_start, steps-1))
        
        # Create two subplots: potential change and weight change
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Potential change chart
        ax1.plot(time_steps, goal_pot, 'g-', linewidth=2, alpha=0.7, label='Goal Potential')
        ax1.plot(time_steps, memory_pot, 'b-', linewidth=2, alpha=0.7, label='Memory Potential')
        ax1.plot(time_steps, trap_pot, 'r-', linewidth=2, alpha=0.7, label='Trap Potential')
        ax1.plot(time_steps, total_pot, 'k-', linewidth=2.5, alpha=0.8, label='Total Potential')
        
        # Weight change chart
        ax2.plot(time_steps, w_goal, 'g-', linewidth=2, alpha=0.7, label='Goal Weight')
        ax2.plot(time_steps, w_memory, 'b-', linewidth=2, alpha=0.7, label='Memory Weight')
        ax2.plot(time_steps, w_trap, 'r-', linewidth=2, alpha=0.7, label='Trap Weight')
        
        # Add shading for stuck regions
        for start, end in stuck_periods:
            ax1.axvspan(start, end, color='red', alpha=0.15, label='Stuck Region' if start==stuck_periods[0][0] else "")
            ax2.axvspan(start, end, color='red', alpha=0.15)
            
            # Add markers - carefully positioned to avoid overlap
            if end - start > 5:  # Only mark longer stuck periods
                mid = (start + end) // 2
                
                # Check if there's room for an annotation
                # Position annotation vertically based on potential value
                y_pos = max(total_pot[start:end]) * 1.1
                
                # Determine if this annotation would overlap with previous ones
                # If too close to previous annotation, reposition
                ax1.annotate('Stuck State', xy=(mid, y_pos), 
                           xytext=(mid, y_pos + 100),  # Position text above the line
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, alpha=0.6),
                           ha='center', fontsize=10)
        
        # Set chart properties
        ax1.set_title('Potential Phase Change During Navigation', fontsize=14)
        ax1.set_ylabel('Potential Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Control Weight Adaptation', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Weight Value', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add legends to upper right corner of each plot
        ax1.legend(loc='upper right', framealpha=0.9)
        ax2.legend(loc='upper right', framealpha=0.9)
        
        # Save chart
        plt.tight_layout()
        plt.savefig("results/visualizations/potential_phase.png", dpi=150)
        plt.close()
        
        print("Potential phase change visualization saved")


def run_simplified_experiment():
    """Run simplified multi-scale potential MPPI experiment"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create experiment
    experiment = SimplifiedExperiment()
    
    # Run comparison experiment
    basic_result, ms_result = experiment.run_comparison()
    
    return basic_result, ms_result


if __name__ == "__main__":
    run_simplified_experiment()
