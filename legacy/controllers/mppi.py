"""
Implementation of the Model Predictive Path Integral (MPPI) controller.
"""
import numpy as np
from core.controller_base import ControllerBase

class MPPI(ControllerBase):
    """Model Predictive Path Integral Controller"""
    
    def __init__(self, dynamics_model, running_cost, terminal_cost=None, 
                 state_dim=4, control_dim=2, horizon=15, num_samples=100, 
                 noise_sigma=None, lambda_=1.0, temperature=0.5,
                 u_min=None, u_max=None, dt=0.1):
        """
        Initialize MPPI controller.
        
        Args:
            dynamics_model: Function for forward dynamics
            running_cost: Running cost function
            terminal_cost: Terminal cost function
            state_dim: Dimension of state space
            control_dim: Dimension of control space
            horizon: Planning horizon
            num_samples: Number of trajectory samples
            noise_sigma: Control noise covariance matrix
            lambda_: Path integral weight
            temperature: Temperature parameter for softmax
            u_min: Minimum control input
            u_max: Maximum control input
            dt: Time step
        """
        super().__init__(state_dim, control_dim, horizon, dt)
        
        self.dynamics = dynamics_model
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
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
    
    def compute_control(self, x0, goal=None, obstacles=None):
        """
        Compute optimal control input using MPPI.
        
        Args:
            x0: Current state
            goal: Target state
            obstacles: List of obstacles
            
        Returns:
            control: Control input
        """
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
    
    def reset(self):
        """Reset controller internal state"""
        self.U = np.zeros((self.horizon, self.control_dim))
        self.best_trajectory = None
        self.all_trajectories = None
        self.trajectory_costs = None
