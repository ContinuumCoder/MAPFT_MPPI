"""
Implementation of Iterative Linear Quadratic Regulator (iLQR) controller.
"""
import numpy as np
from core.controller_base import ControllerBase

class iLQR(ControllerBase):
    """Iterative Linear Quadratic Regulator controller"""
    
    def __init__(self, dynamics_model, cost_function, state_dim, control_dim, 
                 horizon=15, max_iterations=10, stopping_threshold=1e-4,
                 regularization=1e-6, line_search_factor=0.5, 
                 u_min=None, u_max=None, dt=0.1):
        """
        Initialize iLQR controller.
        
        Args:
            dynamics_model: Dynamics model function
            cost_function: Cost function
            state_dim: State dimension
            control_dim: Control dimension
            horizon: Planning horizon
            max_iterations: Maximum iterations
            stopping_threshold: Threshold for stopping iteration
            regularization: Regularization parameter for Hessian
            line_search_factor: Line search reduction factor
            u_min: Minimum control
            u_max: Maximum control
            dt: Time step
        """
        super().__init__(state_dim, control_dim, horizon, dt)
        
        self.dynamics = dynamics_model
        self.cost_function = cost_function
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold
        self.regularization = regularization
        self.line_search_factor = line_search_factor
        
        # Control constraints
        self.u_min = u_min if u_min is not None else -np.ones(control_dim) * float('inf')
        self.u_max = u_max if u_max is not None else np.ones(control_dim) * float('inf')
        
        # Initialize nominal trajectory
        self.x_nominal = None
        self.u_nominal = None
        
        # For visualization
        self.best_trajectory = None
    
    def reset(self):
        """Reset controller"""
        self.x_nominal = None
        self.u_nominal = None
        self.best_trajectory = None
    
    def compute_control(self, state, goal=None, obstacles=None):
        """
        Compute optimal control input using iLQR.
        
        Args:
            state: Current state
            goal: Goal state
            obstacles: List of obstacles
            
        Returns:
            control: Control input
        """
        # Initialize nominal trajectory if not available
        if self.x_nominal is None or self.u_nominal is None:
            self._initialize_nominal_trajectory(state, goal)
        
        # Update first state to current state
        self.x_nominal[0] = state.copy()
        
        # Wrapper for dynamics
        def dynamics_step(x, u):
            return self.dynamics(x, u, self.dt)
        
        # Wrapper for cost
        def cost_step(x, u, i):
            # Terminal cost for last step
            if i == self.horizon:
                return self.cost_function(x, None, goal, obstacles, terminal=True)
            return self.cost_function(x, u, goal, obstacles, terminal=False)
        
        # Run iLQR iterations
        for iteration in range(self.max_iterations):
            # Forward pass: roll out nominal trajectory and compute cost
            total_cost = self._rollout_trajectory(dynamics_step, cost_step)
            
            # Backward pass: compute optimal control law
            k_seq, K_seq, expected_cost_reduction = self._backward_pass(dynamics_step, cost_step)
            
            # Check for convergence
            if expected_cost_reduction < self.stopping_threshold:
                break
            
            # Line search for optimal step size
            alpha = 1.0
            line_search_iterations = 0
            max_line_search = 10
            
            while line_search_iterations < max_line_search:
                # Forward pass with new control law
                x_new, u_new, new_cost = self._forward_pass(
                    dynamics_step, cost_step, k_seq, K_seq, alpha
                )
                
                # Accept step if cost is reduced
                if new_cost < total_cost:
                    self.x_nominal = x_new
                    self.u_nominal = u_new
                    break
                
                # Reduce step size
                alpha *= self.line_search_factor
                line_search_iterations += 1
        
        # Store best trajectory for visualization
        self.best_trajectory = self.x_nominal.copy()
        
        # Return first control
        control = self.u_nominal[0].copy()
        
        # Roll control sequence forward
        self.u_nominal = np.vstack([self.u_nominal[1:], np.zeros((1, self.control_dim))])
        
        return control
    
    def _initialize_nominal_trajectory(self, state, goal):
        """Initialize nominal trajectory with zeros or linear interpolation"""
        # Initialize state trajectory
        self.x_nominal = np.zeros((self.horizon + 1, self.state_dim))
        self.x_nominal[0] = state.copy()
        
        # Linear interpolation to goal
        if goal is not None:
            for i in range(1, self.horizon + 1):
                alpha = i / self.horizon
                self.x_nominal[i] = (1 - alpha) * state + alpha * goal
        
        # Initialize control trajectory
        self.u_nominal = np.zeros((self.horizon, self.control_dim))
    
    def _rollout_trajectory(self, dynamics_step, cost_step):
        """
        Roll out nominal trajectory and compute total cost.
        
        Args:
            dynamics_step: Dynamics step function
            cost_step: Cost step function
            
        Returns:
            total_cost: Total trajectory cost
        """
        total_cost = 0.0
        
        # First state is already set
        
        # Roll out the rest of the trajectory
        for i in range(self.horizon):
            # Apply control limits
            u = np.clip(self.u_nominal[i], self.u_min, self.u_max)
            
            # Compute cost
            total_cost += cost_step(self.x_nominal[i], u, i)
            
            # Propagate dynamics
            self.x_nominal[i+1] = dynamics_step(self.x_nominal[i], u)
        
        # Terminal cost
        total_cost += cost_step(self.x_nominal[self.horizon], None, self.horizon)
        
        return total_cost
    
    def _backward_pass(self, dynamics_step, cost_step):
        """
        Backward pass to compute optimal control law.
        
        Args:
            dynamics_step: Dynamics step function
            cost_step: Cost step function
            
        Returns:
            k_seq: Feed-forward term
            K_seq: Feedback gain term
            expected_cost_reduction: Expected cost reduction
        """
        # Initialize arrays
        k_seq = [None] * self.horizon
        K_seq = [None] * self.horizon
        
        # Initialize value function derivatives
        V_x = np.zeros(self.state_dim)
        V_xx = np.zeros((self.state_dim, self.state_dim))
        
        # Terminal cost derivatives
        cost = cost_step(self.x_nominal[self.horizon], None, self.horizon)
        l_x, l_xx = self._numerical_derivatives(
            lambda x: cost_step(x, None, self.horizon), 
            self.x_nominal[self.horizon]
        )
        
        V_x = l_x
        V_xx = l_xx
        
        # Expected cost reduction
        expected_cost_reduction = 0.0
        
        # Backward pass
        for i in range(self.horizon - 1, -1, -1):
            x = self.x_nominal[i]
            u = self.u_nominal[i]
            
            # Compute cost derivatives
            l_x, l_xx, l_u, l_uu, l_xu = self._derivatives_at_point(cost_step, x, u, i)
            
            # Compute dynamics derivatives
            A, B = self._linearize_dynamics(dynamics_step, x, u)
            
            # Compute Q-function derivatives
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_xu = l_xu + A.T @ V_xx @ B
            
            # Add regularization to ensure Q_uu is positive definite
            Q_uu_reg = Q_uu + np.eye(self.control_dim) * self.regularization
            
            # Compute gains
            try:
                # Solve for feedback and feed-forward terms
                k = -np.linalg.solve(Q_uu_reg, Q_u)
                K = -np.linalg.solve(Q_uu_reg, Q_xu.T)
                
                # Store gains
                k_seq[i] = k
                K_seq[i] = K
                
                # Update value function derivatives
                V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_xu.T @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_xu + Q_xu.T @ K
                
                # Ensure V_xx is symmetric
                V_xx = (V_xx + V_xx.T) / 2
                
                # Compute expected cost reduction
                expected_cost_reduction += 0.5 * k.T @ Q_uu @ k + k.T @ Q_u
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudoinverse
                print("Warning: Q_uu matrix is singular, using pseudoinverse")
                Q_uu_inv = np.linalg.pinv(Q_uu_reg)
                k = -Q_uu_inv @ Q_u
                K = -Q_uu_inv @ Q_xu.T
                
                k_seq[i] = k
                K_seq[i] = K
                
                V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_xu.T @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_xu + Q_xu.T @ K
                V_xx = (V_xx + V_xx.T) / 2
                
                expected_cost_reduction += 0.5 * k.T @ Q_uu @ k + k.T @ Q_u
        
        return k_seq, K_seq, expected_cost_reduction
    
    def _forward_pass(self, dynamics_step, cost_step, k_seq, K_seq, alpha):
        """
        Forward pass with new control law.
        
        Args:
            dynamics_step: Dynamics step function
            cost_step: Cost step function
            k_seq: Feed-forward term
            K_seq: Feedback gain term
            alpha: Step size
            
        Returns:
            x_new: New state trajectory
            u_new: New control trajectory
            total_cost: New total cost
        """
        x_new = np.zeros((self.horizon + 1, self.state_dim))
        u_new = np.zeros((self.horizon, self.control_dim))
        
        x_new[0] = self.x_nominal[0].copy()
        total_cost = 0.0
        
        for i in range(self.horizon):
            # State and control at current time step
            x = x_new[i]
            
            # Compute new control with feedback and feed-forward terms
            dx = x - self.x_nominal[i]
            u = self.u_nominal[i] + alpha * k_seq[i] + K_seq[i] @ dx
            
            # Apply control limits
            u = np.clip(u, self.u_min, self.u_max)
            u_new[i] = u
            
            # Compute cost
            total_cost += cost_step(x, u, i)
            
            # Propagate dynamics
            x_new[i+1] = dynamics_step(x, u)
        
        # Terminal cost
        total_cost += cost_step(x_new[self.horizon], None, self.horizon)
        
        return x_new, u_new, total_cost
    
    def _numerical_derivatives(self, func, x, u=None, eps=1e-6):
        """
        Compute numerical derivatives of a function.
        
        Args:
            func: Function to differentiate
            x: State
            u: Control (optional)
            eps: Step size
            
        Returns:
            derivatives: List of derivatives
        """
        if u is None:
            # First-order derivative with respect to x
            n = len(x)
            fx = np.zeros(n)
            fxx = np.zeros((n, n))
            
            f0 = func(x)
            
            # First-order derivatives
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = func(x_plus)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = func(x_minus)
                
                fx[i] = (f_plus - f_minus) / (2 * eps)
            
            # Second-order derivatives
            for i in range(n):
                for j in range(i, n):
                    x_plus_plus = x.copy()
                    x_plus_plus[i] += eps
                    x_plus_plus[j] += eps
                    f_plus_plus = func(x_plus_plus)
                    
                    x_plus_minus = x.copy()
                    x_plus_minus[i] += eps
                    x_plus_minus[j] -= eps
                    f_plus_minus = func(x_plus_minus)
                    
                    x_minus_plus = x.copy()
                    x_minus_plus[i] -= eps
                    x_minus_plus[j] += eps
                    f_minus_plus = func(x_minus_plus)
                    
                    x_minus_minus = x.copy()
                    x_minus_minus[i] -= eps
                    x_minus_minus[j] -= eps
                    f_minus_minus = func(x_minus_minus)
                    
                    fxx[i, j] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * eps**2)
                    fxx[j, i] = fxx[i, j]  # Symmetric
            
            return fx, fxx
        else:
            # Derivatives with respect to x and u
            n, m = len(x), len(u)
            
            # Function wrapper for control
            def f_xu(xu):
                x_part = xu[:n]
                u_part = xu[n:]
                return func(x_part, u_part)
            
            # Concatenate x and u
            xu = np.concatenate([x, u])
            
            # Compute derivatives
            fx_fu, fxx_fuu_fxu = self._numerical_derivatives(f_xu, xu)
            
            # Extract components
            fx = fx_fu[:n]
            fu = fx_fu[n:]
            
            fxx = fxx_fuu_fxu[:n, :n]
            fuu = fxx_fuu_fxu[n:, n:]
            fxu = fxx_fuu_fxu[:n, n:]
            
            return fx, fxx, fu, fuu, fxu
    
    def _derivatives_at_point(self, cost_step, x, u, i):
        """
        Compute cost derivatives at a point.
        
        Args:
            cost_step: Cost step function
            x: State
            u: Control
            i: Time step
            
        Returns:
            l_x, l_xx, l_u, l_uu, l_xu: Cost derivatives
        """
        def cost_func(x_val, u_val):
            return cost_step(x_val, u_val, i)
        
        return self._numerical_derivatives(cost_func, x, u)
    
    def _linearize_dynamics(self, dynamics_step, x, u, eps=1e-6):
        """
        Linearize dynamics around a point.
        
        Args:
            dynamics_step: Dynamics step function
            x: State
            u: Control
            eps: Step size
            
        Returns:
            A: State Jacobian
            B: Control Jacobian
        """
        n, m = len(x), len(u)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        
        # Compute nominal next state
        x_next = dynamics_step(x, u)
        
        # Compute A matrix (df/dx)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_next_plus = dynamics_step(x_plus, u)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            x_next_minus = dynamics_step(x_minus, u)
            
            A[:, i] = (x_next_plus - x_next_minus) / (2 * eps)
        
        # Compute B matrix (df/du)
        for i in range(m):
            u_plus = u.copy()
            u_plus[i] += eps
            x_next_plus = dynamics_step(x, u_plus)
            
            u_minus = u.copy()
            u_minus[i] -= eps
            x_next_minus = dynamics_step(x, u_minus)
            
            B[:, i] = (x_next_plus - x_next_minus) / (2 * eps)
        
        return A, B
