"""
Car-like robot dynamics model with non-holonomic constraints.
"""
import numpy as np

class CarDynamics:
    """Car-like robot dynamics with Ackermann steering"""
    
    def __init__(self, wheelbase=1.0, max_steer=0.5, max_speed=3.0, dt=0.1):
        """
        Initialize car dynamics model.
        
        Args:
            wheelbase: Distance between front and rear axles
            max_steer: Maximum steering angle (radians)
            max_speed: Maximum speed (m/s)
            dt: Time step
        """
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.dt = dt
    
    def step(self, state, control):
        """
        Step dynamics forward.
        
        Args:
            state: [x, y, theta, v] (position, heading, velocity)
            control: [acceleration, steering]
            
        Returns:
            next_state: Next state
        """
        x, y, theta, v = state
        accel, steer = control
        
        # Apply control limits
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        
        # Update velocity with acceleration
        v_next = v + accel * self.dt
        v_next = np.clip(v_next, -self.max_speed, self.max_speed)
        
        # Update position and heading
        if abs(steer) < 1e-4:
            # Nearly straight motion
            x_next = x + v_next * np.cos(theta) * self.dt
            y_next = y + v_next * np.sin(theta) * self.dt
            theta_next = theta
        else:
            # Turning motion
            turn_radius = self.wheelbase / np.tan(steer)
            beta = v_next * self.dt / turn_radius  # Central angle
            
            x_next = x + turn_radius * (np.sin(theta + beta) - np.sin(theta))
            y_next = y + turn_radius * (np.cos(theta) - np.cos(theta + beta))
            theta_next = (theta + beta) % (2 * np.pi)
        
        return np.array([x_next, y_next, theta_next, v_next])
    
    def get_state_dim(self):
        """Get state dimension"""
        return 4
    
    def get_control_dim(self):
        """Get control dimension"""
        return 2
