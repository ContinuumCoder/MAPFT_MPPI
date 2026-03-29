"""
Base controller interface for stochastic optimal control implementations.
"""
import numpy as np
from abc import ABC, abstractmethod

class ControllerBase(ABC):
    """Base class for all controllers"""
    
    def __init__(self, state_dim, control_dim, horizon, dt=0.1):
        """
        Initialize base controller.
        
        Args:
            state_dim: Dimension of state space
            control_dim: Dimension of control space
            horizon: Planning horizon
            dt: Time step
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.dt = dt
        
    @abstractmethod
    def compute_control(self, state, goal=None, obstacles=None):
        """
        Compute control input.
        
        Args:
            state: Current state
            goal: Target state
            obstacles: List of obstacles
            
        Returns:
            control: Control input
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller internal state"""
        pass
