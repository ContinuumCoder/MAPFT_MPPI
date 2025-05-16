"""
Simplified quadrotor dynamics model for 3D navigation.
"""
import numpy as np

class QuadrotorDynamics:
    """Simplified quadrotor dynamics model"""
    
    def __init__(self, mass=1.0, g=9.81, max_thrust=20.0, max_torque=5.0, dt=0.05):
        """
        Initialize quadrotor dynamics.
        
        Args:
            mass: Mass (kg)
            g: Gravity (m/s^2)
            max_thrust: Maximum thrust (N)
            max_torque: Maximum torque (Nm)
            dt: Time step (s)
        """
        self.mass = mass
        self.g = g
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.dt = dt
        
        # Moment of inertia matrix (diagonal)
        self.J = np.array([0.01, 0.01, 0.02])  # [Jx, Jy, Jz]
        
        # Drag coefficients
        self.k_d = 0.1  # Linear drag
        
        # State dimension: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        # x, y, z: position
        # vx, vy, vz: linear velocity
        # qw, qx, qy, qz: quaternion orientation
        # wx, wy, wz: angular velocity
        self.state_dim = 13
        
        # Control dimension: [thrust, tau_x, tau_y, tau_z]
        # thrust: total thrust magnitude
        # tau_x, tau_y, tau_z: body torques
        self.control_dim = 4
    
    def step(self, state, control):
        """
        Step dynamics forward.
        
        Args:
            state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            control: [thrust, tau_x, tau_y, tau_z]
            
        Returns:
            next_state: Next state
        """
        # Unpack state
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        # Unpack control and apply limits
        thrust = np.clip(control[0], 0, self.max_thrust)
        torque = np.clip(control[1:4], -self.max_torque, self.max_torque)
        
        # Convert quaternion to rotation matrix
        R = self._quat_to_rot(quat)
        
        # Compute acceleration
        # Thrust force in body frame (along z-axis)
        thrust_body = np.array([0, 0, thrust])
        # Convert to world frame
        thrust_world = R @ thrust_body
        
        # Add gravity and drag
        acc = thrust_world / self.mass - np.array([0, 0, self.g])
        acc = acc - self.k_d * vel  # Linear drag
        
        # Update position and velocity
        pos_next = pos + vel * self.dt + 0.5 * acc * self.dt**2
        vel_next = vel + acc * self.dt
        
        # Update angular velocity
        omega_dot = torque / self.J - np.cross(omega, self.J * omega) / self.J
        omega_next = omega + omega_dot * self.dt
        
        # Update quaternion using angular velocity
        quat_next = self._update_quaternion(quat, omega, self.dt)
        
        # Assemble next state
        next_state = np.concatenate([pos_next, vel_next, quat_next, omega_next])
        
        return next_state
    
    def _quat_to_rot(self, quat):
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = quat
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def _update_quaternion(self, quat, omega, dt):
        """Update quaternion using angular velocity"""
        qw, qx, qy, qz = quat
        wx, wy, wz = omega
        
        # Quaternion derivative
        quat_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
            qw*wx + qy*wz - qz*wy,
            qw*wy - qx*wz + qz*wx,
            qw*wz + qx*wy - qy*wx
        ])
        
        # Update quaternion
        quat_next = quat + quat_dot * dt
        
        # Normalize
        quat_next = quat_next / np.linalg.norm(quat_next)
        
        return quat_next
    
    def get_state_dim(self):
        """Get state dimension"""
        return self.state_dim
    
    def get_control_dim(self):
        """Get control dimension"""
        return self.control_dim
