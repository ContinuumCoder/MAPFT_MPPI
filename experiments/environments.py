"""
Standard control test environments for MA-MPPI benchmarking.

1. UAV 3D obstacle avoidance (quadrotor dynamics)
2. IEEE 39-bus power system stability
"""
import numpy as np
from typing import Tuple


# ============================================================
# UAV Quadrotor 3D Obstacle Avoidance
# ============================================================
class QuadrotorEnv:
    """
    UAV as 3D double integrator with drag.
    State (6D): [px, py, pz, vx, vy, vz]
    Control (3D): [ax, ay, az] acceleration commands

    Fast vectorized dynamics for MPPI benchmarking.
    """

    def __init__(self, obstacles=None, goal=None, dt=0.05):
        self.dt = dt
        self.k_d = 0.2       # drag

        self.goal = goal if goal is not None else np.array([8.0, 8.0, 5.0])

        if obstacles is None:
            self.obstacles = [
                (2.0, 2.0, 0.8, 8.0),
                (4.0, 5.0, 1.0, 8.0),
                (6.0, 3.0, 0.7, 8.0),
            ]
        else:
            self.obstacles = obstacles

        self.state_dim = 6
        self.control_dim = 3
        self.u_min = np.full(3, -5.0)
        self.u_max = np.full(3, 5.0)

    def dynamics(self, x, u):
        """Vectorized 3D double integrator with drag."""
        if x.ndim == 1:
            pos = x[:3]; vel = x[3:]
            u_c = np.clip(u, self.u_min, self.u_max)
            acc = u_c - self.k_d * vel
            vel_new = np.clip(vel + acc * self.dt, -4.0, 4.0)
            pos_new = pos + vel_new * self.dt
            return np.concatenate([pos_new, vel_new])
        # Batch
        pos = x[:, :3]; vel = x[:, 3:]
        u_c = np.clip(u, self.u_min, self.u_max)
        acc = u_c - self.k_d * vel
        vel_new = np.clip(vel + acc * self.dt, -4.0, 4.0)
        pos_new = pos + vel_new * self.dt
        return np.hstack([pos_new, vel_new])

    def cost(self, x, u):
        """Vectorized cost: goal + obstacles + control."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1); u = u.reshape(1, -1)

        pos = x[:, :3]; vel = x[:, 3:]
        goal_cost = 2.0 * np.sum((pos - self.goal) ** 2, axis=1)
        vel_cost = 0.1 * np.sum(vel ** 2, axis=1)
        ctrl_cost = 0.01 * np.sum(u ** 2, axis=1)

        obs_cost = np.zeros(len(x))
        for cx, cy, r, h in self.obstacles:
            d = np.sqrt((pos[:, 0]-cx)**2 + (pos[:, 1]-cy)**2) - r
            in_height = (pos[:, 2] > 0) & (pos[:, 2] < h)
            close = (d < 0.5) & in_height
            obs_cost[close] += 1000.0 / (d[close] + 0.01) ** 2

        result = goal_cost + obs_cost + vel_cost + ctrl_cost
        return float(result[0]) if single else result

    def reset(self) -> np.ndarray:
        """Start at origin, zero velocity."""
        return np.zeros(6)

    def is_collision(self, pos):
        for cx, cy, r, h in self.obstacles:
            d = np.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2)
            if d < r and 0 < pos[2] < h:
                return True
        return False

    def is_reached(self, pos, tol=0.5):
        return np.linalg.norm(pos - self.goal) < tol


# ============================================================
# IEEE 39-Bus Power System (simplified)
# ============================================================
class IEEE39BusEnv:
    """
    Simplified IEEE 39-bus New England power system model.

    State (20D): 10 generators x [rotor_angle, angular_velocity]
    Control (10D): mechanical power adjustments for each generator
    Disturbances: fault, load change, generator trip

    Simplified swing equation dynamics:
        M_i * d²δ_i/dt² + D_i * dδ_i/dt = P_m_i - P_e_i(δ)
    """

    def __init__(self, dt=0.01):
        self.dt = dt
        self.n_gen = 10
        self.state_dim = 2 * self.n_gen  # (δ, ω) per generator
        self.control_dim = self.n_gen

        # Generator parameters (simplified)
        self.M = np.array([42.0, 30.2, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 500.0])
        self.D = self.M * 0.05  # damping
        self.P_m0 = np.array([2.5, 5.73, 6.32, 5.08, 6.5, 5.6, 5.4, 8.3, 10.0, 2.5])

        # Admittance matrix (simplified: nearest-neighbor coupling)
        self.Y = np.zeros((self.n_gen, self.n_gen))
        coupling = 5.0
        for i in range(self.n_gen - 1):
            self.Y[i, i+1] = coupling
            self.Y[i+1, i] = coupling
        self.Y[0, -1] = coupling * 0.5
        self.Y[-1, 0] = coupling * 0.5
        np.fill_diagonal(self.Y, -np.sum(self.Y, axis=1))

        # Control limits
        self.u_min = np.full(self.control_dim, -5.0)
        self.u_max = np.full(self.control_dim, 5.0)

        # Disturbance state
        self.disturbance = None

    def dynamics(self, x, u):
        """Swing equation dynamics."""
        if x.ndim > 1:
            return np.array([self.dynamics(x[k], u[k]) for k in range(len(x))])

        delta = x[:self.n_gen]
        omega = x[self.n_gen:]
        u = np.clip(u, self.u_min, self.u_max)

        # Electrical power: P_e_i = Σ Y_ij * sin(δ_i - δ_j)
        P_e = np.zeros(self.n_gen)
        for i in range(self.n_gen):
            for j in range(self.n_gen):
                if i != j and self.Y[i, j] != 0:
                    P_e[i] += abs(self.Y[i, j]) * np.sin(delta[i] - delta[j])

        # Mechanical power = base + control adjustment
        P_m = self.P_m0 + u

        # Apply disturbance
        if self.disturbance is not None:
            P_m = P_m + self.disturbance

        # Swing equation
        d_omega = (P_m - P_e - self.D * omega) / self.M
        d_delta = omega

        delta_new = delta + d_delta * self.dt
        omega_new = omega + d_omega * self.dt

        # Reference frame: keep generator 10 (slack bus) angle near 0
        delta_new -= delta_new[-1]

        return np.concatenate([delta_new, omega_new])

    def cost(self, x, u):
        """
        Cost: frequency deviation + voltage stability proxy + control effort.
        """
        if x.ndim > 1:
            return np.array([self.cost(x[k], u[k]) for k in range(len(x))])

        omega = x[self.n_gen:]
        delta = x[:self.n_gen]

        # Frequency deviation (want ω ≈ 0, i.e., synchronized)
        freq_cost = 100.0 * np.sum(omega ** 2)

        # Angle spread (stability margin proxy)
        angle_spread = np.max(delta) - np.min(delta)
        stability_cost = 10.0 * angle_spread ** 2

        # Control effort
        ctrl_cost = 0.1 * np.sum(u ** 2)

        return freq_cost + stability_cost + ctrl_cost

    def reset(self) -> np.ndarray:
        """Steady state: small random perturbation."""
        delta0 = np.random.randn(self.n_gen) * 0.05
        delta0 -= delta0[-1]  # reference
        omega0 = np.random.randn(self.n_gen) * 0.01
        return np.concatenate([delta0, omega0])

    def apply_fault(self, fault_type="three_phase"):
        """Apply a disturbance."""
        if fault_type == "three_phase":
            self.disturbance = np.zeros(self.n_gen)
            self.disturbance[3] = -1.5
            self.disturbance[4] = -1.0
        elif fault_type == "load_change":
            self.disturbance = np.zeros(self.n_gen)
            self.disturbance[1] = -0.8
            self.disturbance[5] = -0.5
        elif fault_type == "generator_trip":
            self.disturbance = np.zeros(self.n_gen)
            self.disturbance[7] = -self.P_m0[7] * 0.5  # partial trip

    def clear_fault(self):
        self.disturbance = None

    def is_stable(self, x, threshold=1.0):
        """Check if system is stable (small frequency deviation)."""
        omega = x[self.n_gen:]
        return np.max(np.abs(omega)) < threshold
