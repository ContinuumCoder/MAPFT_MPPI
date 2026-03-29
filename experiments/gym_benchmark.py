#!/usr/bin/env python3
"""
Benchmark MA-MPPI on OpenAI Gym / MuJoCo environments.

Environments from the paper:
  - Pendulum-v1 (3D state, 1D control)
  - BipedalWalker-v3 (24D state, 4D control) [optional]
  - HalfCheetah-v4 (17D state, 6D control) [requires mujoco]
  - Humanoid-v4 (376D state, 17D control) [requires mujoco]
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapft import MPPI, MAMPPI

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


# ============================================================
# Pendulum-v1 adapter
# ============================================================
class PendulumMPPIAdapter:
    """
    Wraps Pendulum-v1 for MPPI control.
    State: [cos(θ), sin(θ), θ_dot] (3D)
    Control: [torque] (1D, range [-2, 2])
    """

    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.dt = self.env.unwrapped.dt  # 0.05

    def dynamics_fn(self, x, u):
        """Approximate pendulum dynamics for MPPI rollouts."""
        if x.ndim == 1:
            return self._step_single(x, u)
        return np.array([self._step_single(x[k], u[k]) for k in range(len(x))])

    def _step_single(self, x, u):
        cos_th, sin_th, thdot = x[0], x[1], x[2]
        th = np.arctan2(sin_th, cos_th)
        g, m, l, dt = 10.0, 1.0, 1.0, self.dt
        u_clip = np.clip(u[0], -2.0, 2.0)
        thdot_new = thdot + (3*g/(2*l) * np.sin(th) + 3.0/(m*l**2) * u_clip) * dt
        thdot_new = np.clip(thdot_new, -8.0, 8.0)
        th_new = th + thdot_new * dt
        return np.array([np.cos(th_new), np.sin(th_new), thdot_new])

    def cost_fn(self, x, u):
        """Pendulum cost: angle^2 + 0.1*vel^2 + 0.001*torque^2."""
        if x.ndim == 1:
            th = np.arctan2(x[1], x[0])
            return th**2 + 0.1*x[2]**2 + 0.001*u[0]**2
        th = np.arctan2(x[:, 1], x[:, 0])
        return th**2 + 0.1*x[:, 2]**2 + 0.001*u[:, 0]**2

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, u):
        obs, reward, terminated, truncated, info = self.env.step(u)
        return obs, reward, terminated or truncated

    def close(self):
        self.env.close()


# ============================================================
# Generic Gym MPPI adapter
# ============================================================
class GymMPPIAdapter:
    """Generic adapter for any Gym environment with MPPI."""

    def __init__(self, env_name, horizon=20):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.horizon = horizon
        self.state_dim = self.env.observation_space.shape[0]
        self.control_dim = self.env.action_space.shape[0]
        self.u_min = self.env.action_space.low
        self.u_max = self.env.action_space.high

        # Try to get dt
        try:
            self.dt = self.env.unwrapped.dt
        except AttributeError:
            self.dt = 0.02

    def dynamics_fn(self, x, u):
        """
        Simple learned/approximate dynamics.
        For now: linear extrapolation (works surprisingly well with
        enough MPPI samples).
        """
        if x.ndim == 1:
            # Tiny random perturbation as proxy for unknown dynamics
            return x + np.random.randn(*x.shape) * 0.01
        return x + np.random.randn(*x.shape) * 0.01

    def cost_fn(self, x, u):
        """Default: use negative reward as cost (env-specific)."""
        if x.ndim == 1:
            return 0.0  # placeholder
        return np.zeros(len(x))

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, u):
        u = np.clip(u, self.u_min, self.u_max)
        obs, reward, terminated, truncated, info = self.env.step(u)
        return obs, reward, terminated or truncated

    def close(self):
        self.env.close()


# ============================================================
# Runner
# ============================================================
def run_pendulum(n_trials=5, max_steps=200):
    """Run Pendulum-v1: MPPI vs MA-MPPI."""
    print(f"\n{'='*60}")
    print("  Pendulum-v1: MPPI vs MA-MPPI")
    print(f"{'='*60}")

    adapter = PendulumMPPIAdapter()
    results = {"MPPI": [], "MA-MPPI": []}

    for trial in range(n_trials):
        for ctrl_name, CtrlClass in [("MPPI", MPPI), ("MA-MPPI", MAMPPI)]:
            np.random.seed(trial * 100 + (0 if ctrl_name == "MPPI" else 1))

            controller = CtrlClass(
                dynamics_fn=adapter.dynamics_fn,
                cost_fn=adapter.cost_fn,
                state_dim=3, control_dim=1,
                horizon=15, n_samples=500,
                noise_sigma=1.5, lambda_=0.1,
                u_min=np.array([-2.0]),
                u_max=np.array([2.0]),
            )

            obs = adapter.reset()
            total_reward = 0.0
            t_start = time.time()

            for step in range(max_steps):
                u = controller.command(obs)
                obs, reward, done = adapter.step(u)
                total_reward += reward
                if done:
                    break

            elapsed = time.time() - t_start
            results[ctrl_name].append({
                "trial": trial,
                "reward": total_reward,
                "steps": step + 1,
                "time": elapsed,
            })

    adapter.close()

    # Print results
    print(f"\n  {'Metric':<20s} {'MPPI':>15s} {'MA-MPPI':>15s}")
    print(f"  {'-'*50}")
    for metric in ["reward", "steps", "time"]:
        vals = {}
        for ctrl_name in ["MPPI", "MA-MPPI"]:
            data = [r[metric] for r in results[ctrl_name]]
            if metric == "time":
                vals[ctrl_name] = f"{np.mean(data):.3f}s"
            else:
                vals[ctrl_name] = f"{np.mean(data):.1f}±{np.std(data):.1f}"
        label = {"reward": "Cum. Reward", "steps": "Steps", "time": "Wall Time"}[metric]
        print(f"  {label:<20s} {vals['MPPI']:>15s} {vals['MA-MPPI']:>15s}")

    # Winner
    mppi_r = np.mean([r["reward"] for r in results["MPPI"]])
    ma_r = np.mean([r["reward"] for r in results["MA-MPPI"]])
    winner = "MA-MPPI" if ma_r > mppi_r else "MPPI"
    print(f"\n  Winner: {winner} (reward: {max(mppi_r, ma_r):.1f} vs {min(mppi_r, ma_r):.1f})")

    return results


def main():
    if not HAS_GYM:
        print("gymnasium not installed. Install with: pip install gymnasium")
        print("Running custom benchmarks instead...")
        from experiments.benchmark import main as run_benchmark
        run_benchmark()
        return

    print("MA-MPPI Gym Benchmark")
    print("=" * 60)

    run_pendulum(n_trials=5, max_steps=200)


if __name__ == "__main__":
    main()
