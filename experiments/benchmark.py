#!/usr/bin/env python3
"""
Benchmark: MPPI vs MA-MPPI on multiple test scenarios.

Scenarios:
1. Double-well trap (2D): classic local minimum test
2. U-shaped corridor (2D): narrow passage with traps
3. Multi-trap maze (2D): multiple local minima
4. High-dim quadrotor-like (6D): higher dimensional control
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapft import MPPI, MAMPPI


# ============================================================
# Scenario 1: Double-Well Potential (2D)
# ============================================================
def double_well_scenario():
    """
    2D point mass in a double-well potential.
    Local minimum at (-1, 0), global minimum at (2, 0).
    Start near local minimum.
    """
    goal = np.array([2.0, 0.0])
    dt = 0.1

    def dynamics(x, u):
        return x + u * dt

    def cost(x, u):
        # Double-well: V(x) = (x[0]^2 - 1)^2 + x[1]^2 + goal attraction
        if x.ndim == 1:
            well = (x[0]**2 - 1)**2 + x[1]**2
            goal_cost = np.sum((x - goal)**2)
            ctrl = 0.01 * np.sum(u**2)
            return well * 5.0 + goal_cost + ctrl
        else:
            well = (x[:, 0]**2 - 1)**2 + x[:, 1]**2
            goal_cost = np.sum((x - goal)**2, axis=1)
            ctrl = 0.01 * np.sum(u**2, axis=1)
            return well * 5.0 + goal_cost + ctrl

    x0 = np.array([-1.0, 0.0])  # Start near local minimum
    return dynamics, cost, x0, goal, dt, "Double-Well (2D)"


# ============================================================
# Scenario 2: U-Shaped Trap (2D)
# ============================================================
def u_trap_scenario():
    """
    2D navigation with U-shaped obstacle creating a trap.
    """
    goal = np.array([8.0, 5.0])
    dt = 0.1

    obstacles = [
        (3.0, 2.0, 3.0, 7.0),   # left wall
        (3.0, 2.0, 7.0, 2.0),   # bottom wall
        (7.0, 2.0, 7.0, 7.0),   # right wall
    ]

    def dynamics(x, u):
        return x + np.clip(u, -2.0, 2.0) * dt

    def _obstacle_cost(x):
        """Inverse-distance penalty from walls."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        cost = np.zeros(len(x))
        for x1, y1, x2, y2 in obstacles:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2) + 1e-10
            t = np.clip(((x[:, 0] - x1)*dx + (x[:, 1] - y1)*dy) / length**2, 0, 1)
            px, py = x1 + t*dx, y1 + t*dy
            dist = np.sqrt((x[:, 0] - px)**2 + (x[:, 1] - py)**2)
            cost += np.where(dist < 0.5, 1000.0 / (dist + 0.01), 0.0)
        return cost

    def cost(x, u):
        if x.ndim == 1:
            goal_c = np.sum((x - goal)**2)
            obs_c = float(_obstacle_cost(x).item())
            ctrl = 0.01 * np.sum(u**2)
            return goal_c + obs_c + ctrl
        else:
            goal_c = np.sum((x - goal)**2, axis=1)
            obs_c = _obstacle_cost(x)
            ctrl = 0.01 * np.sum(u**2, axis=1)
            return goal_c + obs_c + ctrl

    x0 = np.array([5.0, 5.0])  # Inside the U-trap
    return dynamics, cost, x0, goal, dt, "U-Trap (2D)"


# ============================================================
# Scenario 3: Multi-Trap (2D)
# ============================================================
def multi_trap_scenario():
    """
    Multiple Gaussian traps between start and goal.
    """
    goal = np.array([10.0, 10.0])
    dt = 0.1

    traps = [
        (3.0, 3.0, 2.0),    # (cx, cy, strength)
        (5.0, 7.0, 3.0),
        (7.0, 4.0, 2.5),
        (8.0, 8.0, 1.5),
    ]

    def dynamics(x, u):
        return x + np.clip(u, -2.0, 2.0) * dt

    def cost(x, u):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        goal_c = np.sum((x - goal)**2, axis=1)
        trap_c = np.zeros(len(x))
        for cx, cy, s in traps:
            dist_sq = (x[:, 0] - cx)**2 + (x[:, 1] - cy)**2
            trap_c += s * np.exp(-dist_sq / 2.0)  # attractive trap
        # Trap makes it look like a good spot (negative cost = attraction)
        # We want to model as: goal cost is high but trap gives false low cost
        total = goal_c - trap_c * 10.0  # traps reduce apparent cost
        ctrl = 0.01 * np.sum(u.reshape(-1, u.shape[-1])**2, axis=1)
        result = total + ctrl
        return float(result[0]) if squeeze else result

    x0 = np.array([0.0, 0.0])
    return dynamics, cost, x0, goal, dt, "Multi-Trap (2D)"


# ============================================================
# Scenario 4: Higher-dimensional (6D state, 3D control)
# ============================================================
def high_dim_scenario():
    """
    6D double integrator (pos + vel in 3D) with potential traps.
    """
    goal_pos = np.array([5.0, 5.0, 5.0])
    dt = 0.05

    def dynamics(x, u):
        if x.ndim == 1:
            x = x.reshape(1, -1)
            u = u.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        pos = x[:, :3] + x[:, 3:] * dt
        vel = x[:, 3:] + np.clip(u, -5.0, 5.0) * dt
        vel = np.clip(vel, -3.0, 3.0)  # velocity limit
        result = np.hstack([pos, vel])
        return result[0] if squeeze else result

    def cost(x, u):
        if x.ndim == 1:
            pos = x[:3]
            vel = x[3:]
            goal_c = np.sum((pos - goal_pos)**2)
            # Trap at (2, 2, 2)
            trap_dist = np.sum((pos - np.array([2.0, 2.0, 2.0]))**2)
            trap_c = 5.0 * np.exp(-trap_dist / 2.0)
            vel_c = 0.1 * np.sum(vel**2)
            ctrl = 0.01 * np.sum(u**2)
            return goal_c - trap_c * 10.0 + vel_c + ctrl
        else:
            pos = x[:, :3]
            vel = x[:, 3:]
            goal_c = np.sum((pos - goal_pos)**2, axis=1)
            trap_dist = np.sum((pos - np.array([2.0, 2.0, 2.0]))**2, axis=1)
            trap_c = 5.0 * np.exp(-trap_dist / 2.0)
            vel_c = 0.1 * np.sum(vel**2, axis=1)
            ctrl = 0.01 * np.sum(u**2, axis=1)
            return goal_c - trap_c * 10.0 + vel_c + ctrl

    x0 = np.zeros(6)
    return dynamics, cost, x0, goal_pos, dt, "High-Dim (6D)"


# ============================================================
# Runner
# ============================================================
def run_experiment(scenario_fn, max_steps=300, n_trials=5):
    """Run MPPI vs MA-MPPI comparison."""
    dynamics, cost, x0, goal, dt, name = scenario_fn()
    goal_flat = goal.flatten()

    results = {"MPPI": [], "MA-MPPI": []}

    for trial in range(n_trials):
        np.random.seed(trial * 42)

        for ctrl_name, CtrlClass in [("MPPI", MPPI), ("MA-MPPI", MAMPPI)]:
            state_dim = len(x0)
            ctrl_dim = len(goal_flat) if state_dim <= 3 else 3

            kwargs = dict(
                dynamics_fn=dynamics, cost_fn=cost,
                state_dim=state_dim, control_dim=ctrl_dim,
                horizon=25, n_samples=800,
                noise_sigma=1.0, lambda_=1.0,
                u_min=np.full(ctrl_dim, -2.0),
                u_max=np.full(ctrl_dim, 2.0),
            )
            if CtrlClass == MAMPPI:
                kwargs["device"] = "cpu"

            controller = CtrlClass(**kwargs)
            x = x0.copy()

            trajectory = [x.copy()]
            total_cost = 0.0
            t_start = time.time()
            reached = False

            for step in range(max_steps):
                u = controller.command(x)
                x = dynamics(x, u)
                if np.ndim(x) > 1:
                    x = x[0]
                trajectory.append(x.copy())
                total_cost += cost(x, u)

                # Check goal reached
                pos = x[:len(goal_flat)] if len(x) > len(goal_flat) else x
                if np.linalg.norm(pos - goal_flat) < 0.5:
                    reached = True
                    break

            elapsed = time.time() - t_start
            trajectory = np.array(trajectory)

            results[ctrl_name].append({
                "trial": trial,
                "steps": step + 1,
                "reached": reached,
                "total_cost": total_cost,
                "final_dist": float(np.linalg.norm(
                    trajectory[-1][:len(goal_flat)] - goal_flat
                )),
                "time": elapsed,
                "path_length": float(np.sum(np.linalg.norm(
                    np.diff(trajectory, axis=0), axis=1
                ))),
            })

    return name, results


def print_results(name, results):
    """Pretty-print comparison results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20s} {'MPPI':>15s} {'MA-MPPI':>15s}")
    print(f"  {'-'*50}")

    for metric in ["reached", "steps", "final_dist", "total_cost", "time"]:
        vals = {}
        for ctrl_name in ["MPPI", "MA-MPPI"]:
            data = [r[metric] for r in results[ctrl_name]]
            if metric == "reached":
                vals[ctrl_name] = f"{sum(data)}/{len(data)}"
            elif metric == "time":
                vals[ctrl_name] = f"{np.mean(data):.3f}s"
            else:
                vals[ctrl_name] = f"{np.mean(data):.2f}±{np.std(data):.2f}"

        label = {
            "reached": "Goal Reached",
            "steps": "Steps",
            "final_dist": "Final Distance",
            "total_cost": "Total Cost",
            "time": "Wall Time",
        }[metric]
        print(f"  {label:<20s} {vals['MPPI']:>15s} {vals['MA-MPPI']:>15s}")


def main():
    scenarios = [
        double_well_scenario,
        u_trap_scenario,
        multi_trap_scenario,
        high_dim_scenario,
    ]

    print("MPPI vs MA-MPPI Benchmark")
    print("=" * 60)

    all_results = {}
    for scenario_fn in scenarios:
        name, results = run_experiment(scenario_fn, max_steps=300, n_trials=5)
        print_results(name, results)
        all_results[name] = results

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    mppi_wins, ma_wins = 0, 0
    for name, results in all_results.items():
        mppi_reached = sum(r["reached"] for r in results["MPPI"])
        ma_reached = sum(r["reached"] for r in results["MA-MPPI"])
        mppi_dist = np.mean([r["final_dist"] for r in results["MPPI"]])
        ma_dist = np.mean([r["final_dist"] for r in results["MA-MPPI"]])

        better = "MA-MPPI" if (ma_reached > mppi_reached or
                               (ma_reached == mppi_reached and ma_dist < mppi_dist)) else "MPPI"
        if better == "MA-MPPI":
            ma_wins += 1
        else:
            mppi_wins += 1
        print(f"  {name:<25s} Winner: {better}")

    print(f"\n  MA-MPPI wins: {ma_wins}/{len(all_results)}")


if __name__ == "__main__":
    main()
