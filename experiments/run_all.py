#!/usr/bin/env python3
"""
Run all experiments: custom benchmarks + UAV + IEEE 39-bus.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapft import MPPI, MAMPPI
from experiments.environments import QuadrotorEnv, IEEE39BusEnv


def run_controller(env_dynamics, env_cost, x0, state_dim, control_dim,
                   u_min, u_max, CtrlClass, max_steps, horizon=20,
                   n_samples=500, sigma=1.0, lambda_=1.0):
    """Generic controller runner."""
    controller = CtrlClass(
        dynamics_fn=env_dynamics, cost_fn=env_cost,
        state_dim=state_dim, control_dim=control_dim,
        horizon=horizon, n_samples=n_samples,
        noise_sigma=sigma, lambda_=lambda_,
        u_min=u_min, u_max=u_max,
    )
    x = x0.copy()
    trajectory = [x.copy()]
    total_cost = 0.0
    t0 = time.time()

    for step in range(max_steps):
        u = controller.command(x)
        x = env_dynamics(x, u)
        if x.ndim > 1:
            x = x[0]
        trajectory.append(x.copy())
        total_cost += env_cost(x, u)

    return {
        "trajectory": np.array(trajectory),
        "total_cost": total_cost,
        "time": time.time() - t0,
        "steps": max_steps,
    }


def run_uav_experiment(n_trials=3, max_steps=200):
    """UAV obstacle avoidance: MPPI vs MA-MPPI."""
    print(f"\n{'='*60}")
    print("  UAV 3D Obstacle Avoidance")
    print(f"{'='*60}")

    env = QuadrotorEnv()
    results = {"MPPI": [], "MA-MPPI": []}

    for trial in range(n_trials):
        for name, Cls in [("MPPI", MPPI), ("MA-MPPI", MAMPPI)]:
            np.random.seed(trial * 77 + (0 if name == "MPPI" else 1))
            x0 = env.reset()

            r = run_controller(
                env.dynamics, env.cost, x0,
                env.state_dim, env.control_dim,
                env.u_min, env.u_max, Cls,
                max_steps=max_steps,
                horizon=15, n_samples=400,
                sigma=2.0, lambda_=1.0,
            )

            final_pos = r["trajectory"][-1][:3]
            reached = env.is_reached(final_pos, tol=1.0)
            collisions = sum(1 for t in range(len(r["trajectory"]))
                           if env.is_collision(r["trajectory"][t][:3]))

            results[name].append({
                "trial": trial,
                "reached": reached,
                "final_dist": float(np.linalg.norm(final_pos - env.goal)),
                "collisions": collisions,
                "total_cost": r["total_cost"],
                "time": r["time"],
            })

    print(f"\n  {'Metric':<20s} {'MPPI':>15s} {'MA-MPPI':>15s}")
    print(f"  {'-'*50}")
    for metric in ["reached", "final_dist", "collisions", "total_cost", "time"]:
        vals = {}
        for name in ["MPPI", "MA-MPPI"]:
            data = [r[metric] for r in results[name]]
            if metric == "reached":
                vals[name] = f"{sum(data)}/{len(data)}"
            elif metric == "time":
                vals[name] = f"{np.mean(data):.2f}s"
            else:
                vals[name] = f"{np.mean(data):.2f}±{np.std(data):.2f}"
        label = {"reached": "Goal Reached", "final_dist": "Final Dist",
                 "collisions": "Collisions", "total_cost": "Total Cost",
                 "time": "Wall Time"}[metric]
        print(f"  {label:<20s} {vals['MPPI']:>15s} {vals['MA-MPPI']:>15s}")

    return results


def run_power_experiment(n_trials=3, max_steps=300):
    """IEEE 39-bus power system stability."""
    print(f"\n{'='*60}")
    print("  IEEE 39-Bus Power System Stability")
    print(f"{'='*60}")

    faults = ["three_phase", "load_change", "generator_trip"]
    all_results = {}

    for fault_type in faults:
        results = {"MPPI": [], "MA-MPPI": []}

        for trial in range(n_trials):
            for name, Cls in [("MPPI", MPPI), ("MA-MPPI", MAMPPI)]:
                np.random.seed(trial * 33 + (0 if name == "MPPI" else 1))
                env = IEEE39BusEnv(dt=0.01)
                x0 = env.reset()

                # Run 50 steps normal, then apply fault
                controller = Cls(
                    dynamics_fn=env.dynamics, cost_fn=env.cost,
                    state_dim=env.state_dim, control_dim=env.control_dim,
                    horizon=20, n_samples=400,
                    noise_sigma=0.5, lambda_=0.5,
                    u_min=env.u_min, u_max=env.u_max,
                )

                x = x0.copy()
                total_cost = 0.0
                recovery_step = None
                t0 = time.time()

                for step in range(max_steps):
                    if step == 50:
                        env.apply_fault(fault_type)
                    if step == 65:
                        env.clear_fault()

                    u = controller.command(x)
                    x = env.dynamics(x, u)
                    if x.ndim > 1:
                        x = x[0]
                    total_cost += env.cost(x, u)

                    if step > 65 and recovery_step is None and env.is_stable(x, 0.1):
                        recovery_step = step - 65

                elapsed = time.time() - t0
                omega_final = x[env.n_gen:]

                results[name].append({
                    "trial": trial,
                    "max_freq_dev": float(np.max(np.abs(omega_final))),
                    "recovery_steps": recovery_step if recovery_step else max_steps,
                    "total_cost": total_cost,
                    "stable": env.is_stable(x, 0.5),
                    "time": elapsed,
                })

        all_results[fault_type] = results

        print(f"\n  Fault: {fault_type}")
        print(f"  {'Metric':<20s} {'MPPI':>15s} {'MA-MPPI':>15s}")
        print(f"  {'-'*50}")
        for metric in ["stable", "max_freq_dev", "recovery_steps", "total_cost"]:
            vals = {}
            for nm in ["MPPI", "MA-MPPI"]:
                data = [r[metric] for r in results[nm]]
                if metric == "stable":
                    vals[nm] = f"{sum(data)}/{len(data)}"
                else:
                    vals[nm] = f"{np.mean(data):.2f}±{np.std(data):.2f}"
            label = {"stable": "Stable", "max_freq_dev": "Max Freq Dev",
                     "recovery_steps": "Recovery Steps", "total_cost": "Total Cost"}[metric]
            print(f"  {label:<20s} {vals['MPPI']:>15s} {vals['MA-MPPI']:>15s}")

    return all_results


def main():
    print("MA-MPPI Complete Benchmark Suite")
    print("=" * 60)

    # Custom trap scenarios
    from experiments.benchmark import main as run_custom
    run_custom()

    # UAV
    run_uav_experiment(n_trials=3, max_steps=200)

    # Power system
    run_power_experiment(n_trials=3, max_steps=300)

    print(f"\n{'='*60}")
    print("  All experiments complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
