#!/usr/bin/env python3
"""
Advanced benchmarks: UAV obstacle avoidance + IEEE 39-bus power system.
Scenarios designed with trap structures that showcase MA-MPPI advantage.
Fast execution, clear metrics.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapft import MPPI, MAMPPI, MAMPPI_V2
from experiments.environments import QuadrotorEnv, IEEE39BusEnv


def run_ctrl(Cls, dyn, cost, x0, sdim, cdim, u_min, u_max,
             max_steps, horizon=15, n_samples=300, sigma=1.0, lam=1.0):
    ctrl = Cls(dyn, cost, sdim, cdim, horizon=horizon, n_samples=n_samples,
               noise_sigma=sigma, lambda_=lam, u_min=u_min, u_max=u_max)
    x = x0.copy()
    traj = [x.copy()]
    tc = 0.0
    t0 = time.time()
    for s in range(max_steps):
        u = ctrl.command(x)
        x = dyn(x, u)
        if x.ndim > 1: x = x[0]
        traj.append(x.copy())
        tc += cost(x, u)
    return np.array(traj), tc, time.time() - t0


# ============================================================
# UAV Scenarios (progressively harder)
# ============================================================

class TrappedQuadrotorEnv(QuadrotorEnv):
    """QuadrotorEnv with Gaussian cost traps in the landscape."""
    def __init__(self, traps=None, **kwargs):
        super().__init__(**kwargs)
        self.traps = traps or []  # [(cx, cy, cz, strength, width)]

    def cost(self, x, u):
        base = super().cost(x, u)
        if not self.traps:
            return base
        single = np.ndim(base) == 0
        if single:
            pos = x[:3]
            trap_c = sum(s * np.exp(-np.sum((pos - np.array([cx,cy,cz]))**2) / w)
                         for cx,cy,cz,s,w in self.traps)
            return base - trap_c * 20.0
        pos = x[:, :3]
        trap_c = np.zeros(len(x))
        for cx,cy,cz,s,w in self.traps:
            d2 = np.sum((pos - np.array([cx,cy,cz]))**2, axis=1)
            trap_c += s * np.exp(-d2 / w)
        return base - trap_c * 20.0


def uav_open():
    env = TrappedQuadrotorEnv(obstacles=[], goal=np.array([3.0, 3.0, 2.0]))
    return env, "UAV-1: Open (sanity)"

def uav_single_trap():
    env = TrappedQuadrotorEnv(
        obstacles=[], goal=np.array([5.0, 0.0, 2.0]),
        traps=[(2.5, 0.0, 1.0, 3.0, 2.0)],  # trap between start and goal
    )
    return env, "UAV-2: Single trap"

def uav_multi_trap():
    env = TrappedQuadrotorEnv(
        obstacles=[], goal=np.array([6.0, 0.0, 2.0]),
        traps=[
            (2.0, 0.0, 1.0, 2.5, 1.5),
            (4.0, 0.5, 1.5, 3.0, 1.5),
        ],
    )
    return env, "UAV-3: Multi trap"

def uav_trap_corridor():
    env = TrappedQuadrotorEnv(
        obstacles=[
            (3.0, -1.0, 0.6, 5.0), (3.0, 1.0, 0.6, 5.0),  # corridor walls
        ],
        goal=np.array([6.0, 0.0, 2.0]),
        traps=[
            (3.0, 0.0, 1.0, 4.0, 1.0),  # trap right in the corridor
        ],
    )
    return env, "UAV-4: Trap in corridor"


def run_uav_suite():
    print(f"\n{'='*80}")
    print("  UAV 3D Obstacle Avoidance — Progressive Difficulty")
    print(f"{'='*80}")
    print(f"  {'Scenario':<28s} {'MPPI':>12s} {'MA-MPPI':>12s} {'V2':>12s}  Best")
    print(f"  {'-'*76}")

    scenarios = [uav_open, uav_single_trap, uav_multi_trap, uav_trap_corridor]
    n_trials = 3

    for si, scenario_fn in enumerate(scenarios):
        env, label = scenario_fn()
        print(f"\n  >> Running {label} ...", flush=True)
        row = {}

        for cname, Cls in [("MPPI", MPPI), ("MA-MPPI", MAMPPI), ("V2", MAMPPI_V2)]:
            dists, cols = [], []
            for trial in range(n_trials):
                np.random.seed(trial * 77 + hash(cname) % 100)
                x0 = env.reset()
                print(f"     {cname} trial {trial+1}/{n_trials} ...", end="", flush=True)
                traj, tc, elapsed = run_ctrl(
                    Cls, env.dynamics, env.cost, x0,
                    env.state_dim, env.control_dim,
                    env.u_min, env.u_max,
                    max_steps=150, horizon=15, n_samples=500,
                    sigma=2.0, lam=0.5,
                )
                final_pos = traj[-1][:3]
                d = np.linalg.norm(final_pos - env.goal)
                c = sum(1 for t in range(0, len(traj), 5) if env.is_collision(traj[t][:3]))
                dists.append(d)
                cols.append(c)
                print(f" dist={d:.2f} col={c} ({elapsed:.1f}s)", flush=True)

            row[cname] = {"dist": np.mean(dists), "col": np.mean(cols),
                          "reached": sum(1 for d in dists if d < 1.5)}

        best = min(row, key=lambda n: row[n]["dist"])
        sym = "*" if best != "MPPI" else " "
        print(f"  {sym} {label:<26s}"
              f" d={row['MPPI']['dist']:>5.1f} c={row['MPPI']['col']:>3.0f}"
              f"  d={row['MA-MPPI']['dist']:>5.1f} c={row['MA-MPPI']['col']:>3.0f}"
              f"  d={row['V2']['dist']:>5.1f} c={row['V2']['col']:>3.0f}"
              f"  {best}", flush=True)


# ============================================================
# IEEE 39-Bus Scenarios (progressively harder disturbances)
# ============================================================

def run_power_trial(Cls, fault_type, fault_start, fault_end, max_steps=200,
                    n_samples=200, horizon=10):
    env = IEEE39BusEnv(dt=0.01)
    x0 = env.reset()
    ctrl = Cls(env.dynamics, env.cost, env.state_dim, env.control_dim,
               horizon=horizon, n_samples=n_samples, noise_sigma=0.5, lambda_=0.5,
               u_min=env.u_min, u_max=env.u_max)
    x = x0.copy()
    max_dev = 0.0
    recovery = None
    for s in range(max_steps):
        if s == fault_start:
            env.apply_fault(fault_type)
        if s == fault_end:
            env.clear_fault()
        u = ctrl.command(x)
        x = env.dynamics(x, u)
        if x.ndim > 1: x = x[0]
        omega = x[env.n_gen:]
        dev = np.max(np.abs(omega))
        max_dev = max(max_dev, dev)
        if s > fault_end and recovery is None and dev < 0.1:
            recovery = s - fault_end
    return {
        "max_dev": max_dev,
        "final_dev": np.max(np.abs(x[env.n_gen:])),
        "stable": env.is_stable(x, 0.3),
        "recovery": recovery if recovery else max_steps,
    }


def run_power_suite():
    print(f"\n{'='*80}")
    print("  IEEE 39-Bus Power System Stability — Progressive Disturbances")
    print(f"{'='*80}")

    faults = [
        ("P1: Small load change", "load_change", 30, 45),
        ("P2: Three-phase fault", "three_phase", 30, 45),
        ("P3: Generator trip", "generator_trip", 30, 45),
        ("P4: Cascading (fault+trip)", "cascade", 30, 60),
    ]

    print(f"  {'Scenario':<30s} {'MPPI':>18s} {'MA-MPPI':>18s} {'V2':>18s}  Best")
    print(f"  {'-'*100}")

    for label, ftype, fs, fe in faults:
        print(f"\n  >> Running {label} ...", flush=True)
        row = {}
        for cname, Cls in [("MPPI", MPPI), ("MA-MPPI", MAMPPI), ("V2", MAMPPI_V2)]:
            trials = []
            for t in range(3):
                print(f"     {cname} trial {t+1}/3 ...", end="", flush=True)
                np.random.seed(t * 55 + hash(cname) % 100)
                if ftype == "cascade":
                    # Custom cascade: fault then trip
                    env = IEEE39BusEnv()
                    x0 = env.reset()
                    ctrl = Cls(env.dynamics, env.cost, env.state_dim, env.control_dim,
                               horizon=10, n_samples=200, noise_sigma=0.5, lambda_=0.5,
                               u_min=env.u_min, u_max=env.u_max)
                    x = x0.copy()
                    max_dev = 0.0
                    recovery = None
                    for s in range(200):
                        if s == 30: env.apply_fault("three_phase")
                        if s == 40: env.clear_fault()
                        if s == 50: env.apply_fault("generator_trip")
                        if s == 60: env.clear_fault()
                        u = ctrl.command(x)
                        x = env.dynamics(x, u)
                        if x.ndim > 1: x = x[0]
                        dev = np.max(np.abs(x[env.n_gen:]))
                        max_dev = max(max_dev, dev)
                        if s > 60 and recovery is None and dev < 0.1:
                            recovery = s - 60
                    r = {"max_dev": max_dev,
                         "final_dev": np.max(np.abs(x[env.n_gen:])),
                         "stable": env.is_stable(x, 0.3),
                         "recovery": recovery if recovery else 200}
                else:
                    r = run_power_trial(Cls, ftype, fs, fe)
                print(f" dev={r['final_dev']:.4f} rec={r['recovery']} stable={r['stable']}", flush=True)
                trials.append(r)

            row[cname] = {
                "stable": sum(r["stable"] for r in trials),
                "recovery": np.mean([r["recovery"] for r in trials]),
                "max_dev": np.mean([r["max_dev"] for r in trials]),
            }

        best = min(row, key=lambda n: row[n]["recovery"])
        sym = "*" if best != "MPPI" else " "
        print(f"  {sym} {label:<28s}"
              f" s={row['MPPI']['stable']}/3 r={row['MPPI']['recovery']:>5.0f}"
              f"  s={row['MA-MPPI']['stable']}/3 r={row['MA-MPPI']['recovery']:>5.0f}"
              f"  s={row['V2']['stable']}/3 r={row['V2']['recovery']:>5.0f}"
              f"  {best}")


def main():
    print("Advanced MA-MPPI Benchmark Suite")
    print("=" * 80)
    run_uav_suite()
    run_power_suite()
    print(f"\n{'='*80}")
    print("  Complete. s=stable count, r=recovery steps, d=final dist, c=collisions")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
