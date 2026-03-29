#!/usr/bin/env python3
"""
Progressive difficulty benchmark: easy → hard, showing where MA-MPPI shines.
Fast execution, clear metrics.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapft import MPPI, MAMPPI, MAMPPI_V2
from mapft.ma_mppi_v3 import MAMPPI_V3


def make_controller(Cls, dynamics, cost, sdim, cdim, **kw):
    return Cls(dynamics, cost, sdim, cdim,
               horizon=15, n_samples=300, noise_sigma=1.0, lambda_=1.0,
               u_min=np.full(cdim, -2.0), u_max=np.full(cdim, 2.0), **kw)


def run(dynamics, cost, x0, goal, name, max_steps=200, n_trials=3):
    sdim, cdim = len(x0), len(goal)
    res = {}
    for cname, Cls in [("MPPI", MPPI), ("V2", MAMPPI_V2), ("V3", MAMPPI_V3)]:
        reached, dists, costs, times = [], [], [], []
        for t in range(n_trials):
            np.random.seed(t * 99)
            ctrl = make_controller(Cls, dynamics, cost, sdim, cdim)
            x = x0.copy()
            tc = 0.0
            t0 = time.time()
            ok = False
            for s in range(max_steps):
                u = ctrl.command(x)
                x = dynamics(x, u)
                tc += cost(x, u)
                if np.linalg.norm(x[:len(goal)] - goal) < 0.5:
                    ok = True; break
            reached.append(ok)
            dists.append(np.linalg.norm(x[:len(goal)] - goal))
            costs.append(tc)
            times.append(time.time() - t0)
        res[cname] = {"reach": sum(reached), "n": n_trials,
                      "dist": np.mean(dists), "time": np.mean(times)}
    return name, res


def print_row(name, res):
    m, v2, v3 = res["MPPI"], res["V2"], res["V3"]
    best = min(["MPPI", "V2", "V3"],
               key=lambda n: (-res[n]["reach"], res[n]["dist"]))
    sym = "*" if best != "MPPI" else " "
    print(f"  {sym} {name:<26s} {m['reach']}/{m['n']} {m['dist']:>5.2f}  "
          f"{v2['reach']}/{v2['n']} {v2['dist']:>5.2f}  "
          f"{v3['reach']}/{v3['n']} {v3['dist']:>5.2f}  {best}")


# ============================================================
# Level 1: No trap (sanity check — both should succeed)
# ============================================================
def level1():
    goal = np.array([5.0, 5.0])
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1: return np.sum((x-goal)**2) + 0.01*np.sum(u**2)
        return np.sum((x-goal)**2, axis=1) + 0.01*np.sum(u**2, axis=1)
    return run(dyn, cost, np.zeros(2), goal, "L1: Open field (no trap)")


# ============================================================
# Level 2: Single shallow trap
# ============================================================
def level2():
    goal = np.array([6.0, 6.0])
    trap = np.array([3.0, 3.0])
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x-goal)**2)
            t = 3.0 * np.exp(-np.sum((x-trap)**2)/2.0)
            return g - t*5 + 0.01*np.sum(u**2)
        g = np.sum((x-goal)**2, axis=1)
        t = 3.0 * np.exp(-np.sum((x-trap)**2, axis=1)/2.0)
        return g - t*5 + 0.01*np.sum(u**2, axis=1)
    return run(dyn, cost, np.zeros(2), goal, "L2: Single shallow trap")


# ============================================================
# Level 3: Deep U-shaped trap
# ============================================================
def level3():
    goal = np.array([8.0, 5.0])
    dt = 0.1
    walls = [(3,2,3,7), (3,2,7,2), (7,2,7,7)]
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def wall_cost(x):
        if x.ndim == 1: x = x.reshape(1,-1)
        c = np.zeros(len(x))
        for x1,y1,x2,y2 in walls:
            dx,dy = x2-x1, y2-y1
            L = np.sqrt(dx**2+dy**2)+1e-10
            t = np.clip(((x[:,0]-x1)*dx+(x[:,1]-y1)*dy)/L**2, 0, 1)
            d = np.sqrt((x[:,0]-x1-t*dx)**2 + (x[:,1]-y1-t*dy)**2)
            c += np.where(d < 0.5, 500.0/(d+0.01), 0.0)
        return c
    def cost(x, u):
        if x.ndim == 1:
            return np.sum((x-goal)**2) + wall_cost(x).item() + 0.01*np.sum(u**2)
        return np.sum((x-goal)**2,axis=1) + wall_cost(x) + 0.01*np.sum(u**2,axis=1)
    return run(dyn, cost, np.array([5.0, 5.0]), goal, "L3: U-shaped trap")


# ============================================================
# Level 4: Multiple traps in sequence
# ============================================================
def level4():
    goal = np.array([10.0, 10.0])
    traps = [(2,2,2), (4,6,3), (6,3,2.5), (8,8,2)]
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x-goal)**2)
            t = sum(s*np.exp(-((x[0]-cx)**2+(x[1]-cy)**2)/2) for cx,cy,s in traps)
            return g - t*8 + 0.01*np.sum(u**2)
        g = np.sum((x-goal)**2, axis=1)
        t = sum(s*np.exp(-((x[:,0]-cx)**2+(x[:,1]-cy)**2)/2) for cx,cy,s in traps)
        return g - t*8 + 0.01*np.sum(u**2, axis=1)
    return run(dyn, cost, np.zeros(2), goal, "L4: 4 sequential traps")


# ============================================================
# Level 5: Narrow corridor with dead-end
# ============================================================
def level5():
    goal = np.array([9.0, 5.0])
    dt = 0.1
    walls = [
        (2,1,2,4), (2,6,2,9),    # left gap at y=4-6
        (5,1,5,3), (5,5,5,9),    # middle gap at y=3-5
        (5,3,7,3), (7,3,7,1),    # dead-end branch down
        (8,1,8,4), (8,6,8,9),    # right gap at y=4-6
    ]
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def wall_cost(x):
        if x.ndim == 1: x = x.reshape(1,-1)
        c = np.zeros(len(x))
        for x1,y1,x2,y2 in walls:
            dx,dy = x2-x1, y2-y1
            L = np.sqrt(dx**2+dy**2)+1e-10
            t = np.clip(((x[:,0]-x1)*dx+(x[:,1]-y1)*dy)/L**2, 0, 1)
            d = np.sqrt((x[:,0]-x1-t*dx)**2 + (x[:,1]-y1-t*dy)**2)
            c += np.where(d < 0.4, 800.0/(d+0.01), 0.0)
        return c
    def cost(x, u):
        if x.ndim == 1:
            return np.sum((x-goal)**2) + wall_cost(x).item() + 0.01*np.sum(u**2)
        return np.sum((x-goal)**2,axis=1) + wall_cost(x) + 0.01*np.sum(u**2,axis=1)
    return run(dyn, cost, np.array([0.0, 5.0]), goal, "L5: Corridor + dead-end", max_steps=300)


# ============================================================
# Level 6: Repeated traps (memory reuse test)
# ============================================================
def level6():
    goal = np.array([12.0, 0.0])
    traps = [(i*3.0, 0.0, 2.0) for i in range(1, 5)]  # 4 evenly spaced
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x-goal)**2)
            t = sum(s*np.exp(-((x[0]-cx)**2+(x[1]-cy)**2)/1.5) for cx,cy,s in traps)
            return g - t*10 + 0.01*np.sum(u**2)
        g = np.sum((x-goal)**2, axis=1)
        t = sum(s*np.exp(-((x[:,0]-cx)**2+(x[:,1]-cy)**2)/1.5) for cx,cy,s in traps)
        return g - t*10 + 0.01*np.sum(u**2, axis=1)
    return run(dyn, cost, np.zeros(2), goal, "L6: Repeated traps (memory reuse)", max_steps=400)


# ============================================================
# Level 7: UAV 3D — multi-episode learning
# ============================================================
def level7():
    """UAV with traps, 5 episodes. MA-MPPI keeps memory across episodes."""
    goal = np.array([5.0, 0.0, 2.0])
    traps_3d = [(2.0, 0.0, 1.0, 3.0), (3.5, 0.5, 1.5, 2.5)]
    dt = 0.05
    def dyn(x, u):
        if x.ndim == 1:
            pos, vel = x[:3], x[3:]
            acc = np.clip(u, -5, 5) - 0.2 * vel
            v_new = np.clip(vel + acc * dt, -3, 3)
            return np.concatenate([pos + v_new * dt, v_new])
        pos, vel = x[:, :3], x[:, 3:]
        acc = np.clip(u, -5, 5) - 0.2 * vel
        v_new = np.clip(vel + acc * dt, -3, 3)
        return np.hstack([pos + v_new * dt, v_new])
    def cost(x, u):
        if x.ndim == 1:
            pos = x[:3]
            g = np.sum((pos - goal)**2)
            t = sum(s*np.exp(-np.sum((pos-np.array([cx,cy,cz]))**2)/2.0) for cx,cy,cz,s in traps_3d)
            return g - t*15 + 0.1*np.sum(x[3:]**2) + 0.01*np.sum(u**2)
        pos = x[:, :3]
        g = np.sum((pos-goal)**2, axis=1)
        t = sum(s*np.exp(-np.sum((pos-np.array([cx,cy,cz]))**2, axis=1)/2.0) for cx,cy,cz,s in traps_3d)
        return g - t*15 + 0.1*np.sum(x[:,3:]**2, axis=1) + 0.01*np.sum(u**2, axis=1)
    return run_iterative(dyn, cost, np.zeros(6), goal, "L7: UAV 3D (5 episodes)",
                         n_episodes=5, max_steps=200)


# ============================================================
# Level 8: Power system — multi-episode learning
# ============================================================
def level8():
    """4-gen power system with faults, 5 episodes. Memory accumulates."""
    n_gen = 4; dt = 0.02
    M = np.array([30.0, 25.0, 35.0, 100.0])
    D = M * 0.05
    P_m0 = np.array([3.0, 4.0, 5.0, 2.0])
    Y = np.array([[0,5,0,3],[5,0,4,0],[0,4,0,5],[3,0,5,0]], dtype=float)
    fault_power = [np.array([-2,0,0,0]), np.array([0,-2.5,0,0]), np.array([0,0,-2,0])]
    goal = np.zeros(4)

    def make_dyn(episode):
        """Each episode has fault at different generator."""
        fault_gen = episode % 3
        step_c = [0]
        def dyn(x, u):
            step_c[0] += 1
            if x.ndim == 1:
                delta, omega = x[:n_gen], x[n_gen:]
                u_c = np.clip(u, -3, 3)
                P_e = np.zeros(n_gen)
                for i in range(n_gen):
                    for j in range(n_gen):
                        if Y[i,j] > 0: P_e[i] += Y[i,j]*np.sin(delta[i]-delta[j])
                P_m = P_m0 + u_c
                s = step_c[0]
                if 30 <= s < 40: P_m = P_m + fault_power[fault_gen]
                d_omega = (P_m - P_e - D*omega) / M
                d_new = delta + omega*dt; o_new = omega + d_omega*dt
                d_new -= d_new[-1]
                return np.concatenate([d_new, o_new])
            return np.array([dyn(x[k], u[k]) for k in range(len(x))])
        return dyn

    def cost(x, u):
        if x.ndim == 1:
            return 50*np.sum(x[n_gen:]**2) + 5*(np.max(x[:n_gen])-np.min(x[:n_gen]))**2 + 0.1*np.sum(u**2)
        return 50*np.sum(x[:,n_gen:]**2,axis=1) + 5*(np.max(x[:,:n_gen],axis=1)-np.min(x[:,:n_gen],axis=1))**2 + 0.1*np.sum(u**2,axis=1)

    return run_iterative_power(make_dyn, cost, 8, 4, goal,
                               "L8: Power 4-gen (5 episodes)", n_episodes=5, max_steps=100)


# ============================================================
# Multi-episode runner: memory persists across episodes
# ============================================================
def run_iterative(dynamics, cost, x0, goal, name, n_episodes=5, max_steps=200):
    """Run multiple episodes. MPPI resets each time, MA-MPPI keeps memory."""
    sdim, cdim = len(x0), len(goal)
    print(f"\n  >> {name}: {n_episodes} episodes, memory persists for MA-MPPI/V2", flush=True)

    res = {}
    for cname, Cls in [("MPPI", MPPI), ("V2", MAMPPI_V2), ("V3", MAMPPI_V3)]:
        np.random.seed(42)
        ctrl = make_controller(Cls, dynamics, cost, sdim, cdim)
        ep_dists = []
        for ep in range(n_episodes):
            x = x0.copy()
            if hasattr(ctrl, 'new_episode'):
                ctrl.new_episode()  # V3: decay memory, keep learned features
            else:
                ctrl.U = np.zeros((ctrl.H, ctrl.n_u))
            for s in range(max_steps):
                u = ctrl.command(x)
                x = dynamics(x, u)
                if np.linalg.norm(x[:len(goal)] - goal) < 0.5:
                    break
            d = np.linalg.norm(x[:len(goal)] - goal)
            ep_dists.append(d)

        print(f"     {cname:>8s}: " + " → ".join(f"{d:.2f}" for d in ep_dists), flush=True)
        res[cname] = {"reach": sum(1 for d in ep_dists if d < 0.5), "n": n_episodes,
                      "dist": ep_dists[-1], "curve": ep_dists}
    return name, res


def run_iterative_power(make_dyn, cost, sdim, cdim, goal, name, n_episodes=5, max_steps=100):
    """Multi-episode for power system with varying faults."""
    print(f"\n  >> {name}: {n_episodes} episodes, rotating faults", flush=True)

    res = {}
    for cname, Cls in [("MPPI", MPPI), ("V2", MAMPPI_V2), ("V3", MAMPPI_V3)]:
        np.random.seed(42)
        ctrl = Cls(make_dyn(0), cost, sdim, cdim,
                   horizon=15, n_samples=300, noise_sigma=1.0, lambda_=1.0,
                   u_min=np.full(cdim, -3.0), u_max=np.full(cdim, 3.0))
        ep_devs = []
        for ep in range(n_episodes):
            dyn = make_dyn(ep)
            ctrl.dynamics = dyn
            if hasattr(ctrl, 'new_episode'):
                ctrl.new_episode()
            else:
                ctrl.U = np.zeros((ctrl.H, ctrl.n_u))
            x = np.random.randn(sdim) * 0.02
            for s in range(max_steps):
                u = ctrl.command(x)
                x = dyn(x, u)
                if x.ndim > 1: x = x[0]
            dev = np.max(np.abs(x[cdim:]))
            ep_devs.append(dev)

        print(f"     {cname:>8s}: " + " → ".join(f"{d:.3f}" for d in ep_devs), flush=True)
        res[cname] = {"reach": sum(1 for d in ep_devs if d < 0.1), "n": n_episodes,
                      "dist": ep_devs[-1], "curve": ep_devs}
    return name, res


def main():
    print("Progressive Difficulty Benchmark: MPPI vs MA-MPPI")
    print("=" * 78)
    print(f"  {'Level':<28s} {'MPPI':>9s}  {'V2':>9s}  {'V3':>9s}  Winner")
    print(f"  {'':28s} {'r  dist':>9s}  {'r  dist':>9s}  {'r  dist':>9s}")
    print(f"  {'-'*78}")

    levels = [level1, level2, level3, level4, level5, level6, level7, level8]
    for lvl in levels:
        name, res = lvl()
        print_row(name, res)
        print(flush=True)

    print(f"\n  * = MA-MPPI advantage (trap/non-convex scenario)")
    print("=" * 78)


if __name__ == "__main__":
    main()
