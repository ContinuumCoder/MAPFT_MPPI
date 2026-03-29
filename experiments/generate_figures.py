#!/usr/bin/env python3
"""
Generate publication-quality figures for the MA-MPPI paper.

Produces trajectory comparison plots (MPPI vs MA-MPPI) for:
  - L2: Single shallow trap (2D)
  - L4: Multiple sequential traps (2D)
  - L6: Repeated traps (2D)
  - UAV: 3D double-integrator with traps (XY projection)

Saves all figures to ../figures/ as PNG.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mapft import MPPI, MAMPPI_V2

# ── Global settings ──────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZON = 15
N_SAMPLES = 300
SIGMA = 1.0
LAMBDA = 1.0
MAX_STEPS = 200
SEED = 42

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.6",
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_controller(Cls, dynamics, cost, sdim, cdim, u_lim=2.0):
    return Cls(dynamics, cost, sdim, cdim,
               horizon=HORIZON, n_samples=N_SAMPLES,
               noise_sigma=SIGMA, lambda_=LAMBDA,
               u_min=np.full(cdim, -u_lim), u_max=np.full(cdim, u_lim))


def run_trajectory(Cls, dynamics, cost, x0, sdim, cdim, max_steps=MAX_STEPS,
                   u_lim=2.0):
    """Run a controller and return the state trajectory as (N, sdim)."""
    np.random.seed(SEED)
    ctrl = make_controller(Cls, dynamics, cost, sdim, cdim, u_lim=u_lim)
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(max_steps):
        u = ctrl.command(x)
        x = dynamics(x, u)
        traj.append(x.copy())
    return np.array(traj)


def draw_traps_2d(ax, traps, radius=1.0):
    """Draw shaded circles for trap locations. traps: list of (cx, cy, ...)."""
    for t in traps:
        cx, cy = t[0], t[1]
        circle = Circle((cx, cy), radius, facecolor="orange", edgecolor="darkorange",
                         alpha=0.25, linewidth=1.2, linestyle="--")
        ax.add_patch(circle)


def finalize_2d(ax, start, goal, traps, trap_radius=1.0, pad=1.0):
    """Add start/goal markers, traps, axis labels, and legend."""
    draw_traps_2d(ax, traps, radius=trap_radius)
    ax.plot(*start, "ko", markersize=8, zorder=5, label="Start")
    ax.plot(*goal, "k*", markersize=14, zorder=5, label="Goal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    # auto-range with padding
    ax.margins(0.05)


# ── Scenario definitions ────────────────────────────────────────────────────

# L2 -----------------------------------------------------------------------
def l2_scenario():
    goal = np.array([6.0, 6.0])
    trap = np.array([3.0, 3.0])
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x - goal)**2)
            t = 3.0 * np.exp(-np.sum((x - trap)**2) / 2.0)
            return g - t * 5 + 0.01 * np.sum(u**2)
        g = np.sum((x - goal)**2, axis=1)
        t = 3.0 * np.exp(-np.sum((x - trap)**2, axis=1) / 2.0)
        return g - t * 5 + 0.01 * np.sum(u**2, axis=1)
    traps = [(3.0, 3.0)]
    return dyn, cost, np.zeros(2), goal, traps, "L2: Single Shallow Trap"


# L4 -----------------------------------------------------------------------
def l4_scenario():
    goal = np.array([10.0, 10.0])
    traps_raw = [(2, 2, 2), (4, 6, 3), (6, 3, 2.5), (8, 8, 2)]
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x - goal)**2)
            t = sum(s * np.exp(-((x[0] - cx)**2 + (x[1] - cy)**2) / 2)
                    for cx, cy, s in traps_raw)
            return g - t * 8 + 0.01 * np.sum(u**2)
        g = np.sum((x - goal)**2, axis=1)
        t = sum(s * np.exp(-((x[:, 0] - cx)**2 + (x[:, 1] - cy)**2) / 2)
                for cx, cy, s in traps_raw)
        return g - t * 8 + 0.01 * np.sum(u**2, axis=1)
    traps = [(cx, cy) for cx, cy, _ in traps_raw]
    return dyn, cost, np.zeros(2), goal, traps, "L4: Multiple Sequential Traps"


# L6 -----------------------------------------------------------------------
def l6_scenario():
    goal = np.array([12.0, 0.0])
    traps_raw = [(i * 3.0, 0.0, 2.0) for i in range(1, 5)]
    dt = 0.1
    def dyn(x, u): return x + np.clip(u, -2, 2) * dt
    def cost(x, u):
        if x.ndim == 1:
            g = np.sum((x - goal)**2)
            t = sum(s * np.exp(-((x[0] - cx)**2 + (x[1] - cy)**2) / 1.5)
                    for cx, cy, s in traps_raw)
            return g - t * 10 + 0.01 * np.sum(u**2)
        g = np.sum((x - goal)**2, axis=1)
        t = sum(s * np.exp(-((x[:, 0] - cx)**2 + (x[:, 1] - cy)**2) / 1.5)
                for cx, cy, s in traps_raw)
        return g - t * 10 + 0.01 * np.sum(u**2, axis=1)
    traps = [(cx, cy) for cx, cy, _ in traps_raw]
    return dyn, cost, np.zeros(2), goal, traps, "L6: Repeated Traps"


# UAV -----------------------------------------------------------------------
def uav_scenario():
    goal = np.array([3.0, 0.0, 0.0])
    traps_3d = [(1.0, 0.0, 0.0, 3.0), (2.0, 0.2, 0.0, 2.5)]
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
            t = sum(s * np.exp(-np.sum((pos - np.array([cx, cy, cz]))**2) / 2.0)
                    for cx, cy, cz, s in traps_3d)
            return g - t * 15 + 0.1 * np.sum(x[3:]**2) + 0.01 * np.sum(u**2)
        pos = x[:, :3]
        g = np.sum((pos - goal)**2, axis=1)
        t = sum(s * np.exp(-np.sum((pos - np.array([cx, cy, cz]))**2, axis=1) / 2.0)
                for cx, cy, cz, s in traps_3d)
        return g - t * 15 + 0.1 * np.sum(x[:, 3:]**2, axis=1) + 0.01 * np.sum(u**2, axis=1)
    traps = [(cx, cy, cz) for cx, cy, cz, _ in traps_3d]
    return dyn, cost, np.zeros(6), goal, traps, "UAV: 3D Double Integrator"


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_2d_scenario(scenario_fn, filename, max_steps=MAX_STEPS):
    dyn, cost, x0, goal, traps, title = scenario_fn()
    sdim, cdim = len(x0), len(goal)

    print(f"  Running MPPI for {title}...", flush=True)
    traj_mppi = run_trajectory(MPPI, dyn, cost, x0, sdim, cdim,
                               max_steps=max_steps)
    print(f"  Running MA-MPPI for {title}...", flush=True)
    traj_ma = run_trajectory(MAMPPI_V2, dyn, cost, x0, sdim, cdim,
                             max_steps=max_steps)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(traj_mppi[:, 0], traj_mppi[:, 1], "r-", linewidth=1.5,
            alpha=0.85, label="MPPI", zorder=3)
    ax.plot(traj_mppi[-1, 0], traj_mppi[-1, 1], "rx", markersize=8, zorder=4)
    ax.plot(traj_ma[:, 0], traj_ma[:, 1], "b-", linewidth=1.5,
            alpha=0.85, label="MA-MPPI", zorder=3)
    ax.plot(traj_ma[-1, 0], traj_ma[-1, 1], "b^", markersize=7, zorder=4)
    finalize_2d(ax, x0, goal, traps)
    ax.set_title(title, fontsize=12)

    fpath = os.path.join(FIG_DIR, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fpath}")
    return traj_mppi, traj_ma


def plot_uav_scenario(filename):
    dyn, cost, x0, goal, traps, title = uav_scenario()
    sdim, cdim = 6, 3

    print(f"  Running MPPI for {title}...", flush=True)
    traj_mppi = run_trajectory(MPPI, dyn, cost, x0, sdim, cdim,
                               max_steps=MAX_STEPS, u_lim=5.0)
    print(f"  Running MA-MPPI for {title}...", flush=True)
    traj_ma = run_trajectory(MAMPPI_V2, dyn, cost, x0, sdim, cdim,
                             max_steps=MAX_STEPS, u_lim=5.0)

    # XY projection
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(traj_mppi[:, 0], traj_mppi[:, 1], "r-", linewidth=1.5,
            alpha=0.85, label="MPPI", zorder=3)
    ax.plot(traj_mppi[-1, 0], traj_mppi[-1, 1], "rx", markersize=8, zorder=4)
    ax.plot(traj_ma[:, 0], traj_ma[:, 1], "b-", linewidth=1.5,
            alpha=0.85, label="MA-MPPI", zorder=3)
    ax.plot(traj_ma[-1, 0], traj_ma[-1, 1], "b^", markersize=7, zorder=4)

    traps_2d = [(cx, cy) for cx, cy, _ in traps]
    finalize_2d(ax, x0[:2], goal[:2], traps_2d, trap_radius=0.5)
    ax.set_title(f"{title} (XY Projection)", fontsize=12)

    fpath = os.path.join(FIG_DIR, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fpath}")
    return traj_mppi, traj_ma


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating MA-MPPI paper figures...")
    print("=" * 50)

    plot_2d_scenario(l2_scenario, "l2_single_trap.png")
    plot_2d_scenario(l4_scenario, "l4_multi_traps.png")
    plot_2d_scenario(l6_scenario, "l6_repeated_traps.png", max_steps=400)
    plot_uav_scenario("uav_3d_xy.png")

    print("=" * 50)
    print("All figures saved to", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
