# Memory-Augmented Potential Field Theory for MPPI Control

A unified framework integrating historical experience into stochastic optimal control through dynamic potential fields that automatically identify and encode topological features of the state space.

Paper: *Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains* (NeurIPS 2025)

---

## Install

```bash
pip install -e .
```

Or use `mapft/` directly:

```python
from mapft import MPPI, MAMPPI
```

## Quick Start

MA-MPPI is a **drop-in replacement** for standard MPPI. Same interface, same observations, zero extra cost function calls.

```python
from mapft import MPPI

# Standard MPPI
controller = MPPI(
    dynamics_fn=my_dynamics,    # x_{t+1} = f(x_t, u_t)
    cost_fn=my_cost,            # c(x_t, u_t)
    state_dim=6, control_dim=3,
    horizon=20, n_samples=500,
    noise_sigma=1.0, lambda_=1.0,
    u_min=np.array([-2, -2, -2]),
    u_max=np.array([2, 2, 2]),
)

x = x0
for t in range(max_steps):
    u = controller.command(x)
    x = dynamics(x, u)
```

**Switch to MA-MPPI** -- just change the import:

```python
from mapft import MAMPPI_Reactive as MAMPPI

# Drop-in replacement: automatically detects traps and escapes
controller = MAMPPI(
    dynamics_fn=my_dynamics,
    cost_fn=my_cost,
    state_dim=6, control_dim=3,
    horizon=20, n_samples=500,
    noise_sigma=1.0, lambda_=1.0,
    u_min=np.array([-2, -2, -2]),
    u_max=np.array([2, 2, 2]),
)

# Exact same control loop -- memory is fully automatic
x = x0
for t in range(max_steps):
    u = controller.command(x)
    x = dynamics(x, u)
```

---

## MA-MPPI Implementations

| Implementation | Import | Strategy | Best For |
|---|---|---|---|
| **MA-MPPI** | `MAMPPI` | State-variance detection, single-timescale | General use |
| **MA-MPPI Reactive** | `MAMPPI_Reactive` | Dual-timescale + directional bias sampling | Trap-heavy environments |
| **MA-MPPI Adaptive** | `MAMPPI_Adaptive` | Progress-aware + online neural parameter tuning | Multi-episode, long-horizon |

```python
from mapft import MAMPPI              # Standard
from mapft import MAMPPI_Reactive     # or MAMPPI_R
from mapft import MAMPPI_Adaptive     # or MAMPPI_A
```

### MA-MPPI Reactive

Dual-timescale detection: fast 5-step detection for immediate reaction + confirmed 15-step detection for persistent features. Includes directional bias sampling that steers a fraction of trajectory perturbations toward escape directions near known traps. Uses both state-variance and cost-variance for stagnation detection, enabling operation in any-dimensional state spaces.

### MA-MPPI Adaptive

Progress-aware detection that only triggers when cost improvement stalls (suppresses false positives). Features have a maximum lifetime and decay between episodes. Integrates an online-learned MLP that predicts optimal temperature, memory weight, and exploration parameters from sampling statistics. Designed for multi-episode scenarios:

```python
controller = MAMPPI_Adaptive(dynamics, cost, ...)

for episode in range(n_episodes):
    controller.new_episode()   # decay old memory, keep learned features
    x = x0
    for t in range(max_steps):
        u = controller.command(x)
        x = dynamics(x, u)
```

---

## Key Insight: Zero Extra Observations

Standard MPPI already samples K trajectories and computes their costs at every step. MA-MPPI extracts landscape topology from these existing statistics:

| Information | Extracted from MPPI sampling |
|---|---|
| Local minimum detection | ESS (effective sample size) drops when all samples are trapped |
| Gradient direction | Weighted mean of perturbations: `sum(w_k * noise_k)` |
| Landscape curvature | Weight entropy: uniform = flat, peaked = sharp |
| Stagnation | Cost coefficient of variation stops decreasing |

No external gradient computation, no Hessian, no extra cost calls. The sampling **is** the observation.

---

## Benchmark Results

### 2D Progressive Difficulty (3 trials each)

| Scenario | MPPI | MA-MPPI Reactive | Improvement |
|---|---|---|---|
| L1: Open field (no trap) | **3/3** (dist 0.32) | 3/3 (dist 0.32) | Baseline |
| L2: Single shallow trap | 0/3 (stuck at 3.60) | **3/3** (dist 0.36) | **0% -> 100%** |
| L3: U-shaped trap | 0/3 (stuck at 1.62) | **2/3** (dist 0.86) | **0% -> 67%** |
| L4: 4 sequential traps | 0/3 (stuck at 6.44) | **3/3** (dist 0.37) | **0% -> 100%** |
| L5: Corridor + dead-end | 3/3 (dist 0.43) | **3/3** (dist 0.40) | -7% distance |
| L6: Repeated traps | 0/3 (stuck at 2.78) | **3/3** (dist 0.32) | **0% -> 100%** |

MA-MPPI achieves **100% escape rate** on 3/4 trap scenarios where standard MPPI has **0% success**.

### UAV 3D Navigation

6D double-integrator UAV with Gaussian cost traps on path:

| | MPPI | MA-MPPI Reactive |
|---|---|---|
| **Success** | 0/6 | **6/6** |
| **Final dist** | 1.43 (stuck) | **0.42** (reached) |

### Trajectory Visualizations

Generate figures locally:

```bash
python experiments/generate_figures.py
```

### Multi-Episode Learning

MA-MPPI accumulates experience across episodes. MPPI shows no improvement:

```
Episode:   1     2     3     4     5
MPPI:    2.37 → 2.37 → 2.37 → 2.37 → 2.38  (no learning)
MA-MPPI: 1.67 → 1.70 → ...                   (30% closer from ep1)
```

---

## Architecture

```
Standard MPPI Sampling (K trajectories, costs S(τ^k))
    │
    ├─ Weights w_k = softmax(-S(τ^k)/λ)
    │
    ├─ Sampling Statistics Extraction (zero extra cost)
    │     ├─ ESS = 1/Σw² → local curvature
    │     ├─ H(w) entropy → landscape flatness
    │     ├─ Σw·δu → gradient direction
    │     └─ CV(cost) → stagnation detection
    │
    ├─ Feature Detection (dual-timescale)
    │     ├─ Fast (5-step): immediate reaction
    │     └─ Confirmed (15-step): persistent features
    │
    ├─ Memory Repository M = {(m_i, r_i, γ_i, κ_i, d_i)}
    │     ├─ Local minima → repulsive potential φ₁
    │     ├─ Low-gradient → directional guide φ₂
    │     └─ High-curvature → saddle potential φ₃
    │
    ├─ Adaptive Potential Field
    │     V(x,M) = α·V_base + (1-α)·V_mem
    │     λ(x,M) = λ₀·(1 + η·(1-α))    (temperature)
    │     Σ_u(x,M) = Σ₀·(1 + μ·(1-α))  (exploration)
    │
    └─ Optimal Control: u* = Σ w_k · u_k
```

---

## RL Baselines

Standalone RL baseline implementations for comparison:

```bash
python baselines/sac.py --env Pendulum-v1 --steps 2000
python baselines/ppo.py --env Pendulum-v1 --steps 2000
python baselines/ddpg.py --env Pendulum-v1 --steps 2000
```

---

## Repository Structure

```
MAPFT_MPPI/
├── mapft/                          # Core library
│   ├── mppi.py                     # Standard MPPI
│   ├── ma_mppi.py                  # MA-MPPI
│   ├── ma_mppi_v2.py               # MA-MPPI Reactive
│   ├── ma_mppi_v3.py               # MA-MPPI Adaptive
│   ├── memory.py                   # Feature detection + memory repository
│   ├── potentials.py               # Type-specific potential fields
│   └── adaptive.py                 # Online parameter learning network
├── baselines/                      # RL comparison methods
│   ├── sac.py                      # Soft Actor-Critic
│   ├── ppo.py                      # Proximal Policy Optimization
│   └── ddpg.py                     # Deep Deterministic Policy Gradient
├── experiments/
│   ├── progressive_test.py         # L1-L8 progressive benchmark
│   ├── benchmark.py                # Basic trap benchmarks
│   ├── environments.py             # UAV + IEEE 39-bus
│   ├── generate_figures.py         # Visualization generation
│   └── run_all.py                  # Complete experiment suite
├── figures/                        # Generated visualizations
├── pyproject.toml
└── main.py
```

## Run Experiments

```bash
# Progressive difficulty benchmark
python experiments/progressive_test.py

# Generate trajectory figures
python experiments/generate_figures.py

# Full benchmark suite
python experiments/run_all.py
```

## License

MIT
