"""
Main entry point for MA-MPPI experiments.

Usage:
    python main.py                    # Run all benchmarks
    python main.py --experiment trap  # Run specific scenario
    python main.py --experiment gym   # Run Gym environments
"""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="MA-MPPI Experiments")
    parser.add_argument("--experiment", type=str, default="benchmark",
                        choices=["benchmark", "trap", "gym"],
                        help="Experiment to run")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--steps", type=int, default=300, help="Max steps per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.experiment == "benchmark":
        from experiments.benchmark import main as run_benchmark
        run_benchmark()
    elif args.experiment == "trap":
        from experiments.benchmark import run_experiment, print_results, u_trap_scenario
        name, results = run_experiment(u_trap_scenario, args.steps, args.trials)
        print_results(name, results)
    elif args.experiment == "gym":
        from experiments.gym_benchmark import main as run_gym
        run_gym()


if __name__ == "__main__":
    main()
