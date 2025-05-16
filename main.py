"""
Main script for running Memory-Augmented Potential Field experiments.
"""
import argparse
import numpy as np
import os
import time
from examples.trap_navigation import run_comparison
from utils.common import create_directory

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Memory-Augmented Potential Field Framework")
    parser.add_argument("--experiment", type=str, default="trap", 
                        choices=["trap", "maze", "obstacles"],
                        help="Experiment to run")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--steps", type=int, default=None, help="Maximum steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create results directory
    create_directory("results")
    
    # Run experiment
    if args.experiment == "trap":
        print("Running trap navigation experiment...")
        standard_result, ma_result = run_comparison()
    elif args.experiment == "maze":
        print("Maze experiment not implemented yet.")
    elif args.experiment == "obstacles":
        print("Obstacles experiment not implemented yet.")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
