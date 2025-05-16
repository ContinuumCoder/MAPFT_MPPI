"""
Benchmark utilities for evaluating controller performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from collections import defaultdict

class PerformanceMetrics:
    """Utility for measuring controller performance"""
    
    def __init__(self, output_dir="results/benchmarks"):
        """
        Initialize performance metrics.
        
        Args:
            output_dir: Output directory for benchmark results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics tracked
        self.metrics = {
            'success_rate': [],
            'collision_rate': [],
            'stuck_rate': [],
            'avg_path_length': [],
            'avg_control_energy': [],
            'avg_steps': [],
            'avg_computation_time': [],
            'path_optimality': [],
            'local_minima_escape_rate': []
        }
        
        # Keep track of individual experiment results
        self.experiment_results = []
    
    def record_experiment(self, controller_name, scenario_name, result_dict):
        """
        Record experiment result.
        
        Args:
            controller_name: Name of controller
            scenario_name: Name of scenario
            result_dict: Dictionary of experiment results
        """
        # Add metadata
        result_dict['controller'] = controller_name
        result_dict['scenario'] = scenario_name
        result_dict['timestamp'] = time.time()
        
        # Add to results
        self.experiment_results.append(result_dict)
    
    def compute_metrics(self, controller_name=None, scenario_name=None):
        """
        Compute aggregate metrics.
        
        Args:
            controller_name: Filter by controller name
            scenario_name: Filter by scenario name
            
        Returns:
            metrics: Dictionary of metrics
        """
        # Filter results
        filtered_results = self.experiment_results
        
        if controller_name is not None:
            filtered_results = [r for r in filtered_results if r['controller'] == controller_name]
        
        if scenario_name is not None:
            filtered_results = [r for r in filtered_results if r['scenario'] == scenario_name]
        
        # Compute metrics
        if not filtered_results:
            return None
        
        total = len(filtered_results)
        
        metrics = {
            'success_rate': sum(r['success'] for r in filtered_results) / total * 100,
            'collision_rate': sum(r['collision'] for r in filtered_results) / total * 100,
            'stuck_rate': sum(r.get('stuck', False) for r in filtered_results) / total * 100,
            'avg_path_length': np.mean([r['path_length'] for r in filtered_results]),
            'avg_control_energy': np.mean([r['control_energy'] for r in filtered_results]),
            'avg_steps': np.mean([r['steps'] for r in filtered_results]),
            'avg_computation_time': np.mean([r['time'] for r in filtered_results])
        }
        
        # Compute path optimality if optimal path is available
        optimal_paths = [r.get('optimal_path_length') for r in filtered_results if 'optimal_path_length' in r]
        if optimal_paths:
            actual_paths = [r['path_length'] for r in filtered_results if 'optimal_path_length' in r]
            optimality = np.mean([opt/act for opt, act in zip(optimal_paths, actual_paths)])
            metrics['path_optimality'] = optimality * 100
        
        # Compute local minima escape rate if available
        escape_results = [r.get('escaped_local_minimum') for r in filtered_results if 'escaped_local_minimum' in r]
        if escape_results:
            metrics['local_minima_escape_rate'] = sum(escape_results) / len(escape_results) * 100
        
        return metrics
    
    def generate_comparison_report(self, controllers, scenarios=None, output_file=None):
        """
        Generate comparison report between controllers.
        
        Args:
            controllers: List of controller names
            scenarios: List of scenario names (None for all)
            output_file: Output file name
            
        Returns:
            df: Pandas DataFrame with comparison results
        """
        # Default output file
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'controller_comparison.csv')
        
        # Get unique scenarios if not specified
        if scenarios is None:
            scenarios = list(set(r['scenario'] for r in self.experiment_results))
        
        # Initialize results table
        results = []
        
        # For each scenario
        for scenario in scenarios:
            scenario_results = {'Scenario': scenario}
            
            # For each controller
            for controller in controllers:
                # Compute metrics
                metrics = self.compute_metrics(controller, scenario)
                
                if metrics is None:
                    continue
                
                # Add to results with controller name prefix
                for key, value in metrics.items():
                    scenario_results[f"{controller}_{key}"] = value
            
            results.append(scenario_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Comparison report saved to {output_file}")
        
        return df
    
    def visualize_comparison(self, controllers, metrics_to_plot=None, output_file=None):
        """
        Visualize controller comparison.
        
        Args:
            controllers: List of controller names
            metrics_to_plot: List of metrics to plot (None for default set)
            output_file: Output file name
        """
        # Default metrics
        if metrics_to_plot is None:
            metrics_to_plot = ['success_rate', 'avg_path_length', 'avg_control_energy', 'avg_computation_time']
        
        # Default output file
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'controller_comparison.png')
        
        # Get unique scenarios
        scenarios = list(set(r['scenario'] for r in self.experiment_results))
        
        # Create figure
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        # For each metric
        for i, metric in enumerate(metrics_to_plot):
            # Collect data
            data = defaultdict(list)
            
            for scenario in scenarios:
                for controller in controllers:
                    metrics = self.compute_metrics(controller, scenario)
                    
                    if metrics is None or metric not in metrics:
                        continue
                    
                    data[controller].append(metrics[metric])
            
            # Create bar chart
            ax = axes[i]
            
            # Set x positions for bars
            x = np.arange(len(scenarios))
            width = 0.8 / len(controllers)
            
            # Plot bars for each controller
            for j, controller in enumerate(controllers):
                if controller in data:
                    ax.bar(x + j*width - 0.4 + width/2, data[controller], width, label=controller)
            
            # Set labels
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"Comparison visualization saved to {output_file}")
    
    def save_results(self, filename=None):
        """
        Save results to file.
        
        Args:
            filename: Output filename
        """
        if filename is None:
            filename = os.path.join(self.output_dir, 'benchmark_results.npy')
        
        np.save(filename, self.experiment_results)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename):
        """
        Load results from file.
        
        Args:
            filename: Input filename
        """
        self.experiment_results = np.load(filename, allow_pickle=True).tolist()
        print(f"Loaded {len(self.experiment_results)} results from {filename}")
