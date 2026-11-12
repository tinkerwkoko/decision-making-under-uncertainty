"""
Bar chart comparison: GA vs Q-learning vs Value Iteration on GridWorld.

Metrics: success rate, average steps, average reward.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np


def plot_ga_comparison():
    """
    Generate and save a bar chart comparing GA, Q-learning, and Value Iteration.
    """
    # These are example results - in practice, these would come from running the actual experiments
    algorithms = ['Genetic Algorithm', 'Q-Learning', 'Value Iteration']
    
    # Example metrics (these should be replaced with actual experiment results)
    success_rates = [0.85, 0.95, 1.00]      # Success rate
    avg_steps = [12.5, 6.2, 5.0]            # Average steps to goal
    avg_rewards = [-8.5, -5.5, -5.0]        # Average reward (negative = cost)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Colors for each algorithm
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Success rate
    bars = axes[0].bar(algorithms, success_rates, color=colors)
    axes[0].set_title('Success Rate')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.2f}', ha='center', va='bottom')
    
    # Average steps
    bars = axes[1].bar(algorithms, avg_steps, color=colors)
    axes[1].set_title('Average Steps to Goal')
    axes[1].set_ylabel('Steps')
    for bar, val in zip(bars, avg_steps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                     f'{val:.1f}', ha='center', va='bottom')
    
    # Average reward
    bars = axes[2].bar(algorithms, avg_rewards, color=colors)
    axes[2].set_title('Average Reward')
    axes[2].set_ylabel('Reward')
    for bar, val in zip(bars, avg_rewards):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.2, 
                     f'{val:.1f}', ha='center', va='top')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'ga_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Also save data as CSV
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'ga_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Success Rate', 'Avg Steps', 'Avg Reward'])
        for i, alg in enumerate(algorithms):
            writer.writerow([alg, success_rates[i], avg_steps[i], avg_rewards[i]])
    print(f"Saved data to {csv_path}")
    
    plt.close()


if __name__ == "__main__":
    plot_ga_comparison()