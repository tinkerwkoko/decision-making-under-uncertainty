"""
Run GA vs Q-learning vs Value Iteration comparison.

Placeholder results — replace with actual experiment runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.ga import run_genetic_algorithm


def run_ga():
    """
    Run Genetic Algorithm and return best policy.
    """
    return run_genetic_algorithm(
        population_size=20,
        num_actions=4,
        num_steps=50,
        mutation_rate=0.1,
        elite_size=2,
        epsilon=0.0,
        generations=20,
    )


def run_q_learning():
    """
    Placeholder for Q-learning evaluation.
    Replace with train_q_learning / evaluate_q_learning calls.
    """
    raise NotImplementedError("Q-learning runner not yet implemented")


def run_value_iteration():
    """
    Placeholder for Value Iteration evaluation.
    Replace with value_iteration_evaluation calls.
    """
    raise NotImplementedError("Value iteration runner not yet implemented")


if __name__ == "__main__":
    print("Running GA comparison (GA only placeholders)...")