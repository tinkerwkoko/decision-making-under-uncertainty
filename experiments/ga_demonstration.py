"""
Single-run Genetic Algorithm demonstration.

Tracks best policy across generations and prints progress.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.ga import GeneticAlgorithm


def run_ga_demo(
    population_size: int = 20,
    num_actions: int = 4,
    num_steps: int = 50,
    mutation_rate: float = 0.1,
    elite_size: int = 2,
    epsilon: float = 0.0,
    generations: int = 20,
    seed: int = 42,
) -> list:
    """
    Run a single GA demonstration.

    Args:
        population_size: Number of individuals in population.
        num_actions: Number of possible actions.
        num_steps: Maximum policy length.
        mutation_rate: Probability of mutating an action.
        elite_size: Number of elite individuals to preserve.
        epsilon: Exploration parameter.
        generations: Number of generations to evolve.
        seed: Random seed for reproducibility.

    Returns:
        Best policy found.
    """
    import random
    random.seed(seed)

    ga = GeneticAlgorithm(
        population_size=population_size,
        num_actions=num_actions,
        num_steps=num_steps,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        epsilon=epsilon,
    )

    print(f"Running GA demo: pop={population_size}, steps={num_steps}, "
          f"mutation={mutation_rate}, elite={elite_size}, gens={generations}")
    print(f"Seed: {seed}")

    best_policy = ga.run()

    # Evaluate final best policy
    fitness = ga.evaluate_fitness(best_policy)
    print(f"Best policy fitness: {fitness:.4f}")
    print(f"Best policy (first 20 actions): {best_policy[:20]}")

    return best_policy


if __name__ == "__main__":
    run_ga_demo()