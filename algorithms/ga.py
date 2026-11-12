"""
Genetic Algorithm implementation for policy optimisation in GridWorld.
"""

import random
from typing import List, Tuple, Dict, Any


class GeneticAlgorithm:
    """Genetic Algorithm for optimising GridWorld policies."""

    def __init__(
        self,
        population_size: int = 20,
        num_actions: int = 4,
        num_steps: int = 50,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        epsilon: float = 0.0,
    ):
        """
        Initialise the genetic algorithm.

        Args:
            population_size: Number of individuals in the population.
            num_actions: Number of possible actions per state.
            num_steps: Maximum length of the policy (action sequence).
            mutation_rate: Probability of mutating an action.
            elite_size: Number of best individuals to carry over to next generation.
            epsilon: Exploration parameter (0.0 = greedy, >0.0 = stochastic).
        """
        self.population_size = population_size
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.epsilon = epsilon

        # Create initial random population
        self.population: List[List[int]] = []
        for _ in range(population_size):
            policy = [random.randint(0, self.num_actions - 1) for _ in range(self.num_steps)]
            self.population.append(policy)

    def evaluate_fitness(self, policy: List[int]) -> float:
        """
        Evaluate the fitness of a policy by running it on GridWorld.

        Args:
            policy: Action sequence of length num_steps.

        Returns:
            Fitness score (higher is better).
        """
        # Greedy evaluation: simulate the policy and count successful steps
        steps = 0
        for action in policy:
            if steps < self.num_steps:
                success = random.random() > 0.3
                if success:
                    steps += 1
        
        # Normalize fitness by maximum possible steps
        max_steps = self.num_steps
        return min(steps / max_steps, 1.0)

    def selection(self) -> List[List[int]]:
        """
        Select parents using tournament selection.

        Returns:
            List of selected parent policies.
        """
        selected = []
        for _ in range(self.population_size // 2):
            tournament = random.sample(self.population, 3)
            best = max(tournament, key=self.evaluate_fitness)
            selected.append(best)
        return selected

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform uniform crossover between two parents.

        Args:
            parent1: First parent policy.
            parent2: Second parent policy.

        Returns:
            Child policy resulting from uniform crossover.
        """
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def evolve(self, generations: int = 20) -> List[List[int]]:
        """
        Run the genetic algorithm for a fixed number of generations.

        Args:
            generations: Number of generations to evolve.

        Returns:
            Best policy found during evolution.
        """
        best_policy = None
        best_fitness = -1.0

        for gen in range(generations):
            # Evaluate all policies
            fitnesses = [(policy, self.evaluate_fitness(policy)) for policy in self.population]
            # Sort by fitness (descending)
            fitnesses.sort(key=lambda x: x[1], reverse=True)
            
            # Select parents
            parents = self.selection()
            
            # Create next generation
            new_population = []
            # Elitism: keep the best individuals
            for i, (policy, _) in enumerate(fitnesses):
                if i < self.elite_size:
                    new_population.append(policy)
            
            # Fill the rest with offspring
            while len(new_population) < self.population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            # Check for improvement
            current_best = max(self.population, key=self.evaluate_fitness)
            current_fitness = self.evaluate_fitness(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_policy = current_best
        
        return best_policy

    def run(self) -> List[int]:
        """
        Run the full genetic algorithm and return the best policy.

        Returns:
            Best policy found.
        """
        return self.evolve()


def run_genetic_algorithm(
    population_size: int = 20,
    num_actions: int = 4,
    num_steps: int = 50,
    mutation_rate: float = 0.1,
    elite_size: int = 2,
    epsilon: float = 0.0,
    generations: int = 20,
) -> List[int]:
    """
    Convenience function to run the genetic algorithm.

    Args:
        population_size: Number of individuals.
        num_actions: Number of actions per state.
        num_steps: Policy length.
        mutation_rate: Mutation probability.
        elite_size: Number of elites.
        epsilon: Exploration parameter.
        generations: Number of generations.

    Returns:
        Best policy found.
    """
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_actions=num_actions,
        num_steps=num_steps,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        epsilon=epsilon,
    )
    return ga.run()