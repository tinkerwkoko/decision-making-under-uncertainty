"""
Reusable simulation engine for the Decision-Making Under Uncertainty toolkit.

Provides a unified interface to run experiments across different algorithms,
environments, and hyperparameters. Supports programmatic and UI-driven execution.
"""

import json
import csv
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    algorithm: str
    environment: str
    population_size: int = 20
    num_actions: int = 4
    num_steps: int = 50
    mutation_rate: float = 0.1
    elite_size: int = 2
    epsilon: float = 0.0
    generations: int = 20
    seed: int = 42
    output_dir: str = "experiments"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    best_policy: List[int]
    fitness: float
    success_rate: float
    avg_steps: float
    avg_reward: float
    generations: int
    elapsed_time: float


class ExperimentEngine:
    """Reusable engine for running experiments across the toolkit."""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[ExperimentResult] = []

    def run_experiment(
        self,
        config: ExperimentConfig,
        seed: Optional[int] = None,
    ) -> ExperimentResult:
        """
        Run a single experiment and store the results.

        Args:
            config: Experiment configuration
            seed: Random seed for reproducibility

        Returns:
            ExperimentResult with all metrics
        """
        import random
        import time

        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)

        start_time = time.time()

        # Import and run the algorithm
        from algorithms.ga import GeneticAlgorithm
        from algorithms.q_learning import QLearning
        from algorithms.value_iteration import ValueIteration

        # Dispatch to the appropriate algorithm
        if config.algorithm == "genetic_algorithm":
            ga = GeneticAlgorithm(
                population_size=config.population_size,
                num_actions=config.num_actions,
                num_steps=config.num_steps,
                mutation_rate=config.mutation_rate,
                elite_size=config.elite_size,
                epsilon=config.epsilon,
                generations=config.generations,
            )
            best_policy = ga.run()
            fitness = best_policy
        elif config.algorithm == "q_learning":
            q_learn = QLearning(
                population_size=config.population_size,
                num_actions=config.num_actions,
                num_steps=config.num_steps,
                epsilon=config.epsilon,
                generations=config.generations,
            )
            best_policy = q_learn.run()
            fitness = best_policy
        elif config.algorithm == "value_iteration":
            vi = ValueIteration(
                population_size=config.population_size,
                num_actions=config.num_actions,
                num_steps=config.num_steps,
                epsilon=config.epsilon,
                generations=config.generations,
            )
            best_policy = vi.run()
            fitness = best_policy
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        elapsed = time.time() - start_time

        result = ExperimentResult(
            config=config,
            best_policy=best_policy,
            fitness=fitness,
            success_rate=0.0,  # Would be computed from environment
            avg_steps=0.0,
            avg_reward=0.0,
            generations=config.generations,
            elapsed_time=elapsed,
        )

        self.experiments.append(result)
        return result

    def run_multiple_experiments(
        self,
        configs: List[ExperimentConfig],
        seed: Optional[int] = None,
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments in parallel.

        Args:
            configs: List of ExperimentConfig objects
            seed: Global random seed

        Returns:
            List of ExperimentResult objects
        """
        results = []
        for config in configs:
            result = self.run_experiment(config, seed=seed)
            results.append(result)
        return results

    def save_results(self, filename: str = "experiment_results.json"):
        """Save all experiment results to a JSON file."""
        data = [asdict(exp) for exp in self.experiments]
        with open(self.output_dir / filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {self.output_dir / filename}")

    def load_results(self, filename: str = "experiment_results.json"):
        """Load experiment results from a JSON file."""
        if not Path(self.output_dir).exists():
            print("No experiment results found.")
            return []

        with open(self.output_dir / filename, "r") as f:
            data = json.load(f)

        self.experiments = [ExperimentResult(**d) for d in data]
        print(f"Loaded {len(self.experiments)} experiments from {filename}")
        return self.experiments

    def get_best_performance(self) -> Dict[str, Any]:
        """Get the experiment with the best fitness."""
        if not self.experiments:
            return {}
        return max(self.experiments, key=lambda e: e.fitness)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with their metrics."""
        return [asdict(exp) for exp in self.experiments]
