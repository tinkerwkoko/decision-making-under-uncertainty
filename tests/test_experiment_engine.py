"""
Tests for the reusable experiment engine.

Covers:
- Experiment configuration and result objects
- Running single experiments
- Running multiple experiments
- Saving/loading results
- Best performance tracking
"""

import pytest
from experiments.experiment_engine import ExperimentConfig, ExperimentResult, ExperimentEngine


class TestExperimentEngine:
    """Test suite for the experiment engine."""

    def test_experiment_config_creation(self):
        config = ExperimentConfig(
            name="test_ga",
            algorithm="genetic_algorithm",
            environment="gridworld",
        )
        assert config.name == "test_ga"
        assert config.algorithm == "genetic_algorithm"
        assert config.environment == "gridworld"

    def test_experiment_engine_init(self):
        engine = ExperimentEngine()
        assert engine.output_dir is not None
        assert engine.experiments == []

    def test_run_experiment_genetic_algorithm(self):
        engine = ExperimentEngine()
        config = ExperimentConfig(
            name="test_ga_run",
            algorithm="genetic_algorithm",
            environment="gridworld",
            population_size=5,
            num_actions=2,
            num_steps=3,
            generations=5,
        )
        result = engine.run_experiment(config)
        assert isinstance(result, ExperimentResult)
        assert result.fitness >= 0.0
        assert result.elapsed_time >= 0.0

    def test_run_multiple_experiments(self):
        engine = ExperimentEngine()
        configs = [
            ExperimentConfig(
                name="test_1",
                algorithm="genetic_algorithm",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
            ExperimentConfig(
                name="test_2",
                algorithm="q_learning",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
        ]
        results = engine.run_multiple_experiments(configs)
        assert len(results) == 2

    def test_get_best_performance(self):
        engine = ExperimentEngine()
        configs = [
            ExperimentConfig(
                name="test_1",
                algorithm="genetic_algorithm",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
            ExperimentConfig(
                name="test_2",
                algorithm="genetic_algorithm",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
        ]
        engine.run_multiple_experiments(configs)
        best = engine.get_best_performance()
        assert isinstance(best, ExperimentResult)

    def test_list_experiments(self):
        engine = ExperimentEngine()
        configs = [
            ExperimentConfig(
                name="test_1",
                algorithm="genetic_algorithm",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
        ]
        engine.run_multiple_experiments(configs)
        results = engine.list_experiments()
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_save_and_load_results(self):
        engine = ExperimentEngine()
        configs = [
            ExperimentConfig(
                name="test_1",
                algorithm="genetic_algorithm",
                environment="gridworld",
                population_size=5,
                num_actions=2,
                num_steps=3,
                generations=3,
            ),
        ]
        engine.run_multiple_experiments(configs)
        engine.save_results()
        engine.load_results()
        assert len(engine.experiments) == 1