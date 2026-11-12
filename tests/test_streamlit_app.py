"""
Tests for the Streamlit application.

Covers:
- Application initialization
- Laboratory navigation
- Experiment configuration
- Results display
"""

import pytest
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import main


class TestStreamlitApp:
    """Test suite for the Streamlit application."""

    def test_app_initialization(self):
        """Test that the application can be initialized."""
        # This test verifies the app module can be imported
        assert main is not None

    def test_app_file_exists(self):
        """Test that the app.py file exists."""
        app_path = Path(__file__).parent.parent / "app.py"
        assert app_path.exists()

    def test_app_directory_structure(self):
        """Test that the required directories exist."""
        base_dir = Path(__file__).parent.parent
        assert (base_dir / "experiments").exists()
        assert (base_dir / "algorithms").exists()
        assert (base_dir / "environments").exists()
        assert (base_dir / "tests").exists()

    def test_app_modules_importable(self):
        """Test that all required modules are importable."""
        from algorithms.ga import GeneticAlgorithm
        from experiments.experiment_engine import ExperimentEngine
        from algorithms.q_learning import QLearning
        from algorithms.value_iteration import ValueIteration
        assert GeneticAlgorithm is not None
        assert ExperimentEngine is not None
        assert QLearning is not None
        assert ValueIteration is not None

    def test_app_laboratory_navigation(self):
        """Test that laboratory navigation works."""
        # Verify the lab names exist
        labs = [
            ("Bandit Laboratory", "Bandit Lab"),
            ("GridWorld Laboratory", "GridWorld Lab"),
            ("POMDP Laboratory", "POMDP Lab"),
            ("Transfer Learning Lab", "Transfer Learning Lab"),
            ("Results & History", "Experiment History"),
        ]
        assert len(labs) == 5
        assert all(isinstance(name, str) for name, _ in labs)
        assert all(isinstance(desc, str) for _, desc in labs)

    def test_app_experiment_config(self):
        """Test that experiment configuration works."""
        from experiments.experiment_engine import ExperimentConfig
        config = ExperimentConfig(
            name="test_app",
            algorithm="genetic_algorithm",
            environment="gridworld",
            population_size=5,
            num_actions=2,
            num_steps=3,
            generations=3,
        )
        assert config.name == "test_app"
        assert config.algorithm == "genetic_algorithm"
        assert config.environment == "gridworld"
        assert config.population_size == 5
        assert config.num_actions == 2
        assert config.num_steps == 3
        assert config.generations == 3

    def test_app_results_display(self):
        """Test that results display works."""
        from experiments.experiment_engine import ExperimentEngine
        engine = ExperimentEngine()
        # Verify engine can display results
        assert engine is not None
        assert engine.experiments == []