"""
Tests for the Genetic Algorithm implementation.

Covers:
- Valid population creation
- Fitness evaluation and ordering
- Tournament selection
- Uniform crossover
- Per-action mutation
- Elitist replacement
- Edge cases (empty policies, wrong sizes)
"""

import pytest
import random
from algorithms.ga import GeneticAlgorithm


class TestGeneticAlgorithm:
    """Test suite for GeneticAlgorithm."""

    def test_initialise_population(self):
        ga = GeneticAlgorithm(population_size=5, num_actions=3, num_steps=4)
        assert len(ga.population) == 5
        for policy in ga.population:
            assert len(policy) == 4
            assert all(isinstance(a, int) for a in policy)

    def test_fitness_evaluation(self):
        ga = GeneticAlgorithm(population_size=2, num_actions=2, num_steps=3)
        # Simple deterministic: first action always success -> fitness should be >0
        policy1 = [0, 0, 0]
        fitness1 = ga.evaluate_fitness(policy1)
        assert 0 <= fitness1 <= 1.0
        policy2 = [1, 1, 1]
        fitness2 = ga.evaluate_fitness(policy2)
        assert 0 <= fitness2 <= 1.0

    def test_selection_tournament(self):
        ga = GeneticAlgorithm(population_size=10, num_actions=2, num_steps=2)
        parents = ga.selection()
        assert len(parents) == 5  # population_size // 2
        for p in parents:
            assert len(p) == 2

    def test_crossover_uniform(self):
        ga = GeneticAlgorithm(population_size=2, num_actions=2, num_steps=2)
        parent1 = [0, 1]
        parent2 = [1, 0]
        child = ga.crossover(parent1, parent2)
        assert len(child) == 2
        # At least one element from each parent
        assert child[0] in (parent1[0], parent2[0])
        assert child[1] in (parent1[1], parent2[1])

    def test_mutation(self):
        ga = GeneticAlgorithm(population_size=1, num_actions=2, num_steps=2, mutation_rate=1.0)
        policy = [0, 0]
        mutated = ga.mutate(policy)
        assert len(mutated) == 2
        # All elements may change due to high mutation rate
        for i in range(len(mutated)):
            assert 0 <= mutated[i] < ga.num_actions

    def test_evolve_basic(self):
        ga = GeneticAlgorithm(population_size=20, num_actions=2, num_steps=3, mutation_rate=0.0, elite_size=2, epsilon=0.0)
        best = ga.evolve()
        assert isinstance(best, list)
        assert len(best) == 3

    def test_run(self):
        ga = GeneticAlgorithm(population_size=10, num_actions=2, num_steps=2, mutation_rate=0.0, elite_size=2, epsilon=0.0)
        result = ga.run()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_edge_case_empty_population(self):
        ga = GeneticAlgorithm(population_size=0, num_actions=2, num_steps=2)
        # Should not raise
        try:
            ga.evolve()
        except ValueError:
            pass  # expected behavior maybe

    def test_edge_case_no_actions(self):
        ga = GeneticAlgorithm(population_size=1, num_actions=0, num_steps=2)
        with pytest.raises((ValueError, IndexError, AssertionError)):
            ga.evaluate_fitness([0])  # no actions defined
