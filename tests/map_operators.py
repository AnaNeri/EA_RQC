from __future__ import annotations

import pytest

from evolution.cluster.operators import crossover, fitness, generate_combinations, mutation


class TestFitness:
    def test_prioritizes_qubit_errors_by_default(self, noise_model, coupling_graph):
        score = fitness([0, 2], noise_model, coupling_graph=coupling_graph, prioritize="qubits")

        assert score == pytest.approx(1.0 / (1.0 + ((0.10 + 0.30) / 2)))

    def test_can_prioritize_coupling_errors(self, noise_model, coupling_graph):
        score = fitness([0, 2], noise_model, coupling_graph=coupling_graph, prioritize="coupling")

        assert score == pytest.approx(1.0 / (1.0 + 1.0))

    def test_can_combine_both_error_sources(self, noise_model, coupling_graph):
        score = fitness([0, 2], noise_model, coupling_graph=coupling_graph, prioritize="both")

        assert score == pytest.approx(1.0 / (1.0 + (((0.10 + 0.30) / 2) + 1.0)))


class TestGenerateCombinations:
    def test_generates_unique_population_of_fixed_size(self):
        population = generate_combinations(population_size=4, device_qubits=6, target_qubits=3, seed=12)

        assert len(population) == 4
        assert len({tuple(individual) for individual in population}) == 4
        assert all(len(individual) == 3 for individual in population)
        assert all(all(0 <= qubit < 6 for qubit in individual) for individual in population)


class TestCrossover:
    def test_preserves_shared_qubits_and_cluster_size(self):
        child1, child2 = crossover([0, 1, 2], [1, 2, 3], device_qubits=5, crossover_rate=1.0, seed=3)

        assert len(child1) == 3
        assert len(child2) == 3
        assert 1 in child1 and 2 in child1
        assert 1 in child2 and 2 in child2

    def test_can_skip_crossover(self):
        child1, child2 = crossover([0, 1, 2], [1, 2, 3], device_qubits=5, crossover_rate=0.0, seed=3)

        assert child1 == [0, 1, 2]
        assert child2 == [1, 2, 3]


class TestMutation:
    def test_keeps_size_and_bounds(self):
        mutated = mutation([0, 1, 2], device_qubits=5, target_qubits=3, mutation_rate=1.0, seed=11)

        assert len(mutated) == 3
        assert all(0 <= qubit < 5 for qubit in mutated)