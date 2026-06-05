from __future__ import annotations

from evolution.cluster.selection import selection_new_generation, selection_rank


class TestSelectionRank:
    def test_deterministic_ranking_picks_best_two(self):
        population = [[0, 1], [2, 3], [4, 5]]
        fitness_scores = [0.2, 0.9, 0.5]

        parent1, parent2 = selection_rank(population, fitness_scores, "deterministic")

        assert parent1 == [2, 3]
        assert parent2 == [4, 5]

    def test_ranked_selection_returns_population_members(self):
        population = [[0, 1], [2, 3], [4, 5]]
        fitness_scores = [0.2, 0.9, 0.5]

        parent1, parent2 = selection_rank(population, fitness_scores, "linear", seed=8)

        assert parent1 in population
        assert parent2 in population


class TestSelectionNewGeneration:
    def test_merges_population_and_children_without_duplicates(self):
        population = [[0, 1], [2, 3]]
        children = [[2, 3], [4, 5]]
        fitness_scores = [0.2, 0.9]
        children_fitness = [0.9, 0.7]

        next_generation = selection_new_generation(
            population,
            children,
            fitness_scores,
            children_fitness,
            device_qubits=6,
            seed=5,
        )

        assert len(next_generation) == 2
        assert len({tuple(individual) for individual in next_generation}) == 2
        assert [2, 3] in next_generation