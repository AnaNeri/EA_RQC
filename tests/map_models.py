from __future__ import annotations

from evolution.cluster.models import EvolutionaryMappingResult


class TestEvolutionaryMappingResult:
    def test_dataclass_fields(self):
        result = EvolutionaryMappingResult(
            population=[[0, 1], [2, 3]],
            fitness_scores=[1.5, 0.8],
            best_cluster=[0, 1],
            best_fitness=1.5,
            generations_run=4,
        )

        assert result.population == [[0, 1], [2, 3]]
        assert result.fitness_scores == [1.5, 0.8]
        assert result.best_cluster == [0, 1]
        assert result.best_fitness == 1.5
        assert result.generations_run == 4
        assert result.top_clusters == []
        assert result.top_fitness_scores == []